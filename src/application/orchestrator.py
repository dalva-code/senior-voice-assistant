import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any

from loguru import logger

from src.application.context_manager import ContextManager
from src.application.contact_manager import ContactManager
from src.application.inbox_manager import InboxManager
from src.application.intent_router import IntentRouter
from src.application.state_machine import VoiceStateMachine
from src.application.tools.weather_tool import get_current_weather
from src.domain.interfaces.audio_io import AudioCaptureInterface
from src.domain.interfaces.llm import LLMInterface
from src.domain.interfaces.messaging import MessagingBridgeInterface
from src.domain.interfaces.persistence import MessageRepositoryInterface
from src.domain.interfaces.stt import STTInterface
from src.domain.interfaces.tts import TTSInterface
from src.domain.interfaces.vad import VADInterface
from src.domain.interfaces.wakeword import WakeWordInterface
from src.domain.models import (
    AudioChunk,
    IncomingMessage,
    IntentType,
    SpeechSegment,
    StoredVoiceMessage,
    VoiceState,
)


class VoiceOrchestrator:
    _MESSAGE_RESPONSE_TIMEOUT_SECONDS = 10.0
    _BARGE_IN_VAD_THRESHOLD = 0.90

    def __init__(
        self,
        audio_capture: AudioCaptureInterface,
        vad: VADInterface,
        wakeword: WakeWordInterface,
        stt: STTInterface,
        llm: LLMInterface,
        tts: TTSInterface,
        messaging: MessagingBridgeInterface,
        repository: MessageRepositoryInterface,
        contact_manager: ContactManager,
        sample_rate: int,
        silence_duration_seconds: float = 2.0,
        wake_word_name: str = "alexa",
        max_context_turns: int = 6,
        follow_up_timeout_seconds: float = 6.0,
        barge_in_enabled: bool = False,
        barge_in_min_speech_duration_ms: int = 350,
        post_tts_guard_seconds: float = 0.4,
        default_latitude: float = 28.1235,
        default_longitude: float = -15.4363,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate debe ser positivo")
        if silence_duration_seconds <= 0.0:
            raise ValueError("silence_duration_seconds debe ser positivo")
        if follow_up_timeout_seconds <= 0.0:
            raise ValueError("follow_up_timeout_seconds debe ser positivo")
        if barge_in_min_speech_duration_ms <= 0:
            raise ValueError("barge_in_min_speech_duration_ms debe ser positivo")
        if post_tts_guard_seconds < 0.0:
            raise ValueError("post_tts_guard_seconds no puede ser negativo")
        if not -90.0 <= default_latitude <= 90.0:
            raise ValueError("default_latitude está fuera de rango")
        if not -180.0 <= default_longitude <= 180.0:
            raise ValueError("default_longitude está fuera de rango")

        self._audio_capture = audio_capture
        self._vad = vad
        self._wakeword = wakeword
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._messaging = messaging
        self._repository = repository
        self._contact_manager = contact_manager
        self._sample_rate = sample_rate
        self._silence_duration_seconds = silence_duration_seconds
        self._wake_word_name = wake_word_name
        self._follow_up_timeout_seconds = follow_up_timeout_seconds
        self._barge_in_enabled = barge_in_enabled
        self._barge_in_min_seconds = barge_in_min_speech_duration_ms / 1_000
        self._post_tts_guard_seconds = post_tts_guard_seconds
        self._default_latitude = default_latitude
        self._default_longitude = default_longitude

        self._state_machine = VoiceStateMachine()
        self._context = ContextManager(max_context_turns)
        self._inbox = InboxManager()
        self._intent_router = IntentRouter(contact_manager)
        self._awaiting_contact_selection = False
        self._active_message: IncomingMessage | None = None

        self._recording = bytearray()
        self._speech_started = False
        self._silence_seconds = 0.0
        self._capture_enabled = False
        self._capture_deadline: float | None = None
        self._follow_up_deadline: float | None = None

        self._barge_in_audio = bytearray()
        self._barge_in_speech_seconds = 0.0
        self._barge_target: VoiceState | None = None
        self._conversation_task: asyncio.Task[None] | None = None
        self._playback_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> VoiceState:
        return self._state_machine.state

    @staticmethod
    def _chunk_duration(chunk: AudioChunk) -> float:
        return len(chunk.data) / (2 * chunk.sample_rate)

    def _reset_barge_in(self) -> None:
        self._barge_in_audio.clear()
        self._barge_in_speech_seconds = 0.0

    def _reset_capture(self) -> None:
        self._recording.clear()
        self._speech_started = False
        self._silence_seconds = 0.0
        self._capture_enabled = False
        self._capture_deadline = None

    @staticmethod
    def _stored_to_incoming(message: StoredVoiceMessage) -> IncomingMessage:
        return IncomingMessage(
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            is_voice=True,
            audio_bytes=message.audio_bytes,
            duration=message.duration,
            stored_message_id=message.id,
            received_at=message.created_at,
        )

    def _transition_to_idle(
        self,
        *,
        clear_context: bool = False,
        clear_message: bool = False,
    ) -> None:
        if self.state is not VoiceState.IDLE:
            self._state_machine.transition_to(VoiceState.IDLE)
        self._reset_capture()
        self._reset_barge_in()
        self._follow_up_deadline = None
        self._awaiting_contact_selection = False
        if clear_context:
            self._context.clear()
        if clear_message:
            self._active_message = None

    def _begin_capture(
        self,
        initial_audio: bytes = b"",
        *,
        contains_speech: bool = False,
        timeout_seconds: float | None = None,
        include_pre_roll: bool = False,
    ) -> None:
        pre_roll = (
            self._audio_capture.get_pre_roll_audio()
            if contains_speech and include_pre_roll
            else b""
        )
        self._recording = bytearray(pre_roll + initial_audio)
        self._speech_started = contains_speech
        self._silence_seconds = 0.0
        self._capture_enabled = True
        self._capture_deadline = (
            asyncio.get_running_loop().time() + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        self._follow_up_deadline = None

    def _build_segment(self) -> SpeechSegment:
        segment = SpeechSegment(
            audio_data=bytes(self._recording),
            duration=len(self._recording) / (2 * self._sample_rate),
        )
        self._reset_capture()
        logger.bind(
            component="orchestrator",
            event="speech_complete",
            duration=segment.duration,
            state=self.state.value,
        ).info("Segmento de voz completado")
        return segment

    def _start_workflow(
        self,
        coroutine: Coroutine[Any, Any, None],
        *,
        name: str,
    ) -> None:
        if self._conversation_task is not None:
            raise RuntimeError("Ya existe un flujo conversacional activo")
        self._conversation_task = asyncio.create_task(coroutine, name=name)

    async def _apply_post_tts_guard(self) -> None:
        logger.bind(
            component="orchestrator",
            event="post_tts_guard",
            duration_seconds=self._post_tts_guard_seconds,
        ).debug("Aplicando guarda acústica posterior a la reproducción")
        if self._post_tts_guard_seconds > 0.0:
            await asyncio.sleep(self._post_tts_guard_seconds)
        self._reset_capture()
        self._reset_barge_in()
        self._vad.reset()

    async def _run_playback(
        self,
        coroutine: Coroutine[Any, Any, None],
        *,
        barge_target: VoiceState | None,
        name: str,
    ) -> bool:
        playback_task = asyncio.create_task(coroutine, name=name)
        self._playback_task = playback_task
        self._barge_target = barge_target
        self._vad.reset()
        self._reset_barge_in()
        try:
            await playback_task
            await self._apply_post_tts_guard()
            return True
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            logger.bind(
                component="orchestrator",
                event="playback_interrupted",
            ).info("Reproducción interrumpida por el usuario")
            return False
        finally:
            if self._playback_task is playback_task:
                self._playback_task = None
                self._barge_target = None

    async def _speak(
        self,
        text: str,
        *,
        barge_target: VoiceState | None = None,
        name: str = "tts-playback",
    ) -> bool:
        synthesized_audio = await self._tts.synthesize(text)
        return await self._run_playback(
            self._tts.play(synthesized_audio),
            barge_target=barge_target,
            name=name,
        )

    async def _handle_idle_chunk(self, chunk: AudioChunk) -> None:
        wake_detected = await asyncio.to_thread(
            self._wakeword.detect,
            chunk.data,
        )
        if not wake_detected:
            return
        self._state_machine.transition_to(VoiceState.WAKE_DETECTED)
        logger.bind(
            component="wakeword",
            event="wake_detected",
            wake_word=self._wake_word_name,
        ).info('Palabra de activación detectada: "{}"', self._wake_word_name)
        self._vad.reset()
        contains_speech = await asyncio.to_thread(self._vad.is_speech, chunk.data)
        self._state_machine.transition_to(VoiceState.LISTENING)
        self._begin_capture(
            chunk.data if contains_speech else b"",
            contains_speech=contains_speech,
            include_pre_roll=True,
        )

    async def _handle_capture_timeout(self) -> None:
        expired_state = self.state
        self._reset_capture()
        if expired_state is VoiceState.PROMPTING_MESSAGE:
            self._start_workflow(
                self._decline_active_message(),
                name="message-prompt-timeout",
            )
        elif expired_state is VoiceState.PROMPTING_REPLY:
            self._start_workflow(
                self._cancel_reply(),
                name="reply-prompt-timeout",
            )

    async def _handle_capture_chunk(self, chunk: AudioChunk) -> None:
        if not self._capture_enabled:
            return
        if (
            self._capture_deadline is not None
            and asyncio.get_running_loop().time() >= self._capture_deadline
        ):
            await self._handle_capture_timeout()
            return

        contains_speech = await asyncio.to_thread(self._vad.is_speech, chunk.data)
        if not self._speech_started:
            if not contains_speech:
                return
            self._recording = bytearray(
                self._audio_capture.get_pre_roll_audio() + chunk.data
            )
            self._speech_started = True
            self._silence_seconds = 0.0
            return

        self._recording.extend(chunk.data)
        if contains_speech:
            self._silence_seconds = 0.0
            return

        self._silence_seconds += self._chunk_duration(chunk)
        if self._silence_seconds < self._silence_duration_seconds:
            return

        capture_state = self.state
        segment = self._build_segment()
        if capture_state is VoiceState.LISTENING:
            self._state_machine.transition_to(VoiceState.THINKING)
            self._start_workflow(
                self._run_conversation(segment),
                name="voice-conversation",
            )
        elif capture_state is VoiceState.PROMPTING_MESSAGE:
            self._start_workflow(
                self._process_message_permission(segment),
                name="message-permission",
            )
        elif capture_state is VoiceState.PROMPTING_REPLY:
            self._start_workflow(
                self._process_reply_permission(segment),
                name="reply-permission",
            )
        elif capture_state is VoiceState.RECORDING_REPLY:
            self._state_machine.transition_to(VoiceState.SENDING_REPLY)
            self._start_workflow(
                self._send_reply(segment.audio_data),
                name="send-recorded-reply",
            )

    async def _run_conversation(self, segment: SpeechSegment) -> None:
        try:
            transcript = (
                await self._stt.transcribe(segment.audio_data, self._sample_rate)
            ).strip()
            logger.bind(component="stt", event="transcription").info(
                '[STT] Usuario dijo: "{}"',
                transcript,
            )
            if not transcript:
                self._transition_to_idle()
                return

            self._context.add_user_message(transcript)
            intent = await self._intent_router.classify_intent_async(transcript)
            logger.bind(
                component="intent_router",
                intent=intent.value,
                state=VoiceState.THINKING.value,
            ).info("Intención de conversación clasificada")
            if intent in {
                IntentType.QUERY_INBOX,
                IntentType.PLAY_PENDING_MESSAGE,
            }:
                await self._query_persistent_inbox()
                return
            if intent is IntentType.PLAY_CONTACT_MESSAGE:
                await self._play_contact_message(transcript)
                return
            if intent is IntentType.CHECK_WEATHER:
                response = await get_current_weather(
                    self._default_latitude,
                    self._default_longitude,
                )
            else:
                response = (
                    await self._llm.generate_response(
                        transcript,
                        self._context.get_messages_for_llm(),
                    )
                ).strip()
            logger.bind(
                component="assistant",
                event="response",
                source=(
                    "weather"
                    if intent is IntentType.CHECK_WEATHER
                    else "llm"
                ),
            ).info(
                '[ASSISTANT] Respuesta: "{}"',
                response,
            )
            if not response:
                self._transition_to_idle()
                return

            self._context.add_assistant_message(response)
            self._state_machine.transition_to(VoiceState.SPEAKING)
            completed = await self._speak(
                response,
                barge_target=VoiceState.LISTENING,
            )
            if completed and self.state is VoiceState.SPEAKING:
                self._enter_follow_up_mode()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.bind(component="orchestrator").exception(
                "Falló el ciclo conversacional"
            )
            self._transition_to_idle()

    async def _query_persistent_inbox(self) -> None:
        unread_messages = await self._repository.get_unread_messages()
        if not unread_messages:
            self._state_machine.transition_to(VoiceState.SPEAKING)
            completed = await self._speak(
                "No tienes ningún mensaje nuevo pendiente.",
                name="empty-inbox",
            )
            if completed:
                self._transition_to_idle()
            return

        if len(unread_messages) == 1:
            message = unread_messages[0]
            self._active_message = self._stored_to_incoming(message)
            self._awaiting_contact_selection = False
            self._state_machine.transition_to(VoiceState.PROMPTING_MESSAGE)
            completed = await self._speak(
                f"Tienes un mensaje sin escuchar de {message.sender_name}. "
                "¿Quieres que te lo ponga?",
                barge_target=VoiceState.PROMPTING_MESSAGE,
                name="single-unread-message",
            )
            if completed and self.state is VoiceState.PROMPTING_MESSAGE:
                self._begin_capture(
                    timeout_seconds=self._MESSAGE_RESPONSE_TIMEOUT_SECONDS
                )
            return

        sender_names = list(
            dict.fromkeys(message.sender_name for message in unread_messages)
        )
        names_text = ", ".join(sender_names)
        self._awaiting_contact_selection = True
        self._state_machine.transition_to(VoiceState.PROMPTING_MESSAGE)
        completed = await self._speak(
            f"Tienes {len(unread_messages)} mensajes pendientes: "
            f"de {names_text}. ¿De quién quieres escucharlo?",
            barge_target=VoiceState.PROMPTING_MESSAGE,
            name="multiple-unread-messages",
        )
        if completed and self.state is VoiceState.PROMPTING_MESSAGE:
            self._begin_capture(
                timeout_seconds=self._MESSAGE_RESPONSE_TIMEOUT_SECONDS
            )

    async def _play_contact_message(self, transcript: str) -> None:
        contact = await self._contact_manager.resolve_contact(transcript)
        if contact is None:
            self._state_machine.transition_to(VoiceState.SPEAKING)
            completed = await self._speak(
                "No he podido identificar el contacto.",
                name="unknown-contact",
            )
            if completed:
                self._transition_to_idle()
            return

        stored_message = await self._repository.get_unread_by_sender_id(
            contact.platform_id
        )
        if stored_message is None:
            self._state_machine.transition_to(VoiceState.SPEAKING)
            completed = await self._speak(
                f"No tienes mensajes nuevos de {contact.canonical_name}.",
                name="no-contact-messages",
            )
            if completed:
                self._transition_to_idle()
            return

        pending_message = self._stored_to_incoming(stored_message)
        self._active_message = pending_message
        self._state_machine.transition_to(VoiceState.PLAYING_MESSAGE)
        logger.bind(
            component="inbox",
            event="pending_message_playback",
            sender_id=pending_message.sender_id,
            message_id=stored_message.id,
        ).info("Reproduciendo mensaje pendiente")
        await self._play_active_message_and_prompt_reply()

    def _enter_follow_up_mode(self) -> None:
        self._state_machine.transition_to(VoiceState.AWAITING_FOLLOWUP)
        self._vad.reset()
        self._follow_up_deadline = (
            asyncio.get_running_loop().time() + self._follow_up_timeout_seconds
        )
        logger.bind(
            component="orchestrator",
            event="awaiting_followup",
            timeout_seconds=self._follow_up_timeout_seconds,
        ).info("Esperando una pregunta de seguimiento")

    async def _handle_follow_up_chunk(self, chunk: AudioChunk) -> None:
        deadline = self._follow_up_deadline
        if deadline is None:
            raise RuntimeError("El modo follow-up no tiene plazo")
        if asyncio.get_running_loop().time() >= deadline:
            self._transition_to_idle(clear_context=True)
            return
        contains_speech = await asyncio.to_thread(self._vad.is_speech, chunk.data)
        if contains_speech:
            self._state_machine.transition_to(VoiceState.LISTENING)
            self._begin_capture(
                chunk.data,
                contains_speech=True,
                include_pre_roll=True,
            )

    async def _announce_active_message(self, message: IncomingMessage) -> None:
        prompt = (
            f"{message.sender_name} te ha enviado un mensaje de voz. "
            "¿Quieres que te lo ponga?"
            if message.is_voice
            else f"{message.sender_name} te ha escrito un mensaje. "
            "¿Quieres que te lo lea?"
        )
        try:
            completed = await self._speak(
                prompt,
                barge_target=VoiceState.PROMPTING_MESSAGE,
                name="message-announcement",
            )
            if completed and self.state is VoiceState.PROMPTING_MESSAGE:
                self._begin_capture(
                    timeout_seconds=self._MESSAGE_RESPONSE_TIMEOUT_SECONDS
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.bind(component="messaging").exception(
                "No se pudo anunciar el mensaje"
            )
            self._transition_to_idle(clear_message=True)

    async def _poll_incoming_message(self) -> None:
        if self.state not in {
            VoiceState.IDLE,
            VoiceState.AWAITING_FOLLOWUP,
        }:
            return
        message = await self._messaging.get_incoming_message()
        if message is None:
            return

        self._stt.update_context_prompt(
            await self._contact_manager.get_canonical_names_prompt()
        )
        self._inbox.push_message(message)
        pending_message = self._inbox.get_pending_message()
        if pending_message is None:
            return
        self._active_message = pending_message
        self._follow_up_deadline = None
        logger.bind(
            component="messaging",
            event="message_received",
            sender_id=pending_message.sender_id,
            is_voice=pending_message.is_voice,
        ).info(
            "[MESSAGING] Mensaje pendiente de: {}",
            pending_message.sender_name,
        )
        self._state_machine.transition_to(VoiceState.PROMPTING_MESSAGE)
        self._start_workflow(
            self._announce_active_message(pending_message),
            name="announce-incoming-message",
        )

    async def _transcribe_intent(
        self,
        segment: SpeechSegment,
    ) -> tuple[str, IntentType]:
        transcript = (
            await self._stt.transcribe(segment.audio_data, self._sample_rate)
        ).strip()
        intent = await self._intent_router.classify_intent_async(transcript)
        logger.bind(
            component="intent_router",
            intent=intent.value,
            state=self.state.value,
        ).info('Intención detectada para: "{}"', transcript)
        return transcript, intent

    async def _process_message_permission(
        self,
        segment: SpeechSegment,
    ) -> None:
        try:
            transcript, intent = await self._transcribe_intent(segment)
            if self._awaiting_contact_selection:
                await self._process_contact_selection(transcript)
                return
            if intent is IntentType.AFFIRMATIVE:
                self._state_machine.transition_to(VoiceState.PLAYING_MESSAGE)
                await self._play_active_message_and_prompt_reply()
            elif intent is IntentType.NEGATIVE:
                await self._decline_active_message()
            else:
                completed = await self._speak(
                    "No te he entendido. ¿Quieres que te ponga el mensaje?",
                    barge_target=VoiceState.PROMPTING_MESSAGE,
                )
                if completed and self.state is VoiceState.PROMPTING_MESSAGE:
                    self._begin_capture(
                        timeout_seconds=self._MESSAGE_RESPONSE_TIMEOUT_SECONDS
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.bind(component="messaging").exception(
                "Falló la respuesta al aviso de mensaje"
            )
            self._transition_to_idle(clear_message=True)

    async def _process_contact_selection(self, transcript: str) -> None:
        contact = await self._contact_manager.resolve_contact(transcript)
        if contact is None:
            contact_names = (
                await self._contact_manager.get_canonical_names_prompt()
            )
            completed = await self._speak(
                f"No he entendido el nombre. Puedes decir: {contact_names}.",
                barge_target=VoiceState.PROMPTING_MESSAGE,
                name="retry-contact-selection",
            )
            if completed and self.state is VoiceState.PROMPTING_MESSAGE:
                self._begin_capture(
                    timeout_seconds=self._MESSAGE_RESPONSE_TIMEOUT_SECONDS
                )
            return

        stored_message = await self._repository.get_unread_by_sender_id(
            contact.platform_id
        )
        if stored_message is None:
            self._awaiting_contact_selection = False
            completed = await self._speak(
                f"No tienes mensajes nuevos de {contact.canonical_name}.",
                name="no-selected-contact-messages",
            )
            if completed:
                self._transition_to_idle()
            return

        self._awaiting_contact_selection = False
        self._active_message = self._stored_to_incoming(stored_message)
        self._state_machine.transition_to(VoiceState.PLAYING_MESSAGE)
        await self._play_active_message_and_prompt_reply()

    async def _decline_active_message(self) -> None:
        completed = await self._speak(
            "De acuerdo, te lo guardo para después.",
            name="decline-message",
        )
        if completed:
            self._transition_to_idle(clear_message=True)

    async def _play_active_message_and_prompt_reply(self) -> None:
        message = self._active_message
        if message is None:
            raise RuntimeError("No hay un mensaje activo")

        if message.is_voice:
            if message.audio_bytes is None:
                raise RuntimeError("El mensaje de voz no contiene audio")
            completed = await self._run_playback(
                self._audio_capture.play_audio_stream(message.audio_bytes),
                barge_target=VoiceState.PROMPTING_REPLY,
                name="incoming-voice-message",
            )
        else:
            if message.text_content is None:
                raise RuntimeError("El mensaje de texto está vacío")
            completed = await self._speak(
                message.text_content,
                barge_target=VoiceState.PROMPTING_REPLY,
                name="incoming-text-message",
            )
        if not completed:
            return

        if message.stored_message_id is not None:
            await self._repository.mark_as_read(message.stored_message_id)
        inbox_message = self._inbox.get_pending_message()
        if (
            inbox_message is not None
            and inbox_message.stored_message_id == message.stored_message_id
        ):
            self._inbox.clear_current()
        self._state_machine.transition_to(VoiceState.PROMPTING_REPLY)
        completed = await self._speak(
            f"¿Quieres responderle algo a {message.sender_name}?",
            barge_target=VoiceState.PROMPTING_REPLY,
            name="reply-question",
        )
        if completed and self.state is VoiceState.PROMPTING_REPLY:
            self._begin_capture(
                timeout_seconds=self._MESSAGE_RESPONSE_TIMEOUT_SECONDS
            )

    async def _process_reply_permission(
        self,
        segment: SpeechSegment,
    ) -> None:
        try:
            transcript, intent = await self._transcribe_intent(segment)
            if intent is IntentType.NEGATIVE:
                await self._cancel_reply()
            elif intent is IntentType.AFFIRMATIVE:
                completed = await self._speak(
                    "Dime qué quieres decirle y se lo envío.",
                    barge_target=VoiceState.RECORDING_REPLY,
                    name="record-reply-prompt",
                )
                if completed and self.state is VoiceState.PROMPTING_REPLY:
                    self._state_machine.transition_to(VoiceState.RECORDING_REPLY)
                    self._begin_capture()
            elif intent is IntentType.DIRECT_REPLY or (
                bool(transcript)
                and segment.duration > 0.5
            ):
                logger.bind(
                    component="intent_router",
                    event="direct_reply_selected",
                    intent=intent.value,
                    audio_duration=segment.duration,
                ).info("La locución actual se enviará como respuesta directa")
                self._state_machine.transition_to(VoiceState.SENDING_REPLY)
                await self._send_reply(segment.audio_data)
            else:
                completed = await self._speak(
                    "No te he entendido. ¿Quieres responderle?",
                    barge_target=VoiceState.PROMPTING_REPLY,
                    name="retry-reply-question",
                )
                if completed and self.state is VoiceState.PROMPTING_REPLY:
                    self._begin_capture(
                        timeout_seconds=self._MESSAGE_RESPONSE_TIMEOUT_SECONDS
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.bind(component="messaging").exception(
                "Falló la preparación de la respuesta"
            )
            self._transition_to_idle(clear_message=True)

    async def _cancel_reply(self) -> None:
        completed = await self._speak(
            "Entendido, no le respondo.",
            name="cancel-reply",
        )
        if completed:
            self._inbox.clear_current()
            self._transition_to_idle(clear_message=True)

    async def _send_reply(self, audio_pcm: bytes) -> None:
        message = self._active_message
        if message is None:
            raise RuntimeError("No hay destinatario para la respuesta")
        sent = False
        try:
            await self._speak(
                f"Enviando respuesta a {message.sender_name}...",
                name="sending-reply",
            )
            sent = await self._messaging.send_voice_note(
                recipient_id=message.sender_id,
                audio_pcm_bytes=audio_pcm,
                sample_rate=self._sample_rate,
            )
            confirmation = (
                "Mensaje enviado."
                if sent
                else "No he podido enviar el mensaje."
            )
            await self._speak(confirmation, name="reply-result")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.bind(component="messaging").exception(
                "Falló el envío de la respuesta"
            )
        finally:
            if sent:
                self._inbox.clear_current()
            self._transition_to_idle(clear_message=True)

    async def _handle_playback_barge_in(self, chunk: AudioChunk) -> None:
        if (
            not self._barge_in_enabled
            or self._playback_task is None
            or self._playback_task.done()
            or self._barge_target is None
        ):
            return
        speech_probability = await asyncio.to_thread(
            self._vad.speech_probability,
            chunk.data,
        )
        if speech_probability < self._BARGE_IN_VAD_THRESHOLD:
            self._reset_barge_in()
            return

        self._barge_in_audio.extend(chunk.data)
        self._barge_in_speech_seconds += self._chunk_duration(chunk)
        if self._barge_in_speech_seconds < self._barge_in_min_seconds:
            return

        source_state = self.state
        target_state = self._barge_target
        interrupted_audio = bytes(self._barge_in_audio)
        if source_state is VoiceState.SPEAKING:
            self._state_machine.transition_to(VoiceState.INTERRUPTED)
        elif source_state is not target_state:
            self._state_machine.transition_to(target_state)

        logger.bind(
            component="orchestrator",
            event="barge_in",
            source_state=source_state.value,
            target_state=target_state.value,
            vad_probability=speech_probability,
        ).info("Barge-in confirmado")

        playback_task = self._playback_task
        if playback_task is not None:
            playback_task.cancel()
            with suppress(asyncio.CancelledError):
                await playback_task
        workflow = self._conversation_task
        if workflow is not None:
            with suppress(asyncio.CancelledError):
                await workflow
        self._conversation_task = None

        if source_state is VoiceState.SPEAKING:
            self._state_machine.transition_to(VoiceState.LISTENING)
        self._vad.reset()
        self._reset_barge_in()
        timeout = (
            self._MESSAGE_RESPONSE_TIMEOUT_SECONDS
            if target_state
            in {VoiceState.PROMPTING_MESSAGE, VoiceState.PROMPTING_REPLY}
            else None
        )
        self._begin_capture(
            interrupted_audio,
            contains_speech=True,
            timeout_seconds=timeout,
        )

    async def _reap_workflow(self) -> None:
        task = self._conversation_task
        if task is None or not task.done():
            return
        with suppress(asyncio.CancelledError):
            await task
        self._conversation_task = None

    async def _cancel_active_tasks(self) -> None:
        tasks = {
            task
            for task in (self._playback_task, self._conversation_task)
            if task is not None and not task.done()
        }
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._playback_task = None
        self._conversation_task = None

    async def run(self) -> None:
        messaging_start_task = asyncio.create_task(
            self._messaging.start(),
            name="messaging-bridge-start",
        )
        try:
            await messaging_start_task
            async for raw_chunk in self._audio_capture.stream_audio():
                await self._reap_workflow()
                await self._poll_incoming_message()
                chunk = AudioChunk(data=raw_chunk, sample_rate=self._sample_rate)

                if self._playback_task is not None:
                    await self._handle_playback_barge_in(chunk)
                elif self.state is VoiceState.IDLE:
                    await self._handle_idle_chunk(chunk)
                elif self.state is VoiceState.AWAITING_FOLLOWUP:
                    await self._handle_follow_up_chunk(chunk)
                elif self.state in {
                    VoiceState.LISTENING,
                    VoiceState.PROMPTING_MESSAGE,
                    VoiceState.PROMPTING_REPLY,
                    VoiceState.RECORDING_REPLY,
                }:
                    await self._handle_capture_chunk(chunk)
        finally:
            if not messaging_start_task.done():
                messaging_start_task.cancel()
                with suppress(asyncio.CancelledError):
                    await messaging_start_task
            await self._cancel_active_tasks()
            await self._messaging.stop()
