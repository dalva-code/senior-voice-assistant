import asyncio

from loguru import logger

from src.domain.interfaces.audio_io import AudioCaptureInterface
from src.domain.interfaces.llm import ConversationMessage, LLMInterface
from src.domain.interfaces.stt import STTInterface
from src.domain.interfaces.tts import TTSInterface
from src.domain.interfaces.vad import VADInterface
from src.domain.interfaces.wakeword import WakeWordInterface
from src.domain.models import AudioChunk, AudioState, SpeechSegment


class AudioPipeline:
    def __init__(
        self,
        audio_capture: AudioCaptureInterface,
        vad: VADInterface,
        wakeword: WakeWordInterface,
        stt: STTInterface,
        llm: LLMInterface,
        tts: TTSInterface,
        sample_rate: int,
        silence_duration_seconds: float = 2.0,
        wake_word_name: str = "alexa",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate debe ser positivo")
        if silence_duration_seconds <= 0.0:
            raise ValueError("silence_duration_seconds debe ser positivo")
        self._audio_capture = audio_capture
        self._vad = vad
        self._wakeword = wakeword
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._sample_rate = sample_rate
        self._silence_duration_seconds = silence_duration_seconds
        self._wake_word_name = wake_word_name
        self._state = AudioState.IDLE
        self._recording = bytearray()
        self._speech_started = False
        self._silence_seconds = 0.0
        self._history: list[ConversationMessage] = []

    @property
    def state(self) -> AudioState:
        return self._state

    def _chunk_duration(self, chunk: AudioChunk) -> float:
        bytes_per_sample = 2
        return len(chunk.data) / (bytes_per_sample * chunk.sample_rate)

    def _start_listening(self, chunk: AudioChunk, contains_speech: bool) -> None:
        self._state = AudioState.LISTENING
        self._recording = bytearray(chunk.data)
        self._speech_started = contains_speech
        self._silence_seconds = 0.0
        logger.info('Palabra de activación detectada: "{}"', self._wake_word_name)

    def _finish_listening(self) -> SpeechSegment:
        duration = len(self._recording) / (2 * self._sample_rate)
        segment = SpeechSegment(audio_data=bytes(self._recording), duration=duration)
        self._state = AudioState.PROCESSING
        logger.info(
            "La persona terminó de hablar tras {:.1f} s de silencio; "
            "segmento capturado: {:.2f} s",
            self._silence_duration_seconds,
            segment.duration,
        )
        self._recording.clear()
        self._speech_started = False
        self._silence_seconds = 0.0
        return segment

    async def _process_segment(self, segment: SpeechSegment) -> None:
        try:
            transcript = (
                await self._stt.transcribe(segment.audio_data, self._sample_rate)
            ).strip()
            logger.bind(component="stt", event="transcription").info(
                '[STT] Usuario dijo: "{}"',
                transcript,
            )
            if not transcript:
                logger.bind(component="stt").warning(
                    "Transcripción vacía; volviendo a espera"
                )
                return

            response = (
                await self._llm.generate_response(transcript, self._history)
            ).strip()
            logger.bind(component="llm", event="response").info(
                '[LLM] Respuesta: "{}"',
                response,
            )
            if not response:
                logger.bind(component="llm").warning(
                    "Ollama devolvió una respuesta vacía; volviendo a espera"
                )
                return

            self._history.extend(
                (
                    {"role": "user", "content": transcript},
                    {"role": "assistant", "content": response},
                )
            )
            self._history = self._history[-12:]
            self._state = AudioState.SPEAKING
            synthesized_audio = await self._tts.synthesize(response)
            await self._tts.play(synthesized_audio)
        except Exception:
            logger.bind(component="conversation").exception(
                "Falló el procesamiento speech-to-speech"
            )
        finally:
            self._state = AudioState.IDLE
            logger.bind(component="pipeline", state=self._state.value).info(
                'Pipeline listo; esperando "{}"',
                self._wake_word_name,
            )

    async def run(self) -> None:
        conversation_task: asyncio.Task[None] | None = None
        try:
            async for raw_chunk in self._audio_capture.stream_audio():
                if conversation_task is not None and conversation_task.done():
                    await conversation_task
                    conversation_task = None

                chunk = AudioChunk(data=raw_chunk, sample_rate=self._sample_rate)
                if self._state is AudioState.IDLE:
                    wake_detected = await asyncio.to_thread(
                        self._wakeword.detect,
                        chunk.data,
                    )
                    if not wake_detected:
                        continue
                    self._vad.reset()
                    contains_speech = await asyncio.to_thread(
                        self._vad.is_speech,
                        chunk.data,
                    )
                    self._start_listening(chunk, contains_speech)
                    continue

                if self._state is not AudioState.LISTENING:
                    continue

                self._recording.extend(chunk.data)
                contains_speech = await asyncio.to_thread(
                    self._vad.is_speech,
                    chunk.data,
                )
                if contains_speech:
                    self._speech_started = True
                    self._silence_seconds = 0.0
                    continue

                if self._speech_started:
                    self._silence_seconds += self._chunk_duration(chunk)
                    if self._silence_seconds >= self._silence_duration_seconds:
                        segment = self._finish_listening()
                        conversation_task = asyncio.create_task(
                            self._process_segment(segment)
                        )
        finally:
            if conversation_task is not None:
                await conversation_task
