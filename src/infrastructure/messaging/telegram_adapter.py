import asyncio
import io
import time
from typing import Any

import numpy as np
import soundfile as sf
from loguru import logger
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

from config.settings import Settings
from src.domain.interfaces.messaging import MessagingBridgeInterface
from src.domain.interfaces.persistence import MessageRepositoryInterface
from src.domain.models import IncomingMessage


class TelegramAdapter(MessagingBridgeInterface):
    def __init__(
        self,
        settings: Settings,
        repository: MessageRepositoryInterface,
    ) -> None:
        self._token = settings.TELEGRAM_BOT_TOKEN.strip()
        self._allowed_user_ids = frozenset(settings.ALLOWED_TELEGRAM_USER_IDS)
        self._auto_approve_new_contacts = settings.AUTO_APPROVE_NEW_CONTACTS
        self._repository = repository
        self._incoming_messages: asyncio.Queue[IncomingMessage] = asyncio.Queue()
        self._application: Application[Any, Any, Any, Any, Any, Any] | None = None

    async def _handle_incoming_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        del context
        user = update.effective_user
        message = update.effective_message
        if user is None or message is None:
            return
        if (
            user.id not in self._allowed_user_ids
            and not self._auto_approve_new_contacts
        ):
            logger.bind(
                component="telegram",
                event="unauthorized_message",
                sender_id=user.id,
            ).warning("Mensaje rechazado de un usuario no autorizado")
            return

        sender_name = user.first_name or "Usuario de Telegram"
        if message.voice is None:
            text_content = (message.text or "").strip()
            if not text_content:
                return
            incoming_message = IncomingMessage(
                sender_id=user.id,
                sender_name=sender_name,
                is_voice=False,
                text_content=text_content,
                received_at=time.time(),
            )
            await self._incoming_messages.put(incoming_message)
            logger.bind(
                component="telegram",
                event="text_message_queued",
                sender_id=user.id,
            ).info("Mensaje de texto autorizado encolado")
            return

        try:
            telegram_file = await message.voice.get_file()
            audio_buffer = io.BytesIO()
            await telegram_file.download_to_memory(audio_buffer)
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.read()
        except (TelegramError, OSError):
            logger.bind(
                component="telegram",
                event="voice_note_download_failed",
                sender_id=user.id,
            ).exception("No se pudo descargar la nota de voz")
            return

        if not audio_bytes:
            logger.bind(
                component="telegram",
                event="empty_voice_note",
                sender_id=user.id,
            ).warning("Telegram devolvió una nota de voz vacía")
            return

        incoming_message = IncomingMessage(
            sender_id=user.id,
            sender_name=sender_name,
            is_voice=True,
            audio_bytes=audio_bytes,
            duration=float(message.voice.duration),
            received_at=time.time(),
        )
        try:
            contact = await self._repository.get_or_create_contact(
                platform_id=user.id,
                raw_name=sender_name,
            )
            if not contact.is_trusted:
                logger.bind(
                    component="telegram",
                    event="untrusted_contact",
                    sender_id=user.id,
                ).warning("Nota de voz rechazada de un contacto no confiable")
                return
            stored_message_id = await self._repository.save_message(
                incoming_message
            )
            incoming_message = incoming_message.model_copy(
                update={"stored_message_id": stored_message_id}
            )
        except Exception:
            logger.bind(
                component="telegram",
                event="message_persistence_failed",
                sender_id=user.id,
            ).exception("No se pudo persistir la nota de voz")
            return

        await self._incoming_messages.put(incoming_message)
        logger.bind(
            component="telegram",
            event="voice_note_queued",
            sender_id=incoming_message.sender_id,
            audio_size=len(audio_bytes),
        ).info("Nota de voz autorizada descargada y encolada")

    async def start(self) -> None:
        if self._application is not None:
            return
        if not self._token:
            logger.bind(component="telegram", event="disabled").warning(
                "Telegram desactivado: TELEGRAM_BOT_TOKEN no está configurado"
            )
            return

        application = ApplicationBuilder().token(self._token).build()
        application.add_handler(
            MessageHandler(
                filters.VOICE | filters.TEXT,
                self._handle_incoming_message,
            )
        )
        self._application = application
        try:
            await application.initialize()
            await application.start()
            updater = application.updater
            if updater is None:
                raise RuntimeError("Telegram Application no tiene Updater")
            await updater.start_polling()
        except Exception:
            logger.bind(component="telegram").exception(
                "No se pudo iniciar el puente de Telegram"
            )
            await self.stop()
            raise

        logger.bind(component="telegram", event="started").info(
            "Puente de Telegram iniciado"
        )

    async def stop(self) -> None:
        application = self._application
        if application is None:
            return
        self._application = None
        try:
            updater = application.updater
            if updater is not None and updater.running:
                await updater.stop()
            if application.running:
                await application.stop()
            await application.shutdown()
        except Exception:
            logger.bind(component="telegram").exception(
                "Error durante el apagado del puente de Telegram"
            )
        else:
            logger.bind(component="telegram", event="stopped").info(
                "Puente de Telegram detenido"
            )

    async def get_incoming_message(self) -> IncomingMessage | None:
        try:
            return self._incoming_messages.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @staticmethod
    def _encode_pcm_as_ogg(audio_pcm_bytes: bytes, sample_rate: int) -> bytes:
        if sample_rate <= 0:
            raise ValueError("sample_rate debe ser positivo")
        if not audio_pcm_bytes:
            raise ValueError("audio_pcm_bytes no puede estar vacío")
        if len(audio_pcm_bytes) % np.dtype(np.int16).itemsize != 0:
            raise ValueError("El audio PCM debe contener muestras int16 completas")

        samples = np.frombuffer(audio_pcm_bytes, dtype=np.int16)
        output = io.BytesIO()
        sf.write(
            output,
            samples,
            sample_rate,
            format="OGG",
            subtype="OPUS",
        )
        return output.getvalue()

    async def send_voice_note(
        self,
        recipient_id: int,
        audio_pcm_bytes: bytes,
        sample_rate: int = 16_000,
    ) -> bool:
        application = self._application
        if application is None or not application.running:
            logger.bind(component="telegram", event="send_skipped").warning(
                "No se puede enviar la nota: Telegram no está activo"
            )
            return False

        try:
            encoded_audio = await asyncio.to_thread(
                self._encode_pcm_as_ogg,
                audio_pcm_bytes,
                sample_rate,
            )
            audio_stream = io.BytesIO(encoded_audio)
            audio_stream.seek(0)
            await application.bot.send_voice(
                chat_id=recipient_id,
                voice=audio_stream,
                filename="voice.ogg",
            )
        except (TelegramError, OSError, RuntimeError, ValueError):
            logger.bind(
                component="telegram",
                event="send_failed",
                recipient_id=recipient_id,
            ).exception("No se pudo enviar la nota de voz")
            return False

        logger.bind(
            component="telegram",
            event="voice_note_sent",
            recipient_id=recipient_id,
        ).info("Nota de voz enviada")
        return True
