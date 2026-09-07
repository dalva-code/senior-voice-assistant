import asyncio
import signal
import sys
from contextlib import suppress

from loguru import logger

from config.settings import Settings, get_settings
from src.application.contact_manager import ContactManager
from src.application.orchestrator import VoiceOrchestrator
from src.infrastructure.audio.openwakeword_adapter import (
    OpenWakeWordAdapter,
    ensure_openwakeword_models,
)
from src.infrastructure.audio.silero_vad_adapter import SileroVADAdapter
from src.infrastructure.audio.sounddevice_adapter import SoundDeviceAudioCapture
from src.infrastructure.llm.ollama_adapter import OllamaAdapter
from src.infrastructure.messaging.telegram_adapter import TelegramAdapter
from src.infrastructure.persistence.sqlite_adapter import SQLiteAdapter
from src.infrastructure.stt.faster_whisper_adapter import FasterWhisperAdapter
from src.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra} | {message}",
    )


async def run_until_shutdown(orchestrator: VoiceOrchestrator) -> None:
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    registered_signals: list[signal.Signals] = []
    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(shutdown_signal, shutdown_event.set)
            registered_signals.append(shutdown_signal)
        except NotImplementedError:
            logger.bind(component="shutdown").warning(
                "El event loop no admite señales POSIX"
            )
            break

    orchestrator_task = asyncio.create_task(
        orchestrator.run(),
        name="voice-orchestrator",
    )
    shutdown_task = asyncio.create_task(
        shutdown_event.wait(),
        name="shutdown-signal",
    )
    try:
        completed, _ = await asyncio.wait(
            {orchestrator_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if shutdown_task in completed:
            logger.bind(component="shutdown", signal="SIGINT/SIGTERM").info(
                "Señal de apagado recibida"
            )
            orchestrator_task.cancel()
            with suppress(asyncio.CancelledError):
                await orchestrator_task
        else:
            await orchestrator_task
    finally:
        shutdown_task.cancel()
        with suppress(asyncio.CancelledError):
            await shutdown_task
        if not orchestrator_task.done():
            orchestrator_task.cancel()
            with suppress(asyncio.CancelledError):
                await orchestrator_task
        for registered_signal in registered_signals:
            loop.remove_signal_handler(registered_signal)


async def run(settings: Settings) -> None:
    repository = SQLiteAdapter(
        settings.DATABASE_PATH,
        auto_approve_new_contacts=settings.AUTO_APPROVE_NEW_CONTACTS,
        trusted_platform_ids=frozenset(settings.ALLOWED_TELEGRAM_USER_IDS),
    )
    await repository.initialize()
    contact_manager = ContactManager(repository)
    contact_names_prompt = await contact_manager.get_canonical_names_prompt()
    logger.info("Comprobando modelos de VAD y wake word")
    await asyncio.to_thread(
        ensure_openwakeword_models,
        settings.wake_word_model_name,
    )
    logger.bind(component="startup").info(
        "Cargando modelos de VAD, wake word y Whisper"
    )
    vad, wakeword, stt = await asyncio.gather(
        asyncio.to_thread(
            SileroVADAdapter,
            threshold=settings.vad_threshold,
            sample_rate=settings.sample_rate,
        ),
        asyncio.to_thread(
            OpenWakeWordAdapter,
            model_name=settings.wake_word_model_name,
        ),
        asyncio.to_thread(
            FasterWhisperAdapter,
            settings,
            contact_names_prompt,
        ),
    )
    audio_capture = SoundDeviceAudioCapture(
        sample_rate=settings.sample_rate,
        chunk_size=settings.chunk_size,
        pre_roll_buffer_ms=settings.PRE_ROLL_BUFFER_MS,
    )
    llm = OllamaAdapter(settings)
    tts = EdgeTTSAdapter(settings)
    messaging = TelegramAdapter(settings, repository)
    orchestrator = VoiceOrchestrator(
        audio_capture=audio_capture,
        vad=vad,
        wakeword=wakeword,
        stt=stt,
        llm=llm,
        tts=tts,
        messaging=messaging,
        repository=repository,
        contact_manager=contact_manager,
        sample_rate=settings.sample_rate,
        silence_duration_seconds=settings.silence_duration_seconds,
        wake_word_name=settings.wake_word_model_name,
        max_context_turns=settings.MAX_CONTEXT_TURNS,
        follow_up_timeout_seconds=settings.FOLLOW_UP_TIMEOUT_SECONDS,
        barge_in_enabled=settings.BARGE_IN_ENABLED,
        barge_in_min_speech_duration_ms=settings.BARGE_IN_MIN_SPEECH_DURATION_MS,
        post_tts_guard_seconds=settings.POST_TTS_GUARD_SECONDS,
        default_latitude=settings.DEFAULT_LATITUDE,
        default_longitude=settings.DEFAULT_LONGITUDE,
    )
    logger.bind(component="orchestrator", state=orchestrator.state.value).info(
        'Orquestador listo; esperando "{}"',
        settings.wake_word_model_name,
    )
    try:
        await run_until_shutdown(orchestrator)
    finally:
        await llm.aclose()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Asistente detenido por el usuario")


if __name__ == "__main__":
    main()
