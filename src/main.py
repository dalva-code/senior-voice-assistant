import asyncio
import sys

from loguru import logger

from config.settings import Settings, get_settings
from src.application.audio_pipeline import AudioPipeline
from src.infrastructure.audio.openwakeword_adapter import (
    OpenWakeWordAdapter,
    ensure_openwakeword_models,
)
from src.infrastructure.audio.silero_vad_adapter import SileroVADAdapter
from src.infrastructure.audio.sounddevice_adapter import SoundDeviceAudioCapture


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    )


async def run(settings: Settings) -> None:
    logger.info("Comprobando modelos de VAD y wake word")
    await asyncio.to_thread(
        ensure_openwakeword_models,
        settings.wake_word_model_name,
    )
    logger.info("Cargando modelos de VAD y wake word")
    vad, wakeword = await asyncio.gather(
        asyncio.to_thread(
            SileroVADAdapter,
            threshold=settings.vad_threshold,
            sample_rate=settings.sample_rate,
        ),
        asyncio.to_thread(
            OpenWakeWordAdapter,
            model_name=settings.wake_word_model_name,
        ),
    )
    audio_capture = SoundDeviceAudioCapture(
        sample_rate=settings.sample_rate,
        chunk_size=settings.chunk_size,
    )
    pipeline = AudioPipeline(
        audio_capture=audio_capture,
        vad=vad,
        wakeword=wakeword,
        sample_rate=settings.sample_rate,
        silence_duration_seconds=settings.silence_duration_seconds,
        wake_word_name=settings.wake_word_model_name,
    )
    logger.info('Pipeline listo; esperando "{}"', settings.wake_word_model_name)
    await pipeline.run()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Asistente detenido por el usuario")


if __name__ == "__main__":
    main()
