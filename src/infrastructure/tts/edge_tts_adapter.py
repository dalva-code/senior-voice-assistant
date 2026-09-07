import asyncio
import io
import threading
from _thread import LockType
from typing import Any, cast

import edge_tts
import numpy as np
import numpy.typing as npt
import sounddevice as sd
import soundfile as sf
from loguru import logger

from config.settings import Settings
from src.domain.interfaces.tts import TTSInterface


class EdgeTTSAdapter(TTSInterface):
    def __init__(self, settings: Settings) -> None:
        self._voice = settings.TTS_VOICE
        self._rate = settings.TTS_RATE

    async def synthesize(self, text: str) -> bytes:
        clean_text = text.strip()
        if not clean_text:
            return b""

        communicate = edge_tts.Communicate(
            clean_text,
            voice=self._voice,
            rate=self._rate,
        )
        output = io.BytesIO()
        async for raw_chunk in communicate.stream():
            chunk = cast(dict[str, Any], raw_chunk)
            if chunk.get("type") == "audio":
                output.write(cast(bytes, chunk["data"]))

        audio_bytes = output.getvalue()
        if not audio_bytes:
            raise RuntimeError("Edge TTS no devolvió datos de audio")
        logger.bind(component="tts", voice=self._voice).debug(
            "Audio sintetizado: {} bytes",
            len(audio_bytes),
        )
        return audio_bytes

    @staticmethod
    def _decode_and_play(
        audio_bytes: bytes,
        cancel_event: threading.Event,
        playback_lock: LockType,
    ) -> None:
        with io.BytesIO(audio_bytes) as audio_buffer:
            decoded = cast(
                tuple[npt.NDArray[np.float32], int],
                sf.read(audio_buffer, dtype="float32", always_2d=False),
            )
        audio_data, sample_rate = decoded
        with playback_lock:
            if cancel_event.is_set():
                return
            sd.play(audio_data, samplerate=sample_rate, blocking=False)
        sd.wait()

    @staticmethod
    def _stop_playback_safely() -> None:
        try:
            sd.stop()
        except Exception as error:
            logger.bind(
                component="tts",
                event="coreaudio_stop_suppressed",
                error=str(error),
            ).warning("Se ignoró un error interno al cerrar el stream TTS")

    async def play(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            raise ValueError("audio_bytes no puede estar vacío")
        cancel_event = threading.Event()
        playback_lock = threading.Lock()
        try:
            await asyncio.to_thread(
                self._decode_and_play,
                audio_bytes,
                cancel_event,
                playback_lock,
            )
        except asyncio.CancelledError:
            cancel_event.set()
            with playback_lock:
                self._stop_playback_safely()
            logger.bind(component="tts", event="playback_cancelled").info(
                "Reproducción TTS detenida inmediatamente"
            )
            raise
        finally:
            await asyncio.to_thread(self._stop_playback_safely)
