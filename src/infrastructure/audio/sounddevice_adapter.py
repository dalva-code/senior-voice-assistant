import asyncio
import io
import threading
from _thread import LockType
from collections import deque
from collections.abc import AsyncGenerator, Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import soundfile as sf
from loguru import logger

from src.domain.interfaces.audio_io import AudioCaptureInterface


class SoundDeviceAudioCapture(AudioCaptureInterface):
    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_size: int = 1_280,
        queue_capacity: int = 32,
        device: int | str | None = None,
        pre_roll_buffer_ms: int = 300,
    ) -> None:
        if sample_rate <= 0 or chunk_size <= 0 or queue_capacity <= 0:
            raise ValueError("sample_rate, chunk_size y queue_capacity deben ser positivos")
        if pre_roll_buffer_ms < 0:
            raise ValueError("pre_roll_buffer_ms no puede ser negativo")
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._queue_capacity = queue_capacity
        self._device = device
        pre_roll_frame_count = (
            (sample_rate * pre_roll_buffer_ms + chunk_size * 1_000 - 1)
            // (chunk_size * 1_000)
            if pre_roll_buffer_ms > 0
            else 0
        )
        self._pre_roll_frames: deque[bytes] = deque(
            maxlen=pre_roll_frame_count
        )

    def get_pre_roll_audio(self) -> bytes:
        return b"".join(self._pre_roll_frames)

    @staticmethod
    def _enqueue_chunk(queue: asyncio.Queue[bytes], chunk: bytes) -> None:
        if queue.full():
            queue.get_nowait()
            logger.warning("Cola de audio saturada; se descartó el bloque más antiguo")
        queue.put_nowait(chunk)

    def _build_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[bytes],
    ) -> Callable[[Any, int, Any, Any], None]:
        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info
            if status:
                logger.warning("Estado del flujo de entrada: {}", status)
            loop.call_soon_threadsafe(self._enqueue_chunk, queue, bytes(indata))

        return callback

    @staticmethod
    def _run_stream_action_safely(
        action: Callable[[], Any],
        action_name: str,
    ) -> None:
        try:
            action()
        except Exception as error:
            logger.bind(
                component="audio_capture",
                event="coreaudio_action_suppressed",
                action=action_name,
                error=str(error),
            ).warning("Se ignoró un error interno de CoreAudio")

    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        self._pre_roll_frames.clear()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=self._queue_capacity)
        callback = self._build_callback(loop, queue)

        stream = await asyncio.to_thread(
            sd.RawInputStream,
            samplerate=self._sample_rate,
            blocksize=self._chunk_size,
            channels=1,
            dtype="int16",
            callback=callback,
            device=self._device,
        )
        started = False
        aborted = False
        try:
            await asyncio.to_thread(stream.start)
            started = True
            logger.bind(component="audio_capture").info(
                "Micrófono activo: {} Hz, {} muestras por bloque",
                self._sample_rate,
                self._chunk_size,
            )
            while True:
                chunk = await queue.get()
                yield chunk
                self._pre_roll_frames.append(chunk)
        except asyncio.CancelledError:
            if started:
                aborted = True
                await asyncio.to_thread(
                    self._run_stream_action_safely,
                    stream.abort,
                    "abort",
                )
            logger.bind(component="audio_capture").info(
                "Captura de micrófono cancelada"
            )
            raise
        finally:
            if started and not aborted:
                await asyncio.to_thread(
                    self._run_stream_action_safely,
                    stream.stop,
                    "stop",
                )
            await asyncio.to_thread(
                self._run_stream_action_safely,
                stream.close,
                "close",
            )
            logger.bind(component="audio_capture").info(
                "Captura de micrófono detenida"
            )

    @staticmethod
    def _decode_and_play(
        audio_bytes: bytes,
        cancel_event: threading.Event,
        playback_lock: LockType,
    ) -> None:
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio_data, sample_rate = cast(
                tuple[npt.NDArray[np.float32], int],
                sf.read(audio_buffer, dtype="float32", always_2d=False),
            )
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
                component="audio_playback",
                event="coreaudio_stop_suppressed",
                error=str(error),
            ).warning("Se ignoró un error interno al cerrar el stream de audio")

    async def play_audio_stream(self, audio_bytes: bytes) -> None:
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
            logger.bind(
                component="audio_playback",
                event="cancelled",
            ).info("Reproducción de mensaje detenida")
            raise
        finally:
            await asyncio.to_thread(self._stop_playback_safely)
