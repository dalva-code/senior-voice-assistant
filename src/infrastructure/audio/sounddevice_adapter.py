import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

import sounddevice as sd
from loguru import logger

from src.domain.interfaces.audio_io import AudioCaptureInterface


class SoundDeviceAudioCapture(AudioCaptureInterface):
    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_size: int = 1_280,
        queue_capacity: int = 32,
        device: int | str | None = None,
    ) -> None:
        if sample_rate <= 0 or chunk_size <= 0 or queue_capacity <= 0:
            raise ValueError("sample_rate, chunk_size y queue_capacity deben ser positivos")
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._queue_capacity = queue_capacity
        self._device = device

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

    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
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
        await asyncio.to_thread(stream.start)
        logger.info(
            "Micrófono activo: {} Hz, {} muestras por bloque",
            self._sample_rate,
            self._chunk_size,
        )

        try:
            while True:
                yield await queue.get()
        finally:
            await asyncio.to_thread(stream.stop)
            await asyncio.to_thread(stream.close)
            logger.info("Captura de micrófono detenida")
