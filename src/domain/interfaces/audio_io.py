from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class AudioCaptureInterface(ABC):
    @abstractmethod
    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError
        yield b""
