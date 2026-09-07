from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class AudioCaptureInterface(ABC):
    @abstractmethod
    def get_pre_roll_audio(self) -> bytes:
        raise NotImplementedError

    @abstractmethod
    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError
        yield b""

    @abstractmethod
    async def play_audio_stream(self, audio_bytes: bytes) -> None:
        raise NotImplementedError
