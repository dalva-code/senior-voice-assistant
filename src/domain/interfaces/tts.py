from abc import ABC, abstractmethod


class TTSInterface(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    async def play(self, audio_bytes: bytes) -> None:
        raise NotImplementedError
