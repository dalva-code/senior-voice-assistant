from abc import ABC, abstractmethod


class STTInterface(ABC):
    @abstractmethod
    def update_context_prompt(self, context_prompt: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16_000) -> str:
        raise NotImplementedError
