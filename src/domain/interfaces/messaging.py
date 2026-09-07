from abc import ABC, abstractmethod
from typing import Optional

from src.domain.models import IncomingMessage


class MessagingBridgeInterface(ABC):
    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_incoming_message(self) -> Optional[IncomingMessage]:
        raise NotImplementedError

    @abstractmethod
    async def send_voice_note(
        self,
        recipient_id: int,
        audio_pcm_bytes: bytes,
        sample_rate: int = 16_000,
    ) -> bool:
        raise NotImplementedError
