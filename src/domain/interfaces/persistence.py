from abc import ABC, abstractmethod
from typing import Optional

from src.domain.models import Contact, IncomingMessage, StoredVoiceMessage


class MessageRepositoryInterface(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_or_create_contact(
        self,
        platform_id: int,
        raw_name: str,
    ) -> Contact:
        raise NotImplementedError

    @abstractmethod
    async def get_all_contacts(self) -> list[Contact]:
        raise NotImplementedError

    @abstractmethod
    async def save_message(self, msg: IncomingMessage) -> int:
        raise NotImplementedError

    @abstractmethod
    async def get_unread_messages(self) -> list[StoredVoiceMessage]:
        raise NotImplementedError

    @abstractmethod
    async def get_unread_by_sender_id(
        self,
        sender_id: int,
    ) -> Optional[StoredVoiceMessage]:
        raise NotImplementedError

    @abstractmethod
    async def mark_as_read(self, message_id: int) -> None:
        raise NotImplementedError
