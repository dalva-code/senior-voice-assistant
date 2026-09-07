from collections import deque
from typing import Optional

from loguru import logger

from src.domain.models import IncomingMessage


class InboxManager:
    def __init__(self) -> None:
        self._messages: deque[IncomingMessage] = deque()
        self._current: IncomingMessage | None = None

    def push_message(self, msg: IncomingMessage) -> None:
        self._messages.append(msg)
        logger.bind(
            component="inbox",
            event="message_stored",
            sender_id=msg.sender_id,
            unread_count=len(self._messages) + int(self._current is not None),
        ).info("Mensaje guardado en el buzón")

    def get_pending_message(self) -> Optional[IncomingMessage]:
        if self._current is None and self._messages:
            self._current = self._messages.popleft()
        return self._current

    def has_unread(self) -> bool:
        return self._current is not None or bool(self._messages)

    def clear_current(self) -> None:
        if self._current is not None:
            logger.bind(
                component="inbox",
                event="message_cleared",
                sender_id=self._current.sender_id,
            ).info("Mensaje actual retirado del buzón")
        self._current = None
