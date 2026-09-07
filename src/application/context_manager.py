import time

from src.domain.models import ChatMessage


class ContextManager:
    def __init__(self, max_messages: int = 6) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages debe ser positivo")
        self._max_messages = max_messages
        self._messages: tuple[ChatMessage, ...] = ()

    def _add_message(self, role: str, content: str) -> None:
        clean_text = content.strip()
        if not clean_text:
            raise ValueError("El mensaje no puede estar vacío")
        message = ChatMessage(
            role=role,
            content=clean_text,
            timestamp=time.time(),
        )
        self._messages = (*self._messages, message)[-self._max_messages :]

    def add_user_message(self, content: str) -> None:
        self._add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self._add_message("assistant", content)

    def get_messages_for_llm(self) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in self._messages
        ]

    def clear(self) -> None:
        self._messages = ()
