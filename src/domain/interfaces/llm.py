from abc import ABC, abstractmethod


ConversationMessage = dict[str, str]


class LLMInterface(ABC):
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        history: list[ConversationMessage] | None = None,
    ) -> str:
        raise NotImplementedError
