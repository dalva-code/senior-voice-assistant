import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict

from config.settings import Settings
from src.domain.interfaces.llm import ConversationMessage, LLMInterface


SYSTEM_PROMPT = (
    "Eres un compañero de conversación empático, atento y paciente para una "
    "persona mayor. Responde en español de forma natural, en máximo 2 frases "
    "breves, sin usar formato markdown ni listas."
)


class _OllamaMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str


class _OllamaChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: _OllamaMessage


class OllamaAdapter(LLMInterface):
    def __init__(self, settings: Settings) -> None:
        self._model = settings.OLLAMA_MODEL
        self._endpoint = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def generate_response(
        self,
        prompt: str,
        history: list[ConversationMessage] | None = None,
    ) -> str:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            return ""

        conversation_history = list(history or [])
        prompt_already_in_history = bool(
            conversation_history
            and conversation_history[-1].get("role") == "user"
            and conversation_history[-1].get("content") == clean_prompt
        )
        messages: list[ConversationMessage] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation_history,
        ]
        if not prompt_already_in_history:
            messages.append({"role": "user", "content": clean_prompt})
        logger.bind(component="llm", model=self._model).debug(
            "Enviando solicitud a Ollama"
        )
        response = await self._client.post(
            self._endpoint,
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = _OllamaChatResponse.model_validate(response.json())
        return payload.message.content.strip()

    async def aclose(self) -> None:
        await self._client.aclose()
