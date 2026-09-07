import re
import unicodedata
from typing import TYPE_CHECKING

from src.domain.models import IntentType

if TYPE_CHECKING:
    from src.application.contact_manager import ContactManager


class IntentRouter:
    _AFFIRMATIVE_EXACT = re.compile(
        r"^(si|claro|por supuesto|ponlo|ponla|reproducelo|reproducir|"
        r"escuchar|escuchalo|vale|bueno|dale|venga|ok|adelante)$"
    )
    _AFFIRMATIVE_WORD = re.compile(
        r"(?<!\w)(si|claro|ponlo|ponla|vale|dale|venga|ok|adelante)(?!\w)"
    )
    _NEGATIVE_EXACT = re.compile(
        r"^(no|luego|mas tarde|ahora no|dejalo|cancela|cancelar|para|"
        r"nada|despues)$"
    )
    _NEGATIVE_WORD = re.compile(
        r"(?<!\w)(no|ahora no|dejalo|cancela|cancelar|para|nada)(?!\w)"
    )
    _DIRECT_REPLY_PREFIX = re.compile(
        r"^(responde|responder|dile|di|contestale|contesta|cuentale|cuenta|"
        r"mandale|manda|avisale|avisa)\s+(diciendo\s+)?(que\s+)?"
    )
    _CONVERSATIONAL_REPLY = re.compile(
        r"^(yo\s+)?(estoy|soy|tengo|voy|puedo|quiero|he|me encuentro|"
        r"me siento|nosotros|nosotras)\b"
    )
    _WEATHER = re.compile(
        r"(?<!\w)(tiempo|clima|temperatura|va a llover|que dia hace|"
        r"que tiempo hace)(?!\w)"
    )
    _PENDING_MESSAGE = re.compile(
        r"(?<!\w)(pon el mensaje|leer mensaje|"
        r"reproduce el mensaje|que mensaje tengo|escuchar mensaje|"
        r"reproducir audio)(?!\w)"
    )
    _QUERY_INBOX = re.compile(
        r"(?<!\w)(tengo mensajes|hay mensajes|que mensajes tengo|"
        r"mensajes pendientes|buzon|ver mensajes)(?!\w)"
    )
    _CONTACT_MESSAGE = re.compile(
        r"(?<!\w)(pon el de|mensaje de|escuchar a|que dijo)\s+\w+"
    )

    def __init__(self, contact_manager: "ContactManager | None" = None) -> None:
        self._contact_manager = contact_manager

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowercase = text.casefold()
        without_punctuation = re.sub(r'[¿?¡!.,;:\-"“”]', " ", lowercase)
        decomposed = unicodedata.normalize("NFKD", without_punctuation)
        without_diacritics = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return re.sub(r"\s+", " ", without_diacritics).strip()

    def classify_intent(self, text: str) -> IntentType:
        normalized = self._normalize_text(text)
        if not normalized:
            return IntentType.UNKNOWN
        if self._DIRECT_REPLY_PREFIX.match(normalized):
            return IntentType.DIRECT_REPLY
        if self._NEGATIVE_EXACT.fullmatch(normalized):
            return IntentType.NEGATIVE
        if self._NEGATIVE_WORD.search(normalized):
            return IntentType.NEGATIVE
        if self._WEATHER.search(normalized):
            return IntentType.CHECK_WEATHER
        if self._QUERY_INBOX.search(normalized):
            return IntentType.QUERY_INBOX
        if self._PENDING_MESSAGE.search(normalized):
            return IntentType.PLAY_PENDING_MESSAGE
        if self._AFFIRMATIVE_EXACT.fullmatch(normalized):
            return IntentType.AFFIRMATIVE
        if self._AFFIRMATIVE_WORD.search(normalized):
            return IntentType.AFFIRMATIVE
        if self._CONVERSATIONAL_REPLY.match(normalized):
            return IntentType.DIRECT_REPLY
        return IntentType.UNKNOWN

    async def classify_intent_async(self, text: str) -> IntentType:
        normalized = self._normalize_text(text)
        if self._contact_manager is not None and self._CONTACT_MESSAGE.search(
            normalized
        ):
            contact = await self._contact_manager.resolve_contact(normalized)
            return (
                IntentType.PLAY_CONTACT_MESSAGE
                if contact is not None
                else IntentType.UNKNOWN
            )
        return self.classify_intent(normalized)
