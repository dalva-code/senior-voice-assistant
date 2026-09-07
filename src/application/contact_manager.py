import asyncio
import re
import unicodedata

from loguru import logger
from rapidfuzz import fuzz

from src.domain.interfaces.persistence import MessageRepositoryInterface
from src.domain.models import Contact


class ContactManager:
    _MATCH_THRESHOLD = 75.0

    def __init__(self, repository: MessageRepositoryInterface) -> None:
        self._repository = repository

    @staticmethod
    def _normalize(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text.casefold())
        without_diacritics = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return re.sub(r"\s+", " ", without_diacritics).strip()

    @classmethod
    def _find_best_contact(
        cls,
        normalized_query: str,
        contacts: list[Contact],
    ) -> tuple[Contact | None, float]:
        best_contact: Contact | None = None
        best_score = 0.0
        for contact in contacts:
            if not contact.is_trusted:
                continue
            candidates = {
                cls._normalize(contact.canonical_name),
                *(cls._normalize(alias) for alias in contact.aliases),
            }
            contact_score = max(
                (
                    max(
                        float(fuzz.token_set_ratio(normalized_query, candidate)),
                        float(fuzz.partial_ratio(normalized_query, candidate)),
                    )
                    for candidate in candidates
                    if candidate
                ),
                default=0.0,
            )
            if contact_score > best_score:
                best_score = contact_score
                best_contact = contact
        return best_contact, best_score

    async def resolve_contact(self, query_text: str) -> Contact | None:
        normalized_query = self._normalize(query_text)
        if not normalized_query:
            return None
        contacts = await self._repository.get_all_contacts()
        best_contact, best_score = await asyncio.to_thread(
            self._find_best_contact,
            normalized_query,
            contacts,
        )

        if best_contact is None or best_score <= self._MATCH_THRESHOLD:
            logger.bind(
                component="contacts",
                event="contact_not_resolved",
                query=normalized_query,
                score=best_score,
            ).info("No se encontró un contacto suficientemente similar")
            return None

        logger.bind(
            component="contacts",
            event="contact_resolved",
            contact_id=best_contact.id,
            canonical_name=best_contact.canonical_name,
            score=best_score,
        ).info("Contacto resuelto mediante similitud fonética")
        return best_contact

    async def get_canonical_names_prompt(self) -> str:
        contacts = await self._repository.get_all_contacts()
        return ", ".join(
            contact.canonical_name
            for contact in contacts
            if contact.is_trusted
        )
