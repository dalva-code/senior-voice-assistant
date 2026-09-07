import unittest

from src.application.intent_router import IntentRouter
from src.domain.models import IntentType


class IntentRouterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.router = IntentRouter()

    def test_short_affirmative_with_punctuation(self) -> None:
        self.assertIs(
            self.router.classify_intent("Si."),
            IntentType.AFFIRMATIVE,
        )

    def test_short_affirmative(self) -> None:
        self.assertIs(
            self.router.classify_intent("vale"),
            IntentType.AFFIRMATIVE,
        )

    def test_explicit_direct_reply(self) -> None:
        self.assertIs(
            self.router.classify_intent(
                "responde diciendo que estoy muy bien, que como cuando nos vemos."
            ),
            IntentType.DIRECT_REPLY,
        )

    def test_conversational_direct_reply(self) -> None:
        self.assertIs(
            self.router.classify_intent("Yo estoy bien como estás tú."),
            IntentType.DIRECT_REPLY,
        )

    def test_compound_negative(self) -> None:
        self.assertIs(
            self.router.classify_intent("no, déjalo para luego"),
            IntentType.NEGATIVE,
        )

    def test_weather_request(self) -> None:
        self.assertIs(
            self.router.classify_intent("¿Qué tiempo hace?"),
            IntentType.CHECK_WEATHER,
        )

    def test_pending_message_request(self) -> None:
        self.assertIs(
            self.router.classify_intent("pon el mensaje"),
            IntentType.PLAY_PENDING_MESSAGE,
        )

    def test_inbox_query(self) -> None:
        self.assertIs(
            self.router.classify_intent("¿Tengo mensajes pendientes?"),
            IntentType.QUERY_INBOX,
        )


if __name__ == "__main__":
    unittest.main()
