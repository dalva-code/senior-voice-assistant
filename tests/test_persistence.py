import tempfile
import unittest
from pathlib import Path

from src.application.contact_manager import ContactManager
from src.application.intent_router import IntentRouter
from src.domain.models import IncomingMessage, IntentType
from src.infrastructure.persistence.sqlite_adapter import SQLiteAdapter


class SQLiteAdapterTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "messages.db"
        self.repository = SQLiteAdapter(str(database_path))
        await self.repository.initialize()

    async def asyncTearDown(self) -> None:
        self.temporary_directory.cleanup()

    async def test_contact_resolution_and_message_lifecycle(self) -> None:
        contact = await self.repository.get_or_create_contact(
            platform_id=12345,
            raw_name="David",
        )
        self.assertIn("david", contact.aliases)

        contact_manager = ContactManager(self.repository)
        resolved = await contact_manager.resolve_contact(
            "pon el mensaje de Davi"
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.platform_id, 12345)
        router = IntentRouter(contact_manager)
        self.assertIs(
            await router.classify_intent_async("pon el de Davi"),
            IntentType.PLAY_CONTACT_MESSAGE,
        )

        incoming = IncomingMessage(
            sender_id=12345,
            sender_name="David",
            is_voice=True,
            audio_bytes=b"\x01\x02",
            duration=1.25,
            received_at=1_000.0,
        )
        message_id = await self.repository.save_message(incoming)
        unread = await self.repository.get_unread_messages()
        self.assertEqual([message.id for message in unread], [message_id])

        await self.repository.mark_as_read(message_id)
        self.assertEqual(await self.repository.get_unread_messages(), [])


if __name__ == "__main__":
    unittest.main()
