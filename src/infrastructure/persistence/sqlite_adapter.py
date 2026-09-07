import asyncio
import re
import time
import unicodedata
from pathlib import Path

import aiosqlite
from loguru import logger

from src.domain.interfaces.persistence import MessageRepositoryInterface
from src.domain.models import Contact, IncomingMessage, StoredVoiceMessage


class SQLiteAdapter(MessageRepositoryInterface):
    def __init__(
        self,
        database_path: str,
        *,
        auto_approve_new_contacts: bool = True,
        trusted_platform_ids: set[int] | frozenset[int] | None = None,
    ) -> None:
        if not database_path.strip():
            raise ValueError("database_path no puede estar vacío")
        self._database_path = Path(database_path).expanduser()
        self._auto_approve_new_contacts = auto_approve_new_contacts
        self._trusted_platform_ids = frozenset(trusted_platform_ids or ())

    @staticmethod
    def _normalize_name(name: str) -> str:
        decomposed = unicodedata.normalize("NFKD", name.casefold())
        without_diacritics = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return re.sub(r"\s+", " ", without_diacritics).strip()

    @classmethod
    def _generate_aliases(cls, raw_name: str) -> list[str]:
        normalized = cls._normalize_name(raw_name)
        return sorted({normalized, *normalized.split()} - {""})

    async def initialize(self) -> None:
        await asyncio.to_thread(
            self._database_path.parent.mkdir,
            parents=True,
            exist_ok=True,
        )
        async with aiosqlite.connect(self._database_path) as database:
            await database.execute("PRAGMA foreign_keys = ON")
            await database.execute("PRAGMA journal_mode = WAL")
            await database.executescript(
                """
                CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform_id INTEGER UNIQUE NOT NULL,
                    canonical_name TEXT NOT NULL,
                    is_trusted INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS contact_aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contact_id INTEGER NOT NULL,
                    alias TEXT NOT NULL,
                    FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
                    UNIQUE(contact_id, alias)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id INTEGER NOT NULL,
                    sender_name TEXT NOT NULL,
                    audio_bytes BLOB NOT NULL,
                    duration REAL NOT NULL,
                    created_at REAL NOT NULL,
                    is_read INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_messages_unread
                ON messages(is_read, created_at);

                CREATE INDEX IF NOT EXISTS idx_messages_sender_unread
                ON messages(sender_id, is_read, created_at);
                """
            )
            await database.commit()
        logger.bind(
            component="sqlite",
            event="initialized",
            database_path=str(self._database_path),
        ).info("Base de datos inicializada")

    @staticmethod
    async def _get_aliases(
        database: aiosqlite.Connection,
        contact_id: int,
    ) -> list[str]:
        async with database.execute(
            """
            SELECT alias
            FROM contact_aliases
            WHERE contact_id = ?
            ORDER BY alias
            """,
            (contact_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    async def get_or_create_contact(
        self,
        platform_id: int,
        raw_name: str,
    ) -> Contact:
        canonical_name = raw_name.strip()
        if platform_id <= 0:
            raise ValueError("platform_id debe ser positivo")
        if not canonical_name:
            raise ValueError("raw_name no puede estar vacío")

        should_trust = (
            self._auto_approve_new_contacts
            or platform_id in self._trusted_platform_ids
        )
        aliases = self._generate_aliases(canonical_name)
        async with aiosqlite.connect(self._database_path) as database:
            database.row_factory = aiosqlite.Row
            await database.execute("PRAGMA foreign_keys = ON")
            await database.execute("BEGIN IMMEDIATE")
            async with database.execute(
                "SELECT * FROM contacts WHERE platform_id = ?",
                (platform_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                created_at = time.time()
                cursor = await database.execute(
                    """
                    INSERT INTO contacts (
                        platform_id, canonical_name, is_trusted, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        platform_id,
                        canonical_name,
                        int(should_trust),
                        created_at,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite no devolvió el ID del contacto")
                contact_id = cursor.lastrowid
            else:
                contact_id = int(row["id"])
                created_at = float(row["created_at"])
                should_trust = bool(row["is_trusted"]) or should_trust
                await database.execute(
                    """
                    UPDATE contacts
                    SET canonical_name = ?, is_trusted = ?
                    WHERE id = ?
                    """,
                    (canonical_name, int(should_trust), contact_id),
                )

            await database.executemany(
                """
                INSERT OR IGNORE INTO contact_aliases (contact_id, alias)
                VALUES (?, ?)
                """,
                [(contact_id, alias) for alias in aliases],
            )
            await database.commit()
            stored_aliases = await self._get_aliases(database, contact_id)

        contact = Contact(
            id=contact_id,
            platform_id=platform_id,
            canonical_name=canonical_name,
            aliases=stored_aliases,
            is_trusted=should_trust,
            created_at=created_at,
        )
        logger.bind(
            component="sqlite",
            event="contact_upserted",
            contact_id=contact.id,
            platform_id=contact.platform_id,
            is_trusted=contact.is_trusted,
        ).info("Contacto registrado")
        return contact

    async def get_all_contacts(self) -> list[Contact]:
        async with aiosqlite.connect(self._database_path) as database:
            database.row_factory = aiosqlite.Row
            async with database.execute(
                "SELECT * FROM contacts ORDER BY canonical_name"
            ) as cursor:
                rows = await cursor.fetchall()
            contacts = [
                Contact(
                    id=int(row["id"]),
                    platform_id=int(row["platform_id"]),
                    canonical_name=str(row["canonical_name"]),
                    aliases=await self._get_aliases(database, int(row["id"])),
                    is_trusted=bool(row["is_trusted"]),
                    created_at=float(row["created_at"]),
                )
                for row in rows
            ]
        return contacts

    async def save_message(self, msg: IncomingMessage) -> int:
        if not msg.is_voice or msg.audio_bytes is None:
            raise ValueError("Solo se pueden persistir mensajes de voz")
        async with aiosqlite.connect(self._database_path) as database:
            cursor = await database.execute(
                """
                INSERT INTO messages (
                    sender_id,
                    sender_name,
                    audio_bytes,
                    duration,
                    created_at,
                    is_read
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    msg.sender_id,
                    msg.sender_name,
                    msg.audio_bytes,
                    msg.duration,
                    msg.received_at,
                ),
            )
            await database.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite no devolvió el ID del mensaje")
            message_id = cursor.lastrowid
        logger.bind(
            component="sqlite",
            event="message_saved",
            message_id=message_id,
            sender_id=msg.sender_id,
        ).info("Mensaje de voz persistido")
        return message_id

    @staticmethod
    def _to_stored_message(row: aiosqlite.Row) -> StoredVoiceMessage:
        return StoredVoiceMessage(
            id=int(row["id"]),
            sender_id=int(row["sender_id"]),
            sender_name=str(row["sender_name"]),
            audio_bytes=bytes(row["audio_bytes"]),
            duration=float(row["duration"]),
            created_at=float(row["created_at"]),
            is_read=bool(row["is_read"]),
        )

    async def get_unread_messages(self) -> list[StoredVoiceMessage]:
        async with aiosqlite.connect(self._database_path) as database:
            database.row_factory = aiosqlite.Row
            async with database.execute(
                """
                SELECT *
                FROM messages
                WHERE is_read = 0
                ORDER BY created_at ASC, id ASC
                """
            ) as cursor:
                rows = await cursor.fetchall()
        return [self._to_stored_message(row) for row in rows]

    async def get_unread_by_sender_id(
        self,
        sender_id: int,
    ) -> StoredVoiceMessage | None:
        async with aiosqlite.connect(self._database_path) as database:
            database.row_factory = aiosqlite.Row
            async with database.execute(
                """
                SELECT *
                FROM messages
                WHERE sender_id = ? AND is_read = 0
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                (sender_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return self._to_stored_message(row) if row is not None else None

    async def mark_as_read(self, message_id: int) -> None:
        async with aiosqlite.connect(self._database_path) as database:
            await database.execute(
                "UPDATE messages SET is_read = 1 WHERE id = ?",
                (message_id,),
            )
            await database.commit()
        logger.bind(
            component="sqlite",
            event="message_marked_read",
            message_id=message_id,
        ).info("Mensaje marcado como leído")
