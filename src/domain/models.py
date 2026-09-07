from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AudioState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class VoiceState(StrEnum):
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    AWAITING_FOLLOWUP = "awaiting_followup"
    INTERRUPTED = "interrupted"
    PROMPTING_MESSAGE = "prompting_message"
    PLAYING_MESSAGE = "playing_message"
    PROMPTING_REPLY = "prompting_reply"
    RECORDING_REPLY = "recording_reply"
    SENDING_REPLY = "sending_reply"


class IntentType(StrEnum):
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    DIRECT_REPLY = "direct_reply"
    CHECK_WEATHER = "check_weather"
    PLAY_PENDING_MESSAGE = "play_pending_message"
    QUERY_INBOX = "query_inbox"
    PLAY_CONTACT_MESSAGE = "play_contact_message"
    UNKNOWN = "unknown"


class AudioChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: bytes = Field(min_length=1)
    sample_rate: int = Field(gt=0)


class SpeechSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_data: bytes = Field(min_length=1)
    duration: float = Field(gt=0.0)


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1)
    timestamp: float = Field(ge=0.0)


class IncomingMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    sender_id: int = Field(gt=0)
    sender_name: str = Field(min_length=1)
    is_voice: bool
    audio_bytes: Optional[bytes] = None
    text_content: Optional[str] = None
    duration: float = Field(default=0.0, ge=0.0)
    stored_message_id: Optional[int] = Field(default=None, gt=0)
    received_at: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_content(self) -> "IncomingMessage":
        if self.is_voice:
            if not self.audio_bytes:
                raise ValueError("Un mensaje de voz requiere audio_bytes")
            if self.text_content is not None:
                raise ValueError("Un mensaje de voz no puede contener texto")
        else:
            if not self.text_content or not self.text_content.strip():
                raise ValueError("Un mensaje de texto requiere text_content")
            if self.audio_bytes is not None:
                raise ValueError("Un mensaje de texto no puede contener audio")
        return self


class Contact(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int = Field(gt=0)
    platform_id: int = Field(gt=0)
    canonical_name: str = Field(min_length=1)
    aliases: list[str]
    is_trusted: bool = True
    created_at: float = Field(ge=0.0)


class StoredVoiceMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int = Field(gt=0)
    sender_id: int = Field(gt=0)
    sender_name: str = Field(min_length=1)
    audio_bytes: bytes = Field(min_length=1)
    duration: float = Field(ge=0.0)
    created_at: float = Field(ge=0.0)
    is_read: bool = False
