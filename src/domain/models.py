from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AudioState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class AudioChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: bytes = Field(min_length=1)
    sample_rate: int = Field(gt=0)


class SpeechSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_data: bytes = Field(min_length=1)
    duration: float = Field(gt=0.0)
