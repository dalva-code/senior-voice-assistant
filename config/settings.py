from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sample_rate: int = Field(default=16_000, gt=0)
    chunk_size: int = Field(default=1_280, gt=0)
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    silence_duration_seconds: float = Field(default=2.5, gt=0.0)
    PRE_ROLL_BUFFER_MS: int = Field(default=300, ge=0)
    wake_word_model_name: str = Field(default="alexa", min_length=1)
    log_level: str = Field(default="INFO", min_length=1)
    WHISPER_MODEL_SIZE: str = Field(default="base", min_length=1)
    WHISPER_LANGUAGE: str = Field(default="es", min_length=2)
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", min_length=1)
    WHISPER_INITIAL_PROMPT: str = (
        "Sí, no, vale, claro, ponlo, escuchar, dile que, responde."
    )
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", min_length=1)
    OLLAMA_MODEL: str = Field(default="qwen2.5:7b", min_length=1)
    TTS_VOICE: str = Field(default="es-ES-AlvaroNeural", min_length=1)
    TTS_RATE: str = Field(default="-8%", pattern=r"^[+-]\d+%$")
    MAX_CONTEXT_TURNS: int = Field(default=6, gt=0)
    FOLLOW_UP_TIMEOUT_SECONDS: float = Field(default=6.0, gt=0.0)
    BARGE_IN_ENABLED: bool = False
    BARGE_IN_MIN_SPEECH_DURATION_MS: int = Field(default=350, gt=0)
    POST_TTS_GUARD_SECONDS: float = Field(default=0.4, ge=0.0)
    TELEGRAM_BOT_TOKEN: str = ""
    ALLOWED_TELEGRAM_USER_IDS: list[int] = Field(
        default_factory=lambda: [1_112_763_099]
    )
    DEFAULT_LATITUDE: float = Field(default=28.1235, ge=-90.0, le=90.0)
    DEFAULT_LONGITUDE: float = Field(default=-15.4363, ge=-180.0, le=180.0)
    DATABASE_PATH: str = Field(default="data/messages.db", min_length=1)
    AUTO_APPROVE_NEW_CONTACTS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
