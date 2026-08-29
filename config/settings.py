from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sample_rate: int = Field(default=16_000, gt=0)
    chunk_size: int = Field(default=1_280, gt=0)
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    silence_duration_seconds: float = Field(default=2.0, gt=0.0)
    wake_word_model_name: str = Field(default="alexa", min_length=1)
    log_level: str = Field(default="INFO", min_length=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
