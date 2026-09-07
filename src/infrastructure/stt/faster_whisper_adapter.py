import asyncio
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from faster_whisper import WhisperModel
from loguru import logger

from config.settings import Settings
from src.domain.interfaces.stt import STTInterface


class FasterWhisperAdapter(STTInterface):
    _WHISPER_SAMPLE_RATE = 16_000

    def __init__(
        self,
        settings: Settings,
        contact_names_prompt: str = "",
    ) -> None:
        self._language = settings.WHISPER_LANGUAGE
        self._base_initial_prompt = settings.WHISPER_INITIAL_PROMPT
        self._initial_prompt = self._base_initial_prompt
        self.update_context_prompt(contact_names_prompt)
        self._model = WhisperModel(
            model_size_or_path=settings.WHISPER_MODEL_SIZE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        logger.bind(component="stt").info(
            "Modelo Whisper cargado: model={}, compute_type={}",
            settings.WHISPER_MODEL_SIZE,
            settings.WHISPER_COMPUTE_TYPE,
        )

    def update_context_prompt(self, context_prompt: str) -> None:
        contact_hint = context_prompt.strip()
        self._initial_prompt = self._base_initial_prompt
        if contact_hint:
            self._initial_prompt = (
                f"{self._base_initial_prompt} Contactos: {contact_hint}."
            )

    @classmethod
    def _normalize_pcm(
        cls,
        audio_pcm: bytes,
        sample_rate: int,
    ) -> npt.NDArray[np.float32]:
        if sample_rate <= 0:
            raise ValueError("sample_rate debe ser positivo")
        if len(audio_pcm) % np.dtype(np.int16).itemsize != 0:
            raise ValueError("El audio PCM debe contener muestras int16 completas")

        audio = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32_768.0
        if audio.size == 0 or sample_rate == cls._WHISPER_SAMPLE_RATE:
            return audio

        target_size = max(
            1,
            round(audio.size * cls._WHISPER_SAMPLE_RATE / sample_rate),
        )
        source_positions = np.arange(audio.size, dtype=np.float64) / sample_rate
        target_positions = (
            np.arange(target_size, dtype=np.float64) / cls._WHISPER_SAMPLE_RATE
        )
        return np.interp(target_positions, source_positions, audio).astype(np.float32)

    def _transcribe_sync(self, audio_pcm: bytes, sample_rate: int) -> str:
        audio = self._normalize_pcm(audio_pcm, sample_rate)
        if audio.size == 0:
            return ""

        raw_segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            initial_prompt=self._initial_prompt,
            temperature=0.0,
            condition_on_previous_text=False,
            beam_size=5,
        )
        segments = cast(Iterable[Any], raw_segments)
        return " ".join(
            text
            for segment in segments
            if (text := str(segment.text).strip())
        ).strip()

    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16_000) -> str:
        return await asyncio.to_thread(
            self._transcribe_sync,
            audio_pcm,
            sample_rate,
        )
