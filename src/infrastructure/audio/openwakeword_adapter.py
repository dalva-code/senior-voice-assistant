from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from openwakeword.model import Model
from openwakeword.utils import download_models

from src.domain.interfaces.wakeword import WakeWordInterface


def ensure_openwakeword_models(model_name: str) -> None:
    if not model_name.strip():
        raise ValueError("model_name no puede estar vacío")
    download_models([model_name])


class OpenWakeWordAdapter(WakeWordInterface):
    _FRAME_SAMPLES = 1_280

    def __init__(self, model_name: str = "alexa", threshold: float = 0.5) -> None:
        if not model_name.strip():
            raise ValueError("model_name no puede estar vacío")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold debe estar entre 0 y 1")

        self._model_name = model_name.casefold()
        self._threshold = threshold
        self._model = Model(
            wakeword_models=[model_name],
            inference_framework="onnx",
        )
        self._pending: npt.NDArray[np.int16] = np.empty(0, dtype=np.int16)

    def detect(self, chunk: bytes) -> bool:
        if len(chunk) % np.dtype(np.int16).itemsize != 0:
            raise ValueError("El bloque PCM debe contener muestras int16 completas")
        samples = np.frombuffer(chunk, dtype=np.int16)
        self._pending = np.concatenate((self._pending, samples))

        while self._pending.size >= self._FRAME_SAMPLES:
            frame = self._pending[: self._FRAME_SAMPLES]
            self._pending = self._pending[self._FRAME_SAMPLES :]
            raw_prediction = cast(Any, self._model.predict(frame))
            prediction = cast(Mapping[str, float], raw_prediction)
            matching_scores = [
                float(score)
                for name, score in prediction.items()
                if self._model_name in name.casefold()
            ]
            scores = matching_scores or [float(score) for score in prediction.values()]
            if scores and max(scores) >= self._threshold:
                return True
        return False
