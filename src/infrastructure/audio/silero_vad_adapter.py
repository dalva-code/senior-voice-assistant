from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from src.domain.interfaces.vad import VADInterface


class SileroVADAdapter(VADInterface):
    _FRAME_SAMPLES = 512

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16_000,
        model_path: str | Path | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold debe estar entre 0 y 1")
        if sample_rate not in (8_000, 16_000):
            raise ValueError("Silero VAD solo admite 8000 o 16000 Hz")

        self._threshold = threshold
        self._sample_rate = sample_rate
        self._session = self._load_session(model_path)
        self._input_names = {item.name for item in self._session.get_inputs()}
        self._state: npt.NDArray[np.float32]
        self._hidden: npt.NDArray[np.float32]
        self._cell: npt.NDArray[np.float32]
        self._pending = np.empty(0, dtype=np.float32)
        self.reset()

    @staticmethod
    def _load_session(model_path: str | Path | None) -> ort.InferenceSession:
        if model_path is not None:
            path = Path(model_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"No existe el modelo Silero VAD: {path}")
            return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        packaged_model = resources.files("openwakeword").joinpath(
            "resources", "models", "silero_vad.onnx"
        )
        if not packaged_model.is_file():
            raise FileNotFoundError(
                "openwakeword no incluye resources/models/silero_vad.onnx"
            )
        with resources.as_file(packaged_model) as path:
            return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._hidden = np.zeros((2, 1, 64), dtype=np.float32)
        self._cell = np.zeros((2, 1, 64), dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)

    def _infer_frame(self, frame: npt.NDArray[np.float32]) -> float:
        inputs: dict[str, Any] = {}
        audio_input_name = "input" if "input" in self._input_names else "audio"
        if audio_input_name not in self._input_names:
            raise RuntimeError("El modelo Silero no expone una entrada de audio compatible")
        inputs[audio_input_name] = frame.reshape(1, -1)

        if "sr" in self._input_names:
            inputs["sr"] = np.asarray(self._sample_rate, dtype=np.int64)
        if "state" in self._input_names:
            inputs["state"] = self._state
        elif {"h", "c"}.issubset(self._input_names):
            inputs["h"] = self._hidden
            inputs["c"] = self._cell
        else:
            raise RuntimeError("El modelo Silero no expone entradas de estado compatibles")

        outputs = self._session.run(None, inputs)
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if "state" in self._input_names:
            self._state = np.asarray(outputs[1], dtype=np.float32)
        else:
            self._hidden = np.asarray(outputs[1], dtype=np.float32)
            self._cell = np.asarray(outputs[2], dtype=np.float32)
        return probability

    def speech_probability(self, chunk: bytes) -> float:
        if len(chunk) % np.dtype(np.int16).itemsize != 0:
            raise ValueError("El bloque PCM debe contener muestras int16 completas")
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32_768.0
        self._pending = np.concatenate((self._pending, samples))

        maximum_probability = 0.0
        while self._pending.size >= self._FRAME_SAMPLES:
            frame = self._pending[: self._FRAME_SAMPLES]
            self._pending = self._pending[self._FRAME_SAMPLES :]
            maximum_probability = max(
                maximum_probability,
                self._infer_frame(frame),
            )
        return maximum_probability

    def is_speech(self, chunk: bytes) -> bool:
        return self.speech_probability(chunk) >= self._threshold
