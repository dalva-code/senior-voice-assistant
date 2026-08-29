import asyncio

from loguru import logger

from src.domain.interfaces.audio_io import AudioCaptureInterface
from src.domain.interfaces.vad import VADInterface
from src.domain.interfaces.wakeword import WakeWordInterface
from src.domain.models import AudioChunk, AudioState, SpeechSegment


class AudioPipeline:
    def __init__(
        self,
        audio_capture: AudioCaptureInterface,
        vad: VADInterface,
        wakeword: WakeWordInterface,
        sample_rate: int,
        silence_duration_seconds: float = 2.0,
        wake_word_name: str = "alexa",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate debe ser positivo")
        if silence_duration_seconds <= 0.0:
            raise ValueError("silence_duration_seconds debe ser positivo")
        self._audio_capture = audio_capture
        self._vad = vad
        self._wakeword = wakeword
        self._sample_rate = sample_rate
        self._silence_duration_seconds = silence_duration_seconds
        self._wake_word_name = wake_word_name
        self._state = AudioState.IDLE
        self._recording = bytearray()
        self._speech_started = False
        self._silence_seconds = 0.0

    @property
    def state(self) -> AudioState:
        return self._state

    def _chunk_duration(self, chunk: AudioChunk) -> float:
        bytes_per_sample = 2
        return len(chunk.data) / (bytes_per_sample * chunk.sample_rate)

    def _start_listening(self, chunk: AudioChunk, contains_speech: bool) -> None:
        self._state = AudioState.LISTENING
        self._recording = bytearray(chunk.data)
        self._speech_started = contains_speech
        self._silence_seconds = 0.0
        logger.info('Palabra de activación detectada: "{}"', self._wake_word_name)

    def _finish_listening(self) -> SpeechSegment:
        duration = len(self._recording) / (2 * self._sample_rate)
        segment = SpeechSegment(audio_data=bytes(self._recording), duration=duration)
        self._state = AudioState.PROCESSING
        logger.info(
            "La persona terminó de hablar tras {:.1f} s de silencio; "
            "segmento capturado: {:.2f} s",
            self._silence_duration_seconds,
            segment.duration,
        )
        self._recording.clear()
        self._speech_started = False
        self._silence_seconds = 0.0
        self._state = AudioState.IDLE
        return segment

    async def run(self) -> None:
        async for raw_chunk in self._audio_capture.stream_audio():
            chunk = AudioChunk(data=raw_chunk, sample_rate=self._sample_rate)

            if self._state is AudioState.IDLE:
                wake_detected = await asyncio.to_thread(self._wakeword.detect, chunk.data)
                if not wake_detected:
                    continue
                self._vad.reset()
                contains_speech = await asyncio.to_thread(self._vad.is_speech, chunk.data)
                self._start_listening(chunk, contains_speech)
                continue

            if self._state is not AudioState.LISTENING:
                continue

            self._recording.extend(chunk.data)
            contains_speech = await asyncio.to_thread(self._vad.is_speech, chunk.data)
            if contains_speech:
                self._speech_started = True
                self._silence_seconds = 0.0
                continue

            if self._speech_started:
                self._silence_seconds += self._chunk_duration(chunk)
                if self._silence_seconds >= self._silence_duration_seconds:
                    self._finish_listening()
