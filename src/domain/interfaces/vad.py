from abc import ABC, abstractmethod


class VADInterface(ABC):
    @abstractmethod
    def is_speech(self, chunk: bytes) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
