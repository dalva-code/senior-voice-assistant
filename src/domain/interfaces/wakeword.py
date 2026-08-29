from abc import ABC, abstractmethod


class WakeWordInterface(ABC):
    @abstractmethod
    def detect(self, chunk: bytes) -> bool:
        raise NotImplementedError
