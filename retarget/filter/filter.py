import abc
from typing import Dict


class Filter(abc.ABC):
    def __init__(self, config: Dict):
        pass

    @abc.abstractmethod
    def __call__(self, data) -> dict:
        pass

    def calibrate(self, data):
        pass

    def reset_history(self):
        pass
