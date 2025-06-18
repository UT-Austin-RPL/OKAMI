from abc import ABC, abstractmethod


class Retargeter(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def __call__(self, data) -> dict:
        pass

    @abstractmethod
    def calibrate(self, data):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def visualize(self, target):
        pass

    @abstractmethod
    def control(self, weights, relative_trans):
        pass
