import abc

from retarget.robot import Robot


class PreProcessor(abc.ABC):
    def __init__(self, config, robot: Robot):
        self.config = config
        self.robot = robot

    @abc.abstractmethod
    def __call__(self, data) -> dict:
        pass

    @abc.abstractmethod
    def calibrate(self, data):
        pass
