from abc import ABC, abstractmethod
from typing import Dict

from retarget.robot import Robot


class Solver(ABC):
    def __init__(self, config: Dict, robot: Robot):
        self.robot = robot
        self.config = config.copy()

    @abstractmethod
    def __call__(self, target):
        pass

    @abstractmethod
    def update_weights(self, weights):
        pass
