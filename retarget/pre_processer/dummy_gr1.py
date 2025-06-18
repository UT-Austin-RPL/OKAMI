from retarget.robot import Robot

from .pre_processor import PreProcessor

class DummyGR1Preprocessor(PreProcessor):
    def __init__(self, config, robot: Robot):
        super().__init__(config, robot)

    def calibrate(self, data):
        pass

    def __call__(self, data) -> dict:
        return data

