from .gripper_model import GripperModel
from .gripper_factory import gripper_factory
from .gripper_tester import GripperTester

from .panda_gripper import PandaGripper
from .rethink_gripper import RethinkGripper
from .robotiq_85_gripper import Robotiq85Gripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper, RobotiqThreeFingerDexterousGripper
from .panda_gripper import PandaGripper
from .jaco_three_finger_gripper import JacoThreeFingerGripper, JacoThreeFingerDexterousGripper
from .robotiq_140_gripper import Robotiq140Gripper
from .wiping_gripper import WipingGripper
from .google_gripper import GoogleGripper
from .z1_gripper import Z1Gripper
from .yumi_gripper import YumiLeftGripper, YumiRightGripper
from .bd_gripper import BDGripper
from .null_gripper import NullGripper
from .aloha_gripper import AlohaGripper
from .inspire_hands import InspireLeftHand, InspireRightHand
from .g1_three_finger_gripper import G1ThreeFingerLeftGripper, G1ThreeFingerRightGripper

GRIPPER_MAPPING = {
    "RethinkGripper": RethinkGripper,
    "PandaGripper": PandaGripper,
    "JacoThreeFingerGripper": JacoThreeFingerGripper,
    "JacoThreeFingerDexterousGripper": JacoThreeFingerDexterousGripper,
    "WipingGripper": WipingGripper,
    "Robotiq85Gripper": Robotiq85Gripper,
    "Robotiq140Gripper": Robotiq140Gripper,
    "RobotiqThreeFingerGripper": RobotiqThreeFingerGripper,
    "RobotiqThreeFingerDexterousGripper": RobotiqThreeFingerDexterousGripper,
    "GoogleGripper": GoogleGripper,
    "Z1Gripper": Z1Gripper,
    "BDGripper": BDGripper,
    "YumiLeftGripper": YumiLeftGripper,
    "YumiRightGripper": YumiRightGripper,
    "AlohaGripper": AlohaGripper,
    "InspireLeftHand": InspireLeftHand,
    "InspireRightHand": InspireRightHand,
    "G1ThreeFingerLeftGripper": G1ThreeFingerLeftGripper,
    "G1ThreeFingerRightGripper": G1ThreeFingerRightGripper,
    None: NullGripper,
}

ALL_GRIPPERS = GRIPPER_MAPPING.keys()
