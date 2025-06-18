from typing import Any, Dict

import numpy as np
import os

from retarget.robot import Robot

from .solver import Solver
from retarget.utils.misc import fix_urdf_joints, get_root_dir, import_class

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from pathlib import Path

class DexRetargetSolver(Solver):
    def __init__(self, config, robot):
        super().__init__(config, robot)

        root_dir = get_root_dir()
        # print("root_dir=", root_dir)
        urdf_dir = os.path.join(root_dir, config['default_urdf_dir'])
        RetargetingConfig.set_default_urdf_dir(urdf_dir)
        dex_retarget_config = config['retargeting'].copy()
        retargeting_config = RetargetingConfig.from_dict(dex_retarget_config)
        self.dex_retargeter = retargeting_config.build()

        print("self.config=", self.config['retargeting'])

    def __call__(self, targets):
        indices = self.dex_retargeter.optimizer.target_link_human_indices

        joint_pos = targets['finger_positions']
        ref_value = joint_pos[indices[1, :], :] - joint_pos[indices[0, :], :]
        thetas = self.dex_retargeter.retarget(ref_value)

        # print("thetas=", thetas)
        return thetas

    def update_weights(self, weights):
        pass