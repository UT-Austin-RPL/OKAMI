import numpy as np
import os
import json
import time

from gr1_interface.gr1_control.gr1_client import gr1_interface
from gr1_interface.gr1_control.utils.variables import (
    finger_joints,
    name_to_limits,
    name_to_sign,
    name_to_urdf_idx,
)

from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface

from utils.real_robot_utils import real_to_urdf

class StateSaver:
    def __init__(self, task_name, exp_name):
        self.task_name = task_name
        self.exp_name = exp_name
        self.folder = os.path.join("data", task_name, exp_name)
        os.makedirs(self.folder, exist_ok=True)

        assert_camera_ref_convention('rs_0')
        camera_info = get_camera_info('rs_0')
        camera_id = camera_info.camera_id
        self.cr_interface = CameraRedisSubInterface(redis_host="localhost", camera_info=camera_info, use_depth=True)
        self.cr_interface.start()

        self.res = []

    def add_state(self, joint_command):
        '''
        Add state from real robot joint commands. 
        '''
        assert len(joint_command) == 56
        
        img_idx = self.cr_interface.get_img_info()["image_idx"]

        self.res.append({'image_idx': img_idx, 'robot_state': real_to_urdf(joint_command).tolist()})
        
        # filename = os.path.join(self.folder, f"states_{len(self.res):07d}.json")
        # with open(filename, 'w') as f:
        #     json.dump(self.res[-1], f, indent=4)
    
    def save(self):
        filename = os.path.join(self.folder, "all_states.json")
        with open(filename, 'w') as f:
            json.dump(self.res, f, indent=4)
        print("state and image paired data saved in ", filename, 'number of states=', len(self.res))