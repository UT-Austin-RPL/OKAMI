#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

"""Load a robot description in Pinocchio."""

import os
from typing import List, Optional, Union

import pinocchio as pin
from easydict import EasyDict

PinocchioJoint = Union[
    pin.JointModelRX,
    pin.JointModelRY,
    pin.JointModelRZ,
    pin.JointModelPX,
    pin.JointModelPY,
    pin.JointModelPZ,
    pin.JointModelFreeFlyer,
    pin.JointModelSpherical,
    pin.JointModelSphericalZYX,
    pin.JointModelPlanar,
    pin.JointModelTranslation,
]


def get_package_dirs(module) -> List[str]:
    """Get package directories for a given module.

    Args:
        module: Robot description module.

    Returns:
        Package directories.
    """
    return [
        module.PACKAGE_PATH,
        module.REPOSITORY_PATH,
        os.path.dirname(module.PACKAGE_PATH),
        os.path.dirname(module.REPOSITORY_PATH),
        os.path.dirname(module.URDF_PATH),  # e.g. laikago_description
    ]


def load_robot_description(
    config: EasyDict,
    root_joint: Optional[PinocchioJoint] = None,
) -> pin.RobotWrapper:
    """Load a robot description in Pinocchio.

    Args:
        description_name: Name of the robot description.
        root_joint (optional): First joint of the kinematic chain, for example
            a free flyer between the floating base of a mobile robot and an
            inertial frame. Defaults to no joint, i.e., a fixed base.
        commit: If specified, check out that commit from the cloned robot
            description repository.

    Returns:
        Robot model for Pinocchio.
    """
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=config.URDF_PATH,
        package_dirs=get_package_dirs(config),
        root_joint=root_joint,
    )
    return robot
