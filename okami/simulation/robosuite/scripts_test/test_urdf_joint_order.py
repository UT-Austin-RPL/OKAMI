import os
import retarget

import xml.etree.ElementTree as ET

def parse_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    joints = []
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joints.append(joint_name)
    
    return joints

# 使用示例
retarget_repo_dir = os.path.dirname(retarget.__file__)
urdf_path = os.path.join(retarget_repo_dir, "assets/gr1/urdf/gr1_dex.urdf")

joint_order = parse_urdf(urdf_path)
print("Joint Order:", joint_order)
