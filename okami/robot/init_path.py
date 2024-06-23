import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(path, "./"))
sys.path.insert(0, os.path.join(path, "../"))
sys.path.insert(0, os.path.join(path, "../dora_deoxys_vision_example"))
sys.path.insert(0, os.path.join(path, "../deoxys_vision"))
sys.path.insert(0, os.path.join(path, "../dora"))