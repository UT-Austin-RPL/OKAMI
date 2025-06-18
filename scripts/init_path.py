import sys
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(path, '../'))

sys.path.append("./third_party/co-tracker")
sys.path.append("./third_party/hamer")
sys.path.append("./okami/simulation")