import yaml
from easydict import EasyDict


def load_config(config_path):
    with open(config_path) as f:
        config = EasyDict(yaml.safe_load(f))
    return config
