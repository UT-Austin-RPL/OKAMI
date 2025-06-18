import logging
import logging.config
import os

from termcolor import colored

from okami.plan_generation import ROOT_PATH
from okami.plan_generation.utils import YamlConfig

class OrionDefaultLogger:
    def __init__(self, logger_config_path, project_name):
        if logger_config_path is None:
            logger_config_path = os.path.join(ROOT_PATH, "../configs/orion_default_logger.yml")
        config = YamlConfig(logger_config_path).as_easydict()
        config["loggers"][project_name] = config["loggers"]["project"]
        os.makedirs("logs", exist_ok=True)
        logging.config.dictConfig(config)

class OrionColorFormatter(logging.Formatter):
    """This color format is for logging user's project wise information"""

    format_str = "[%(levelname)s - %(filename)s:%(lineno)d] "
    message_str = "%(message)s"
    FORMATS = {
        logging.DEBUG: format_str + message_str,
        logging.INFO: colored(format_str, "green", attrs=["bold"]) + message_str,
        logging.WARNING: colored(format_str, "yellow", attrs=["bold"]) + message_str,
        logging.ERROR: colored(format_str, "red", attrs=["bold"]) + message_str,
        logging.CRITICAL: colored(format_str, "red", attrs=["bold", "reverse"])
        + message_str,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_orion_logger(project_name="project", logger_config_path=None):
    """This function returns a logger that follows the deoxys convention"""
    OrionDefaultLogger(logger_config_path, project_name)
    logger = logging.getLogger(project_name)
    return logger