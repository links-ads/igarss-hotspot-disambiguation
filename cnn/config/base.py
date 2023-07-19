import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseSettings
import calendar
import time


def prepare_logging(log_level: int,
                    log_format: str,
                    date_format: str = "%Y-%m-%d %H:%M",
                    suppress: Dict[str, str] = {}):
    """Initializes logging to print infos and with standard format.
    """
    required_level = logging.getLevelName(log_level.upper())
    current_GMT = time.gmtime()

    time_stamp = calendar.timegm(current_GMT)
    logging.basicConfig(filename=f"logs/{time_stamp}_logs.log", level=required_level, format=log_format, datefmt=date_format)
    for log_name, log_level in suppress.items():
        logging.getLogger(log_name).setLevel(logging.getLevelName(log_level.upper()))



def yaml_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """
    A simple settings source that loads variables from a JSON file
    at the project's root.

    Here we happen to choose to use the `env_file_encoding` from Config
    when reading `config.json`
    """
    filename = settings.Config.env_file
    encoding = settings.Config.env_file_encoding
    return yaml.safe_load(Path(filename).read_text(encoding=encoding))


class Settings(BaseSettings):
    class Config:
        env_file = "config.yml"
        env_file_encoding = "utf-8"

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                yaml_config_settings_source,
                file_secret_settings,
            )
