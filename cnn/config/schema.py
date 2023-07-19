from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum, IntEnum
from pydantic import BaseModel, validator

from .base import Settings

class StatusCode(IntEnum):
    success = 200
    accepted = 201
    processing = 202
    missing_images = 404
    invalid_entity = 422
    internal_error = 500


class GeoApiConfig(BaseModel):
    # bbox max size in pixels (width, height)
    max_bbox_size: Tuple[int, int] = (15000, 15000)
    max_tile_dim: int = 2000
    shub_client_id: str
    shub_client_secret: str
    max_cloud_coverage: float = 0.1
    mosaicking_order: str = "leastCC"
    cache_dir: Path = Path("data/data")
    evalscript_dir: Path = Path("resources/evalscripts")
    list: Path
    id: int

    @validator('max_cloud_coverage')
    def cc_must_be_percentage(cls, val):
        if val < 0 or val > 1:
            raise ValueError("Max Cloud Coverage must be a float in the range [0, 1]")
        return val

class HotspotsDB(BaseModel):
    host: str
    database: str
    user: str
    password: str
    port: str
    cache_dir: str


class Configuration(Settings):
    log_level: str = "info"
    log_format: str = "[%(asctime)s] [PID %(process)d] [%(threadName)s] [%(name)s] [%(levelname)s] %(message)s"
    log_suppress: dict = {}

    geoapis: List[GeoApiConfig]
    hotspotdb: HotspotsDB
