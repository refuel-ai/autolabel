from autolabel.utils import calculate_md5
from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, Optional
import json


class TransformType(str, Enum):
    """Enum containing all Transforms supported by autolabel"""

    WEBPAGE_TRANSFORM = "webpage_scrape"
    PDF = "pdf"
    IMAGE = "image"
    WEB_SEARCH_SERP_API = "web_search_serp_api"
    WEB_SEARCH_SERPER = "web_search"


class TransformCacheEntry(BaseModel):
    transform_name: TransformType
    transform_params: Dict[str, Any]
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    creation_time_ms: Optional[int] = -1
    ttl_ms: Optional[int] = -1

    class Config:
        orm_mode = True

    def get_id(self) -> str:
        """
        Generates a unique ID for the given transform cache configuration
        """
        return calculate_md5([self.transform_name, self.transform_params, self.input])

    def get_serialized_output(self) -> str:
        """
        Returns the serialized cache entry output
        """
        return json.dumps(self.output)

    @classmethod
    def deserialize_output(cls, output: str) -> Dict[str, Any]:
        """
        Deserializes the cache entry output
        """
        return json.loads(output)


class TransformErrorType(str, Enum):
    """Transform error types"""

    INVALID_INPUT = "INVALID_INPUT"
    TRANSFORM_ERROR = "TRANSFORM_ERROR"
    TRANSFORM_TIMEOUT = "TRANSFORM_TIMEOUT"
    MAX_RETRIES_REACHED = "MAX_RETRIES_REACHED"
    SERP_API_ERROR = "SERP_API_ERROR"
    SERPER_API_ERROR = "SERPER_API_ERROR"


class TransformError(Exception):
    """Class representing an error occurred when running transformation on a dataset row"""

    def __init__(self, error_type: TransformErrorType, error_message: str):
        self.error_type = error_type
        self.error_message = error_message
        super().__init__(f"{self.error_type.value}: {self.error_message}")
