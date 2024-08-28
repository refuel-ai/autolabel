import logging

from .base import BaseTransform
from .pdf import PDFTransform
from .serp_api import SerpApi
from .serper_api import SerperApi
from .serper_maps import SerperMaps
from .custom_api import CustomApi
from .webpage_transform import WebpageTransform
from .webpage_scrape import WebpageScrape
from .image import ImageTransform
from typing import Dict
from autolabel.transforms.schema import TransformType
from autolabel.cache import BaseCache

logger = logging.getLogger(__name__)

TRANSFORM_REGISTRY = {
    TransformType.PDF: PDFTransform,
    TransformType.WEBPAGE_TRANSFORM: WebpageTransform,
    TransformType.WEBPAGE_SCRAPE: WebpageScrape,
    TransformType.IMAGE: ImageTransform,
    TransformType.WEB_SEARCH_SERP_API: SerpApi,
    TransformType.WEB_SEARCH_SERPER: SerperApi,
    TransformType.CUSTOM_API: CustomApi,
    TransformType.MAPS_SEARCH: SerperMaps,
}


class TransformFactory:
    """The ModelFactory class is used to create a BaseModel object from the given AutoLabelConfig configuration."""

    @staticmethod
    def from_dict(dict: Dict, cache: BaseCache = None) -> BaseTransform:
        """
        Returns a Transform object based on the given name and parameters
        """
        assert "name" in dict, f"Transform name must be specified in transform {dict}"
        assert (
            "output_columns" in dict
        ), f"Output columns must be specified in transform {dict}"

        name = dict["name"]
        params = dict.get("params", {})
        output_columns = dict["output_columns"]

        if name not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown transform type {name}")
        model_cls = TRANSFORM_REGISTRY[name]
        return model_cls(
            output_columns=output_columns,
            cache=cache,
            **params,
        )


def register_transform(name, transform_cls):
    TRANSFORM_REGISTRY[name] = transform_cls
