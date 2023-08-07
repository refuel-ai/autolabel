import logging

from .base import BaseTransform
from .webpage_transform import WebpageTransform
from typing import Dict, List
from autolabel.schema import TransformType

logger = logging.getLogger(__name__)

TRANSFORM_REGISTRY = {
    TransformType.WEBPAGE_TRANSFORM: WebpageTransform,
}


class TransformFactory:
    """The ModelFactory class is used to create a BaseModel object from the given AutoLabelConfig configuration."""

    @staticmethod
    def from_dict(dict: Dict) -> BaseTransform:
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
            **params,
        )


def register_transform(name, transform_cls):
    TRANSFORM_REGISTRY[name] = transform_cls
