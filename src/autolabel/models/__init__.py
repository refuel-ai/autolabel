import logging
from .base import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.schema import ModelProvider
from autolabel.cache import BaseCache

logger = logging.getLogger(__name__)

from autolabel.models.openai import OpenAILLM
from autolabel.models.anthropic import AnthropicLLM
from autolabel.models.cohere import CohereLLM
from autolabel.models.palm import PaLMLLM
from autolabel.models.hf_pipeline import HFPipelineLLM
from autolabel.models.refuel import RefuelLLM

MODEL_REGISTRY = {
    ModelProvider.OPENAI: OpenAILLM,
    ModelProvider.ANTHROPIC: AnthropicLLM,
    ModelProvider.COHERE: CohereLLM,
    ModelProvider.HUGGINGFACE_PIPELINE: HFPipelineLLM,
    ModelProvider.GOOGLE: PaLMLLM,
    ModelProvider.REFUEL: RefuelLLM,
}


def register_model(name, model_cls):
    """Register Model class"""
    MODEL_REGISTRY[name] = model_cls


class ModelFactory:
    """The ModelFactory class is used to create a BaseModel object from the given AutoLabelConfig configuration."""

    @staticmethod
    def from_config(config: AutolabelConfig, cache: BaseCache = None) -> BaseModel:
        """
        Returns a BaseModel object configured with the settings found in the provided AutolabelConfig.
        Args:
            config: AutolabelConfig object containing project settings
            cache: cache allows for saving results in between labeling runs for future use
        Returns:
            model: a fully configured BaseModel object
        """
        provider = ModelProvider(config.provider())
        model_name = config.model_name()
        try:
            model_cls = MODEL_REGISTRY[provider]
            model_obj = model_cls(config=config, cache=cache)
            # The below ensures that users should based off of the BaseModel
            # when creating/registering custom models.
            assert isinstance(
                model_obj, BaseModel
            ), f"{model_obj} should inherit from autolabel.models.BaseModel"
        except ValueError as e:
            logger.error(
                f"provider={provider}, model={model_name} is not in the list of supported "
                f"providers {list(ModelProvider.__members__.keys())} or their respective models "
                f"\nCheckout \n\t`from autolabel.models import MODEL_REGISTRY; print(MODEL_REGISTRY)` "
                f"to fetch the list of supported providers and their models"
            )
            return None
        return model_obj
