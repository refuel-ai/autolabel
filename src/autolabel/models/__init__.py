import logging
from .base import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.schema import ModelProvider
from autolabel.cache import BaseCache

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {}


def register_model(name, model_cls):
    MODEL_REGISTRY[name] = model_cls


def _register_openai() -> None:
    """Register OpenAI models"""
    from autolabel.models.openai import OpenAILLM

    register_model(
        name=ModelProvider.OPENAI,
        model_cls=OpenAILLM,
    )


def _register_anothropic() -> None:
    """Register Anthropic models"""
    from autolabel.models.anthropic import AnthropicLLM

    register_model(name=ModelProvider.ANTHROPIC, model_cls=AnthropicLLM)


def _register_cohere() -> None:
    """Register Cohere models"""
    from autolabel.models.cohere import CohereLLM

    register_model(
        name=ModelProvider.COHERE,
        model_cls=CohereLLM,
    )


def _register_hugging_face_models() -> None:
    """Register Cohere models"""
    from autolabel.models.hf_pipeline import HFPipelineLLM

    register_model(
        name=ModelProvider.HUGGINGFACE_PIPELINE,
        model_cls=HFPipelineLLM,
    )


def _register_palm() -> None:
    """Register Google models"""
    from autolabel.models.palm import PaLMLLM

    register_model(
        name=ModelProvider.GOOGLE,
        model_cls=PaLMLLM,
    )


def _register_refuel() -> None:
    """Register Refuel models"""
    from autolabel.models.refuel import RefuelLLM

    register_model(
        name=ModelProvider.REFUEL,
        model_cls=RefuelLLM,
    )


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


_register_openai()
_register_anothropic()
_register_cohere()
_register_hugging_face_models()
_register_palm()
_register_refuel()
