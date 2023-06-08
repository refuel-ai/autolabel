from typing import Dict
from loguru import logger

from .base import BaseModel
from .anthropic import AnthropicLLM
from .openai import OpenAILLM
from .hf_pipeline import HFPipelineLLM
from .refuel import RefuelLLM
from .palm import PaLMLLM

from autolabel.configs import AutolabelConfig
from autolabel.schema import ModelProvider
from autolabel.cache import BaseCache

MODEL_PROVIDER_TO_IMPLEMENTATION: Dict[ModelProvider, BaseModel] = {
    ModelProvider.OPENAI: OpenAILLM,
    ModelProvider.ANTHROPIC: AnthropicLLM,
    ModelProvider.HUGGINGFACE_PIPELINE: HFPipelineLLM,
    ModelProvider.REFUEL: RefuelLLM,
    ModelProvider.GOOGLE: PaLMLLM,
    # We will add more providers here in the future. See roadmap at [TODO]
}


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
        model_provider = ModelProvider(config.provider())
        try:
            model_cls = MODEL_PROVIDER_TO_IMPLEMENTATION[model_provider]
        except ValueError as e:
            logger.error(
                f"{config.provider()} is not in the list of supported providers: \
                {MODEL_PROVIDER_TO_IMPLEMENTATION.keys()}"
            )
            return None
        return model_cls(config, cache)
