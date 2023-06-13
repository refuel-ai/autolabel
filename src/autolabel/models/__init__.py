import logging

from .base import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.schema import ModelProvider
from autolabel.cache import BaseCache

logger = logging.getLogger(__name__)


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
            if model_provider == ModelProvider.OPENAI:
                from .openai import OpenAILLM

                model_cls = OpenAILLM
            elif model_provider == ModelProvider.ANTHROPIC:
                from .anthropic import AnthropicLLM

                model_cls = AnthropicLLM
            elif model_provider == ModelProvider.HUGGINGFACE_PIPELINE:
                from .hf_pipeline import HFPipelineLLM

                model_cls = HFPipelineLLM
            elif model_provider == ModelProvider.REFUEL:
                from .refuel import RefuelLLM

                model_cls = RefuelLLM
            elif model_provider == ModelProvider.GOOGLE:
                from .palm import PaLMLLM

                model_cls = PaLMLLM
            else:
                raise ValueError
        except ValueError as e:
            logger.error(
                f"{config.provider()} is not in the list of supported providers: \
                {list(ModelProvider.__members__.keys())}"
            )
            return None
        return model_cls(config, cache)
