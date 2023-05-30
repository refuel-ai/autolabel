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
    ModelProvider.REFUEL: RefuelLLM
    # We will add more providers here in the future. See roadmap at [TODO]
}


class ModelFactory:
    @staticmethod
    def from_config(config: AutolabelConfig, cache: BaseCache = None) -> BaseModel:
        try:
            model_provider = ModelProvider(config.provider())
            model_cls = MODEL_PROVIDER_TO_IMPLEMENTATION[model_provider]
            return model_cls(config, cache)
        except ValueError as e:
            logger.error(
                f"{config.provider()} is not in the list of supported providers: \
                {MODEL_PROVIDER_TO_IMPLEMENTATION.keys()}"
            )
            return None
