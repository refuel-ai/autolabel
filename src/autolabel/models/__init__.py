from typing import Dict
from loguru import logger

from .base import BaseModel
from .anthropic import AnthropicLLM
from .openai import OpenAILLM
from .hf_pipeline import HFPipelineLLM
from .refuel import RefuelLLM

from autolabel.configs import ModelConfig
from autolabel.cache import BaseCache

MODEL_PROVIDER_TO_IMPLEMENTATION: Dict[str, BaseModel] = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "huggingface_pipeline": HFPipelineLLM,
    "refuel": RefuelLLM
    # We will add more providers here in the future. See roadmap at [TODO]
}


class ModelFactory:
    @staticmethod
    def from_config(config: ModelConfig, cache: BaseCache = None) -> BaseModel:
        model_provider = config.get_provider()
        if model_provider not in MODEL_PROVIDER_TO_IMPLEMENTATION:
            logger.error(
                f"Model provider {model_provider} is not in the list of supported providers: {MODEL_PROVIDER_TO_IMPLEMENTATION.keys()}"
            )
            return None
        model_cls = MODEL_PROVIDER_TO_IMPLEMENTATION[model_provider]
        return model_cls(config, cache)
