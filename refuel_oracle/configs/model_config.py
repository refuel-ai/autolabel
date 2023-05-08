from typing import Dict, Union

from .base import BaseConfig

class ModelConfig(BaseConfig):
    # config keys
    LLM_PROVIDER_KEY = "provider_name"
    LLM_MODEL_KEY = "model_name"
    MODEL_PARAMS_KEY = "model_params"
    HAS_LOGPROB_KEY = "has_logprob"

    # default values
    DEFAULT_LLM_PROVIDER = "openai"

    def __init__(self, config: Union[str, Dict]) -> None:
        super().__init__(config)

    def get_provider(self) -> str:
        provider_name = self.config.get(
            self.LLM_PROVIDER_KEY, self.DEFAULT_LLM_PROVIDER)
        return provider_name

    def get_model_name(self) -> str:
        return self.config.get(self.LLM_MODEL_KEY, None)

    def get_model_params(self) -> Dict:
        return self.config.get(self.MODEL_PARAMS_KEY, {})

    def get_has_logprob(self) -> bool:
        """
        Returns whether or not current task supports returning LogProb confidence of its response
        """
        return self.config.get(self.HAS_LOGPROB_KEY, False)
