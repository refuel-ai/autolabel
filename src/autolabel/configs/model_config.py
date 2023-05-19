from typing import Dict, Union

from .base import BaseConfig


class ModelConfig(BaseConfig):
    """Sets the model configuration of Autolabel.

    Attributes:
        provider_name (str): Name of the model provider. One of "openai", "anthropic", "huggingface_pipeline", "refuel".
        model_name (str): Name of the model. For eg:- "text-davinci-003" for OpenAI's davinci model.
        model_params (Dict): Model parameters. For eg:- {"max_tokens": 1000, "temperature": 0.0, "model_kwargs": {"logprobs": 1}} for OpenAI's davinci model.
        has_logprob (bool): Whether or not the model has logprob. For eg:- True for OpenAI's davinci model.
    """

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
            self.LLM_PROVIDER_KEY, self.DEFAULT_LLM_PROVIDER
        )
        return provider_name

    def get_model_name(self) -> str:
        return self.config.get(self.LLM_MODEL_KEY, None)

    def get_model_params(self) -> Dict:
        return self.config.get(self.MODEL_PARAMS_KEY, {})

    def get_has_logprob(self) -> bool:
        return self.config.get(self.HAS_LOGPROB_KEY, False)
