import logging
from tabulate import tabulate
from typing import Union, Callable, Any
from .base import BaseModel
from collections import namedtuple
from autolabel.configs import AutolabelConfig
from autolabel.schema import ModelProvider
from autolabel.cache import BaseCache

logger = logging.getLogger(__name__)

MODEL_NAME_IDENTIFIER = lambda provider, model_name: f"{provider}{model_name}"


class ModelRegistry:
    """Registry."""

    def __init__(self) -> None:
        """Initialize."""
        self._data = {}
        self._model_meta = []
        self._headers = ["provider", "model_name", "model"]

    def __getitem__(self, key: str) -> Any:
        """Fetch Value given key"""
        return self._data[key]

    def register(
        self, provider: str, model_name: str, model: Union[type, Callable]
    ) -> None:
        """Register the model object."""
        model_meta = namedtuple("ModelMeta", self._headers)
        self._model_meta += [
            model_meta(
                provider=provider,
                model_name=model_name,
                model=model,
            )
        ]

        model_ref_name = MODEL_NAME_IDENTIFIER(provider=provider, model_name=model_name)
        assert (
            model_ref_name not in self._data
        ), f"An provider = {provider} with model_name = {model_name} was already registered!"
        self._data[model_ref_name] = model

    def __repr__(self) -> str:
        """Generate Module string."""
        table_data = tabulate(self._model_meta, headers=self._headers, tablefmt="grid")
        return "MODEL Registry:\n" + table_data


MODEL_REGISTRY = ModelRegistry()


def _register_openai() -> None:
    """Register OpenAI models"""

    from autolabel.models.openai import OpenAILLM

    all_openai_models = OpenAILLM.CHAT_ENGINE_MODELS + OpenAILLM.MODELS_WITH_TOKEN_PROBS

    for model_name in all_openai_models:
        MODEL_REGISTRY.register(
            provider=ModelProvider.OPENAI,
            model_name=model_name,
            model=OpenAILLM,
        )


def _register_anothropic() -> None:
    """Register Anthropic models"""

    from autolabel.models.anthropic import AnthropicLLM

    model_name = AnthropicLLM.DEFAULT_MODEL

    MODEL_REGISTRY.register(
        provider=ModelProvider.ANTHROPIC,
        model_name=model_name,
        model=AnthropicLLM
    )


def _register_cohere() -> None:
    """Register Cohere models"""

    from autolabel.models.cohere import CohereLLM

    model_name = CohereLLM.DEFAULT_MODEL

    MODEL_REGISTRY.register(
        provider=ModelProvider.COHERE,
        model_name=model_name,
        model=CohereLLM,
    )


def _register_hugging_face_models() -> None:
    """Register Cohere models"""

    from autolabel.models.hf_pipeline import HFPipelineLLM

    model_name = HFPipelineLLM.DEFAULT_MODEL

    MODEL_REGISTRY.register(
        provider=ModelProvider.HUGGINGFACE_PIPELINE,
        model_name=model_name,
        model=HFPipelineLLM,
    )


def _register_palm() -> None:
    """Register Google models"""

    from autolabel.models.palm import PaLMLLM

    model_names = PaLMLLM.CHAT_ENGINE_MODELS + [PaLMLLM.DEFAULT_MODEL]

    for model_name in model_names:
        MODEL_REGISTRY.register(
            provider=ModelProvider.GOOGLE,
            model_name=model_name,
            model=PaLMLLM,
        )


def _register_refuel() -> None:
    """Register Refuel models"""

    from autolabel.models.refuel import RefuelLLM

    model_name = "refuel"
    MODEL_REGISTRY.register(
        provider=ModelProvider.REFUEL,
        model_name=model_name,
        model=RefuelLLM,
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
            model_cls = MODEL_REGISTRY[
                MODEL_NAME_IDENTIFIER(provider=provider, model_name=model_name)
            ]
            model_obj = model_cls(config=config, cache=cache)
            # The below ensures that users should based off of the BaseModel
            # when creating/registering custom models.
            assert isinstance(model_obj, BaseModel)
        except ValueError as e:
            logger.error(
                f"{config.provider()} is not in the list of supported providers: \
                {list(ModelProvider.__members__.keys())}"
            )
            return None
        return model_obj


_register_openai()
_register_anothropic()
_register_cohere()
_register_hugging_face_models()
_register_palm()
_register_refuel()