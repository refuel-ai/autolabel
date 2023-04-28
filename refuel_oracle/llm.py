import copy
from enum import Enum
from typing import Dict, List, Any
from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic, BaseLLM, Cohere, HuggingFacePipeline, OpenAI
from langchain.schema import Generation, HumanMessage, LLMResult
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import json


# All available LLM providers
class LLMProvider(str, Enum):
    openai = "openai"
    openai_chat = "openai_chat"
    anthropic = "anthropic"
    cohere = "cohere"
    huggingface = "huggingface"


# All chat models
CHAT_MODELS = {LLMProvider.openai: (["gpt-3.5-turbo"], LLMProvider.openai_chat)}

# Default model names for each provider
DEFAULT_MODEL_NAMES = {
    LLMProvider.openai: "gpt-3.5-turbo",
    LLMProvider.anthropic: "claude-v1",
    LLMProvider.huggingface: "google/flant-t5-xxl",
}


class LLMConfig:
    LLM_PROVIDER_KEY = "provider_name"
    LLM_MODEL_KEY = "model_name"
    QUANTIZE_BITS_KEY = "quantize"
    MODEL_PARAMS_KEY = "model_params"
    HAS_LOGPROB_KEY = "has_logprob"
    DEFAULT_LLM_CONFIG = {
        "model_name": "gpt-3.5-turbo",
        "provider_name": "openai",
    }

    def __init__(self, config_dict: Dict) -> None:
        if config_dict is None:
            config_dict = self.DEFAULT_LLM_CONFIG
        self.dict = config_dict

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.dict.get(key, default_value)

    def keys(self) -> List:
        return list(self.dict.keys())

    def __getitem__(self, key):
        return self.dict[key]

    def get_provider(self) -> str:
        provider_name = self.dict.get(self.LLM_PROVIDER_KEY, LLMProvider.openai)
        model_name = self.get_model_name()
        # Converting an open ai provider to openai_chat internally to handle
        # chat models separately
        if provider_name in CHAT_MODELS and model_name in CHAT_MODELS[provider_name][0]:
            provider_name = CHAT_MODELS[provider_name][1]

        return provider_name

    def get_model_name(self) -> str:
        provider_name = self.dict.get(self.LLM_PROVIDER_KEY, LLMProvider.openai)
        model_name = self.dict.get(
            self.LLM_MODEL_KEY, DEFAULT_MODEL_NAMES[provider_name]
        )
        return model_name

    def get_model_params(self) -> Dict:
        return self.dict.get(self.MODEL_PARAMS_KEY, {})

    def get_quantize_bits(self) -> str:
        # Support quantization default value as 16 bits.
        return self.dict.get(self.QUANTIZE_BITS_KEY, "16")

    def get_has_logprob(self) -> bool:
        """
        Returns whether or not current task supports returning LogProb confidence of its response
        """
        return self.dict.get(self.HAS_LOGPROB_KEY, "False") == "True"

    @classmethod
    def from_json(cls, json_file_path: str, **kwargs):
        try:
            config_dict = json.load(open(json_file_path))
        except ValueError:
            logger.error("JSON file: {} not loaded successfully", json_file_path)
            return None

        config_dict.update(kwargs)

        return LLMConfig(config_dict)


class LLMLabeler:
    def __init__(self, config: LLMConfig, base_llm: BaseLLM) -> None:
        self.config = config
        self.base_llm = base_llm

    def is_chat_model(self, llm_provider: LLMProvider) -> bool:
        # Add more models here that are in `langchain.chat_models.*`
        return llm_provider == LLMProvider.openai_chat

    def generate(self, prompts: List) -> LLMResult:
        llm_provider = self.config.get_provider()
        if self.is_chat_model(llm_provider):
            # Need to convert list[prompts] -> list[messages]
            # Currently the entire prompt is stuck into the "human message"
            # We might consider breaking this up into human vs system message in future
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
        try:
            return self.base_llm.generate(prompts)
        except Exception as e:
            print(f"Error generating from LLM: {e}, returning empty result")
            generations = [[Generation(text="")] for prompt in prompts]
            return LLMResult(generations=generations)


class LLMFactory:
    # Default parameters that we will use to initialize LLMs from a provider
    PROVIDER_TO_DEFAULT_PARAMS = {
        LLMProvider.openai: {
            "max_tokens": 100,
            "temperature": 0.0,
            "model_kwargs": {"logprobs": 1},
        },
        LLMProvider.openai_chat: {
            "max_tokens": 100,
            "temperature": 0.0,
        },
        LLMProvider.cohere: {
            # TODO
        },
        LLMProvider.anthropic: {
            "max_tokens_to_sample": 100,
            "temperature": 0.0,
        },
        LLMProvider.huggingface: {
            "max_tokens": 30,
            "temperature": 0.0,
        },
    }

    # Provider mapping to the langchain LLM wrapper
    PROVIDER_TO_LLM = {
        LLMProvider.anthropic: Anthropic,
        LLMProvider.cohere: Cohere,
        LLMProvider.openai: OpenAI,
        LLMProvider.openai_chat: ChatOpenAI,
        LLMProvider.huggingface: HuggingFacePipeline,
    }

    @staticmethod
    def _resolve_params(default_params: Dict, provided_params: Dict) -> Dict:
        final_params = copy.deepcopy(default_params)

        if not provided_params or not isinstance(provided_params, dict):
            return final_params

        for key, val in provided_params.items():
            final_params[key] = val
        return final_params

    @staticmethod
    def from_config(config: LLMConfig) -> LLMLabeler:
        # TODO: llm_model might need to be rolled up into llm_params in the future
        llm_provider = config.get_provider()
        llm_model = config.get_model_name()
        llm_params = config.get_model_params()
        llm_cls = LLMFactory.PROVIDER_TO_LLM[llm_provider]
        llm_params = LLMFactory._resolve_params(
            LLMFactory.PROVIDER_TO_DEFAULT_PARAMS[llm_provider], llm_params
        )
        if llm_provider in [LLMProvider.openai, LLMProvider.openai_chat]:
            base_llm = llm_cls(model_name=llm_model, **llm_params)
        elif llm_provider == LLMProvider.huggingface:
            tokenizer = AutoTokenizer.from_pretrained(llm_model)
            quantize_bits = config.get_quantize_bits()
            if quantize_bits == "8":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    llm_model, load_in_8bit=True, device_map="auto"
                )
            elif quantize_bits == "16":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    llm_model, torch_dtype=torch.float16, device_map="auto"
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    llm_model, device_map="auto"
                )
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
            # TODO: Does not support batch inference yet
            base_llm = llm_cls(pipeline=pipe, model_kwargs=llm_params)
        else:
            base_llm = llm_cls(model=llm_model, **llm_params)
        return LLMLabeler(config, base_llm)
