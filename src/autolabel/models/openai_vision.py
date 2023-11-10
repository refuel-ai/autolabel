from functools import cached_property
from typing import List, Optional
import logging
import json

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import RefuelLLMResult
from langchain.schema import HumanMessage, Generation
from functools import partial

import os

logger = logging.getLogger(__name__)


class OpenAIVisionLLM(BaseModel):
    CHAT_ENGINE_MODELS = ["gpt-4-vision-preview"]
    MODELS_WITH_TOKEN_PROBS = []

    # Default parameters for OpenAIVisionLLM
    DEFAULT_MODEL = "gpt-4-vision-preview"
    DEFAULT_PARAMS_CHAT_ENGINE = {
        "max_tokens": 300,
        "temperature": 0.0,
        "request_timeout": 30,
    }

    # Reference: https://openai.com/pricing
    COST_PER_PROMPT_TOKEN = {
        "gpt-4-vision-preview": 0.01 / 1000,
    }
    COST_PER_COMPLETION_TOKEN = {
        "gpt-4-vision-preview": 0.03 / 1000,
    }

    @cached_property
    def _engine(self) -> str:
        if self.model_name is not None and self.model_name in self.CHAT_ENGINE_MODELS:
            return "chat"
        else:
            return "completion"

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        try:
            from openai import OpenAI
            import tiktoken
        except ImportError:
            raise ImportError(
                "openai is required to use the OpenAIVisionLLM. Please install it with the following command: pip install 'refuel-autolabel[openai]'"
            )

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        self.model_params = self.DEFAULT_PARAMS_CHAT_ENGINE

        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        client = OpenAI()
        self.llm = partial(
            client.chat.completions.create,
            model=self.model_name,
            max_tokens=self.DEFAULT_PARAMS_CHAT_ENGINE["max_tokens"],
        )
        self.tiktoken = tiktoken

    def _generate_logit_bias(self) -> None:
        """Generates logit bias for the labels specified in the config

        Returns:
            Dict: logit bias and max tokens
        """
        if len(self.config.labels_list()) == 0:
            logger.warning(
                "No labels specified in the config. Skipping logit bias generation."
            )
            return {}
        encoding = self.tiktoken.encoding_for_model(self.model_name)
        logit_bias = {}
        max_tokens = 0
        for label in self.config.labels_list():
            if label not in logit_bias:
                tokens = encoding.encode(label)
                for token in tokens:
                    logit_bias[token] = self.config.logit_bias()
                max_tokens = max(max_tokens, len(tokens))

        return {"logit_bias": logit_bias, "max_tokens": max_tokens}

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        for prompt in prompts:
            parsed_prompt = json.loads(prompt)
            result = self.llm(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": parsed_prompt["text"]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": parsed_prompt["image_url"],
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ]
            )
            generations.append(
                [
                    Generation(
                        text=result.choices[0].message.content,
                        generation_info=None,
                    )
                ]
            )
        return RefuelLLMResult(
            generations=generations, errors=[None] * len(generations)
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        encoding = self.tiktoken.encoding_for_model(self.model_name)
        num_prompt_toks = len(encoding.encode(prompt))
        if label:
            num_label_toks = len(encoding.encode(label))
        else:
            # get an upper bound
            num_label_toks = self.model_params["max_tokens"]

        cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN[self.model_name]
        cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN[self.model_name]
        return (num_prompt_toks * cost_per_prompt_token) + (
            num_label_toks * cost_per_completion_token
        )

    def returns_token_probs(self) -> bool:
        return (
            self.model_name is not None
            and self.model_name in self.MODELS_WITH_TOKEN_PROBS
        )
