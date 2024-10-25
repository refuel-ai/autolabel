import json
import logging
import os
from functools import cached_property, partial
from time import time
from typing import Dict, List, Optional

from langchain.schema import Generation
from transformers import AutoTokenizer

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import RefuelLLMResult

logger = logging.getLogger(__name__)


class OpenAIVisionLLM(BaseModel):
    CHAT_ENGINE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-08-06",
        "gpt-4-vision-preview",
    ]
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
        "gpt-4o": 0.005 / 1000,
        "gpt-4o-mini": 0.000015 / 1000,
        "gpt-4o-2024-08-06": 0.0025 / 1000,
    }
    COST_PER_COMPLETION_TOKEN = {
        "gpt-4-vision-preview": 0.03 / 1000,
        "gpt-4o": 0.015 / 1000,
        "gpt-4o-mini": 0.00006 / 1000,
        "gpt-4o-2024-08-06": 0.01 / 1000,
    }

    @cached_property
    def _engine(self) -> str:
        if self.model_name is not None and self.model_name in self.CHAT_ENGINE_MODELS:
            return "chat"
        else:
            return "completion"

    def __init__(
        self,
        config: AutolabelConfig,
        cache: Optional[BaseCache] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__(config, cache, tokenizer)
        try:
            import tiktoken
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required to use the OpenAIVisionLLM. Please install it with the following command: pip install 'refuel-autolabel[openai]'"
            )

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS_CHAT_ENGINE, **model_params}

        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI()
        self.llm = partial(
            self.client.chat.completions.create,
            model=self.model_name,
            max_tokens=self.model_params["max_tokens"],
        )
        self.tiktoken = tiktoken
        self.image_cols = config.image_columns()

    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        generations = []
        start_time = time()
        for prompt in prompts:
            parsed_prompt = json.loads(prompt)
            try:
                content = [{"type": "text", "text": parsed_prompt["text"]}]
                if self.image_cols:
                    for col in self.image_cols:
                        if (
                            parsed_prompt.get(col) is not None
                            and len(parsed_prompt[col]) > 0
                        ):
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": parsed_prompt[col],
                                        "detail": "high",
                                    },
                                }
                            )
                result = self.llm(
                    messages=[
                        {
                            "role": "user",
                            "content": content,
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
            except Exception as e:
                logger.error(f"Error generating label: {e}")
                generations.append(
                    [
                        Generation(
                            text="",
                            generation_info=None,
                        )
                    ]
                )
        return RefuelLLMResult(
            generations=generations,
            errors=[None] * len(generations),
            latencies=[time() - start_time] * len(generations),
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        encoding = self.tiktoken.encoding_for_model(self.model_name)
        num_prompt_toks = len(encoding.encode(prompt))
        if label:
            num_label_toks = len(encoding.encode(label))
        else:
            # get an upper bound
            num_label_toks = self.model_params["max_tokens"]

        # upper bound on image cost with high detail
        cost_per_image = 765 * self.COST_PER_PROMPT_TOKEN[self.model_name]
        cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN[self.model_name]
        cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN[self.model_name]
        return (
            (num_prompt_toks * cost_per_prompt_token)
            + (num_label_toks * cost_per_completion_token)
            + cost_per_image
        )

    def returns_token_probs(self) -> bool:
        return (
            self.model_name is not None
            and self.model_name in self.MODELS_WITH_TOKEN_PROBS
        )

    def get_num_tokens(self, prompt: str) -> int:
        encoding = self.tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(prompt))
