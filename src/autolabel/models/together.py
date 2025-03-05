import logging
import os
from time import time
from typing import Dict, List, Optional

from langchain.schema import Generation
from transformers import AutoTokenizer

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import ErrorType, LabelingError, RefuelLLMResult

logger = logging.getLogger(__name__)


class TogetherLLM(BaseModel):
    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    DEFAULT_PARAMS = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "request_timeout": 30,
        "logprobs": 1
    }

    # Reference: https://www.together.ai/pricing
    COST_PER_PROMPT_TOKEN = {
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88 / 1_000_000)
    }
    COST_PER_COMPLETION_TOKEN = {
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88 / 1_000_000)
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__(config, cache, tokenizer)

        try:
            from together import Together
        except ImportError:
            raise ImportError(
                "together is required to use the Together.ai LLM. Please install it with the following command: pip install 'refuel-autolabel[together]'"
            )

        if os.getenv("TOGETHER_API_KEY") is None:
            raise ValueError("TOGETHER_API_KEY environment variable not set")

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        # populate model params
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        
        self.client = Together()

    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        
        for prompt in prompts:
            try:
                start_time = time()
                
                # Format the prompt based on the model
                messages = [{"role": "user", "content": prompt}]
                
                # Call the Together.ai API
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    **self.model_params,
                )
                
                end_time = time()
                
                # Extract the generated text
                generated_text = response.choices[0].message.content
                
                # TODO: logprobs
                
                generations.append([Generation(text=generated_text)])
                errors.append(None)
                latencies.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Error generating label: {e}")
                generations.append([Generation(text="")])
                errors.append(
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR,
                        error_message=str(e),
                    )
                )
                latencies.append(0)
        
        return RefuelLLMResult(
            generations=generations,
            errors=errors,
            latencies=latencies,
        )

    async def _alabel(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        # For now, we'll use the synchronous implementation
        return self._label(prompts, output_schema)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        num_prompt_tokens = self.get_num_tokens(prompt)
        
        if label:
            num_completion_tokens = self.get_num_tokens(label)
        else:
            num_completion_tokens = self.model_params["max_tokens"]
        
        # Get the cost per token for this model
        cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN.get(
            self.model_name
        )
        cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN.get(
            self.model_name
        )
        
        return (num_prompt_tokens * cost_per_prompt_token) + (
            num_completion_tokens * cost_per_completion_token
        )

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        if not prompt:
            return 0    
        return len(prompt) // 4