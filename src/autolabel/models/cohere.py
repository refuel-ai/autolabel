import os
from time import time
from typing import Dict, List, Optional

from langchain.schema import Generation

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import ErrorType, LabelingError, RefuelLLMResult


class CohereLLM(BaseModel):
    # Default parameters for OpenAILLM
    DEFAULT_MODEL = "command"
    DEFAULT_MODEL_PARAMS = {
        "max_tokens": 512,
        "temperature": 0.0,
    }

    # Reference: https://cohere.com/pricing
    COST_PER_TOKEN = 15 / 1_000_000

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        try:
            import cohere
            from langchain_community.llms import Cohere
        except ImportError:
            raise ImportError(
                "cohere is required to use the cohere LLM. Please install it with the following command: pip install 'refuel-autolabel[cohere]'"
            )

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL

        # populate model params and initialize the LLM
        model_params = config.model_params()
        self.model_params = {
            **self.DEFAULT_MODEL_PARAMS,
            **model_params,
        }
        self.llm = Cohere(model=self.model_name, **self.model_params)
        self.co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        try:
            start_time = time()
            result = self.llm.generate(prompts)
            end_time = time()
            return RefuelLLMResult(
                generations=result.generations,
                errors=[None] * len(result.generations),
                latencies=[end_time - start_time] * len(result.generations),
            )
        except Exception as e:
            return RefuelLLMResult(
                generations=[[Generation(text="")] for _ in prompts],
                errors=[
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR,
                        error_message=str(e),
                    )
                    for _ in prompts
                ],
                latencies=[0 for _ in prompts],
            )

    def get_cost(
        self, prompt: str, label: Optional[str] = "", llm_output: Optional[Dict] = None
    ) -> float:
        num_prompt_toks = len(self.co.tokenize(prompt).tokens)
        if label:
            num_label_toks = len(self.co.tokenize(label).tokens)
        else:
            num_label_toks = self.model_params["max_tokens"]

        return self.COST_PER_TOKEN * (num_prompt_toks + num_label_toks)

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.co.tokenize(prompt).tokens)
