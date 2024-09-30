import logging
from typing import List, Optional, Dict
from time import time
from langchain.schema import Generation

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import ErrorType, LabelingError, RefuelLLMResult


logger = logging.getLogger(__name__)


class HFPipelineLLM(BaseModel):
    DEFAULT_MODEL = "google/flan-t5-xxl"
    DEFAULT_PARAMS = {"temperature": 0.0, "quantize": 8}

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)

        from langchain.llms import HuggingFacePipeline

        try:
            from transformers import (
                AutoConfig,
                AutoModelForSeq2SeqLM,
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
            )
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            )
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )

        try:
            import torch
        except ImportError:
            raise ValueError(
                "Could not import torch package. "
                "Please it install it with `pip install torch`."
            )
        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL

        # populate model params
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        # initialize HF pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=False, add_prefix_space=True
        )
        quantize_bits = self.model_params["quantize"]
        model_config = AutoConfig.from_pretrained(self.model_name)
        if isinstance(model_config, tuple(MODEL_FOR_CAUSAL_LM_MAPPING)):
            AutoModel = AutoModelForCausalLM
        elif isinstance(model_config, tuple(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING)):
            AutoModel = AutoModelForSeq2SeqLM
        else:
            raise ValueError(
                "model_name is neither a causal LM nor a seq2seq LM. Please check the model_name."
            )

        if not torch.cuda.is_available():
            model = AutoModel.from_pretrained(self.model_name)
        elif quantize_bits == 8:
            model = AutoModel.from_pretrained(
                self.model_name, load_in_8bit=True, device_map="auto"
            )
        elif quantize_bits == "16":
            model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = AutoModel.from_pretrained(self.model_name, device_map="auto")

        model_kwargs = dict(self.model_params)  # make a copy of the model params
        model_kwargs.pop("quantize", None)  # remove quantize from the model params
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=self.tokenizer,
            **model_kwargs,
        )

        # initialize LLM
        self.llm = HuggingFacePipeline(pipeline=pipe, model_kwargs=model_kwargs)

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
            logger.exception(f"Unable to generate prediction: {e}")
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

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        # Model inference for this model is being run locally
        # Revisit this in the future when we support HF inference endpoints
        return 0.0

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
