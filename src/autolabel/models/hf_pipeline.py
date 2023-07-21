from typing import List, Optional
from langchain.llms import HuggingFacePipeline
from langchain.schema import LLMResult

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import RefuelLLMResult


class HFPipelineLLM(BaseModel):
    DEFAULT_MODEL = "google/flan-t5-xxl"
    DEFAULT_PARAMS = {"temperature": 0.0, "quantize": 8}

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
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
        super().__init__(config, cache)
        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL

        # populate model params
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}

        # initialize HF pipeline
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
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
            tokenizer=tokenizer,
            **model_kwargs,
        )

        # initialize LLM
        self.llm = HuggingFacePipeline(pipeline=pipe, model_kwargs=model_kwargs)

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        try:
            result = self.llm.generate(prompts)
            return RefuelLLMResult(
                generations=result.generations, errors=[None] * len(result.generations)
            )
        except Exception as e:
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        # Model inference for this model is being run locally
        # Revisit this in the future when we support HF inference endpoints
        return 0.0

    def returns_token_probs(self) -> bool:
        return False
