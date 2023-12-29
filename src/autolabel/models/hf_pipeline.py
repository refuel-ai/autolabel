import logging
from typing import List, Optional, Dict
from time import time
from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import RefuelLLMResult


logger = logging.getLogger(__name__)


class HFPipelineLLM(BaseModel):
    DEFAULT_MODEL = "google/flan-t5-xxl"
    DEFAULT_PARAMS = {"temperature": 0.0, "quantize": 16, "max_new_tokens": 128}

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
        if config.logit_bias() != 0:
            self.model_params = {
                **self._generate_sequence_bias(),
                **self.model_params,
            }

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
                self.model_name, load_in_8bit=True, device_map="auto", cache_dir="/workspace"
            )
        elif quantize_bits == "16":
            model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto", cache_dir="/workspace"
            )
        else:
            model = AutoModel.from_pretrained(self.model_name, device_map="auto", cache_dir="/workspace")

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

    def _generate_sequence_bias(self) -> Dict:
        """Generates sequence bias dict to add to the config for the labels specified

        Returns:
            Dict: sequence bias, max new tokens, and num beams
        """
        if len(self.config.labels_list()) == 0:
            logger.warning(
                "No labels specified in the config. Skipping logit bias generation."
            )
            return {}
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )
        sequence_bias = {tuple([self.tokenizer.eos_token_id]): self.config.logit_bias()}
        max_new_tokens = 0
        for label in self.config.labels_list():
            tokens = tuple(
                self.tokenizer([label], add_special_tokens=False).input_ids[0]
            )
            for token in tokens:
                sequence_bias[tuple([token])] = self.config.logit_bias()
            max_new_tokens = max(max_new_tokens, len(tokens))

        return {
            "sequence_bias": sequence_bias,
            "max_new_tokens": max_new_tokens,
        }

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        try:
            start_time = time()
            result = self.llm.generate(prompts)
            logger.error(result)
            for idx in range(len(result.generations)):
                for gen in range(len(result.generations[idx])):
                    result.generations[idx][gen].text = result.generations[idx][gen].text.replace(prompts[idx], "").replace("<|im_end|>", "").strip("\n")
            end_time = time()
            return RefuelLLMResult(
                generations=result.generations,
                errors=[None] * len(result.generations),
                latencies=[end_time - start_time] * len(result.generations),
            )
        except Exception as e:
            logger.error(f"Error while generating output via HF Pipeline: {e}")
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        # Model inference for this model is being run locally
        # Revisit this in the future when we support HF inference endpoints
        return 0.0

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
