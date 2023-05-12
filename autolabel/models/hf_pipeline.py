from typing import List, Optional
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

from langchain.schema import LLMResult, Generation

from autolabel.models import BaseModel
from autolabel.configs import ModelConfig


class HFPipelineLLM(BaseModel):
    DEFAULT_MODEL = "google/flan-t5-xxl"
    DEFAULT_PARAMS = {"max_new_tokens": 1000, "temperature": 0.0, "quantize": 8}

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # populate model name
        self.model_name = config.get_model_name() or self.DEFAULT_MODEL

        # populate model params
        model_params = config.get_model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}

        # initialize HF pipeline
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        quantize_bits = self.model_params["quantize"]
        if quantize_bits == 8:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, load_in_8bit=True, device_map="auto"
            )
        elif quantize_bits == "16":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, device_map="auto"
            )

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

    def label(self, prompts: List[str]) -> LLMResult:
        try:
            return self.llm.generate(prompts)
        except Exception as e:
            print(f"Error generating from LLM: {e}, returning empty result")
            generations = [[Generation(text="")] for _ in prompts]
            return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        # Model inference for this model is being run locally
        # Revisit this in the future when we support HF inference endpoints
        return 0.0
