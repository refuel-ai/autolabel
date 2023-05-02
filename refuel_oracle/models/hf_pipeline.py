from typing import List, Optional
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

from refuel_oracle.models import ModelConfig, BaseModel
from langchain.schema import LLMResult, Generation


class HFPipelineLLM(BaseModel):
    DEFAULT_MODEL = "google/flan-t5-xxl"
    DEFAULT_PARAMS = {"max_tokens": 100, "temperature": 0.0, "quantize": 16}

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
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            **self.model_params,
        )

        # initialize LLM
        self.llm = HuggingFacePipeline(pipeline=pipe, model_kwargs=self.model_params)

    def label(self, prompts: List[str]) -> List[LLMResult]:
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
