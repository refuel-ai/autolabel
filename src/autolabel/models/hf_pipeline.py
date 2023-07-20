from typing import List, Optional
from langchain.llms import HuggingFacePipeline
from langchain.schema import LLMResult, Generation

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache


class HFPipelineLLM(BaseModel):
    DEFAULT_MODEL = "google/flan-t5-xxl"
    DEFAULT_PARAMS = {"temperature": 0.0, "quantize": 8}

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, LlamaForCausalLM
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
        self.model_name = "NousResearch/Llama-2-13b-chat-hf"

        # populate model params
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}

        self.llm = LlamaForCausalLM.from_pretrained(self.model_name, device_map="auto", cache_dir="/workspace/dhruva")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def _label(self, prompts: List[str]) -> LLMResult:
        def sanitize_out(str):
            str = str.split(":")[-1].split("(")[-1].split(")")[0]
            blacklist = ["Sure! Here's my answer:", "Sure! Here's the answer:", "<s>", "</s>", "\n"]
            for word in blacklist:
                str = str.replace(word, "")
            return str.strip().lower()
            
        try:
            prompts = [f"[INST]{prompt}[/INST]" for prompt in prompts]
            input_ids = self.tokenizer(prompts, return_tensors='pt').input_ids.cuda()
            output = self.llm.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512, top_p=0.95, repetition_penalty=1.15)
            decoded_outputs = self.tokenizer.batch_decode(output)
            to_ret = LLMResult(generations=[[Generation(text=sanitize_out(decoded_outputs[i].replace(prompts[i], "")))] for i in range(len(decoded_outputs))])
            return to_ret
        except Exception as e:
            print(f"Error generating from LLM: {e}, retrying each prompt individually")
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        # Model inference for this model is being run locally
        # Revisit this in the future when we support HF inference endpoints
        return 0.0

    def returns_token_probs(self) -> bool:
        return False
