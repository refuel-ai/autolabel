import json
import logging
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import Generation, RefuelLLMResult

logger = logging.getLogger(__name__)


class HFPipelineMultimodal(BaseModel):
    DEFAULT_MODEL = "HuggingFaceM4/idefics-9b-instruct"
    DEFAULT_PARAMS = {"temperature": 0.0, "quantize": 8}

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None, tokenizer: Optional[AutoTokenizer] = None) -> None:
        super().__init__(config, cache, tokenizer)
        try:
            from transformers import (
                AutoConfig,
                AutoModelForPreTraining,
                AutoProcessor,
                pipeline,
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

        processor = AutoProcessor.from_pretrained(self.model_name)
        quantize_bits = self.model_params["quantize"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not torch.cuda.is_available():
            model = AutoModelForPreTraining.from_pretrained(self.model_name)
        elif quantize_bits == 8:
            model = AutoModelForPreTraining.from_pretrained(
                self.model_name, load_in_8bit=True, device_map="auto"
            )
        elif quantize_bits == "16":
            model = AutoModelForPreTraining.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = AutoModelForPreTraining.from_pretrained(
                self.model_name, device_map="auto"
            )

        self.preprocessor = processor
        self.llm = model
        self.exit_condition = processor.tokenizer(
            "<end_of_utterance>", add_special_tokens=False
        ).input_ids
        self.bad_words_ids = processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids

    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        generations = []
        for prompt in prompts:
            parsed_prompt = json.loads(prompt)
            prompt = [
                "User: ",
                parsed_prompt["image_url"],
                parsed_prompt["text"],
                "<end_of_utterance>",
                "\nAssistant:",
            ]
            inputs = self.preprocessor(
                prompt, add_end_of_utterance_token=False, return_tensors="pt"
            ).to(self.device)
            generated_ids = self.llm.generate(
                **inputs,
                eos_token_id=self.exit_condition,
                bad_words_ids=self.bad_words_ids,
                max_length=512,
            )
            generated_text = self.preprocessor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            generated_text = generated_text.split(prompt[-1])[-1].strip()
            generations.append(
                [
                    Generation(
                        text=generated_text,
                        generation_info=None,
                    )
                ]
            )
        return RefuelLLMResult(
            generations=generations, errors=[None] * len(generations)
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        # Model inference for this model is being run locally
        # Revisit this in the future when we support HF inference endpoints
        return 0.0

    def returns_token_probs(self) -> bool:
        return False
