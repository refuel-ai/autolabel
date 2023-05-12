from typing import List, Optional
import json
from langchain.schema import LLMResult, Generation
from botocore.config import Config
from loguru import logger

from autolabel.models import BaseModel
from autolabel.configs import ModelConfig
from autolabel.utils import retry_session


class RefuelLLM(BaseModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # populate model name
        # This is unused today, but in the future could
        # be used to decide which refuel model is queried
        self.model_name = config.get_model_name()

        # initialize runtime
        config = Config(retries={"max_attempts": 10, "mode": "standard"})
        self.BASE_API = "https://api.refuel.ai/llm"
        self.RETRY_LIMIT = 5
        self.SESSION = retry_session(self.RETRY_LIMIT)

    def label(self, prompts: List[str]) -> LLMResult:
        try:
            generations = []
            for prompt in prompts:
                payload = json.dumps(
                    {"model_input": prompt, "task": "generate"}
                ).encode("utf-8")
                response = self.SESSION.post(self.BASE_API, data=payload)
                if response.status_code == 200:
                    generations.append([Generation(text=response.text.strip('"'))])
                else:
                    # This signifies an error in generating the response using RefuelLLm
                    logger.error(
                        "Unable to generate prediction: ",
                        response.text,
                        response.status_code,
                    )
                    generations.append([Generation(text="")])
            return LLMResult(generations=generations)
        except Exception as e:
            print(f"Error generating from LLM: {e}, returning empty result")
            generations = [[Generation(text="")] for _ in prompts]
            return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0
