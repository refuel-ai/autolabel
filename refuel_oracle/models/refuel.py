from typing import List, Optional
import boto3
import json
from langchain.schema import LLMResult, Generation

from refuel_oracle.models import BaseModel
from refuel_oracle.configs import ModelConfig


class RefuelLLM(BaseModel):
    DEFAULT_MODEL = "huggingface-pytorch-inference-2023-05-09-21-00-24-326"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # populate model name
        self.model_name = config.get_model_name() or self.DEFAULT_MODEL
        # initialize runtime
        self.RUNTIME = boto3.client("sagemaker-runtime")

    def label(self, prompts: List[str]) -> LLMResult:
        try:
            generations = []
            for prompt in prompts:
                payload = json.dumps(
                    {
                        "input_type": "text",
                        "data": {
                            "input_text": prompt,
                        },
                    }
                ).encode("utf-8")
                response = self.RUNTIME.invoke_endpoint(
                    EndpointName=self.model_name,
                    ContentType="text/plain",
                    Body=payload,
                )
                generations.append(
                    [
                        Generation(
                            text=response["Body"].read().decode("utf-8").strip('"')
                        )
                    ]
                )
            return LLMResult(generations=generations)
        except Exception as e:
            print(f"Error generating from LLM: {e}, returning empty result")
            generations = [[Generation(text="")] for _ in prompts]
            return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0
