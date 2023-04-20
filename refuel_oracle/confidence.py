from typing import Optional
import math

from refuel_oracle.schema import LLMAnnotation
from refuel_oracle.llm import LLMLabeler


class ConfidenceCalculator:
    def __init__(
        self, score_type: str = "logprob_average", llm: Optional[LLMLabeler] = None
    ):
        self.score_type = score_type
        self.llm = llm

    def logprob_average(
        self, model_generation: LLMAnnotation, empty_response: str, **kwargs
    ) -> float:
        """
        This function has only been tested with the Davinci model so far
        It expects that model_generation contains generation info which has a key
        called "logprobs". This dictionary further must contain "top_logprobs"
        that is a list of JSONs mapping tokens to their logprobs
        """
        if (
            model_generation.generation_info is not None
            and "logprobs" in model_generation.generation_info
        ):
            token_to_prob = model_generation.generation_info["logprobs"]["top_logprobs"]
            empty_response_template = empty_response
            logprob_cumulative = [0.0, 0.0]
            for token in token_to_prob:
                token_str = list(token.keys())[0]
                if empty_response_template.startswith(token_str):
                    empty_response_template = empty_response_template.removeprefix(
                        token_str
                    )
                else:
                    logprob_cumulative[0] += token[token_str]
                    logprob_cumulative[1] += 1
            return math.e ** logprob_cumulative[0] / logprob_cumulative[1]
        else:
            raise NotImplementedError()

    def p_true(self, model_generation: LLMAnnotation, prompt: str, **kwargs) -> float:
        p_true_prompt = f"{prompt}{model_generation.raw_text} \n Is the answer to the last example correct? Answer in one word on the same line [Yes/No]: "
        response = self.llm.generate([p_true_prompt])
        response_logprobs = response.generations[0][0].generation_info["logprobs"][
            "top_logprobs"
        ]
        for token in response_logprobs:
            token_str = list(token.keys())[0]
            if token_str.lower() == "yes":
                return math.e ** token[token_str]
            elif token_str.lower() == "no":
                return -math.e ** token[token_str]
        return 0.5

    def calculate(self, model_generation: LLMAnnotation, **kwargs) -> LLMAnnotation:
        SUPPORTED_CALCULATORS = {
            "logprob_average": self.logprob_average,
            "p_true": self.p_true,
        }

        if self.score_type not in SUPPORTED_CALCULATORS:
            raise NotImplementedError()
        confidence = SUPPORTED_CALCULATORS[self.score_type](
            model_generation=model_generation, **kwargs
        )
        model_generation.confidence_score = confidence
        return model_generation
