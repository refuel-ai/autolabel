from typing import List, Optional, Tuple, Union
import math
import numpy as np
import pickle as pkl
import json
import requests
import scipy.stats as stats
import os
import logging

from autolabel.schema import LLMAnnotation, ConfidenceCacheEntry
from autolabel.models import BaseModel
from autolabel.cache import BaseCache

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    TTL_MS = 60 * 60 * 24 * 365 * 3 * 1000  # 3 years

    def __init__(
        self,
        score_type: str = "logprob_average",
        llm: Optional[BaseModel] = None,
        cache: Optional[BaseCache] = None,
    ) -> None:
        self.score_type = score_type
        self.llm = llm
        self.cache = cache
        self.tokens_to_ignore = {"<unk>", "", "\\n"}
        self.SUPPORTED_CALCULATORS = {
            "logprob_average": self.logprob_average,
            "p_true": self.p_true,
            "logprob_average_per_key": self.logprob_average_per_key,
        }
        self.BASE_API = "https://llm.refuel.ai/models/refuelllm/confidence"
        self.REFUEL_API_ENV = "REFUEL_API_KEY"
        if self.REFUEL_API_ENV in os.environ and os.environ[self.REFUEL_API_ENV]:
            self.REFUEL_API_KEY = os.environ[self.REFUEL_API_ENV]
        else:
            self.REFUEL_API_KEY = None

    def logprob_average(
        self,
        logprobs: list,
        **kwargs,
    ) -> float:
        """
        This function has only been tested with the Davinci model so far
        It expects that model_generation contains generation info which has a key
        called "logprobs". This dictionary further must contain "top_logprobs"
        that is a list of JSONs mapping tokens to their logprobs
        """
        logprob_cumulative, count = 0, 0
        for token in logprobs:
            token_str = list(token.keys())[0]
            if (
                token_str.strip() not in self.tokens_to_ignore
                and token[token_str] is not None
            ):
                logprob_cumulative += (
                    token[token_str]
                    if token[token_str] >= 0
                    else math.e ** (token[token_str])
                )
                count += 1
        return logprob_cumulative / count if count > 0 else 0

    def logprob_average_per_key(
        self,
        logprobs: list,
        keys: list,
        **kwargs,
    ):
        """
        This function calculates the confidence score per key. This will return
        a confidence score per key.
        """
        # Find the logprob for each key
        logprob_per_key = {}

        mapping = []
        full_string = ""
        for i in range(len(logprobs)):
            for char in list(logprobs[i].keys())[0]:
                mapping.append(i)
                full_string += char

        # Find the locations of each key in the logprobs as indices
        # into the logprobs list
        locations = []
        for key in keys:
            key_to_find = f'"{key}":'
            loc = full_string.find(key_to_find)
            if loc == -1:
                logprob_per_key[key] = 0
            start_token = mapping[loc]
            end_token = mapping[loc + len(key_to_find) - 1]
            locations.append((start_token, end_token, key))
        locations.sort()

        if len(logprob_per_key) != 0:
            logger.warning("Some keys not found in logprobs")

        for i in range(len(locations) - 1):
            # Average the logprobs from the end of this to the start of the next token
            logprob_per_key[locations[i][2]] = self.logprob_average(
                logprobs[locations[i][1] + 1 : locations[i + 1][0]]
            )
        # Average the logprobs from the end of the last token to the end of the logprobs
        logprob_per_key[locations[-1][2]] = self.logprob_average(
            logprobs[locations[-1][1] + 1 :]
        )

        return logprob_per_key

    def p_true(self, model_generation: LLMAnnotation, **kwargs) -> float:
        p_true_prompt = f"{model_generation.raw_response} \n Is the answer to the last example correct? Answer in one word on the same line [Yes/No]: "

        if self.llm.returns_token_probs():
            p_true_prompt = model_generation.prompt + p_true_prompt
            response = self.llm.label([p_true_prompt])
            response_logprobs = response.generations[0][0].generation_info["logprobs"][
                "top_logprobs"
            ]
        else:
            p_true_prompt = model_generation.confidence_prompt + p_true_prompt
            yes_logprob = self.compute_confidence(p_true_prompt, "Yes")
            no_logprob = self.compute_confidence(p_true_prompt, "No")
            response_logprobs = (
                yes_logprob
                if list(yes_logprob[0].values())[0] > list(no_logprob[0].values())[0]
                else no_logprob
            )
        for token in response_logprobs:
            token_str = list(token.keys())[0]
            if token_str.lower() == "yes":
                return math.e ** token[token_str]
            elif token_str.lower() == "no":
                return -math.e ** token[token_str]
        return 0

    def calculate(self, model_generation: LLMAnnotation, **kwargs) -> float:
        if self.score_type not in self.SUPPORTED_CALCULATORS:
            raise NotImplementedError()

        logprobs = None
        if not self.llm.returns_token_probs():
            if model_generation.raw_response == "":
                model_generation.confidence_score = 0
                return model_generation.confidence_score
            if self.cache:
                cache_entry = ConfidenceCacheEntry(
                    prompt=model_generation.confidence_prompt,
                    raw_response=model_generation.raw_response,
                    score_type=self.score_type,
                )
                logprobs = self.cache.lookup(cache_entry)

                # On cache miss, compute logprobs using API call and update cache
                if logprobs == None:
                    logprobs = self.compute_confidence(
                        model_generation.confidence_prompt,
                        model_generation.raw_response,
                    )
                    cache_entry = ConfidenceCacheEntry(
                        prompt=model_generation.confidence_prompt,
                        raw_response=model_generation.raw_response,
                        logprobs=logprobs,
                        score_type=self.score_type,
                        ttl_ms=self.TTL_MS,
                    )
                    self.cache.update(cache_entry)
            else:
                logprobs = self.compute_confidence(
                    model_generation.confidence_prompt, model_generation.raw_response
                )
        else:
            if model_generation.generation_info is None:
                logger.debug("No generation info found")
                model_generation.confidence_score = 0
                return model_generation
            logprobs = model_generation.generation_info["logprobs"]["top_logprobs"]

        keys = None
        if self.score_type == "logprob_average_per_key":
            assert isinstance(
                model_generation.label, dict
            ), "logprob_average_per_key requires a dict label from attribute extraction"
            keys = model_generation.label.keys()

        confidence = self.SUPPORTED_CALCULATORS[self.score_type](
            model_generation=model_generation,
            logprobs=logprobs,
            keys=keys,
            **kwargs,
        )
        model_generation.confidence_score = confidence
        return confidence

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_with_retry(self, model_input, model_output) -> requests.Response:
        payload = {"input": model_input, "output": model_output}
        headers = {"refuel_api_key": self.REFUEL_API_KEY}
        response = requests.post(
            self.BASE_API, json=payload, headers=headers, timeout=30
        )
        # raise Exception if status != 200
        response.raise_for_status()
        return response

    def compute_confidence(self, model_input, model_output) -> Union[dict, List[dict]]:
        try:
            if self.REFUEL_API_KEY is None:
                logger.error(
                    f"Did not find {self.REFUEL_API_ENV}, please add an environment variable"
                    f" `{self.REFUEL_API_ENV}` which contains it"
                )
                return [{"": 0.5}]
            else:
                response = self._call_with_retry(model_input, model_output)
                return json.loads(response.json())["logprobs"]
        except Exception as e:
            # This signifies an error in computing confidence score
            # using the API. We give it a score of 0.5 and go ahead
            # for now.
            logger.error(
                f"Unable to compute confidence score: {e}",
            )
            return [{"": 0.5}]

    @classmethod
    def compute_completion(cls, confidence: List[float], threshold: float) -> float:
        return sum([i > threshold for i in confidence]) / float(len(confidence))

    @classmethod
    def compute_auroc(
        cls, match: List[int], confidence: List[float], plot: bool = False
    ) -> Tuple[float, List[float]]:
        try:
            import sklearn
        except ImportError:
            raise ValueError(
                "Could not import sklearn python package. "
                "Please it install it with `pip install scikit-learn`."
            )

        if len(set(match)) == 1:
            # ROC AUC score is not defined for a label list with
            # just one prediction
            return 1.0, [0]
        area = sklearn.metrics.roc_auc_score(match, confidence)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(match, confidence, pos_label=1)
        fpr, tpr, thresholds = (
            fpr[1:],
            tpr[1:],
            thresholds[1:],
        )  # first element is always support = 0. Can safely ignore.
        if plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ValueError(
                    "Could not import matplotlib python package. "
                    "Please it install it with `pip install matplotlib`."
                )
            print(f"FPR: {fpr}")
            print(f"TPR: {tpr}")
            print(f"Thresholds: {thresholds}")
            print(
                f"Completion Rate: {[ConfidenceCalculator.compute_completion(confidence, i) for i in thresholds]}"
            )
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                label="ROC curve (area = %0.2f)" % area,
            )
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic example")
            plt.legend(loc="lower right")
            plt.savefig("AUROC_CURVE.png")
            plt.close()
        return area, thresholds

    @classmethod
    def plot_data_distribution(
        cls,
        match: List[int],
        confidence: List[float],
        plot_name: str = "temp.png",
        save_data: bool = True,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ValueError(
                "Could not import matplotlib python package. "
                "Please it install it with `pip install matplotlib`."
            )

        if save_data:
            pkl.dump(match, open("matches.pkl", "wb"))
            pkl.dump(confidence, open("confidences.pkl", "wb"))

        normalized_conf = stats.zscore(np.array(confidence))

        correct = [
            normalized_conf[i] for i in range(len(normalized_conf)) if match[i] == 1
        ]
        incorrect = [
            normalized_conf[i] for i in range(len(normalized_conf)) if match[i] == 0
        ]

        bins = np.linspace(-3, 1, 100)
        plt.hist(correct, bins, alpha=0.5, label="Correct Conf")
        plt.hist(incorrect, bins, alpha=0.5, label="Incorrect Conf")
        plt.legend(loc="upper right")
        plt.savefig(plot_name)
        plt.close()
