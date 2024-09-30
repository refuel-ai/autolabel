from typing import List, Optional, Tuple, Union, Dict
import math
import numpy as np
import pickle as pkl
import json
import httpx
import scipy.stats as stats
import os
import logging
import difflib

from autolabel.schema import LLMAnnotation, ConfidenceCacheEntry, TaskType
from autolabel.models import BaseModel
from autolabel.cache import BaseCache

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    TTL_MS = 60 * 60 * 24 * 7 * 1000  # 1 week

    def __init__(
        self,
        score_type: str = "logprob_average",
        endpoint: str = None,
        llm: Optional[BaseModel] = None,
        cache: Optional[BaseCache] = None,
    ) -> None:
        self.score_type = score_type
        self.endpoint = endpoint
        self.llm = llm
        self.cache = cache
        self.tokens_to_ignore = {"<unk>", "", "\\n"}
        self.SUPPORTED_CALCULATORS = {
            "logprob_average": self.logprob_average,
            "p_true": self.p_true,
            "logprob_average_per_key": self.logprob_average_per_key,
            "logprob_average_per_label": self.logprob_average_per_label,
        }
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
        logprob_cumulative, count = 1.0, 0
        for token in logprobs:
            token_str = list(token.keys())[0]
            if (
                token_str.strip() not in self.tokens_to_ignore
                and token[token_str] is not None
            ):
                logprob_cumulative *= (
                    token[token_str]
                    if token[token_str] > 0
                    else math.e ** (token[token_str])
                )
                count += 1
        return logprob_cumulative ** (1.0 / count) if count > 0 else 0

    def _logprob_average_per_label(
        self,
        logprobs: list,
        label: str,
        delimiter: str = ";",
        **kwargs,
    ) -> Dict[str, float]:
        logprob_per_label = {}
        curr_logprob_average = self.logprob_average(logprobs)
        logprob_per_label = {
            curr_label: curr_logprob_average for curr_label in label.split(delimiter)
        }
        conf_label_keys, prev_key_index, curr_key = {}, 0, ""
        for i in range(len(logprobs)):
            for curr_chars in list(logprobs[i].keys())[0]:
                if delimiter in curr_chars:
                    curr_key += curr_chars.split(delimiter)[0]
                    conf_label_keys[curr_key] = self.logprob_average(
                        logprobs[prev_key_index:i]
                    )
                    prev_key_index = i
                    curr_key = curr_chars.split(delimiter)[-1]
                else:
                    curr_key += curr_chars
        if len(curr_key) > 0:
            conf_label_keys[curr_key] = self.logprob_average(logprobs[prev_key_index:])

        for conf_label_candiate in conf_label_keys:
            closest_match, closest_match_score = None, 0
            for label in logprob_per_label:
                # The SequenceMatcher class is used to compare two sequences. It is especially useful for comparing sequences of characters.
                # None - This is a function that is used to compare the two sequences. If it is None, the default function is used.
                # label - The first sequence to compare
                # conf_label_candiate - The second sequence to compare

                # The find_longest_match function returns a named tuple with the following fields:
                # a - The start of the matching subsequence in the first sequence
                # b - The start of the matching subsequence in the second sequence
                # size - The length of the matching subsequence

                longest_substring = difflib.SequenceMatcher(
                    None, label, conf_label_candiate
                ).find_longest_match(0, len(label), 0, len(conf_label_candiate))
                if (
                    longest_substring.size
                    / (1e-6 + max(len(label), len(conf_label_candiate)))
                ) > closest_match_score:
                    closest_match = label
                    closest_match_score = longest_substring.size / (
                        1e-6 + max(len(label), len(conf_label_candiate))
                    )
            if closest_match is not None:
                logprob_per_label[closest_match] = conf_label_keys[conf_label_candiate]

        return logprob_per_label

    def logprob_average_per_label(
        self,
        model_generation: LLMAnnotation,
        delimiter: str = ";",
        **kwargs,
    ) -> float:
        """
        This function calculates the confidence score per label when there are multiple labels in the response (i.e. multilabel tasks). This will return
        a confidence score per label.
        """
        logprobs = model_generation.generation_info["logprobs"]["top_logprobs"]
        if logprobs is None or len(logprobs) == 0:
            return {}
        return self._logprob_average_per_label(
            logprobs=logprobs,
            label=model_generation.label,
            delimiter=delimiter,
        )

    def logprob_average_per_key(
        self,
        model_generation: LLMAnnotation,
        logprobs: Union[list, dict],
        keys: Dict[str, str],
        **kwargs,
    ):
        """
        This function calculates the confidence score per key. This will return
        a confidence score per key.
        """
        # Find the logprob for each key
        logprob_per_key = {}
        if logprobs is None or len(logprobs) == 0:
            return logprob_per_key

        # Suppose the output for which we compute confidence is {"A": "B", "C": "D"}
        # In this case the logprobs can be a list of dictionaries like
        # [{"{": -1.2}, {"\"A": -1.3}, {"\"": -1.4}, {":": -1.5}, {"\"B\"": -1.6}, ...]
        mapping = []
        full_string = ""
        for i in range(len(logprobs)):
            for char in list(logprobs[i].keys())[0]:
                mapping.append(i)
                full_string += char
        # Here full string would be the actual output string reconstructed i.e {"A": "B", "C": "D"}
        # The mapping here maps every character in this output string to the corresponding token
        # index in the logprobs list. This is because a single token can have multiple characters.
        # Using this, from a given character index we can map to the token responsible for that character.

        # Find the locations of each key in the logprobs as indices
        # into the logprobs list
        locations = [(len(logprobs), len(logprobs), "")]
        for key in keys:
            key_to_find = f'"{key}":'
            loc = full_string.find(key_to_find)
            if loc == -1:
                # We did not find the key in the logprobs so we set its confidence as 0
                # This should not be possible if the LLM followed its guidelines.
                logprob_per_key[key] = 0
            else:
                start_token = mapping[loc]
                end_token = mapping[loc + len(key_to_find) - 1]
                locations.append((start_token, end_token, key))
        locations.sort()
        # Here, the locations consist of the start and end *token* indices for each key
        # i.e for the keys A and B, we find the start and end tokens where they are found in the logprobs list
        # and store them in locations. For eg - locations can be [(1, 3, "A"), (9, 12, "C")]

        if len(logprob_per_key) != 0:
            logger.warning("Some keys not found in logprobs")
        for i in range(len(locations) - 1):
            # Average the logprobs from the end of this to the start of the next token
            # This means that we average the logprobs of all tokens from the end of the key token
            # to the start of the next key token thus for the key "A" this would average the tokens
            # responsible for generating "B",
            curr_key = locations[i][2]
            if (
                curr_key in keys
                and keys[curr_key] == TaskType.MULTILABEL_CLASSIFICATION
            ):
                logprob_per_key[curr_key] = self._logprob_average_per_label(
                    logprobs[locations[i][1] + 1 : locations[i + 1][0]],
                    label=model_generation.label[curr_key],
                )
            else:
                logprob_per_key[curr_key] = self.logprob_average(
                    logprobs[locations[i][1] + 1 : locations[i + 1][0]]
                )
        return logprob_per_key

    async def p_true(self, model_generation: LLMAnnotation, **kwargs) -> float:
        p_true_prompt = f"{model_generation.raw_response} \n Is the answer to the last example correct? Answer in one word on the same line [Yes/No]: "

        if self.llm.returns_token_probs():
            p_true_prompt = model_generation.prompt + p_true_prompt
            response = self.llm.label([p_true_prompt], output_schema=None)
            response_logprobs = response.generations[0][0].generation_info["logprobs"][
                "top_logprobs"
            ]
        else:
            p_true_prompt = model_generation.prompt + p_true_prompt
            yes_logprob = await self.compute_confidence(p_true_prompt, "Yes")
            no_logprob = await self.compute_confidence(p_true_prompt, "No")
            if not yes_logprob or not no_logprob:
                # Error while calculating logprobs
                return 0
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

    def return_empty_logprob(
        self, model_generation: LLMAnnotation
    ) -> Union[float, Dict]:
        if self.score_type == "logprob_average_per_key":
            keys = model_generation.label.keys()
            model_generation.confidence_score = {key: 0 for key in keys}
        else:
            model_generation.confidence_score = 0
        return model_generation.confidence_score

    async def calculate(
        self,
        model_generation: LLMAnnotation,
        keys: Optional[Dict] = None,
        **kwargs,
    ) -> float:
        if self.score_type not in self.SUPPORTED_CALCULATORS:
            raise NotImplementedError()

        logprobs = None
        if not self.llm.returns_token_probs():
            if model_generation.raw_response == "":
                model_generation.confidence_score = 0
                return model_generation.confidence_score
            if self.cache:
                cache_entry = ConfidenceCacheEntry(
                    prompt=model_generation.prompt,
                    raw_response=model_generation.raw_response,
                    score_type=self.score_type,
                )
                logprobs = self.cache.lookup(cache_entry)

                # On cache miss, compute logprobs using API call and update cache
                if logprobs == None:
                    logprobs = await self.compute_confidence(
                        model_generation.prompt,
                        model_generation.raw_response,
                    )
                    if not logprobs:
                        return self.return_empty_logprob(model_generation)
                    cache_entry = ConfidenceCacheEntry(
                        prompt=model_generation.prompt,
                        raw_response=model_generation.raw_response,
                        logprobs=logprobs,
                        score_type=self.score_type,
                        ttl_ms=self.TTL_MS,
                    )
                    self.cache.update(cache_entry)
            else:
                logprobs = await self.compute_confidence(
                    model_generation.prompt, model_generation.raw_response
                )
                if not logprobs:
                    return self.return_empty_logprob(model_generation)
        else:
            if model_generation.generation_info is None:
                logger.debug("No generation info found")
                model_generation.confidence_score = 0
                return model_generation
            logprobs = model_generation.generation_info["logprobs"]["top_logprobs"]

        if self.score_type == "logprob_average_per_key":
            assert isinstance(
                model_generation.label, dict
            ), "logprob_average_per_key requires a dict label from attribute extraction"
            assert keys is not None, "Keys must be provided for logprob_average_per_key"

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
        retry=retry_if_not_exception_type(ValueError),
    )
    async def _call_with_retry(self, model_input, model_output):
        payload = {
            "messages": [
                {"role": "user", "content": model_input},
                {"role": "assistant", "content": model_output},
            ]
        }
        headers = {"refuel_api_key": self.REFUEL_API_KEY}
        if self.endpoint is None:
            raise ValueError("Endpoint not provided")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint, json=payload, headers=headers, timeout=30
            )
            # raise Exception if status != 200
            response.raise_for_status()
            return response

    async def compute_confidence(
        self, model_input, model_output
    ) -> Union[dict, List[dict]]:
        """
        This function computes the confidence score for the given model_input and model_output
        and returns the logprobs for each token in the output. If there is an error, it returns None.
        """
        try:
            if self.REFUEL_API_KEY is None:
                logger.error(
                    f"Did not find {self.REFUEL_API_ENV}, please add an environment variable"
                    f" `{self.REFUEL_API_ENV}` which contains it"
                )
                return None
            else:
                response = await self._call_with_retry(model_input, model_output)
                return json.loads(response.json())["logprobs"]
        except Exception as e:
            # This signifies an error in computing confidence score
            # using the API. We give it a score of 0.5 and go ahead
            # for now.
            logger.error(
                f"Unable to compute confidence score: {e}",
            )
            return None

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
