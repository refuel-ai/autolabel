from typing import List, Optional
import math
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import scipy.stats as stats

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
            logprob_cumulative, count = 0, 0
            for token in token_to_prob:
                token_str = list(token.keys())[0]
                if token_str not in empty_response_template:
                    logprob_cumulative += token[token_str]
                    count += 1
            return math.e ** (logprob_cumulative / count)
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
        return 0

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

    @classmethod
    def compute_completion(cls, confidence: List[float], threshold: float):
        return sum([i > threshold for i in confidence]) / float(len(confidence))

    @classmethod
    def compute_auroc(
        cls, match: List[int], confidence: List[float], plot: bool = False
    ):
        if len(set(match)) == 1:
            # ROC AUC score is not defined for a label list with
            # just one prediction
            return 1.0
        area = sklearn.metrics.roc_auc_score(match, confidence)
        if plot:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                match, confidence, pos_label=1
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
        return area

    @classmethod
    def plot_data_distribution(
        cls,
        match: List[int],
        confidence: List[float],
        plot_name: str = "temp.png",
        save_data: bool = True,
    ):
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
