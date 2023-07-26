import string
import re
from typing import List

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from autolabel.schema import LLMAnnotation


def normalize_text(s: str) -> str:
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def filter_unlabeled_examples(gt_labels: List[str], llm_labels: List[LLMAnnotation]):
    """Filter out unlabeled examples from the ground truth and LLM generated labels.
    This is done by checking the ground truth labels which have nan values.
    The corresponding ground truth and LLM labels are removed from the filtered labels lists.

    Args:
        gt_labels (List[str]): ground truth labels
        llm_labels (List[LLMAnnotation]): llm labels

    Returns:
        Tuple[List[str], List[LLMAnnotation]]: filtered ground truth and LLM generated labels
    """
    filtered_gt_labels = []
    filtered_llm_labels = []
    for gt_label, llm_label in zip(gt_labels, llm_labels):
        if gt_label != "nan":
            filtered_gt_labels.append(gt_label)
            filtered_llm_labels.append(llm_label)
    return filtered_gt_labels, filtered_llm_labels
