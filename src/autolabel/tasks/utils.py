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


def compute_f1(
    truth: List,
    prediction: List,
    average: str = "micro",
    labels: List[str] = None,
    sep: str = None,
) -> float:
    """
    Compute f1 scores based on given ground truth labels.

    If labels are provided, uses sklearn to binarize the labels and compute f1 scores.
    Otherwise, uses bag of words approach to compute f1 scores.

    Args:
        prediction: model generated prediction
        truth: ground truth label to compare against
    Returns:
        f1_score: values range from [0,1], with 1 indicating perfect accuracy
    """
    if labels:
        mlb = MultiLabelBinarizer()
        mlb.fit([labels])

        return f1_score(
            mlb.transform([x.split(sep) for x in truth]),
            mlb.transform([x.split(sep) for x in prediction]),
            average=average,
            zero_division=0,
        )
    else:
        truth = [normalize_text(t).split(sep) for t in truth]
        prediction = [normalize_text(p).split(sep) for p in prediction]

        tp_pairs = zip(truth, prediction)

        f1_scores = []
        for truth_tokens, pred_tokens in tp_pairs:
            # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
            if len(truth_tokens) == 0 or len(pred_tokens) == 0:
                f1_scores.append(int(truth_tokens == pred_tokens))
                continue

            common_tokens = set(truth_tokens) & set(pred_tokens)

            # if there are no common tokens then f1 = 0
            if len(common_tokens) == 0:
                f1_scores.append(0)
                continue

            rec = len(common_tokens) / len(truth_tokens)
            prec = len(common_tokens) / len(pred_tokens)

            f1_scores.append(2 * (prec * rec) / (prec + rec))

        return sum(f1_scores) / len(f1_scores)


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
