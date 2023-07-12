import string
import re
from typing import List

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


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
        mlb.fit(labels)

        def binarize_labels(curr_labels):
            """Generate multilabel array from ground truth and LLM labels"""
            return mlb.transform([x.split(sep) for x in curr_labels])

        return f1_score(
            binarize_labels(truth),
            binarize_labels(prediction),
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
