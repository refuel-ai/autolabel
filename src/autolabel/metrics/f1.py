from typing import List, Optional
import json

from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult, MetricType, F1Type
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import f1_score
from autolabel.utils import normalize_text


class F1Metric(BaseMetric):
    def __init__(
        self,
        type: F1Type,
        labels: Optional[List[str]] = [],
        sep: Optional[str] = " ",
        average: Optional[List[str]] = [MetricType.F1_MICRO],
    ) -> None:
        super().__init__()
        self.type = type
        self.labels = labels
        self.sep = sep
        self.average = average

    def multi_label_compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        filtered_llm_labels = []
        filtered_gt_labels = []
        for llm_label, gt_label in zip(llm_labels, gt_labels):
            if llm_label.error is None and gt_label != "nan":
                filtered_llm_labels.append(llm_label)
                filtered_gt_labels.append(gt_label)

        filtered_llm_labels = [llm_label.label for llm_label in filtered_llm_labels]

        mlb = MultiLabelBinarizer()
        mlb.fit([self.labels])

        value = []
        for average in self.average:
            score = f1_score(
                mlb.transform([x.split(self.sep) for x in filtered_gt_labels]),
                mlb.transform([x.split(self.sep) for x in filtered_llm_labels]),
                average=average.split("_")[-1],
                zero_division=0,
            )
            value.append(MetricResult(name=average, value=score))
        return value

    def text_compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        truth = [normalize_text(gt_label).split(self.sep) for gt_label in gt_labels]
        prediction = [
            normalize_text(llm_label.label).split(self.sep) for llm_label in llm_labels
        ]

        tp_pairs = zip(truth, prediction)

        f1_scores = []
        filtered_datapoints = 0
        for i, (truth_tokens, pred_tokens) in enumerate(tp_pairs):
            # if there is an error filter this datapoint
            if llm_labels[i].error is not None or gt_labels[i] == "nan":
                f1_scores.append(0)
                filtered_datapoints += 1
                continue

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

        values = [
            MetricResult(
                name=MetricType.TEXT_PARTIAL_MATCH,
                value=f1_scores,
            ),
            MetricResult(
                name=MetricType.F1,
                value=sum(f1_scores) / (len(f1_scores) - filtered_datapoints),
            ),
        ]

        return values

    def attribute_extraction_compute(
        self, llm_labels: List[str], gt_labels: List[str]
    ) -> float:
        # Convert JSON strings to dictionaries
        llm_labels = [json.loads(label.label) for label in llm_labels]
        gt_labels = [json.loads(label) for label in gt_labels]

        # Get all unique labels
        all_labels = set()
        for label in llm_labels + gt_labels:
            all_labels.update(label.values())

        # Initialize label encoder
        le = LabelEncoder()
        le.fit(list(all_labels))

        # Compute F1 score for each attribute
        f1_scores = []
        for llm_label, gt_label in zip(llm_labels, gt_labels):
            for attribute in gt_label.keys():
                # Encode labels as integers
                y_true = le.transform([gt_label[attribute]])
                y_pred = le.transform([llm_label.get(attribute, None)])

                # Compute F1 score
                f1 = f1_score(y_true, y_pred, average="micro")
                f1_scores.append(f1)

        # Compute average F1 score
        avg_f1_score = sum(f1_scores) / len(f1_scores)

        return [MetricResult(metric_type=MetricType.F1, value=avg_f1_score)]

    def compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        if self.type == F1Type.MULTI_LABEL:
            return self.multi_label_compute(llm_labels, gt_labels)
        elif self.type == F1Type.ATTRIBUTE_EXTRACTION:
            return self.attribute_extraction_compute(llm_labels, gt_labels)
        else:
            return self.text_compute(llm_labels, gt_labels)
