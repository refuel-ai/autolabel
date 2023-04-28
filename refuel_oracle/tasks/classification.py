import json
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
from langchain.prompts.prompt import PromptTemplate
from loguru import logger
from refuel_oracle.confidence import ConfidenceCalculator
from refuel_oracle.task_config import TaskConfig
from refuel_oracle.schema import LLMAnnotation, Metric, MetricResult
from refuel_oracle.tasks import BaseTask
from refuel_oracle.utils import extract_valid_json_substring
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


class ClassificationTask(BaseTask):
    JSON_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": "can you answer this question. say yes or no", "label": "the correct label"}'
    CSV_OUTPUT_FORMAT_PROMPT = 'You will return the answer in CSV format with two elements: "can you answer this question. say Yes or No", "the correct label"'
    NO_OUTPUT_FORMAT_PROMPT = 'You will return the answer in plain text format with one element: "the correct label"'

    task_prompt = "Your job is to correctly label the provided input example into one of the following {num_labels} categories.\nCategories:\n{labels_list}\n"
    example_prompt_template = "Example: {example}\nOutput: {output}\n"
    example_prompt_variables = ["example", "output"]

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    def initialize_prompt_template(self) -> PromptTemplate:
        # provide context about the prediction task
        pt = PromptTemplate(
            input_variables=self.prompt_template_variables,
            template=self.prompt_template,
        )

        return pt.partial(
            prefix_prompt=self.prefix_prompt,
            output_prompt=self.output_prompt,
        )

    def construct_prompt(self, input: Dict, examples: List) -> str:
        # Create the task prompt based on the dataset config
        labels_list = self.dataset_config.get_labels_list()
        num_labels = len(labels_list)
        task_prompt = self.task_prompt.format(
            num_labels=num_labels, labels_list="\n".join(labels_list)
        )

        # populate seed examples in the prompt
        example_prompt = PromptTemplate(
            input_variables=self.example_prompt_variables,
            template=self.example_prompt_template,
        )
        formatted_examples = []
        for eg in examples:
            expected_output = self._to_output_format(eg["output"])
            formatted_examples.append(
                example_prompt.format(example=eg["example"], output=expected_output)
            )

        # populate the current example in the prompt
        current_example = example_prompt.format(example=input["example"], output="")

        if len(examples):
            seed_examples_prompt = self.seed_examples_prompt
        else:
            seed_examples_prompt = ""

        return self.partial_prompt.format(
            seed_examples_prompt=seed_examples_prompt,
            seed_examples="\n".join(formatted_examples),
            current_example=current_example,
            task_prompt=task_prompt,
        )

    def auroc_score_labels(
        self, gt_labels, llm_labels
    ) -> Tuple[List[int], List[float]]:
        labels = []
        confidences = []
        for index, llm_label in enumerate(llm_labels):
            labels.append(llm_label.label.lower() == gt_labels[index].lower())
            confidences.append(llm_label.confidence_score)

        ConfidenceCalculator.plot_data_distribution(labels, confidences)
        return labels, confidences

    def get_labels_predictions_with_threshold(self, gt_labels, llm_labels, threshold):
        answered_gt_labels, answered_llm_preds = [], []
        for index, l in enumerate(llm_labels):
            if l.label != self.NULL_LABEL_TOKEN and (
                l.confidence_score is None or l.confidence_score >= threshold
            ):
                answered_llm_preds.append(l.label.lower())
                answered_gt_labels.append(gt_labels[index].lower())

        return answered_gt_labels, answered_llm_preds

    def eval(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        """Evaluate the LLM generated labels by comparing them against ground truth

        Args:
            llm_labels (List[LLMAnnotation]): _description_
            gt_labels (List[str]): _description_

        Returns:
            List[MetricResult]: list of metrics and corresponding values
        """

        eval_metrics_map = {
            "support": [Metric.SUPPORT, []],
            "threshold": [Metric.THRESHOLD, []],
            "accuracy": [Metric.ACCURACY, []],
            "completion_rate": [Metric.COMPLETION_RATE, []],
        }
        eval_metrics = []
        thresholds = [float("-inf")]

        if self.config.get_compute_confidence() == "True":
            labels, confidences = self.auroc_score_labels(gt_labels, llm_labels)
            value, meaningful_thresholds = ConfidenceCalculator.compute_auroc(
                labels, confidences
            )
            thresholds.extend(meaningful_thresholds)
            eval_metrics.append(
                MetricResult(
                    metric_type=Metric.AUROC,
                    name="auroc",
                    value=value,
                )
            )

        labels_list = self.dataset_config.get_labels_list()
        labels_list.append(self.NULL_LABEL_TOKEN)
        confusion = confusion_matrix(
            gt_labels, [i.label for i in llm_labels], labels=labels_list
        )
        eval_metrics.append(
            MetricResult(
                metric_type=Metric.CONFUSION_MATRIX,
                name="confusion_matrix",
                value={"labels": labels_list, "value": confusion},
            )
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=labels_list
        )
        disp.plot(cmap="Purples")
        plt.xticks(rotation=90)
        plt.show(block=False)
        input("Displaying confusion matrix. Press enter to continue..")

        for threshold in thresholds:
            (
                curr_gt_labels,
                curr_llm_labels,
            ) = self.get_labels_predictions_with_threshold(
                gt_labels, llm_labels, threshold
            )
            eval_metrics_map["support"][1].append(len(curr_gt_labels))
            eval_metrics_map["completion_rate"][1].append(
                len(curr_gt_labels) / float(len(gt_labels))
            )
            eval_metrics_map["accuracy"][1].append(
                accuracy_score(curr_gt_labels, curr_llm_labels)
            )
            eval_metrics_map["threshold"][1].append(threshold)

        eval_metrics.extend(
            [
                MetricResult(
                    metric_type=eval_metrics_map[i][0],
                    name=i,
                    value=eval_metrics_map[i][1],
                )
                for i in eval_metrics_map.keys()
            ]
        )
        return eval_metrics
