from collections import defaultdict
from typing import List, Dict, Tuple
import json

from langchain.prompts.prompt import PromptTemplate
from sklearn.metrics import accuracy_score

from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import AutolabelConfig
from autolabel.schema import LLMAnnotation, Metric, MetricResult
from autolabel.tasks import BaseTask
from autolabel.utils import get_format_variables


class EntityMatchingTask(BaseTask):
    DEFAULT_OUTPUT_GUIDELINES = (
        'You will return the answer with one element: "the correct option"\n'
    )
    DEFAULT_TASK_GUIDELINES = "Your job is to tell if the two given entities are duplicates or not. You will return the answer from one of the choices. Choices:\n{labels}\n"

    GENERATE_EXPLANATION_PROMPT = "You are an expert at providing a well reasoned explanation for the output of a given task. \n\nBEGIN TASK DESCRIPTION\n{task_guidelines}\nEND TASK DESCRIPTION\nYou will be given an input example and the corresponding output. Your job is to provide an explanation for why the output is correct for the task above.\nThink step by step and generate an explanation. The last line of the explanation should be - So, the answer is <label>.\n{labeled_example}\nExplanation: "

    def __init__(self, config: AutolabelConfig) -> None:
        super().__init__(config)

    def construct_prompt(self, input: Dict, examples: List[Dict]) -> str:
        # Copy over the input so that we can modify it
        input = input.copy()

        # prepare task guideline
        labels_list = self.config.labels_list()
        num_labels = len(labels_list)
        fmt_task_guidelines = self.task_guidelines.format(
            num_labels=num_labels, labels="\n".join(labels_list)
        )

        # prepare seed examples
        example_template = self.config.example_template()
        label_column = self.config.label_column()
        fmt_examples = []
        for eg in examples:
            eg_copy = eg.copy()
            # If chain of thought is enabled
            if label_column and self.config.chain_of_thought():
                eg_copy[label_column] = json.dumps({"label": eg[label_column]})
            fmt_examples.append(example_template.format_map(defaultdict(str, eg_copy)))

        # populate the current example in the prompt
        if label_column:
            input[label_column] = ""

        # populate the explanation column with empty string for current example
        explanation_column = self.config.explanation_column()
        if explanation_column:
            input[explanation_column] = ""

        # populate the current example in the prompt
        current_example = example_template.format_map(defaultdict(str, input))

        if self._is_few_shot_mode():
            return self.prompt_template.format(
                task_guidelines=fmt_task_guidelines,
                output_guidelines=self.output_guidelines,
                seed_examples="\n\n".join(fmt_examples),
                current_example=current_example,
            )
        else:
            return self.prompt_template.format(
                task_guidelines=fmt_task_guidelines,
                output_guidelines=self.output_guidelines,
                current_example=current_example,
            )

    def get_explanation_prompt(self, example: Dict) -> str:
        pt = PromptTemplate(
            input_variables=get_format_variables(self.GENERATE_EXPLANATION_PROMPT),
            template=self.GENERATE_EXPLANATION_PROMPT,
        )

        # prepare task guideline
        labels_list = self.config.labels_list()
        num_labels = len(labels_list)
        fmt_task_guidelines = self.task_guidelines.format(
            num_labels=num_labels, labels="\n".join(labels_list)
        )

        # prepare labeled example
        example_template = self.config.example_template()
        fmt_example = example_template.format_map(defaultdict(str, example))

        return pt.format(
            task_guidelines=fmt_task_guidelines,
            labeled_example=fmt_example,
        )

    def auroc_score_labels(
        self, gt_labels, llm_labels
    ) -> Tuple[List[int], List[float]]:
        labels = []
        confidences = []
        for index, llm_label in enumerate(llm_labels):
            labels.append(llm_label.label.lower() == gt_labels[index].lower())
            confidences.append(llm_label.confidence_score)

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
            Metric.SUPPORT: [],
            Metric.ACCURACY: [],
            Metric.COMPLETION_RATE: [],
        }
        eval_metrics = []
        thresholds = []

        if self.config.confidence():
            eval_metrics_map[Metric.THRESHOLD] = []
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
        else:
            thresholds.append(float("-inf"))

        for index, threshold in enumerate(thresholds):
            (
                curr_gt_labels,
                curr_llm_labels,
            ) = self.get_labels_predictions_with_threshold(
                gt_labels, llm_labels, threshold
            )
            if len(gt_labels) > 0:
                eval_metrics_map[Metric.COMPLETION_RATE].append(
                    len(curr_gt_labels) / float(len(gt_labels))
                )
            eval_metrics_map[Metric.SUPPORT].append(len(curr_gt_labels))
            if len(curr_gt_labels) > 0:
                eval_metrics_map[Metric.ACCURACY].append(
                    accuracy_score(curr_gt_labels, curr_llm_labels)
                )
            else:
                eval_metrics_map[Metric.ACCURACY].append(0.0)

            if self.config.confidence():
                eval_metrics_map[Metric.THRESHOLD].append(threshold)

        eval_metrics.extend(
            [
                MetricResult(
                    metric_type=i,
                    name=i.value,
                    value=eval_metrics_map[i],
                )
                for i in eval_metrics_map.keys()
            ]
        )
        return eval_metrics
