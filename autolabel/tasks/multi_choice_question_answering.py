import ast
from typing import Dict, List, Tuple

from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import TaskConfig
from autolabel.schema import LLMAnnotation, Metric, MetricResult
from autolabel.tasks import BaseTask
from autolabel.tasks.utils import compute_f1, normalize_text
from langchain.prompts.prompt import PromptTemplate
from sklearn.metrics import accuracy_score


class MultiChoiceQATask(BaseTask):
    JSON_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with one key: {"label": "the correct label"}\n'
    CSV_OUTPUT_FORMAT_PROMPT = 'You will return the answer in CSV format with one element: "the correct label"\n'

    task_prompt = "Your job is to answer the following questions using the options provided for each question. Choose the best answer for the question.\n"
    NULL_LABEL_TOKEN = "NO_LABEL"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    def initialize_prompt_template(self) -> PromptTemplate:
        pt = PromptTemplate(
            input_variables=self.prompt_template_variables,
            template=self.prompt_template,
        )

        return pt.partial(
            prefix_prompt=self.prefix_prompt,
            task_prompt=self.task_prompt,
            output_prompt=self.output_prompt,
        )

    def get_context(self, input: Dict) -> str:
        context = input.get("context", "")
        if context:
            context = f"Context: {context}"
        return context

    def get_options(self, input: Dict) -> str:
        options = input.get("options", "")
        # if options is empty, return empty string
        if not options:
            return ""

        if isinstance(options, str):
            options = "\n".join(ast.literal_eval(input["options"]))
            options = f"Options:\n{options}"
        elif isinstance(options, list):
            options = "\n".join(options)
            options = f"Options:\n{options}"
        return options

    def construct_prompt(self, input: Dict, examples: List[Dict]) -> str:
        example_template = self.dataset_config.get_example_template()
        label_column = self.dataset_config.get_label_column()
        explanation_column = self.dataset_config.get_explanation_column()

        formatted_examples = list(
            map(lambda example: example_template.format(**example), examples)
        )
        # Remove label column from formatted input if it exists
        input[label_column] = ""
        input[explanation_column] = ""
        current_example = example_template.format(**input)

        if len(examples):
            seed_examples_prompt = self.seed_examples_prompt
        else:
            seed_examples_prompt = ""

        return self.partial_prompt.format(
            seed_examples_prompt=seed_examples_prompt,
            seed_examples="\n".join(formatted_examples),
            current_example=current_example,
        )

    def auroc_score_labels(
        self, gt_labels, llm_labels
    ) -> Tuple[List[int], List[float]]:
        labels = []
        confidences = []
        for index, llm_label in enumerate(llm_labels):
            labels.append(
                normalize_text(llm_label.label.lower())
                == normalize_text(gt_labels[index].lower())
            )
            confidences.append(llm_label.confidence_score)

        ConfidenceCalculator.plot_data_distribution(labels, confidences)
        return labels, confidences

    def get_labels_predictions_with_threshold(self, gt_labels, llm_labels, threshold):
        answered_gt_labels, answered_llm_preds = [], []
        for index, l in enumerate(llm_labels):
            if l.label != self.NULL_LABEL_TOKEN and (
                l.confidence_score is None or l.confidence_score >= threshold
            ):
                answered_llm_preds.append(normalize_text(l.label.lower()))
                answered_gt_labels.append(normalize_text(gt_labels[index].lower()))

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
            Metric.F1: [],
            Metric.SUPPORT: [],
            Metric.THRESHOLD: [],
            Metric.ACCURACY: [],
            Metric.COMPLETION_RATE: [],
        }
        eval_metrics = []
        thresholds = [float("-inf")]

        if self.config.get_compute_confidence():
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

        for index, threshold in enumerate(thresholds):
            (
                curr_gt_labels,
                curr_llm_labels,
            ) = self.get_labels_predictions_with_threshold(
                gt_labels, llm_labels, threshold
            )
            eval_metrics_map[Metric.SUPPORT].append(
                (len(curr_gt_labels), f"index={index}")
            )
            eval_metrics_map[Metric.COMPLETION_RATE].append(
                (len(curr_gt_labels) / float(len(gt_labels)), f"index={index}")
            )
            eval_metrics_map[Metric.ACCURACY].append(
                (accuracy_score(curr_gt_labels, curr_llm_labels), f"index={index}")
            )
            eval_metrics_map[Metric.THRESHOLD].append(threshold)

            f1 = sum(
                [
                    compute_f1(curr_llm_labels[index], curr_gt_labels[index])
                    for index in range(len(curr_llm_labels))
                ]
            )
            eval_metrics_map[Metric.F1].append(
                (float(f1) / (len(curr_llm_labels) + 1e-5), f"index={index}")
            )

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
