from collections import defaultdict
from typing import List, Dict, Tuple

from langchain.prompts.prompt import PromptTemplate
from sklearn.metrics import accuracy_score

from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import TaskConfig
from autolabel.schema import LLMAnnotation, Metric, MetricResult
from autolabel.tasks import BaseTask


class ClassificationTask(BaseTask):
    JSON_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with one key: {"label": "the correct label"}'
    CSV_OUTPUT_FORMAT_PROMPT = (
        'You will return the answer in CSV format with one element: "the correct label"'
    )

    task_prompt = "Your job is to correctly label the provided input example into one of the following {num_labels} categories.\nCategories:\n{labels_list}\n"

    explanation_generation_prompt = "{prefix_prompt}\n You will be given an input example and the corresponding output. Your job is to provide an explanation for why the output is correct. The label is one of the following {num_labels} categories.\nCategories:\n{labels_list}\n Think step by step and generate an explanation. The last line of the explanation should be - So, the answer is <label>.\n{labeled_example}\nExplanation: "
    explanation_generation_prompt_variables = [
        "prefix_prompt",
        "num_labels",
        "labels_list",
        "labeled_example",
    ]

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
        example_template = self.dataset_config.get_example_template()

        # populate seed examples in the prompt
        formatted_examples = []
        for eg in examples:
            fmt_example = example_template.format_map(defaultdict(str, eg))
            formatted_examples.append(fmt_example)

        if len(examples):
            seed_examples_prompt = self.seed_examples_prompt
        else:
            seed_examples_prompt = ""

        # populate the current example in the prompt
        current_example = example_template.format_map(defaultdict(str, input))

        return self.partial_prompt.format(
            seed_examples_prompt=seed_examples_prompt,
            seed_examples="\n\n".join(formatted_examples),
            current_example=current_example,
            task_prompt=task_prompt,
        )

    def generate_explanation(self, example: Dict) -> str:
        pt = PromptTemplate(
            input_variables=self.explanation_generation_prompt_variables,
            template=self.explanation_generation_prompt,
        )

        labels_list = self.dataset_config.get_labels_list()
        num_labels = len(labels_list)

        example_template = self.dataset_config.get_example_template()
        fmt_example = example_template.format_map(defaultdict(str, example))

        return pt.format(
            prefix_prompt=self.prefix_prompt,
            num_labels=num_labels,
            labels_list="\n".join(labels_list),
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
            eval_metrics_map[Metric.THRESHOLD].append((threshold, f"index={index}"))

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
