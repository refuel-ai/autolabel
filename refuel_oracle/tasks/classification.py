import json
from typing import List, Dict

import matplotlib.pyplot as plt
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from refuel_oracle.task_config import TaskConfig
from refuel_oracle.dataset_config import DatasetConfig
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

    def __init__(self, config: TaskConfig, dataset_config: DatasetConfig) -> None:
        super().__init__(config, dataset_config)

    def initialize_prompt_template(self) -> PromptTemplate:
        # provide context about the prediction task
        labels_list = self.dataset_config.get_labels_list()
        num_labels = len(labels_list)
        task_prompt = self.task_prompt.format(
            num_labels=num_labels, labels_list="\n".join(labels_list)
        )

        pt = PromptTemplate(
            input_variables=self.prompt_template_variables,
            template=self.prompt_template,
        )

        return pt.partial(
            prefix_prompt=self.prefix_prompt,
            task_prompt=task_prompt,
            output_prompt=self.output_prompt,
        )

    def construct_prompt(self, input: Dict, examples: List) -> str:
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
        )

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
        eval_metrics = []

        # support
        support = len(gt_labels)
        eval_metrics.append(
            MetricResult(metric_type=Metric.SUPPORT, name="support", value=support)
        )

        # completion rate
        num_labeled = sum([l.successfully_labeled.lower() == "yes" for l in llm_labels])
        fraction_completed = round(num_labeled * 1.0 / support, 2)
        eval_metrics.append(
            MetricResult(
                metric_type=Metric.COMPLETION_RATE,
                name="completion_rate",
                value=fraction_completed,
            )
        )

        # accuracy
        pred_labels = [l.label for l in llm_labels]
        filtered_gt_labels = []
        filtered_pred_labels = []
        for ind, label in enumerate(pred_labels):
            if label != self.NULL_LABEL_TOKEN:
                filtered_gt_labels.append(gt_labels[ind])
                filtered_pred_labels.append(pred_labels[ind])
        if len(filtered_gt_labels) == 0:
            logger.error("No labels were successfully labeled by the LLM")
            accuracy = 0
        else:
            accuracy = accuracy_score(filtered_gt_labels, filtered_pred_labels)
        eval_metrics.append(
            MetricResult(metric_type=Metric.ACCURACY, name="accuracy", value=accuracy)
        )

        # confusion matrix
        labels_list = self.dataset_config.get_labels_list()
        labels_list.append(self.NULL_LABEL_TOKEN)
        confusion = confusion_matrix(gt_labels, pred_labels, labels=labels_list)
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

        # error examples
        # TODO, need a way to access input dataset in order to display them here

        return eval_metrics
