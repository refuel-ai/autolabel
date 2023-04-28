import json
from typing import List, Dict, Tuple

from langchain.prompts.prompt import PromptTemplate
from loguru import logger
from refuel_oracle.confidence import ConfidenceCalculator
from refuel_oracle.task_config import TaskConfig
from refuel_oracle.schema import LLMAnnotation, Metric, MetricResult
from refuel_oracle.tasks import BaseTask
from sklearn.metrics import accuracy_score
import transformers

transformers.logging.set_verbosity_error()


class EntityMatchingTask(BaseTask):
    JSON_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": "can you answer this question. say yes or no", "label": "duplicate or not duplicate"}\n'
    CSV_OUTPUT_FORMAT_PROMPT = 'You will return the answer in CSV format with two elements: "can you answer this question. say yes or no", "duplicate or not duplicate"\n'
    NO_OUTPUT_FORMAT_PROMPT = 'You will return the answer in plain text format with one element: "duplicate or not duplicate"\n'

    task_prompt = "Your job is to tell if the two given entities are duplicates or not. Say duplicate, if they are duplicate and not duplicate otherwise. Options:\nduplicate\nnot duplicate\n"
    example_prompt_template = (
        "Entity1: {entity1}\nEntity2: {entity2}\nAnswer:{answer}\n"
    )
    example_prompt_variables = ["entity1", "entity2", "answer"]

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

    def construct_prompt(self, input: Dict, examples: List[Dict]) -> str:
        # populate seed examples in the prompt
        example_prompt = PromptTemplate(
            input_variables=self.example_prompt_variables,
            template=self.example_prompt_template,
        )
        formatted_examples = []
        for eg in examples:
            expected_output = self._to_output_format(eg["label"])
            formatted_examples.append(
                example_prompt.format(
                    entity1=eg["entity1"],
                    entity2=eg["entity2"],
                    answer=expected_output,
                )
            )

        # populate the current example in the prompt
        current_example = example_prompt.format(
            entity1=input["entity1"],
            entity2=input["entity2"],
            answer="",  # we don't know the answer yet
        )

        if len(examples):
            seed_examples_prompt = self.seed_examples_prompt
        else:
            seed_examples_prompt = ""

        prompt = self.partial_prompt.format(
            seed_examples="\n".join(formatted_examples),
            current_example=current_example,
            seed_examples_prompt=seed_examples_prompt,
        )
        return prompt

    def auroc_score_labels(
        self, gt_labels, llm_labels_with_conf
    ) -> Tuple[List[int], List[float]]:
        labels = []
        confidences = []
        for index, llm_label in enumerate(llm_labels_with_conf):
            labels.append(gt_labels[index] == llm_label.label)
            confidences.append(llm_label.confidence_score)

        ConfidenceCalculator.plot_data_distribution(labels, confidences)
        return labels, confidences

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

        if llm_labels[0].confidence_score is not None:
            # Assuming that if the first element has a confidence score
            # then all following elements have a confidence score too
            labels, confidences = self.auroc_score_labels(gt_labels, llm_labels)
            eval_metrics.append(
                MetricResult(
                    metric_type=Metric.AUROC,
                    name="auroc",
                    value=ConfidenceCalculator.compute_auroc(labels, confidences),
                )
            )
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
            accuracy = 0.0
        else:
            accuracy = accuracy_score(filtered_gt_labels, filtered_pred_labels)
        eval_metrics.append(
            MetricResult(metric_type=Metric.ACCURACY, name="accuracy", value=accuracy)
        )

        # error examples
        # TODO, need a way to access input dataset in order to display them here

        return eval_metrics
