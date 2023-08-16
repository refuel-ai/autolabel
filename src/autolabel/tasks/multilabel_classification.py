from collections import defaultdict
from typing import List, Dict, Tuple

from langchain.prompts.prompt import PromptTemplate

from autolabel.configs import AutolabelConfig
from autolabel.schema import LLMAnnotation, MetricType, MetricResult, F1Type
from autolabel.tasks import BaseTask
from autolabel.utils import get_format_variables

import json

from autolabel.metrics import (
    AccuracyMetric,
    AUROCMetric,
    SupportMetric,
    CompletionRateMetric,
    F1Metric,
    BaseMetric,
)


class MultilabelClassificationTask(BaseTask):
    DEFAULT_OUTPUT_GUIDELINES = 'You will return the answer as a semicolon-separated list of labels. For example: "label1;label2;label3"'
    DEFAULT_TASK_GUIDELINES = "Your job is to correctly label the provided input example into one or more of the following {num_labels} categories.\nCategories:\n{labels}\n"

    GENERATE_EXPLANATION_PROMPT = "You are an expert at providing a well reasoned explanation for the output of a given task. \n\nBEGIN TASK DESCRIPTION\n{task_guidelines}\nEND TASK DESCRIPTION\nYou will be given an input example and the corresponding output. Your job is to provide an explanation for why the output is correct for the task above.\nThink step by step and generate an explanation. The last line of the explanation should be - So, the answer is <label>.\n{labeled_example}\nExplanation: "

    def __init__(self, config: AutolabelConfig) -> None:
        super().__init__(config)
        self.metrics = [
            AccuracyMetric(),
            SupportMetric(),
            CompletionRateMetric(),
            F1Metric(
                type=F1Type.MULTI_LABEL,
                labels=self.config.labels_list(),
                sep=self.config.label_separator(),
                average=[MetricType.F1_MACRO, MetricType.F1_WEIGHTED],
            ),
        ]

        if self.config.confidence():
            self.metrics.append(AUROCMetric())

    def construct_prompt(self, input: Dict, examples: List) -> str:
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
                eg_copy[label_column] = json.dumps({label_column: eg[label_column]})
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

    def get_generate_dataset_prompt(
        self, label: str, num_rows: int, guidelines: str = None
    ) -> str:
        raise NotImplementedError("Dataset generation not implemented for this task")

    def eval(
        self,
        llm_labels: List[LLMAnnotation],
        gt_labels: List[str],
        additional_metrics: List[BaseMetric] = [],
    ) -> List[MetricResult]:
        """Evaluate the LLM generated labels by comparing them against ground truth

        Args:
            llm_labels (List[LLMAnnotation]): list of LLM generated labels
            gt_labels (List[str]): list of ground truth labels
            additional_metrics (List[BaseMetric], optional): list of additional metrics to compute. Defaults to [].

        Returns:
            List[MetricResult]: list of metrics and corresponding values
        """

        eval_metrics = []
        for metric in self.metrics + additional_metrics:
            eval_metrics.extend(metric.compute(llm_labels, gt_labels))

        return eval_metrics
