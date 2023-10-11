from collections import defaultdict
from typing import List, Dict, Tuple

from langchain.prompts.prompt import PromptTemplate
from sklearn.metrics import accuracy_score

from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import AutolabelConfig
from autolabel.schema import LLMAnnotation, MetricType, MetricResult, ModelProvider
from autolabel.tasks import BaseTask
from autolabel.utils import get_format_variables
from autolabel.tasks.utils import filter_unlabeled_examples
from autolabel.metrics import (
    AccuracyMetric,
    AUROCMetric,
    SupportMetric,
    CompletionRateMetric,
    ClassificationReportMetric,
    BaseMetric,
)

import json


class ClassificationTask(BaseTask):
    DEFAULT_OUTPUT_GUIDELINES = (
        'You will return the answer with just one element: "the correct label"'
    )
    DEFAULT_TASK_GUIDELINES = "Your job is to correctly label the provided input example into one of the following {num_labels} categories.\nCategories:\n{labels}\n"

    GENERATE_EXPLANATION_PROMPT = "You are an expert at providing a well reasoned explanation for the output of a given task. \n\nBEGIN TASK DESCRIPTION\n{task_guidelines}\nEND TASK DESCRIPTION\nYou will be given an input example and the corresponding output. Your job is to provide an explanation for why the output is correct for the task above.\nThink step by step and generate an explanation. The last line of the explanation should be - So, the answer is <label>.\n{labeled_example}\nExplanation: "

    GENERATE_DATASET_TEMPLATE = "{guidelines}\n\nThe inputs must be diverse, covering a wide range of scenarios. You will not generate duplicate inputs. These inputs should be organized in rows in csv format with the columns {columns}.\n\n{label_descriptions}\n\n{format_guidelines}\n\n{output_guidelines}\n\n```csv"
    DEFAULT_DATASET_GENERATION_GUIDELINES = "You are an expert at generating plausible inputs for a given task.\n\nBEGIN TASK DESCRIPTION\n{task_guidelines}\nEND TASK DESCRIPTION"
    LABEL_DESCRIPTIONS_PROMPT = "Each input should fall into one of these {num_labels} categories. These are the only categories that the inputs can belong to."
    GENERATE_DATASET_FORMAT_GUIDELINES = "Your response should be in csv format with the following columns: {columns}.\n\nHere is a template you can follow for your output:\n```csv\n{columns}\n{example_rows}\n```\n\nMake sure to replace the placeholder variables with your own values."
    GENERATE_DATASET_OUTPUT_GUIDELINES = 'Now I want you to generate {num_rows} excerpts that follow the guidelines and all belong to the "{label}" category. They should not belong to any of the other categories.'

    def __init__(self, config: AutolabelConfig) -> None:
        super().__init__(config)
        self.metrics = [
            AccuracyMetric(),
            SupportMetric(),
            CompletionRateMetric(),
            ClassificationReportMetric(),
        ]

        if self.config.confidence():
            self.metrics.append(AUROCMetric())

    def construct_prompt(
        self, input: Dict, examples: List, selected_labels: List[str] = None
    ) -> str:
        # Copy over the input so that we can modify it
        input = input.copy()

        # prepare task guideline
        labels_list = (
            self.config.labels_list() if not selected_labels else selected_labels
        )
        num_labels = len(labels_list)

        is_refuel_llm = self.config.provider() == ModelProvider.REFUEL

        if is_refuel_llm:
            labels = (
                ", ".join([f'\\"{i}\\"' for i in labels_list[:-1]])
                + " or "
                + f'\\"{labels_list[-1]}\\"'
            )
        else:
            labels = "\n".join(labels_list)

        fmt_task_guidelines = self.task_guidelines.format(
            num_labels=num_labels, labels=labels
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
                seed_examples=("\n" if is_refuel_llm else "\n\n").join(fmt_examples),
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

    def get_generate_dataset_prompt(self, label: str) -> str:
        pt = PromptTemplate(
            input_variables=get_format_variables(self.GENERATE_DATASET_TEMPLATE),
            template=self.GENERATE_DATASET_TEMPLATE,
        )

        # prepare task guideline
        labels_list = self.config.labels_list()
        num_labels = len(labels_list)
        fmt_task_guidelines = self.task_guidelines.format(
            num_labels=num_labels, labels="\n".join(labels_list)
        )
        fmt_guidelines = self.dataset_generation_guidelines.format(
            task_guidelines=fmt_task_guidelines
        )

        # prepare columns
        columns = get_format_variables(self.config.example_template())
        columns.remove(self.config.label_column())

        # prepare label descriptions
        fmt_label_descriptions = self.LABEL_DESCRIPTIONS_PROMPT.format(
            num_labels=num_labels
        )
        for i, l in enumerate(labels_list):
            fmt_label_descriptions += f"\n{i+1}. {l}{': ' + self.config.label_descriptions()[l] if self.config.label_descriptions() is not None and l in self.config.label_descriptions() else ''}"

        # prepare format
        example_rows = "\n".join(
            [",".join([f'"{column}_{i+1}"' for column in columns]) for i in range(3)]
        )
        fmt_format_guidelines = self.GENERATE_DATASET_FORMAT_GUIDELINES.format(
            columns=",".join(columns), example_rows=example_rows
        )

        # prepare output guidelines
        fmt_output_guidelines = self.GENERATE_DATASET_OUTPUT_GUIDELINES.format(
            num_rows=self.config.dataset_generation_num_rows(), label=label
        )

        return pt.format(
            guidelines=fmt_guidelines,
            columns=columns,
            label_descriptions=fmt_label_descriptions,
            format_guidelines=fmt_format_guidelines,
            output_guidelines=fmt_output_guidelines,
        )

    def eval(
        self,
        llm_labels: List[LLMAnnotation],
        gt_labels: List[str],
        additional_metrics: List[BaseMetric] = [],
    ) -> List[MetricResult]:
        """Evaluate the LLM generated labels by comparing them against ground truth

        Args:
            llm_labels (List[LLMAnnotation]): _description_
            gt_labels (List[str]): _description_
            additional_metrics (List[BaseMetric], optional): The additional metrics to run. Defaults to [].

        Returns:
            List[MetricResult]: list of metrics and corresponding values
        """

        eval_metrics = []

        for metric in self.metrics + additional_metrics:
            eval_metrics.extend(metric.compute(llm_labels, gt_labels))

        return eval_metrics
