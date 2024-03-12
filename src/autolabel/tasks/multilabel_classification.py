import json
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from langchain.prompts.prompt import PromptTemplate

from autolabel.configs import AutolabelConfig
from autolabel.metrics import (
    AccuracyMetric,
    AUROCMetric,
    BaseMetric,
    CompletionRateMetric,
    F1Metric,
    SupportMetric,
)
from autolabel.schema import F1Type, LLMAnnotation, MetricResult, MetricType
from autolabel.tasks import BaseTask
from autolabel.utils import get_format_variables

logger = logging.getLogger(__name__)


class MultilabelClassificationTask(BaseTask):
    DEFAULT_OUTPUT_GUIDELINES = 'You will return the answer as a semicolon-separated list of labels. For example: "label1;label2;label3"'
    DEFAULT_TASK_GUIDELINES = "Your job is to correctly label the provided input example into one or more of the following {num_labels} categories.\nCategories:\n{labels}\n"

    LABEL_FORMAT_IN_EXPLANATION = " The last line of the explanation should be - So, the answer is <list of label separated by semicolon>."
    EXCLUDE_LABEL_IN_EXPLANATION = " Do not repeat the output of the task - simply provide an explanation for the provided output. The provided label was generated by you in a previous step and your job now is to only provided an explanation for the output. Your job is not verify the output but instead explain why it might have been generated, even if it is incorrect. If you think the provided output is incorrect, give an explanation of why it might have been generated anyway but don't say that the output may be incorrect or incorrectly generated.'"
    GENERATE_EXPLANATION_PROMPT = "You are an expert at providing a well reasoned explanation for the output of a given task. \n\nBEGIN TASK DESCRIPTION\n{task_guidelines}\nEND TASK DESCRIPTION\nYou will be given an input example and the corresponding output (a list of labels seperated by semicolon).\nWhy were these labels given to this input? Output the explanation for each label on a new line, and limit your explanation to one sentence. If there are more than 5 labels, output explanations only for the first 5 labels.{label_format}\n{labeled_example}\nExplanation: "

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

    def construct_prompt(
        self,
        input: Dict,
        examples: List,
        prompt_template_override: PromptTemplate = None,
        refuel_prompt_override: bool = False,
        output_guidelines_override: str = None,
        max_input_tokens: int = None,
        get_num_tokens: Optional[Callable] = None,
        **kwargs,
    ) -> str:
        # Copy over the input so that we can modify it
        input = input.copy()

        # prepare task guideline
        labels_list = self.config.labels_list()
        num_labels = len(labels_list)
        if self.config.label_descriptions():
            labels_list = ""
            for label, description in self.config.label_descriptions().items():
                labels_list = labels_list = f"{label} : {description}\n"
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

        # check if all mapped keys in input are in the example template
        try:
            current_example = example_template.format(**input)
        except KeyError as e:
            current_example = example_template.format_map(defaultdict(str, input))
            logger.warn(
                f'\n\nKey {e} in the "example_template" in the given config'
                f"\n\n{example_template}\n\nis not present in the datsaset columns - {input.keys()}.\n\n"
                f"Input - {input}\n\n"
                "Continuing with the prompt as {current_example}"
            )

        # populate the current example in the prompt
        prompt_template = (
            self.prompt_template
            if prompt_template_override is None
            else prompt_template_override
        )
        output_guidelines = (
            self.output_guidelines
            if output_guidelines_override is None
            else output_guidelines_override
        )
        if self._is_few_shot_mode():
            curr_text_prompt = self.trim_prompt(
                prompt_template,
                task_guidelines=fmt_task_guidelines,
                output_guidelines=output_guidelines,
                seed_examples="\n\n".join(fmt_examples),
                current_example=current_example,
                max_input_tokens=max_input_tokens,
                get_num_tokens=get_num_tokens,
            )
        else:
            curr_text_prompt = self.trim_prompt(
                prompt_template,
                task_guidelines=fmt_task_guidelines,
                output_guidelines=output_guidelines,
                current_example=current_example,
                max_input_tokens=max_input_tokens,
                get_num_tokens=get_num_tokens,
            )
        if self.image_col is not None:
            return json.dumps(
                {"text": curr_text_prompt, "image_url": input[self.image_col]}
            )
        if self.image_cols:
            prompt_dict = {"text": curr_text_prompt}
            for col in self.image_cols:
                if input.get(col) is not None and len(input.get(col)) > 0:
                    prompt_dict[col] = input[col]
                prompt_dict[col] = input[col]
            return json.dumps(prompt_dict)
        else:
            return curr_text_prompt

    def get_explanation_prompt(self, example: Dict, include_label=True) -> str:
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
            label_format=(
                self.LABEL_FORMAT_IN_EXPLANATION
                if include_label
                else self.EXCLUDE_LABEL_IN_EXPLANATION
            ),
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
