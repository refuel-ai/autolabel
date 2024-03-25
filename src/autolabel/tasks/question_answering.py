import json
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from langchain.prompts.prompt import PromptTemplate
from sklearn.metrics import accuracy_score

from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import AutolabelConfig
from autolabel.metrics import (
    AccuracyMetric,
    AUROCMetric,
    BaseMetric,
    CompletionRateMetric,
    F1Metric,
    SupportMetric,
)
from autolabel.schema import (
    F1Type,
    LLMAnnotation,
    MetricResult,
    MetricType,
    ModelProvider,
)
from autolabel.tasks import BaseTask
from autolabel.tasks.utils import filter_unlabeled_examples, normalize_text
from autolabel.utils import get_format_variables

logger = logging.getLogger(__name__)


class QuestionAnsweringTask(BaseTask):
    DEFAULT_OUTPUT_GUIDELINES = (
        'You will return just the answer and nothing else\n'
    )
    LLAMA_DEFAULT_OUTPUT_GUIDELINES = ""
    DEFAULT_TASK_GUIDELINES = "Your job is to answer the following questions using the options provided for each question. Choose the best answer for the question.\n"
    NULL_LABEL_TOKEN = "NO_LABEL"

    LABEL_FORMAT_IN_EXPLANATION = (
        " The last line of the explanation should be - So, the answer is <label>."
    )
    EXCLUDE_LABEL_IN_EXPLANATION = " Do not repeat the output of the task - simply provide an explanation for the provided output. The provided label was generated by you in a previous step and your job now is to only provided an explanation for the output. Your job is not verify the output but instead explain why it might have been generated, even if it is incorrect. If you think the provided output is incorrect, give an explanation of why it might have been generated anyway but don't say that the output may be incorrect or incorrectly generated.'"
    GENERATE_EXPLANATION_PROMPT = "You are an expert at providing a well reasoned explanation for the output of a given task. \n\nBEGIN TASK DESCRIPTION\n{task_guidelines}\nEND TASK DESCRIPTION\nYou will be given an input example and the corresponding output. You will be given a question and an answer. Your job is to provide an explanation for why the answer is correct for the task above.\nThink step by step and generate an explanation.{label_format}\n{labeled_example}\nExplanation: "

    def __init__(self, config: AutolabelConfig) -> None:
        if config.provider() in [ModelProvider.REFUEL, ModelProvider.TGI]:
            self.DEFAULT_OUTPUT_GUIDELINES = self.LLAMA_DEFAULT_OUTPUT_GUIDELINES

        super().__init__(config)
        self.metrics = [
            AccuracyMetric(),
            SupportMetric(),
            CompletionRateMetric(),
            F1Metric(
                type=F1Type.TEXT,
            ),
        ]

        if self.config.confidence():
            self.metrics.append(AUROCMetric())

    def construct_prompt(
        self,
        input: Dict,
        examples: List[Dict],
        prompt_template_override: PromptTemplate = None,
        refuel_prompt_override: bool = False,
        output_guidelines_override: str = None,
        max_input_tokens: int = None,
        get_num_tokens: Optional[Callable] = None,
        **kwargs,
    ) -> str:
        # Copy over the input so that we can modify it
        input = input.copy()

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
            curr_text_prompt = prompt_template.format(
                task_guidelines=self.task_guidelines,
                output_guidelines=output_guidelines,
                seed_examples="\n\n".join(fmt_examples),
                current_example=current_example,
            )
        else:
            curr_text_prompt = prompt_template.format(
                task_guidelines=self.task_guidelines,
                output_guidelines=output_guidelines,
                current_example=current_example,
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

    def construct_confidence_prompt(self, input: str, examples: List, **kwargs) -> str:
        output_guidelines_override = (
            self.config.output_guidelines() or self.LLAMA_DEFAULT_OUTPUT_GUIDELINES
        )
        refuel_prompt = super().construct_confidence_prompt(
            input,
            examples,
            output_guidelines_override=output_guidelines_override,
            **kwargs,
        )
        return refuel_prompt

    def get_explanation_prompt(self, example: Dict, include_label=True) -> str:
        pt = PromptTemplate(
            input_variables=get_format_variables(self.GENERATE_EXPLANATION_PROMPT),
            template=self.GENERATE_EXPLANATION_PROMPT,
        )
        example_template = self.config.example_template()
        fmt_example = example_template.format_map(defaultdict(str, example))

        return pt.format(
            task_guidelines=self.task_guidelines,
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
        additional_metrics: Optional[List[BaseMetric]] = [],
    ) -> List[MetricResult]:
        """Evaluate the LLM generated labels by comparing them against ground truth

        Args:
            llm_labels (List[LLMAnnotation]): _description_
            gt_labels (List[str]): _description_
            additional_metrics (Optional[List[BaseMetric]], optional): _description_. Defaults to [].

        Returns:
            List[MetricResult]: list of metrics and corresponding values
        """
        eval_metrics = []

        for metric in self.metrics + additional_metrics:
            eval_metrics.extend(metric.compute(llm_labels, gt_labels))

        return eval_metrics
