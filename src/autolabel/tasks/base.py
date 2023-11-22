"""Base interface that all prediction tasks will implement."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging
import json
import pickle

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from autolabel.configs import AutolabelConfig
from autolabel.schema import (
    LLMAnnotation,
    MetricResult,
    FewShotAlgorithm,
    TaskType,
    LabelingError,
    ErrorType,
    ModelProvider,
)
from autolabel.utils import (
    get_format_variables,
    extract_valid_json_substring,
)
from autolabel.metrics import BaseMetric

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    ZERO_SHOT_TEMPLATE = "{task_guidelines}\n\n{output_guidelines}\n\nNow I want you to label the following example:\n{current_example}"
    FEW_SHOT_TEMPLATE = "{task_guidelines}\n\n{output_guidelines}\n\nSome examples with their output answers are provided below:\n\n{seed_examples}\n\nNow I want you to label the following example:\n{current_example}"

    ZERO_SHOT_TEMPLATE_REFUEL_LLM = """
    <s>[INST] <<SYS>>
    {task_guidelines}{output_guidelines}
    <</SYS>>
    {current_example}[/INST]\n"""
    FEW_SHOT_TEMPLATE_REFUEL_LLM = """
    <s>[INST] <<SYS>>
    {task_guidelines}{output_guidelines}\n{seed_examples}
    <</SYS>>
    {current_example}[/INST]\n"""

    # Downstream classes should override these
    NULL_LABEL_TOKEN = "NO_LABEL"
    DEFAULT_TASK_GUIDELINES = ""
    DEFAULT_OUTPUT_GUIDELINES = ""
    DEFAULT_DATASET_GENERATION_GUIDELINES = ""

    def __init__(self, config: AutolabelConfig) -> None:
        self.config = config

        # Update the default prompt template with the prompt template from the config
        self.task_guidelines = (
            self.config.task_guidelines() or self.DEFAULT_TASK_GUIDELINES
        )
        self.output_guidelines = (
            self.config.output_guidelines() or self.DEFAULT_OUTPUT_GUIDELINES
        )

        self.dataset_generation_guidelines = (
            self.config.dataset_generation_guidelines()
            or self.DEFAULT_DATASET_GENERATION_GUIDELINES
        )
        self._prompt_schema_init()

    def _prompt_schema_init(self) -> None:
        self.use_refuel_prompt_schema = self.config.provider() == ModelProvider.REFUEL
        if self._is_few_shot_mode():
            self.example_template = (
                self.FEW_SHOT_TEMPLATE_REFUEL_LLM
                if self.use_refuel_prompt_schema
                else self.FEW_SHOT_TEMPLATE
            )
        else:
            self.example_template = (
                self.ZERO_SHOT_TEMPLATE_REFUEL_LLM
                if self.use_refuel_prompt_schema
                else self.ZERO_SHOT_TEMPLATE
            )
        self.prompt_template = PromptTemplate(
            input_variables=get_format_variables(self.example_template),
            template=self.example_template,
        )

    def _is_few_shot_mode(self) -> bool:
        return self.config.few_shot_algorithm() in [x.value for x in FewShotAlgorithm]

    @abstractmethod
    def construct_prompt(
        self,
        input: str,
        examples: List,
        prompt_template_override: PromptTemplate = None,
        refuel_prompt_override: bool = False,
        output_guidelines_override: str = None,
        **kwargs,
    ) -> str:
        pass

    def construct_confidence_prompt(self, input: str, examples: List, **kwargs) -> str:
        prompt_template = PromptTemplate(
            input_variables=get_format_variables(
                self.FEW_SHOT_TEMPLATE_REFUEL_LLM
                if self._is_few_shot_mode()
                else self.ZERO_SHOT_TEMPLATE_REFUEL_LLM
            ),
            template=self.example_template,
        )
        refuel_prompt = self.construct_prompt(
            input,
            examples,
            prompt_template=prompt_template,
            refuel_prompt_override=True,
            **kwargs,
        )
        return refuel_prompt

    @abstractmethod
    def eval(
        self,
        llm_labels: List,
        gt_labels: List,
        additional_metrics: Optional[List[BaseMetric]] = [],
    ) -> List[MetricResult]:
        pass

    @abstractmethod
    def get_explanation_prompt(self, example: Dict) -> str:
        raise NotImplementedError(
            "Explanation generation not implemented for this task"
        )

    @abstractmethod
    def get_generate_dataset_prompt(
        self, label: str, num_rows: int, guidelines: str = None
    ) -> str:
        raise NotImplementedError("Dataset generation not implemented for this task")

    def parse_llm_response(
        self, response: Generation, curr_sample: Dict, prompt: str
    ) -> LLMAnnotation:
        # The last line of the response is the label
        # This is done to handle the case where the model generates an explanation before generating the label
        error = None
        if self.config.chain_of_thought():
            try:
                explanation = response.text.strip().split("\n")[0].strip()
                completion_text = extract_valid_json_substring(
                    response.text.strip().split("\n")[-1].strip()
                )
                completion_text = json.loads(completion_text)["label"]
            except:
                completion_text = None
        else:
            completion_text = response.text.strip().split("\n")[-1].strip()
        if len(response.text.strip()) == 0:
            successfully_labeled = False
            llm_label = self.NULL_LABEL_TOKEN
            logger.warning("LLM response is empty")
            error = LabelingError(
                error_type=ErrorType.EMPTY_RESPONSE_ERROR,
                error_message="Empty response from LLM",
            )
        elif not completion_text:
            successfully_labeled = False
            llm_label = self.NULL_LABEL_TOKEN
            logger.warning(f"Error parsing LLM response: {response.text}")
            error = LabelingError(
                error_type=ErrorType.PARSING_ERROR,
                error_message=f"Error parsing LLM response: {response.text}",
            )
        else:
            llm_label = completion_text.strip()
            if self.config.task_type() in [
                TaskType.CLASSIFICATION,
                TaskType.ENTITY_MATCHING,
            ]:
                if llm_label in self.config.labels_list():
                    successfully_labeled = True
                else:
                    logger.warning(f"LLM response is not in the labels list")
                    llm_label = self.NULL_LABEL_TOKEN
                    successfully_labeled = False
                    error = LabelingError(
                        error_type=ErrorType.OUTPUT_GUIDELINES_NOT_FOLLOWED_ERROR,
                        error_message=f"LLM response is not in the labels list: {llm_label}",
                    )
            elif self.config.task_type() == TaskType.MULTILABEL_CLASSIFICATION:
                llm_multi_labels = llm_label.split(self.config.label_separator())
                llm_multi_labels = list(
                    filter(
                        lambda label: label in self.config.labels_list(),
                        llm_multi_labels,
                    )
                )
                if len(llm_multi_labels) == 0:
                    llm_label = self.NULL_LABEL_TOKEN
                    successfully_labeled = False
                    error = LabelingError(
                        error_type=ErrorType.OUTPUT_GUIDELINES_NOT_FOLLOWED_ERROR,
                        error_message=f"LLM response is not in the labels list: {llm_label}",
                    )
                else:
                    llm_label = self.config.label_separator().join(llm_multi_labels)
                    successfully_labeled = True
            else:
                successfully_labeled = True

        return LLMAnnotation(
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
            raw_response=response.text,
            prompt=prompt,
            curr_sample=pickle.dumps(curr_sample),
            explanation=explanation if self.config.chain_of_thought() else "",
            error=error,
        )
