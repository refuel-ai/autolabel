"""Base interface that all prediction tasks will implement."""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import ChatGeneration, Generation

from autolabel.configs import AutolabelConfig
from autolabel.metrics import BaseMetric
from autolabel.schema import (
    ErrorType,
    FewShotAlgorithm,
    LabelingError,
    LLMAnnotation,
    MetricResult,
)
from autolabel.utils import extract_valid_json_substring, get_format_variables

logger = logging.getLogger(__name__)

REFUEL_LLM_MODEL = "refuel-llm"


class BaseTask(ABC):
    ZERO_SHOT_TEMPLATE = "{task_guidelines}\n\n{output_guidelines}\n\nNow I want you to label the following example:\n{current_example}"
    FEW_SHOT_TEMPLATE = "{task_guidelines}\n\n{output_guidelines}\n\nSome examples with their output answers are provided below:\n\n{seed_examples}\n\nNow I want you to label the following example:\n{current_example}"
    # Downstream classes should override these
    NULL_LABEL_TOKEN = "NO_LABEL"
    DEFAULT_TASK_GUIDELINES = ""
    DEFAULT_OUTPUT_GUIDELINES = ""
    DEFAULT_DATASET_GENERATION_GUIDELINES = ""

    def __init__(self, config: AutolabelConfig) -> None:
        self.config = config
        self.image_cols = self.config.image_columns()
        self.input_cols = self.config.input_columns()
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
        if self._is_few_shot_mode():
            self.example_template = self.FEW_SHOT_TEMPLATE
        else:
            self.example_template = self.ZERO_SHOT_TEMPLATE
        self.prompt_template = PromptTemplate(
            input_variables=get_format_variables(self.example_template),
            template=self.example_template,
        )

    def _is_few_shot_mode(self) -> bool:
        return (
            self.config.few_shot_algorithm() in [x.value for x in FewShotAlgorithm]
            and self.config.few_shot_num_examples() > 0
        )

    @abstractmethod
    def construct_prompt(
        self,
        input: str,
        examples: List,
        prompt_template_override: Optional[PromptTemplate] = None,
        output_guidelines_override: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        get_num_tokens: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        pass

    def trim_prompt(
        self,
        prompt_template: PromptTemplate,
        task_guidelines: str,
        output_guidelines: str,
        current_example: str,
        seed_examples: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        get_num_tokens: Optional[Callable] = None,
    ) -> str:
        complete_prompt = prompt_template.format(
            task_guidelines=task_guidelines,
            output_guidelines=output_guidelines,
            seed_examples=seed_examples,
            current_example=current_example,
        )
        if not max_input_tokens or not get_num_tokens:
            return complete_prompt

        trimming_priority = [
            seed_examples,
            task_guidelines,
            output_guidelines,
            current_example,
        ]
        trimmed_elements = {key: key for key in trimming_priority if key is not None}
        for trimming_candidate in trimming_priority:
            current_prompt_length = get_num_tokens(complete_prompt)
            if current_prompt_length <= max_input_tokens:
                break
            if trimming_candidate is None:
                continue
            extra_tokens = current_prompt_length - max_input_tokens
            trimming_candidate_tokens = get_num_tokens(trimming_candidate)
            max_chars = (
                float(len(trimming_candidate))
                * (trimming_candidate_tokens - extra_tokens - 1)
                / (trimming_candidate_tokens + 1)
            )
            final_candidate_chars = int(max(0, max_chars))
            trimmed_elements[trimming_candidate] = trimming_candidate[
                :final_candidate_chars
            ]
            complete_prompt = prompt_template.format(
                task_guidelines=trimmed_elements[task_guidelines],
                output_guidelines=trimmed_elements[output_guidelines],
                seed_examples=(
                    trimmed_elements[seed_examples]
                    if seed_examples is not None
                    else None
                ),
                current_example=trimmed_elements[current_example],
            )

        return complete_prompt

    @abstractmethod
    def eval(
        self,
        llm_labels: List,
        gt_labels: List,
        additional_metrics: Optional[List[BaseMetric]] = [],
    ) -> List[MetricResult]:
        pass

    @abstractmethod
    def get_explanation_prompt(self, example: Dict, include_label=True) -> str:
        raise NotImplementedError(
            "Explanation generation not implemented for this task",
        )

    @abstractmethod
    def get_generate_dataset_prompt(
        self,
        label: str,
        num_rows: int,
        guidelines: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("Dataset generation not implemented for this task")

    def parse_llm_response(
        self,
        response: Union[Generation, ChatGeneration],
        curr_sample: Dict,
        prompt: str,
        selected_labels_map: Optional[Dict[str, List[str]]] = None,
    ) -> LLMAnnotation:
        # The last line of the response is the label
        # This is done to handle the case where the model generates an explanation before generating the label
        error = None
        if self.config.chain_of_thought():
            try:
                explanation = response.text.strip().split("\n")[0].strip()
                completion_text = extract_valid_json_substring(
                    response.text.strip().split("\n")[-1].strip(),
                )
                completion_text = json.loads(completion_text)["label"]
            except Exception as _:
                completion_text = None
        else:
            completion_text = response.text.strip().split("\n")[-1].strip()
        if len(response.text.strip()) == 0:
            successfully_labeled = False
            llm_label = self.NULL_LABEL_TOKEN
            logger.warning("LLM response is empty")
            error = LabelingError(
                error_type=ErrorType.EMPTY_RESPONSE_ERROR,
                error_message="LLM response was empty.",
            )
        elif not completion_text:
            successfully_labeled = False
            llm_label = self.NULL_LABEL_TOKEN
            logger.warning(f"Error parsing LLM response: {response.text}")
            error = LabelingError(
                error_type=ErrorType.PARSING_ERROR,
                error_message="Error parsing LLM response.",
            )

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
