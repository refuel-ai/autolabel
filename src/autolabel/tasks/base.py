"""Base interface that all prediction tasks will implement."""

from abc import ABC, abstractmethod
from typing import Dict, List
import logging
import json

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from autolabel.configs import AutolabelConfig
from autolabel.schema import LLMAnnotation, MetricResult, FewShotAlgorithm, TaskType
from autolabel.utils import get_format_variables, extract_valid_json_substring

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    ZERO_SHOT_TEMPLATE = "{task_guidelines}\n\n{output_guidelines}\n\nNow I want you to label the following example:\n{current_example}"
    FEW_SHOT_TEMPLATE = "{task_guidelines}\n\n{output_guidelines}\n\nSome examples with their output answers are provided below:\n\n{seed_examples}\n\nNow I want you to label the following example:\n{current_example}"

    # Downstream classes should override these
    NULL_LABEL_TOKEN = "NO_LABEL"
    DEFAULT_TASK_GUIDELINES = ""
    DEFAULT_OUTPUT_GUIDELINES = ""

    def __init__(self, config: AutolabelConfig) -> None:
        self.config = config

        # Update the default prompt template with the prompt template from the config
        self.task_guidelines = (
            self.config.task_guidelines() or self.DEFAULT_TASK_GUIDELINES
        )
        self.output_guidelines = (
            self.config.output_guidelines() or self.DEFAULT_OUTPUT_GUIDELINES
        )

        if self._is_few_shot_mode():
            self.prompt_template = PromptTemplate(
                input_variables=get_format_variables(self.FEW_SHOT_TEMPLATE),
                template=self.FEW_SHOT_TEMPLATE,
            )
        else:
            self.prompt_template = PromptTemplate(
                input_variables=get_format_variables(self.ZERO_SHOT_TEMPLATE),
                template=self.ZERO_SHOT_TEMPLATE,
            )

    def _is_few_shot_mode(self) -> bool:
        return self.config.few_shot_algorithm() in [x.value for x in FewShotAlgorithm]

    @abstractmethod
    def construct_prompt(self, input: str, examples: List) -> str:
        pass

    @abstractmethod
    def eval(self, llm_labels: List, gt_labels: List) -> List[MetricResult]:
        pass

    @abstractmethod
    def get_explanation_prompt(self, example: Dict) -> str:
        raise NotImplementedError(
            "Explanation generation not implemented for this task"
        )

    def parse_llm_response(
        self, response: Generation, curr_sample: Dict, prompt: str
    ) -> LLMAnnotation:
        # The last line of the response is the label
        # This is done to handle the case where the model generates an explanation before generating the label
        if self.config.chain_of_thought():
            try:
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
            logger.warning(f"LLM response is empty")
        elif not completion_text:
            successfully_labeled = False
            llm_label = self.NULL_LABEL_TOKEN
            logger.error(f"Error parsing LLM response: {response.text}")
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
                    successfully_labeled = False
            else:
                successfully_labeled = True

        return LLMAnnotation(
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
            raw_response=response.text,
            prompt=prompt,
            curr_sample=json.dumps(curr_sample),
        )
