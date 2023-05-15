"""Base interface that all prediction tasks will implement."""

import json
from abc import ABC, abstractmethod
from typing import Dict, List

from autolabel.configs import DatasetConfig, TaskConfig
from autolabel.schema import LLMAnnotation, MetricResult
from autolabel.utils import extract_valid_json_substring
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger


class BaseTask(ABC):
    prompt_template = "{prefix_prompt}\n{task_prompt}\n\n{output_prompt}\n\n{seed_examples_prompt}\n{seed_examples}\nNow I want you to label the following example:\n{current_example}"
    prefix_prompt = ""
    task_prompt = ""
    seed_examples_prompt = "Some examples with their output answers are provided below:"
    output_prompt = ""

    output_format = ""

    prompt_template_variables = [
        "prefix_prompt",
        "task_prompt",
        "output_prompt",
        "seed_examples_prompt",
        "seed_examples",
        "current_example",
    ]

    NULL_LABEL_TOKEN = "NO_LABEL"

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

        # Update the default prompt template with the prompt template from the config
        if self.config.get_prompt_template():
            self.prompt_template = self.config.get_prompt_template()

        if self.config.get_prefix_prompt():
            self.prefix_prompt = self.config.get_prefix_prompt()

        if self.config.get_task_prompt():
            self.task_prompt = self.config.get_task_prompt()

        if self.config.get_output_prompt():
            self.output_prompt = self.config.get_output_prompt()

        if self.config.get_output_format():
            self.output_format = self.config.get_output_format()

        # If the output prompt is not passed in, we will generate it based on the output format
        if not self.output_prompt:
            if self.output_format == "json":
                self.output_prompt = self.JSON_OUTPUT_FORMAT_PROMPT
            elif self.output_format == "csv":
                self.output_prompt = self.CSV_OUTPUT_FORMAT_PROMPT

        self.partial_prompt = self.initialize_prompt_template()

    @abstractmethod
    # This initialzes the prompt template for the task but leaves out dataset
    # specific information such as the seed examples and the current example,
    # Dataset specific initialization should be done in the construct prompt
    # method
    def initialize_prompt_template(self) -> PromptTemplate:
        pass

    @abstractmethod
    def construct_prompt(self, input: str, examples: List) -> str:
        pass

    @abstractmethod
    def eval(self, llm_labels: List, gt_labels: List) -> List[MetricResult]:
        pass

    # Should be called before the construct prompt for a specific input is called
    def set_dataset_config(self, dataset_config: DatasetConfig) -> None:
        self.dataset_config = dataset_config

    def _to_output_format(self, label: str) -> str:
        if self.output_format == "json":
            output = {"label": label}
            return json.dumps(output)
        elif self.output_format == "csv":
            return f"{label}"

    def get_explanation(self, example: Dict) -> str:
        if example.get("explanation", ""):
            return f"Let's think step by step.\n{example['explanation']}"
        else:
            return ""

    def parse_llm_response(
        self, response: Generation, curr_sample: Dict, prompt: str
    ) -> LLMAnnotation:
        if self.output_format == "json":
            return self.parse_json_llm_response(
                response, json.dumps(curr_sample), prompt
            )
        elif self.output_format == "csv":
            return self.parse_csv_llm_response(
                response, json.dumps(curr_sample), prompt
            )

    def parse_json_llm_response(
        self, response: Generation, curr_sample: str, prompt: str
    ) -> LLMAnnotation:
        output = {}
        try:
            completion_text = extract_valid_json_substring(response.text)
            if not completion_text:
                raise ValueError(
                    "No valid JSON substring found. Either increase the max_tokens or the model is not able to follow the output format instruction."
                )
            output = json.loads(completion_text.strip())
            successfully_labeled = "yes"
            llm_label = str(output.get("label") or self.NULL_LABEL_TOKEN)
        except Exception as e:
            successfully_labeled = "no"
            llm_label = self.NULL_LABEL_TOKEN
            logger.info(f"Error parsing LLM response: {response.text}. {repr(e)}")

        # TODO: parse generation info correctly to fetch & transform logprobs -> score
        return LLMAnnotation(
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
            raw_response=response.text,
            prompt=prompt,
            curr_sample=curr_sample,
        )

    def parse_csv_llm_response(
        self, response: Generation, curr_sample: str, prompt: str
    ) -> LLMAnnotation:
        # The last line of the response is the label
        # This is done to handle the case where the model generates an explanation before generating the label
        completion_text = response.text.strip().split("\n")[-1].strip()
        if len(completion_text) == 0:
            successfully_labeled = "no"
            llm_label = self.NULL_LABEL_TOKEN
            logger.error(f"Error parsing LLM response: {response.text}")
        else:
            successfully_labeled = "yes"
            llm_label = completion_text.strip()

        # TODO: parse generation info correctly to fetch & transform logprobs -> score
        return LLMAnnotation(
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
            raw_response=response.text,
            prompt=prompt,
            curr_sample=curr_sample,
        )
