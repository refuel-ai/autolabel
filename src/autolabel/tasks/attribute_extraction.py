from typing import List, Dict
from collections import defaultdict
import logging
import json
import pickle

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation

from autolabel.configs import AutolabelConfig
from autolabel.tasks import BaseTask
from autolabel.schema import (
    LLMAnnotation,
    MetricResult,
    F1Type,
    LabelingError,
    ErrorType,
    TaskType,
    MetricType,
)
from autolabel.utils import get_format_variables
from autolabel.metrics import (
    AccuracyMetric,
    SupportMetric,
    CompletionRateMetric,
    BaseMetric,
)

logger = logging.getLogger(__name__)


class AttributeExtractionTask(BaseTask):
    NULL_LABEL = {}
    DEFAULT_TASK_GUIDELINES = "You are an expert at extracting attributes from text. Given a piece of text, extract the required attributes."
    DEFAULT_OUTPUT_GUIDELINES = "You will return the extracted attributes as a json with the following keys:\n{attribute_json}"
    DEFAULT_OUTPUT_GUIDELINES_CSV = "You will return the extracted attributes as a csv with the following rows:\n{attribute_csv}"

    OUTPUT_STR = "output_str"

    def __init__(self, config: AutolabelConfig) -> None:
        super().__init__(config)

        self.metrics = [
            SupportMetric(),
            CompletionRateMetric(),
            AccuracyMetric(),
        ]

    def should_use_csv(self):
        """Returns true if the task should use csv output guidelines"""
        return self.config.provider() in ["refuel", "huggingface_pipeline"]

    def _construct_attribute_json(self) -> str:
        """This function is used to construct the attribute json string for the output guidelines.
        Args:
            attributes (List[Dict]): A list of dictionaries containing the output attributes.

        Returns:
            str: A string containing the output attributes.
        """
        output_json = {}
        for attribute_dict in self.config.attributes():
            if "name" not in attribute_dict or "description" not in attribute_dict:
                raise ValueError(
                    "Attribute dictionary must contain 'name' and 'description' keys"
                )

            attribute_name = attribute_dict["name"]
            attribute_desc = attribute_dict["description"]
            if "options" in attribute_dict:
                attribute_options = attribute_dict["options"]
                attribute_desc += f"\nOptions:\n{','.join(attribute_options)}"

            output_json[attribute_name] = attribute_desc
        return json.dumps(output_json, indent=4)

    def _construct_attribute_csv(self) -> str:
        """This function is used to construct the attribute csv string for the output guidelines.
        Args:
            attributes (List[Dict]): A list of dictionaries containing the output attributes.

        Returns:
            str: A string containing the output attributes.
        """
        output_csv = "attribute,value\n"
        for attribute_dict in self.config.attributes():
            if "name" not in attribute_dict or "description" not in attribute_dict:
                raise ValueError(
                    "Attribute dictionary must contain 'name' and 'description' keys"
                )

            attribute_name = attribute_dict["name"]
            attribute_desc = attribute_dict["description"]
            if "options" in attribute_dict:
                attribute_options = attribute_dict["options"]
                attribute_desc += f" Options: {','.join(attribute_options)}"

            output_csv += f"{attribute_name},{attribute_desc}\n"
        return output_csv

    def _generate_output_str(self, input: Dict) -> Dict:
        """Generate the output dictionary from the input

        Args:
            input (Dict): The input dictionary

        Returns:
            Dict: The output dictionary
        """
        if self.should_use_csv():
            output_csv = "attribute,value\n"
            for attribute in self.config.attributes():
                attribute_name = attribute["name"]
                output_csv += f"{attribute_name},{input[attribute_name]}\n"
            return output_csv
        else:
            output_dict = {}
            for attribute in self.config.attributes():
                attribute_name = attribute["name"]
                output_dict[attribute_name] = input[attribute_name]
            return json.dumps(output_dict)

    def construct_prompt(self, input: Dict, examples: List) -> str:
        fmt_task_guidelines = self.task_guidelines

        if self.should_use_csv():
            attribute_csv = self.config._construct_attribute_csv()
            fmt_output_guidelines = self.output_guidelines_csv.format(
                attribute_csv=attribute_csv
            )
        else:
            attribute_json = self._construct_attribute_json()
            fmt_output_guidelines = self.output_guidelines.format(
                attribute_json=attribute_json
            )

        # prepare seed examples
        example_template = self.config.example_template()
        fmt_examples = []
        for eg in examples:
            output_str = self._generate_output_str(eg)
            eg.update({self.OUTPUT_STR: output_str})
            fmt_examples.append(example_template.format_map(defaultdict(str, eg)))

        input[self.OUTPUT_STR] = ""

        # populate the current example in the prompt
        current_example = example_template.format_map(defaultdict(str, input))

        if self._is_few_shot_mode():
            return self.prompt_template.format(
                task_guidelines=fmt_task_guidelines,
                output_guidelines=fmt_output_guidelines,
                seed_examples="\n\n".join(fmt_examples),
                current_example=current_example,
            )
        else:
            return self.prompt_template.format(
                task_guidelines=fmt_task_guidelines,
                output_guidelines=fmt_output_guidelines,
                current_example=current_example,
            )

    def get_explanation_prompt(self, example: Dict) -> str:
        raise NotImplementedError(
            "Explanation generation not implemented for this task"
        )

    def get_generate_dataset_prompt(
        self, label: str, num_rows: int, guidelines: str = None
    ) -> str:
        raise NotImplementedError("Dataset generation not implemented for this task")

    def parse_llm_response(
        self, response: Generation, curr_sample: Dict, prompt: str
    ) -> LLMAnnotation:
        successfully_labeled = False
        error = None
        try:
            print("Completion text", response.text)
            completion_text = response.text
            llm_label = json.loads(completion_text)
            successfully_labeled = True
        except Exception as e:
            logger.error(f"Error parsing LLM response: {response.text}, Error: {e}")
            llm_label = self.NULL_LABEL
            error = LabelingError(
                error_type=ErrorType.PARSING_ERROR, error_message=str(e)
            )

        # TODO(rajas): Handle output guidelines not followed error (for options case)

        return LLMAnnotation(
            curr_sample=pickle.dumps(curr_sample),
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
            raw_response=response.text,
            prompt=prompt,
            error=error,
        )

    def eval(
        self,
        llm_labels: List[LLMAnnotation],
        gt_labels: List[str],
        additional_metrics: List[BaseMetric] = [],
    ) -> List[MetricResult]:
        """Evaluate the LLM generated labels by comparing them against ground truth"""

        # Convert the llm labels into a mapping from
        # name -> List[LLMAnnotation]
        llm_labels_dict = defaultdict(list)
        for llm_label in llm_labels:
            for attribute, value in llm_label.label.items():
                llm_labels_dict[attribute].append(
                    LLMAnnotation(
                        successfully_labeled=llm_label.successfully_labeled,
                        label=value,
                        raw_response=llm_label.raw_response,
                        curr_sample=llm_label.curr_sample,
                        prompt=llm_label.prompt,
                        error=llm_label.error,
                    )
                )

        eval_metrics = []

        for attribute in llm_labels_dict.keys():
            for metric in self.metrics + additional_metrics:
                if gt_labels[attribute] is None:
                    continue

                computed_metrics = metric.compute(
                    llm_labels_dict[attribute],
                    gt_labels[attribute],
                )
                for m in computed_metrics:
                    eval_metrics.append(
                        MetricResult(
                            name=f"{attribute}:{m.name}",
                            value=m.value,
                        )
                    )

        return eval_metrics
