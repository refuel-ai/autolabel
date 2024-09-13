import json
import json5
import logging
import pickle
import copy
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union, Tuple

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import ChatGeneration, Generation

from autolabel.configs import AutolabelConfig
from autolabel.metrics import (
    AccuracyMetric,
    AUROCMetric,
    BaseMetric,
    CompletionRateMetric,
    SupportMetric,
)
from autolabel.schema import (
    ErrorType,
    LabelingError,
    LLMAnnotation,
    MetricResult,
    TaskType,
)
from autolabel.tasks import BaseTask
from autolabel.utils import get_format_variables

logger = logging.getLogger(__name__)


class AttributeExtractionTask(BaseTask):
    NULL_LABEL = {}
    DEFAULT_TASK_GUIDELINES = "You are an expert at extracting attributes from text. Given a piece of text, extract the required attributes."
    DEFAULT_OUTPUT_GUIDELINES = "You will return the extracted attributes as a json with the following keys:\n{attribute_json}. \n Do not include keys in the final JSON that don't have any valid value extracted."
    LABEL_FORMAT_IN_EXPLANATION = (
        " The explanation should end with - 'so, the answer is <label>.'"
    )
    EXCLUDE_LABEL_IN_EXPLANATION = " Do not repeat the output of the task - simply provide an explanation for the provided output. The provided label was generated by you in a previous step and your job now is to only provided an explanation for the output. Your job is not verify the output but instead explain why it might have been generated, even if it is incorrect. If you think the provided output is incorrect, give an explanation of why it might have been generated anyway but don't say that the output may be incorrect or incorrectly generated.'"
    GENERATE_EXPLANATION_PROMPT = "You are an expert at providing a well reasoned explanation for the output of a given task. \n\nBEGIN TASK DESCRIPTION\n{task_guidelines}\nEND TASK DESCRIPTION\nYou will be given an input example and the output for one of the attributes. Your job is to provide an explanation for why the output for that attribute is correct for the task above.\nYour explanation should be at most two sentences.{label_format}\n{labeled_example}\nCurrent Attribute:{attribute}.\nExplanation: "
    OUTPUT_DICT_KEY = "output_dict"

    def __init__(self, config: AutolabelConfig) -> None:
        super().__init__(config)

        self.metrics = [
            SupportMetric(),
            CompletionRateMetric(),
            AccuracyMetric(),
        ]

        if self.config.confidence():
            self.metrics.append(AUROCMetric())

    def _construct_attribute_json(self) -> Tuple[str, Dict]:
        """This function is used to construct the attribute json string for the output guidelines.
        Args:
            attributes (List[Dict]): A list of dictionaries containing the output attributes.

        Returns:
            str: A string containing the output attributes.
        """
        output_json, output_schema = {}, {
            "title": "AnswerFormat",
            "type": "object",
            "properties": {},
            "required": [],
            "definitions": {},
        }
        for attribute_dict in self.config.attributes():
            curr_property = {"title": attribute_dict["name"], "type": "string"}
            if "name" not in attribute_dict or "description" not in attribute_dict:
                raise ValueError(
                    "Attribute dictionary must contain 'name' and 'description' keys"
                )

            attribute_name = attribute_dict["name"]
            attribute_desc = attribute_dict["description"]
            if "options" in attribute_dict:
                attribute_options = attribute_dict["options"]
                attribute_desc += f"\nOptions:\n{','.join(attribute_options)}"
                if TaskType.CLASSIFICATION == attribute_dict.get("task_type", ""):
                    curr_property = {"$ref": "#/definitions/" + attribute_name}
                    output_schema["definitions"][attribute_name] = {
                        "title": attribute_name,
                        "description": "An enumeration.",
                        "enum": attribute_options,
                    }

            if TaskType.MULTILABEL_CLASSIFICATION == attribute_dict.get(
                "task_type", ""
            ):
                attribute_desc += "Output should be a list of labels from the options provided below, separated by semicolons."

            output_json[attribute_name] = attribute_desc
            output_schema["properties"][attribute_name] = copy.deepcopy(curr_property)
            output_schema["required"].append(attribute_name)
        return json.dumps(output_json, indent=4), output_schema

    def _generate_output_dict(self, input: Dict) -> Optional[str]:
        """Generate the output dictionary from the input

        Args:
            input (Dict): The input dictionary

        Returns:
            Dict: The output dictionary
        """
        output_dict = {}
        for attribute in self.config.attributes():
            attribute_name = attribute["name"]
            output_dict[attribute_name] = input.get(attribute_name, "")
        if not self._validate_output_dict(output_dict):
            logger.warn(
                f"Generated output dict: {output_dict} does not contain all the expected output attributes. Skipping example."
            )
            return None
        return json.dumps(output_dict)

    def _validate_output_dict(self, output_dict: Dict) -> bool:
        """Validate the output dictionary

        Args:
            output_dict (Dict): The output dictionary

        Returns:
            bool: True if the output dictionary is valid, False otherwise
        """
        for attribute in self.config.attributes():
            attribute_name = attribute.get("name")
            attribute_value = output_dict.get(attribute_name)
            if attribute_value is None or len(str(attribute_value)) == 0:
                return False
        return True

    def construct_prompt(
        self,
        input: Dict,
        examples: List,
        prompt_template_override: Optional[PromptTemplate] = None,
        output_guidelines_override: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        get_num_tokens: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        fmt_task_guidelines = self.task_guidelines

        attribute_json, output_schema = self._construct_attribute_json()
        output_guidelines = (
            self.output_guidelines
            if output_guidelines_override is None
            else output_guidelines_override
        )
        fmt_output_guidelines = output_guidelines.format(attribute_json=attribute_json)

        # prepare seed examples
        example_template = self.config.example_template()
        fmt_examples = []
        for eg in examples:
            if self.OUTPUT_DICT_KEY not in eg:
                output_dict = self._generate_output_dict(eg)
                if output_dict is None:
                    continue
                eg.update({self.OUTPUT_DICT_KEY: output_dict})
            fmt_examples.append(example_template.format_map(defaultdict(str, eg)))

        input[self.OUTPUT_DICT_KEY] = ""

        # check if all mapped keys in input are in the example template
        try:
            current_example = example_template.format(**input)
        except KeyError as e:
            try:
                current_example = example_template.format_map(defaultdict(str, input))
                logger.warn(
                    f'\n\nKey {e} in the "example_template" in the given config'
                    f"\n\n{example_template}\n\nis not present in the datsaset columns - {input.keys()}.\n\n"
                    f"Input - {input}\n\n"
                    "Continuing with the prompt as {current_example}"
                )
            except AttributeError as e:
                for key in input.keys():
                    if input[key] is not None:
                        example_template = example_template.replace(
                            f"{{{key}}}", input[key]
                        )
                current_example = example_template

        # populate the current example in the prompt
        prompt_template = (
            self.prompt_template
            if prompt_template_override is None
            else prompt_template_override
        )
        if self._is_few_shot_mode():
            curr_text_prompt = self.trim_prompt(
                prompt_template,
                task_guidelines=fmt_task_guidelines,
                output_guidelines=fmt_output_guidelines,
                seed_examples="\n\n".join(fmt_examples),
                current_example=current_example,
                max_input_tokens=max_input_tokens,
                get_num_tokens=get_num_tokens,
            )
        else:
            curr_text_prompt = self.trim_prompt(
                prompt_template,
                task_guidelines=fmt_task_guidelines,
                output_guidelines=fmt_output_guidelines,
                current_example=current_example,
                max_input_tokens=max_input_tokens,
                get_num_tokens=get_num_tokens,
            )
        if self.image_cols:
            prompt_dict = {"text": curr_text_prompt}
            for col in self.image_cols:
                if input.get(col) is not None and len(input.get(col)) > 0:
                    prompt_dict[col] = input[col]
                prompt_dict[col] = input[col]
            return json.dumps(prompt_dict), output_schema
        else:
            return curr_text_prompt, output_schema

    def get_explanation_prompt(self, example: Dict, include_label=True) -> str:
        pt = PromptTemplate(
            input_variables=get_format_variables(self.GENERATE_EXPLANATION_PROMPT),
            template=self.GENERATE_EXPLANATION_PROMPT,
        )

        fmt_task_guidelines = self.task_guidelines
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
            attribute=example[self.OUTPUT_DICT_KEY],
        )

    def get_generate_dataset_prompt(
        self, label: str, num_rows: int, guidelines: str = None
    ) -> str:
        raise NotImplementedError("Dataset generation not implemented for this task")

    def parse_llm_response(
        self,
        response: Union[Generation, ChatGeneration],
        curr_sample: Dict,
        prompt: str,
    ) -> LLMAnnotation:
        successfully_labeled = False
        error = None
        try:
            completion_text = response.text

            # Remove markdown formatting from the completion text
            completion_text = completion_text.lstrip("```json")
            completion_text = completion_text.rstrip("```")

            llm_label = {k: str(v) for k, v in json5.loads(completion_text).items()}
            successfully_labeled = True
        except Exception as e:
            logger.info(
                f"Error parsing LLM response: {response.text}, Error: {e}. Now searching for valid JSON in response"
            )
            try:
                json_start, json_end = response.text.find("{"), response.text.rfind("}")
                llm_label = {
                    k: str(v)
                    for k, v in json5.loads(
                        response.text[json_start : json_end + 1]
                    ).items()
                }
                successfully_labeled = True
            except Exception as e:
                logger.error(f"Error parsing LLM response: {response.text}, Error: {e}")
                llm_label = self.NULL_LABEL
                error = LabelingError(
                    error_type=ErrorType.INVALID_LLM_RESPONSE_ERROR,
                    error_message=str(e),
                )

        if successfully_labeled:
            for attribute in self.config.attributes():
                attr_options = attribute.get("options")
                if attr_options is not None and len(attr_options) > 0:
                    attr_label = str(llm_label.get(attribute["name"]))
                    if attr_label is not None and attr_label not in attr_options:
                        logger.warning(
                            f"Attribute {attr_label} from the LLM response {llm_label} is not in the labels list"
                        )
                        llm_label.pop(attribute["name"], None)

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
                        confidence_score=(
                            llm_label.confidence_score[attribute]
                            if llm_label.confidence_score
                            else 0
                        ),
                    )
                )

        eval_metrics = []
        macro_metrics = {}

        for attribute in llm_labels_dict.keys():
            for metric in self.metrics + additional_metrics:
                if attribute not in gt_labels or gt_labels[attribute] is None:
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
                    if m.name not in macro_metrics:
                        macro_metrics[m.name] = []
                    macro_metrics[m.name].append(m.value)

        for key in macro_metrics:
            eval_metrics.append(
                MetricResult(
                    name=f"Macro:{key}",
                    value=sum(macro_metrics[key]) / len(macro_metrics[key]),
                )
            )

        return eval_metrics
