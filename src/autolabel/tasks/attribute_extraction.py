from typing import List, Dict
from collections import defaultdict
import logging
import json

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
)
from autolabel.utils import get_format_variables
from autolabel.metrics import (
    AccuracyMetric,
    AUROCMetric,
    SupportMetric,
    CompletionRateMetric,
    BaseMetric,
    F1Metric,
)

logger = logging.getLogger(__name__)


class AttributeExtractionTask(BaseTask):
    NULL_LABEL = {}
    DEFAULT_TASK_GUIDELINES = "You are an expert at extracting attributes from text. Given a piece of text, extract the {num_output_attributes} required attributes. The attributes you need to extract are listed below:\n\n{output_attributes}"
    DEFAULT_OUTPUT_GUIDELINES = 'You will return the answer as a json in the following format:\n{\n\t"attribute_1": "label for attribute 1",\n\t"attribute_2": "label for attribute 2",\n\t...\n\t"attribute_n": "label for attribute n"\n}'

    GENERATE_EXPLANATION_PROMPT = "Given the text and the extracted attributes, explain why you chose these specific attributes. The text is:\n{text}\nThe extracted attributes are:\n{attributes}"

    OUTPUT_DICT_KEY = "output_dict"

    def __init__(self, config: AutolabelConfig) -> None:
        super().__init__(config)
        self.metrics = [
            AccuracyMetric(),
            SupportMetric(),
            CompletionRateMetric(),
            F1Metric(
                type=F1Type.ATTRIBUTE_EXTRACTION,
            ),
        ]

        if self.config.confidence():
            self.metrics.append(AUROCMetric())

    def _construct_output_attributes(self) -> str:
        """Construct the output attributes from the config file

        This assumes that each dict has a "name" and will include the "description" or "options" if it exists.

        Args:
            output_attributes (List[Dict]): A list of dictionaries containing the output attributes.

        Returns:
            str: A string containing the output attributes.
        """
        output_list = []
        for attribute in self.config.output_attributes():
            attribute_str = f'Attribute Name: {attribute["name"]}\n'
            if "type" in attribute:
                attribute_str += f'Type: {attribute["type"]}\n'
            if "description" in attribute:
                attribute_str += f'Description: {attribute["description"]}\n'
            if "options" in attribute:
                options = "\n".join(attribute["options"])
                attribute_str += f"Options:\n{options}\n"
            output_list.append(attribute_str)
        return "\n".join(output_list)

    def _generate_output_dict(self, input: Dict) -> Dict:
        """Generate the output dictionary from the input

        Args:
            input (Dict): The input dictionary

        Returns:
            Dict: The output dictionary
        """
        output_dict = {}
        for attribute in self.config.output_attributes():
            attribute_name = attribute["name"]
            output_dict[attribute_name] = input[attribute_name]
        return json.dumps(output_dict)

    def construct_prompt(self, input: Dict, examples: List) -> str:
        num_output_attributes = len(self.config.output_attributes())
        output_attributes = self._construct_output_attributes()
        fmt_task_guidelines = self.task_guidelines.format(
            num_output_attributes=num_output_attributes,
            output_attributes=output_attributes,
        )

        output_attributes_dict = {
            attribute["name"] + "_" + key: value
            for attribute in self.config.output_attributes()
            for key, value in attribute.items()
        }
        fmt_output_guidelines = self.output_guidelines.format_map(
            defaultdict(str, output_attributes_dict)
        )

        # prepare seed examples
        example_template = self.config.example_template()
        fmt_examples = []
        for eg in examples:
            output_dict = self._generate_output_dict(eg)
            eg.update({self.OUTPUT_DICT_KEY: output_dict})
            fmt_examples.append(example_template.format_map(defaultdict(str, eg)))

        if self.config.label_column():
            input[self.config.label_column()] = ""

        if self.config.explanation_column():
            input[self.config.explanation_column()] = ""

        input[self.OUTPUT_DICT_KEY] = ""

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
        pt = PromptTemplate(
            input_variables=get_format_variables(self.GENERATE_EXPLANATION_PROMPT),
            template=self.GENERATE_EXPLANATION_PROMPT,
        )

        # prepare task guideline
        output_attributes = self.config.output_attributes()
        fmt_task_guidelines = self.task_guidelines.format_map(
            defaultdict(str, output_attributes)
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

    def parse_llm_response(
        self, response: Generation, curr_sample: Dict, prompt: str
    ) -> LLMAnnotation:
        successfully_labeled = False
        error = None
        text_column = self.config.text_column()
        input_str = curr_sample[text_column]
        try:
            completion_text = response.text
            llm_label = json.loads(completion_text)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {response.text}, Error: {e}")
            llm_label = self.NULL_LABEL
            error = LabelingError(error_type=ErrorType.PARSING_ERROR, error_msg=str(e))

        successfully_labeled = False if llm_label == self.NULL_LABEL else True

        return LLMAnnotation(
            curr_sample=input_str,
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
        # Conver gt_labels to list of dictionaries, llm_labels is already a list of dictionaries
        gt_labels = [json.loads(gt_label) for gt_label in gt_labels]
        # Convert llm_labels and gt_labels to dictionary of lists
        llm_labels_dict = defaultdict(list)
        gt_labels_dict = defaultdict(list)

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

        for gt_label in gt_labels:
            for attribute, value in gt_label.items():
                gt_labels_dict[attribute].append(value)

        eval_metrics = []

        for metric in self.metrics + additional_metrics:
            if isinstance(metric, AccuracyMetric):
                for output_attribute in self.config.output_attributes():
                    metrics = metric.compute(
                        llm_labels_dict[output_attribute["name"]],
                        gt_labels_dict[output_attribute["name"]],
                    )
                    for m in metrics:
                        eval_metrics.append(
                            MetricResult(
                                name=f'{m.name} ({output_attribute["name"]})',
                                value=m.value,
                            )
                        )
            else:
                eval_metrics.extend(metric.compute(llm_labels, gt_labels))

        return eval_metrics
