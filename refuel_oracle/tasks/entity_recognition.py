import json
import re
from typing import List

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation, Metric, MetricResult
from refuel_oracle.tasks import BaseTask


class EntityRecognitionTask(BaseTask):

    DEFAULT_TASK_PROMPT = "Your job is to extract named entities mentioned in text, and classify them into one of the following {num_labels} categories.\nCategories:\n{labels_list}\n "
    DEFAULT_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": can you answer this question. say YES or NO, "entities": a JSON list of extracted entities from text, with the text spans}.'
    PROMPT_TEMPLATE = "{prefix_prompt}\n{task_prompt}\n{output_prompt}\n\nSome examples with their output answers are provided below:\n{seed_examples}\nBegin:{current_example}"
    PROMPT_TEMPLATE_VARIABLES = [
        "prefix_prompt",
        "task_prompt",
        "output_prompt",
        "seed_examples",
        "current_example",
    ]
    EXAMPLE_PROMPT_TEMPLATE = "Example: {example}\nOutput: {output}\n"
    EXAMPLE_PROMPT_VARIABLES = ["example", "output"]
    NULL_LABEL = []

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def _to_default_output_format(self, entities: List) -> str:
        output = {"answered": "yes", "entities": entities}
        return json.dumps(output)

    def initialize_prompt_template(self) -> PromptTemplate:
        # provide context about the problem domain
        prefix_prompt = self.config.get("prefix_prompt", "")

        # provide context about the prediction task
        labels_list = self.config.get("labels_list", [])
        num_labels = len(labels_list)
        task_prompt = self.config.get("task_prompt")
        if not task_prompt:
            task_prompt = self.DEFAULT_TASK_PROMPT.format(
                num_labels=num_labels, labels_list="\n".join(labels_list)
            )

        pt = PromptTemplate(
            input_variables=self.PROMPT_TEMPLATE_VARIABLES,
            template=self.PROMPT_TEMPLATE,
        )
        return pt.partial(
            prefix_prompt=prefix_prompt,
            task_prompt=task_prompt,
            output_prompt=self.DEFAULT_OUTPUT_FORMAT_PROMPT,
        )

    def construct_prompt(self, input: str, examples: List) -> str:
        # populate seed examples in the prompt
        example_prompt = PromptTemplate(
            input_variables=self.EXAMPLE_PROMPT_VARIABLES,
            template=self.EXAMPLE_PROMPT_TEMPLATE,
        )
        formatted_examples = []
        for eg in examples:
            expected_output = self._to_default_output_format(eg["output"])
            formatted_examples.append(
                example_prompt.format(example=eg["example"], output=expected_output)
            )

        # populate the current example in the prompt
        current_example = example_prompt.format(example=input, output="")

        return self.prompt_template.format(
            seed_examples="\n".join(formatted_examples), current_example=current_example
        )

    def parse_llm_response(self, response: Generation, input: str) -> LLMAnnotation:
        output = {}
        try:
            completion_text = response.text
            output = json.loads(completion_text.strip())
        except Exception as e:
            logger.info(f"Error parsing LLM response: {response.text}")

        successfully_labeled = output.get("answered", "no")
        if successfully_labeled.lower() == "yes":
            llm_label = output.get("entities") or self.NULL_LABEL

            for label in llm_label:
                text = label["text"]
                matches = [i.start() for i in re.finditer(text, input)]
                if len(matches) > 0:
                    closest_start_idx = matches[0]
                    closest_end_idx = closest_start_idx + len(text)
                    closest_interval_dist = abs(
                        label["start"] - closest_start_idx
                    ) + abs(label["end"] - closest_end_idx)
                    for start_idx in matches:
                        end_idx = start_idx + len(text)
                        interval_dist = abs(label["start"] - start_idx) + abs(
                            label["end"] - end_idx
                        )
                        if interval_dist < closest_interval_dist:
                            closest_start_idx = start_idx
                            closest_end_idx = end_idx
                            closest_interval_dist = interval_dist
                    label["start"] = closest_start_idx
                    label["end"] = closest_end_idx
                else:
                    label["start"] = matches[0]
                    label["end"] = matches[0] + len(label.get("text", ""))
        else:
            llm_label = self.NULL_LABEL

        # TODO: parse generation info correctly to fetch & transform logprobs -> score
        return LLMAnnotation(
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
        )

    def eval(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        """Evaluate the LLM generated labels by comparing them against ground truth

        Args:
            llm_labels (List[LLMAnnotation]): _description_
            gt_labels (List[str]): _description_

        Returns:
            List[MetricResult]: list of metrics and corresponding values
        """
        eval_metrics = []
        logger.info(
            f"llm_labels labeled: {[l.successfully_labeled for l in llm_labels]}"
        )
        logger.info(f"llm_labels: {[l.label for l in llm_labels]}")
        logger.info(f"gt_labels: {gt_labels}")

        # support
        support = len(gt_labels)
        eval_metrics.append(
            MetricResult(metric_type=Metric.SUPPORT, name="support", value=support)
        )

        # completion rate
        num_labeled = sum([l.successfully_labeled.lower() == "yes" for l in llm_labels])
        fraction_completed = round(num_labeled * 1.0 / support, 2)
        eval_metrics.append(
            MetricResult(
                metric_type=Metric.COMPLETION_RATE,
                name="completion_rate",
                value=fraction_completed,
            )
        )

        # TODO: NER label quality metrics

        return eval_metrics
