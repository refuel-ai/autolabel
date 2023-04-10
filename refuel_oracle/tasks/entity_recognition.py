import json
from typing import List

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation, Metric, MetricResult
from refuel_oracle.tasks import BaseTask


class EntityRecognitionTask(BaseTask):

    DEFAULT_TASK_PROMPT = "Your job is to extract location, person or organization named entities present in the provided input text, and the associated text spans.\n"
    DEFAULT_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": can you answer this question. say YES or NO, "entities": a list of extracted entities from text}.'
    PROMPT_TEMPLATE = "{prefix_prompt}\n{task_prompt}\n\n{output_prompt}\n\nSome examples with their output answers are provided below:\n{seed_examples}\n{current_example}"
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
        task_prompt = self.config.get("task_prompt", self.DEFAULT_TASK_PROMPT)

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
                example_prompt.format(
                    example=eg["example"], output=expected_output)
            )

        # populate the current example in the prompt
        current_example = example_prompt.format(example=input, output="")

        return self.prompt_template.format(
            seed_examples="\n".join(formatted_examples), current_example=current_example
        )

    def parse_llm_response(self, response: Generation) -> LLMAnnotation:
        output = {}
        try:
            completion_text = response.text
            output = json.loads(completion_text.strip())
        except Exception as e:
            logger.info(
                f"Error parsing LLM response: {response.text}"
            )

        successfully_labeled = output.get("answered", "no")
        if successfully_labeled.lower() == 'yes':
            llm_label = output.get("entities") or self.NULL_LABEL
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

        # support
        support = len(gt_labels)
        eval_metrics.append(
            MetricResult(metric_type=Metric.SUPPORT,
                         name="support", value=support)
        )

        # completion rate
        num_labeled = sum([l.successfully_labeled.lower()
                          == "yes" for l in llm_labels])
        fraction_completed = round(num_labeled * 1.0 / support, 2)
        eval_metrics.append(
            MetricResult(
                metric_type=Metric.COMPLETION_RATE,
                name="completion_rate",
                value=fraction_completed,
            )
        )

        # error examples
        # TODO, need a way to access input dataset in order to display them here

        return eval_metrics
