import json
from typing import List

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation, Metric, MetricResult
from refuel_oracle.tasks import BaseTask
from sklearn.metrics import accuracy_score


class ClassificationTask(BaseTask):

    DEFAULT_TASK_PROMPT = "Your job is to correctly label the provided input example into one of the following {num_labels} categories.\nCategories:\n{labels_list}\n"
    DEFAULT_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": "can you answer this question. say YES or NO", "label": "the correct label"}'
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

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def _to_default_output_format(self, label: str) -> str:
        output = {"answered": "yes", "label": label}
        return json.dumps(output)

    def _parse_default_output_format(self, output: str) -> dict:
        return json.loads(output.strip())

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

        # output format instructions
        output_prompt = self.config.get(
            "output_format_prompt", self.DEFAULT_OUTPUT_FORMAT_PROMPT
        )

        pt = PromptTemplate(
            input_variables=self.PROMPT_TEMPLATE_VARIABLES,
            template=self.PROMPT_TEMPLATE,
        )
        return pt.partial(
            prefix_prompt=prefix_prompt,
            task_prompt=task_prompt,
            output_prompt=output_prompt,
        )

    def construct_prompt(self, input: str, examples: List) -> str:
        # populate seed examples in the prompt
        example_prompt = PromptTemplate(
            input_variables=self.EXAMPLE_PROMPT_VARIABLES,
            template=self.EXAMPLE_PROMPT_TEMPLATE,
        )
        formatted_examples = []
        for eg in examples:
            # if the user specified an explicit output format prompt, don't preprocess the seed examples
            # we might change this to allow a UDF to do this processing
            if "output_format_prompt" in self.config.keys():
                expected_output = eg["output"]
            else:
                expected_output = self._to_default_output_format(eg["output"])
            formatted_examples.append(
                example_prompt.format(example=eg["example"], output=expected_output)
            )

        # populate the current example in the prompt
        current_example = example_prompt.format(example=input, output="")

        return self.prompt_template.format(
            seed_examples="\n".join(formatted_examples), current_example=current_example
        )

    def parse_llm_response(self, prompt: str, response: Generation) -> LLMAnnotation:
        # if the user specified an explicit output format prompt, don't postprocess the result
        # we will change this to allow a UDF to do this postprocessing
        if "output_format_prompt" in self.config.keys():
            successfully_labeled = True
            llm_label = response.text
        else:
            try:
                output = self._parse_default_output_format(response.text)
                successfully_labeled = output.get("answered", False)
                llm_label = output.get("label", "")
            except Exception as e:
                logger.error(
                    f"Error parsing LLM response: {response.text}. Encountered exception: {e}"
                )

        # TODO: parse generation info correctly to fetch & transform logprobs -> score
        return LLMAnnotation(
            prompt=prompt,
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
            MetricResult(metric_type=Metric.SUPPORT, name="support", value=support)
        )

        # completion rate
        num_labeled = sum([l.successfully_labeled for l in llm_labels])
        fraction_completed = round(num_labeled * 1.0 / support, 2)
        eval_metrics.append(
            MetricResult(
                metric_type=Metric.COMPLETION_RATE,
                name="completion_rate",
                value=fraction_completed,
            )
        )

        # accuracy
        pred_labels = [l.label for l in llm_labels]
        accuracy = accuracy_score(gt_labels, pred_labels)
        eval_metrics.append(
            MetricResult(metric_type=Metric.ACCURACY, name="accuracy", value=accuracy)
        )

        return eval_metrics
