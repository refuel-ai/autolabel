import json
from typing import List

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation, MetricResult
from refuel_oracle.tasks import BaseTask
from refuel_oracle.utils import extract_valid_json_substring


class CompletionTask(BaseTask):
    JSON_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": "can you answer this question. say yes or no", "answer": "the correct answer"}'
    CSV_OUTPUT_FORMAT_PROMPT = 'You will return the answer in CSV format with two elements: "can you answer this question. say Yes or No", "the correct answer"'
    PROMPT_TEMPLATE = "{prefix_prompt}\n\n{output_prompt}\n\nSome examples with their output answers are provided below:\n{seed_examples}\n\nNow I want you to answer the following questions in the same way:\n{current_example}"
    PROMPT_TEMPLATE_VARIABLES = [
        "prefix_prompt",
        "output_prompt",
        "seed_examples",
        "current_example",
    ]
    EXAMPLE_PROMPT_TEMPLATE = "Example: {example}\nOutput: {output}\n"
    EXAMPLE_PROMPT_VARIABLES = ["example", "output"]
    NULL_LABEL_TOKEN = "NO_ANSWER"

    def __init__(self, config: Config) -> None:
        self.output_format = config.get("output_format", "json")
        super().__init__(config)

    def _to_output_format(self, answer: str) -> str:
        if self.output_format == "json":
            output = {"answered": "yes", "answer": answer}
            return json.dumps(output)
        elif self.output_format == "csv":
            return f"yes, {answer}"

    def initialize_prompt_template(self) -> PromptTemplate:
        # provide context about the problem domain
        prefix_prompt = self.config.get("prefix_prompt", "")

        pt = PromptTemplate(
            input_variables=self.PROMPT_TEMPLATE_VARIABLES,
            template=self.PROMPT_TEMPLATE,
        )

        if self.output_format == "csv":
            default_output_prompt = self.CSV_OUTPUT_FORMAT_PROMPT
        else:
            default_output_prompt = self.JSON_OUTPUT_FORMAT_PROMPT
        output_prompt = self.config.get("output_prompt", default_output_prompt)
        return pt.partial(
            prefix_prompt=prefix_prompt,
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
            expected_output = self._to_output_format(eg["output"])
            formatted_examples.append(
                example_prompt.format(example=eg["example"], output=expected_output)
            )

        # populate the current example in the prompt
        current_example = example_prompt.format(example=input, output="")

        return self.prompt_template.format(
            seed_examples="\n".join(formatted_examples), current_example=current_example
        )

    def parse_llm_response(self, response: Generation, input: str) -> LLMAnnotation:
        if self.output_format == "json":
            return self.parse_json_llm_response(response)
        elif self.output_format == "csv":
            return self.parse_csv_llm_response(response)

    def parse_json_llm_response(self, response: Generation) -> LLMAnnotation:
        output = {}
        try:
            completion_text = extract_valid_json_substring(response.text)
            output = json.loads(completion_text.strip())
        except Exception as e:
            logger.info(f"Error parsing LLM response: {response.text}")

        successfully_labeled = output.get("answered", "no")
        if successfully_labeled.lower() == "yes":
            llm_label = output.get("answer") or self.NULL_LABEL_TOKEN
            llm_label = str(llm_label)
        else:
            llm_label = self.NULL_LABEL_TOKEN

        # TODO: parse generation info correctly to fetch & transform logprobs -> score
        return LLMAnnotation(
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
        )

    def parse_csv_llm_response(self, response: Generation) -> LLMAnnotation:
        completion_text = response.text.strip().split(",")
        if len(completion_text) != 2:
            successfully_labeled = "no"
            llm_label = self.NULL_LABEL_TOKEN
            logger.error(f"Error parsing LLM response: {response.text}")
            return LLMAnnotation(
                successfully_labeled=successfully_labeled,
                label=llm_label,
                generation_info=response.generation_info,
            )

        successfully_labeled = completion_text[0].strip().lower()
        if successfully_labeled == "yes":
            llm_label = completion_text[1].strip()
        else:
            llm_label = self.NULL_LABEL_TOKEN

        # TODO: parse generation info correctly to fetch & transform logprobs -> score
        return LLMAnnotation(
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
        )

    def eval(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        return []
