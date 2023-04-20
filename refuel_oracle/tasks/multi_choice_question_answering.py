import json
from typing import List, Dict
import ast

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation, Metric, MetricResult
from refuel_oracle.tasks import BaseTask
from refuel_oracle.utils import extract_valid_json_substring
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class MultiChoiceQATask(BaseTask):
    DEFAULT_TASK_PROMPT = "Your job is to answer the following questions using the options provided for each question. Choose the best answer for the question.\n"
    JSON_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": "can you answer this question. say yes or no", "label": "the correct label"}\n'
    CSV_OUTPUT_FORMAT_PROMPT = 'You will return the answer in CSV format with two elements: "can you answer this question. say Yes or No", "the correct label"\n'
    PROMPT_TEMPLATE = "{prefix_prompt}\n{task_prompt}\n\n{output_prompt}\n\nSome examples with their output answers are provided below:\n{seed_examples}\n Now I want you to label the following example in the same way: {current_example}"
    PROMPT_TEMPLATE_VARIABLES = [
        "prefix_prompt",
        "task_prompt",
        "output_prompt",
        "seed_examples",
        "current_example",
    ]
    EXAMPLE_PROMPT_TEMPLATE = (
        "{context}\nQuestion: {question}\nOptions:\n{options}\nAnswer:{answer}\n"
    )
    EXAMPLE_PROMPT_VARIABLES = ["context", "question", "options", "answer"]
    NULL_LABEL_TOKEN = "NO_LABEL"

    def __init__(self, config: Config) -> None:
        self.output_format = config.get("output_format", "json")
        super().__init__(config)

    def _to_output_format(self, label: str) -> str:
        if self.output_format == "json":
            output = {"answered": "yes", "label": label}
            return json.dumps(output)
        elif self.output_format == "csv":
            return f"yes, {label}"

    def initialize_prompt_template(self) -> PromptTemplate:
        # provide context about the problem domain
        prefix_prompt = self.config.get("prefix_prompt", "")

        # provide context about the prediction task
        task_prompt = self.config.get("task_prompt")
        if not task_prompt:
            task_prompt = self.DEFAULT_TASK_PROMPT

        pt = PromptTemplate(
            input_variables=self.PROMPT_TEMPLATE_VARIABLES,
            template=self.PROMPT_TEMPLATE,
        )

        if self.output_format == "csv":
            output_prompt = self.CSV_OUTPUT_FORMAT_PROMPT
        else:
            output_prompt = self.JSON_OUTPUT_FORMAT_PROMPT
        return pt.partial(
            prefix_prompt=prefix_prompt,
            task_prompt=task_prompt,
            output_prompt=output_prompt,
        )

    def get_context(self, input: Dict) -> str:
        context = input.get("context", "")
        if context:
            context = f"Context: {context}"
        return context

    def construct_prompt(self, input: Dict, examples: List[Dict]) -> str:
        # populate seed examples in the prompt
        example_prompt = PromptTemplate(
            input_variables=self.EXAMPLE_PROMPT_VARIABLES,
            template=self.EXAMPLE_PROMPT_TEMPLATE,
        )
        formatted_examples = []
        for eg in examples:
            expected_output = self._to_output_format(eg["answer"])
            formatted_examples.append(
                example_prompt.format(
                    context=self.get_context(eg),
                    question=eg["question"],
                    options="\n".join(eg["options"]),
                    answer=expected_output,
                )
            )

        # populate the current example in the prompt
        current_example = example_prompt.format(
            context=self.get_context(input),
            question=input["question"],
            options="\n".join(
                ast.literal_eval(input["options"])
            ),  # Arrays sent as a list of strings in the csv right now
            answer="",  # we don't know the answer yet
        )

        return self.prompt_template.format(
            seed_examples="\n".join(formatted_examples), current_example=current_example
        )

    # TODO: Should parsing of responses be moved to a generic class?
    def parse_llm_response(self, response: Generation) -> LLMAnnotation:
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
            llm_label = output.get("label") or self.NULL_LABEL_TOKEN
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
        completion_text = response.text.strip().split(",", 1)
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
        num_labeled = sum([l.successfully_labeled.lower() == "yes" for l in llm_labels])
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
        filtered_gt_labels = []
        filtered_pred_labels = []
        for ind, label in enumerate(pred_labels):
            if label != self.NULL_LABEL_TOKEN:
                filtered_gt_labels.append(gt_labels[ind])
                filtered_pred_labels.append(pred_labels[ind])
        if len(filtered_gt_labels) == 0:
            logger.error("No labels were successfully labeled by the LLM")
            accuracy = 0
        else:
            accuracy = accuracy_score(filtered_gt_labels, filtered_pred_labels)
        eval_metrics.append(
            MetricResult(metric_type=Metric.ACCURACY, name="accuracy", value=accuracy)
        )

        # error examples
        # TODO, need a way to access input dataset in order to display them here

        return eval_metrics
