import json
from typing import List, Dict
import ast

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from refuel_oracle.task_config import TaskConfig
from refuel_oracle.schema import LLMAnnotation, Metric, MetricResult
from refuel_oracle.tasks import BaseTask
from refuel_oracle.utils import extract_valid_json_substring
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import transformers
from refuel_oracle.tasks.utils import normalize_text, compute_f1


class MultiChoiceQATask(BaseTask):
    JSON_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": "can you answer this question. say yes or no", "label": "the correct label"}\n'
    CSV_OUTPUT_FORMAT_PROMPT = 'You will return the answer in CSV format with two elements: "can you answer this question. say Yes or No", "the correct label"\n'
    NO_OUTPUT_FORMAT_PROMPT = 'You will return the answer in plain text format with one element: "the correct label"\n'

    task_prompt = "Your job is to answer the following questions using the options provided for each question. Choose the best answer for the question.\n"
    example_prompt_template = (
        "{context}\nQuestion: {question}\n{options}\nAnswer:{answer}\n"
    )
    example_prompt_variables = ["context", "question", "options", "answer"]
    NULL_LABEL_TOKEN = "NO_LABEL"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    def initialize_prompt_template(self) -> PromptTemplate:
        pt = PromptTemplate(
            input_variables=self.prompt_template_variables,
            template=self.prompt_template,
        )

        return pt.partial(
            prefix_prompt=self.prefix_prompt,
            task_prompt=self.task_prompt,
            output_prompt=self.output_prompt,
        )

    def get_context(self, input: Dict) -> str:
        context = input.get("context", "")
        if context:
            context = f"Context: {context}"
        return context

    def get_options(self, input: Dict) -> str:
        options = input.get("options", "")
        # if options is empty, return empty string
        if not options:
            return ""

        if isinstance(options, str):
            options = "\n".join(ast.literal_eval(input["options"]))
            options = f"Options:\n{options}"
        elif isinstance(options, list):
            options = "\n".join(options)
            options = f"Options:\n{options}"
        return options

    def construct_prompt(self, input: Dict, examples: List[Dict]) -> str:
        # populate seed examples in the prompt
        example_prompt = PromptTemplate(
            input_variables=self.example_prompt_variables,
            template=self.example_prompt_template,
        )

        formatted_examples = []
        for eg in examples:
            expected_output = self._to_output_format(eg["answer"])
            formatted_examples.append(
                example_prompt.format(
                    context=self.get_context(eg),
                    question=eg["question"],
                    options=self.get_options(eg),
                    answer=expected_output,
                )
            )

        # populate the current example in the prompt
        current_example = example_prompt.format(
            context=self.get_context(input),
            question=input["question"],
            options=self.get_options(input),
            answer="",  # we don't know the answer yet
        )

        if len(examples):
            seed_examples_prompt = self.seed_examples_prompt
        else:
            seed_examples_prompt = ""

        return self.partial_prompt.format(
            seed_examples_prompt=seed_examples_prompt,
            seed_examples="\n".join(formatted_examples),
            current_example=current_example,
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
                filtered_gt_labels.append(normalize_text(gt_labels[ind].lower()))
                filtered_pred_labels.append(normalize_text(pred_labels[ind].lower()))
        if len(filtered_gt_labels) == 0:
            logger.error("No labels were successfully labeled by the LLM")
            accuracy = 0
        else:
            accuracy = accuracy_score(filtered_gt_labels, filtered_pred_labels)
        eval_metrics.append(
            MetricResult(metric_type=Metric.ACCURACY, name="accuracy", value=accuracy)
        )

        f1 = 0
        cnt_f1 = 0
        for ind, label in enumerate(pred_labels):
            if label != self.NULL_LABEL_TOKEN:
                f1 += compute_f1(pred_labels[ind], gt_labels[ind])
                cnt_f1 += 1
        eval_metrics.append(
            MetricResult(metric_type=Metric.F1, name="f1", value=f1 / cnt_f1)
        )

        # error examples
        # TODO, need a way to access input dataset in order to display them here

        return eval_metrics
