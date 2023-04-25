import json
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from refuel_oracle.confidence import ConfidenceCalculator
from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation, Metric, MetricResult
from refuel_oracle.tasks import BaseTask
from refuel_oracle.utils import extract_valid_json_substring
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


class ClassificationTask(BaseTask):
    DEFAULT_TASK_PROMPT = "Your job is to correctly label the provided input example into one of the following {num_labels} categories.\nCategories:\n{labels_list}\n"
    JSON_OUTPUT_FORMAT_PROMPT = 'You will return the answer in JSON format with two keys: {"answered": "can you answer this question. say yes or no", "label": "the correct label"}'
    CSV_OUTPUT_FORMAT_PROMPT = 'You will return the answer in CSV format with two elements: "can you answer this question. say Yes or No", "the correct label"'
    SEED_EXAMPLES_PROMPT = "Some examples with their output answers are provided below:"
    PROMPT_TEMPLATE = "{prefix_prompt}\n{task_prompt}\n\n{output_prompt}\n\n{seed_examples_prompt}\n{seed_examples}\nNow I want you to label the following example: {current_example}"
    PROMPT_TEMPLATE_VARIABLES = [
        "prefix_prompt",
        "task_prompt",
        "output_prompt",
        "seed_examples_prompt",
        "seed_examples",
        "current_example",
    ]
    EXAMPLE_PROMPT_TEMPLATE = "Example: {example}\nOutput: {output}\n"
    EXAMPLE_PROMPT_VARIABLES = ["example", "output"]
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

        if self.output_format == "csv":
            default_output_prompt = self.CSV_OUTPUT_FORMAT_PROMPT
        else:
            default_output_prompt = self.JSON_OUTPUT_FORMAT_PROMPT
        output_prompt = self.config.get("output_prompt", default_output_prompt)
        return pt.partial(
            prefix_prompt=prefix_prompt,
            task_prompt=task_prompt,
            output_prompt=output_prompt,
        )

    def construct_prompt(self, input: Dict, examples: List) -> str:
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

        current_input = self.get_single_input(input)

        # populate the current example in the prompt
        current_example = example_prompt.format(example=current_input, output="")

        if len(examples):
            seed_examples_prompt = self.SEED_EXAMPLES_PROMPT
        else:
            seed_examples_prompt = ""

        return self.prompt_template.format(
            seed_examples="\n".join(formatted_examples),
            current_example=current_example,
            seed_examples_prompt=seed_examples_prompt,
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
            logger.info(f"Error parsing LLM response: {response.text}. {e}")

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
            raw_text=response.text,
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
            raw_text=response.text,
        )

    def auroc_score_labels(
        self, gt_labels, llm_labels
    ) -> Tuple[List[int], List[float]]:
        labels = []
        confidences = []
        for index, llm_label in enumerate(llm_labels):
            labels.append(llm_label.label == gt_labels[index])
            confidences.append(llm_label.confidence_score)

        ConfidenceCalculator.plot_data_distribution(labels, confidences)
        return labels, confidences

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

        if llm_labels[0].confidence_score is not None:
            # Assuming that if the first element has a confidence score
            # then all following elements have a confidence score too
            labels, confidences = self.auroc_score_labels(gt_labels, llm_labels)
            eval_metrics.append(
                MetricResult(
                    metric_type=Metric.AUROC,
                    name="auroc",
                    value=ConfidenceCalculator.compute_auroc(labels, confidences),
                )
            )

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

        # confusion matrix
        labels_list = self.config.get("labels_list", None)
        labels_list.append(self.NULL_LABEL_TOKEN)
        confusion = confusion_matrix(gt_labels, pred_labels, labels=labels_list)
        eval_metrics.append(
            MetricResult(
                metric_type=Metric.CONFUSION_MATRIX,
                name="confusion_matrix",
                value={"labels": labels_list, "value": confusion},
            )
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=labels_list
        )
        disp.plot(cmap="Purples")
        plt.xticks(rotation=90)
        plt.show(block=False)
        input("Displaying confusion matrix. Press enter to continue..")

        # error examples
        # TODO, need a way to access input dataset in order to display them here

        return eval_metrics
