import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from loguru import logger
from nervaluate import Evaluator
from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import TaskConfig
from autolabel.schema import LLMAnnotation, Metric, MetricResult
from autolabel.tasks import BaseTask


class NamedEntityRecognitionTask(BaseTask):
    JSON_OUTPUT_FORMAT_PROMPT = "You will return the answer in JSON format. Each key in the JSON dictionary is an entity category, and value is a list of entities of this category detected in the text."
    CSV_OUTPUT_FORMAT_PROMPT = "You will return the answer in CSV format, with two columns seperated by the % character. First column is the extracted entity and second column is the category. Rows in the CSV are separated by new line character."
    NULL_LABEL = {}

    task_prompt = "Your job is to extract named entities mentioned in text, and classify them into one of the following {num_labels} categories.\nCategories:\n{labels_list}\n "

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    def initialize_prompt_template(self) -> PromptTemplate:
        # provide context about the prediction task
        pt = PromptTemplate(
            input_variables=self.prompt_template_variables,
            template=self.prompt_template,
        )
        return pt.partial(
            prefix_prompt=self.prefix_prompt,
            output_prompt=self.output_prompt,
        )

    def construct_prompt(self, input: Dict, examples: List) -> str:
        # Create the task prompt based on the dataset config
        labels_list = self.dataset_config.get_labels_list()
        num_labels = len(labels_list)
        task_prompt = self.task_prompt.format(
            num_labels=num_labels, labels_list="\n".join(labels_list)
        )

        # populate seed examples in the prompt
        example_template = self.dataset_config.get_example_template()

        # populate seed examples in the prompt
        formatted_examples = []
        for eg in examples:
            fmt_example = example_template.format_map(defaultdict(str, eg))
            formatted_examples.append(fmt_example)

        if len(examples):
            seed_examples_prompt = self.seed_examples_prompt
        else:
            seed_examples_prompt = ""

        # populate the label column with empty string for current example
        label_column = self.dataset_config.get_label_column()
        if label_column:
            input[label_column] = ""

        # populate the explanation column with empty string for current example
        explanation_column = self.dataset_config.get_explanation_column()
        if explanation_column:
            input[explanation_column] = ""

        # populate the current example in the prompt
        current_example = example_template.format_map(defaultdict(str, input))

        return self.partial_prompt.format(
            seed_examples="\n\n".join(formatted_examples),
            current_example=current_example,
            seed_examples_prompt=seed_examples_prompt,
            task_prompt=task_prompt,
        )

    def get_explanation_prompt(self, example: Dict) -> str:
        raise NotImplementedError(
            "Explanation generation not implemented for this task"
        )

    def add_text_spans(self, raw_output: dict, input: str) -> list:
        processed_output = []
        for entity_type in raw_output:
            for curr_entity in raw_output[entity_type]:
                processed_output.append({"type": entity_type, "text": curr_entity})

        # create a frequency dict of each named entity in the input to determine text spans for repeated entities
        frequency_count = {label["text"]: 0 for label in processed_output}

        for label in processed_output:
            text = label["text"]
            matches = [i.start() for i in re.finditer(text, input)]
            count = frequency_count[text]
            # if count of the named entity is greater than the number of matches, default to last found match
            if count >= len(matches):
                count = -1

            # if no occurence of named entity in input, default text span to start: -1, end: -1
            if len(matches) == 0:
                label["start"] = -1
                label["end"] = -1
            else:
                label["start"] = matches[count]
                label["end"] = matches[count] + len(text)
            frequency_count[text] += 1
        return processed_output

    def jsonify_csv_output(self, response: str):
        split_response = response.split("\n")
        json_output = {i: [] for i in self.dataset_config.get_labels_list()}

        for row in split_response:
            parts = row.split("%")
            if len(parts) != 2 or parts[1] not in json_output.keys():
                logger.debug(f"Malformed LLM response: {row}")
                continue
            named_entity = parts[0]
            category = parts[1]
            json_output[category].append(named_entity)
        return json_output

    def parse_llm_response(
        self, response: Generation, curr_sample: Dict, prompt: str
    ) -> LLMAnnotation:
        output = {}
        successfully_labeled = "no"
        input_str = curr_sample["example"]
        try:
            completion_text = response.text
            if self.output_format == "csv":
                output = self.jsonify_csv_output(completion_text.strip())
            else:
                output = json.loads(completion_text.strip())

            raw_output = output or self.NULL_LABEL
            llm_label = self.add_text_spans(raw_output, input_str)

        except Exception as e:
            logger.info(f"Error parsing LLM response: {response.text}, Error: {e}")
            llm_label = self.NULL_LABEL

        successfully_labeled = "no" if llm_label == self.NULL_LABEL else "yes"

        # TODO: parse generation info correctly to fetch & transform logprobs -> score
        return LLMAnnotation(
            curr_sample=input_str,
            successfully_labeled=successfully_labeled,
            label=llm_label,
            generation_info=response.generation_info,
            raw_response=response.text,
            prompt=prompt,
        )

    def auroc_score_labels(
        self, gt_labels, llm_labels_with_conf
    ) -> Tuple[List[int], List[float]]:
        labels = []
        confidences = []
        for index, pred_entities in enumerate(llm_labels_with_conf):
            gt_entities = gt_labels[index]
            pred_conf = pred_entities[0]["conf"] if len(pred_entities) > 0 else 0
            for gt_entity in gt_entities:
                match_found = False
                pred_index = 0
                while not match_found and pred_index < len(pred_entities):
                    curr_match = True
                    for key in gt_entity:
                        if gt_entity[key] != pred_entities[pred_index][key]:
                            curr_match = False
                    if curr_match:
                        match_found = True
                    pred_index += 1
                labels.append(int(match_found))
                confidences.append(pred_conf)
        return labels, confidences

    def get_labels_predictions_with_threshold(self, gt_labels, llm_labels, threshold):
        answered_gt_labels, answered_llm_preds = [], []
        for index, l in enumerate(llm_labels):
            if l.successfully_labeled.lower() == "yes" and (
                l.confidence_score is None or l.confidence_score >= threshold
            ):
                answered_gt_labels.append(
                    [{**entity, "label": entity["type"]} for entity in gt_labels[index]]
                )
                answered_llm_preds.append(
                    [
                        {
                            **entity,
                            "label": entity["type"],
                            "conf": l.confidence_score,
                        }
                        for entity in l.label
                    ],
                )

        return answered_gt_labels, answered_llm_preds

    def run_metrics(
        self,
        answered_gt_labels,
        answered_llm_preds,
        entity_types_set,
    ) -> List[MetricResult]:
        eval_metrics = []
        evaluator = Evaluator(
            answered_gt_labels, answered_llm_preds, tags=entity_types_set
        )

        results, _ = evaluator.evaluate()
        # f1 score
        eval_metrics.append(
            MetricResult(
                metric_type=Metric.F1,
                name=f"f1",
                value=results["exact"]["f1"],
            )
        )

        # accuracy
        accuracy = results.get("strict").get("correct") / (
            results.get("strict").get("possible") + 1e-5
        )
        eval_metrics.append(
            MetricResult(
                metric_type=Metric.ACCURACY,
                name=f"accuracy",
                value=accuracy,
            )
        )

        eval_metrics.append(
            MetricResult(
                metric_type=Metric.SUPPORT,
                name=f"support",
                value=len(answered_gt_labels),
            )
        )
        return eval_metrics

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

        gt_labels = [
            self.add_text_spans(
                json.loads(gt_labels[index]), llm_labels[index].curr_sample
            )
            for index in range(len(gt_labels))
        ]

        eval_metrics_map = {
            Metric.F1: [],
            Metric.SUPPORT: [],
            Metric.THRESHOLD: [],
            Metric.ACCURACY: [],
            Metric.COMPLETION_RATE: [],
        }
        eval_metrics = []
        thresholds = [float("-inf")]

        if self.config.get_compute_confidence():
            all_gt_labels, all_llm_preds = self.get_labels_predictions_with_threshold(
                gt_labels, llm_labels, float("-inf")
            )
            labels, confidences = self.auroc_score_labels(all_gt_labels, all_llm_preds)
            value, meaningful_thresholds = ConfidenceCalculator.compute_auroc(
                labels, confidences
            )
            thresholds.extend(meaningful_thresholds)
            eval_metrics.append(
                MetricResult(
                    metric_type=Metric.AUROC,
                    name="auroc",
                    value=value,
                )
            )

        for index, threshold in enumerate(thresholds):
            (
                curr_gt_labels,
                curr_llm_labels,
            ) = self.get_labels_predictions_with_threshold(
                gt_labels, llm_labels, threshold
            )

            entity_types_set = list(
                set(
                    [
                        gt_entity.get("label")
                        for gt_label in curr_gt_labels
                        for gt_entity in gt_label
                    ]
                )
            )

            curr_threshold_metrics = self.run_metrics(
                curr_gt_labels,
                curr_llm_labels,
                entity_types_set,
            )

            for metric in curr_threshold_metrics:
                eval_metrics_map[metric.metric_type].append(
                    (metric.value, f"index={index}")
                )

            eval_metrics_map[Metric.COMPLETION_RATE].append(
                (len(curr_llm_labels) / float(len(gt_labels)), f"index={index}")
            )
            eval_metrics_map[Metric.THRESHOLD].append((threshold, f"index={index}"))
        eval_metrics.extend(
            [
                MetricResult(
                    metric_type=i,
                    name=i.value,
                    value=eval_metrics_map[i],
                )
                for i in eval_metrics_map.keys()
            ]
        )
        return eval_metrics
