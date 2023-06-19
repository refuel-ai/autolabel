import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from copy import deepcopy

from langchain.schema import Generation
import logging
from nervaluate import Evaluator
from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import AutolabelConfig
from autolabel.schema import LLMAnnotation, Metric, MetricResult
from autolabel.tasks import BaseTask

logger = logging.getLogger(__name__)


class NamedEntityRecognitionTask(BaseTask):
    DEFAULT_OUTPUT_GUIDELINES = "You will return the answer in CSV format, with two columns seperated by the % character. First column is the extracted entity and second column is the category. Rows in the CSV are separated by new line character."
    DEFAULT_TASK_GUIDELINES = "Your job is to extract named entities mentioned in text, and classify them into one of the following {num_labels} categories.\nCategories:\n{labels}\n "
    NULL_LABEL = {}

    def __init__(self, config: AutolabelConfig) -> None:
        super().__init__(config)

    def _json_to_llm_format(self, input_label: str) -> str:
        # `label` format: {"entity type": [list of entities of this type]}
        try:
            labels = json.loads(input_label)
            rows = []
            for entity_type, detected_entites in labels.items():
                for e in detected_entites:
                    row = "%".join([e, entity_type])
                    rows.append(row)
            llm_formatted_label = "\n".join(rows)
            return llm_formatted_label
        except json.JSONDecodeError as e:
            logger.error(
                f"Could not parse label: {input_label}. Few-shot examples might be formatted incorrectly"
            )
            return input_label

    def _llm_to_json_format(self, response: str):
        split_response = response.split("\n")
        json_output = {i: [] for i in self.config.labels_list()}

        for row in split_response:
            parts = row.split("%")
            if len(parts) != 2 or parts[1] not in json_output.keys():
                logger.debug(f"Malformed LLM response: {row}")
                continue
            named_entity = parts[0]
            category = parts[1]
            json_output[category].append(named_entity)
        return json_output

    def construct_prompt(self, input: Dict, examples: List) -> str:
        # prepare task guideline
        labels_list = self.config.labels_list()
        num_labels = len(labels_list)
        fmt_task_guidelines = self.task_guidelines.format(
            num_labels=num_labels, labels="\n".join(labels_list)
        )

        # prepare seed examples
        label_column = self.config.label_column()
        example_template = self.config.example_template()
        fmt_examples = []
        for eg in examples:
            eg_copy = deepcopy(eg)
            if label_column:
                eg_copy[label_column] = self._json_to_llm_format(eg_copy[label_column])
            fmt_examples.append(example_template.format_map(defaultdict(str, eg_copy)))

        # populate the current example in the prompt
        if label_column:
            input[label_column] = ""

        # populate the explanation column with empty string for current example
        explanation_column = self.config.explanation_column()
        if explanation_column:
            input[explanation_column] = ""

        # populate the current example in the prompt
        current_example = example_template.format_map(defaultdict(str, input))

        if self._is_few_shot_mode():
            return self.prompt_template.format(
                task_guidelines=fmt_task_guidelines,
                output_guidelines=self.output_guidelines,
                seed_examples="\n\n".join(fmt_examples),
                current_example=current_example,
            )
        else:
            return self.prompt_template.format(
                task_guidelines=fmt_task_guidelines,
                output_guidelines=self.output_guidelines,
                current_example=current_example,
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

    def parse_llm_response(
        self, response: Generation, curr_sample: Dict, prompt: str
    ) -> LLMAnnotation:
        output = {}
        successfully_labeled = False
        text_column = self.config.text_column()
        input_str = curr_sample[text_column]
        try:
            completion_text = response.text
            output = self._llm_to_json_format(completion_text.strip())
            llm_label = self.add_text_spans(output, input_str)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {response.text}, Error: {e}")
            llm_label = self.NULL_LABEL

        successfully_labeled = False if llm_label == self.NULL_LABEL else True

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
            if l.successfully_labeled and (
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
        accuracy = (
            results.get("strict").get("correct")
            / (results.get("strict").get("possible"))
            if results.get("strict").get("possible") > 0
            else 0.0
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
            Metric.ACCURACY: [],
            Metric.COMPLETION_RATE: [],
        }
        eval_metrics = []
        thresholds = []

        if self.config.confidence():
            eval_metrics_map[Metric.THRESHOLD] = []
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
        else:
            thresholds.append(float("-inf"))

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
                eval_metrics_map[metric.metric_type].append(metric.value)

            eval_metrics_map[Metric.COMPLETION_RATE].append(
                len(curr_llm_labels) / float(len(gt_labels))
            )

            if self.config.confidence():
                eval_metrics_map[Metric.THRESHOLD].append(threshold)
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
