import logging
import os
import io
from typing import Dict, List, Optional, Tuple, Union
import json
import pickle
import asyncio
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from rich import print as pprint
from rich.console import Console
from rich.prompt import Confirm

from autolabel.cache import (
    BaseCache,
    SQLAlchemyGenerationCache,
    SQLAlchemyTransformCache,
    SQLAlchemyConfidenceCache,
)
from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import AutolabelConfig
from autolabel.dataset import AutolabelDataset
from autolabel.data_models import AnnotationModel, TaskRunModel
from autolabel.database import StateManager
from autolabel.few_shot import (
    ExampleSelectorFactory,
    BaseExampleSelector,
    DEFAULT_EMBEDDING_PROVIDER,
    PROVIDER_TO_MODEL,
)
from autolabel.few_shot.label_selector import LabelSelector
from autolabel.models import BaseModel, ModelFactory
from autolabel.metrics import BaseMetric
from autolabel.transforms import BaseTransform, TransformFactory
from autolabel.schema import (
    LLMAnnotation,
    MetricResult,
    TaskRun,
    TaskStatus,
    TaskType,
    AggregationFunction,
)
from autolabel.tasks import TaskFactory
from autolabel.utils import (
    maybe_round,
    print_table,
    track,
    track_with_stats,
    gather_async_tasks_with_progress,
    get_format_variables,
    in_notebook,
    safe_serialize_to_string,
)

logger = logging.getLogger(__name__)

COST_TABLE_STYLES = {
    "parameter": "magenta bold",
    "value": "green bold",
}
METRIC_TABLE_STYLE = "cyan bold"

MERGE_FUNCTION = {
    AggregationFunction.MAX: np.max,
    AggregationFunction.MEAN: np.mean,
}


class LabelingAgent:
    COST_KEY = "Cost in $"
    CONFIDENCE_MAX_CONTEXT_LENGTH = 3400

    def __init__(
        self,
        config: Union[AutolabelConfig, str, dict],
        cache: Optional[bool] = True,
        example_selector: Optional[BaseExampleSelector] = None,
        create_task: Optional[bool] = False,
        console_output: Optional[bool] = True,
        generation_cache: Optional[BaseCache] = SQLAlchemyGenerationCache(),
        transform_cache: Optional[BaseCache] = SQLAlchemyTransformCache(),
        confidence_cache: Optional[BaseCache] = SQLAlchemyConfidenceCache(),
        confidence_tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self.create_task = create_task
        self.db = StateManager() if self.create_task else None
        self.generation_cache = generation_cache
        self.transform_cache = transform_cache
        self.confidence_cache = confidence_cache
        if not cache:
            logger.warning(
                f"cache parameter is deprecated and will be removed soon. Please use generation_cache, transform_cache and confidence_cache instead."
            )
            self.generation_cache = None
            self.transform_cache = None
            self.confidence_cache = None

        if self.generation_cache is not None:
            self.generation_cache.initialize()
        if self.transform_cache is not None:
            self.transform_cache.initialize()
        if self.confidence_cache is not None:
            self.confidence_cache.initialize()

        self.console = Console(quiet=not console_output)

        self.config = (
            config if isinstance(config, AutolabelConfig) else AutolabelConfig(config)
        )
        self.task = TaskFactory.from_config(self.config)
        self.llm: BaseModel = ModelFactory.from_config(
            self.config, cache=self.generation_cache
        )

        if self.config.confidence_chunk_column():
            if not confidence_tokenizer:
                self.confidence_tokenizer = AutoTokenizer.from_pretrained(
                    "google/flan-t5-xxl"
                )
            else:
                self.confidence_tokenizer = confidence_tokenizer
        score_type = "logprob_average"
        if self.config.task_type() == TaskType.ATTRIBUTE_EXTRACTION:
            score_type = "logprob_average_per_key"
        self.confidence = ConfidenceCalculator(
            score_type=score_type,
            llm=self.llm,
            cache=self.confidence_cache,
        )

        self.example_selector = example_selector

        # Only used if we don't use task management
        self.all_annotations = []

        if self.create_task:
            logger.warning(
                f"create_task parameter is deprecated and will be removed soon. The LLM calls are getting cached and should handle most use cases."
            )

        if in_notebook():
            import nest_asyncio

            nest_asyncio.apply()

    def run(
        self,
        dataset: AutolabelDataset,
        output_name: Optional[str] = None,
        max_items: Optional[int] = None,
        start_index: int = 0,
        additional_metrics: Optional[List[BaseMetric]] = [],
        skip_eval: Optional[bool] = False,
    ) -> Tuple[pd.Series, pd.DataFrame, List[MetricResult]]:
        """Labels data in a given dataset. Output written to new CSV file.

        Args:
            dataset: path to CSV dataset to be annotated
            max_items: maximum items in dataset to be annotated
            output_name: custom name of output CSV file
            start_index: skips annotating [0, start_index)
        """

        dataset = dataset.get_slice(max_items=max_items, start_index=start_index)

        if self.create_task:
            self.db.initialize()
            self.dataset_obj = self.db.initialize_dataset(dataset.df, self.config)
            self.task_object = self.db.initialize_task(self.config)
        else:
            self.all_annotations = []

        if isinstance(dataset, str):
            csv_file_name = (
                output_name
                if output_name
                else f"{dataset.replace('.csv','')}_labeled.csv"
            )
        else:
            csv_file_name = f"{self.config.task_name()}_labeled.csv"

        if self.create_task:
            # Initialize task run and check if it already exists
            self.task_run = self.db.get_task_run(
                self.task_object.id, self.dataset_obj.id
            )
            # Resume/Delete the task if it already exists or create a new task run
            if self.task_run:
                logger.info("Task run already exists.")
                self.task_run = self.handle_existing_task_run(
                    self.task_run,
                    csv_file_name,
                    gt_labels=dataset.gt_labels,
                    additional_metrics=additional_metrics,
                )
            else:
                self.task_run = self.db.create_task_run(
                    csv_file_name, self.task_object.id, self.dataset_obj.id
                )

        # Get the seed examples from the dataset config
        seed_examples = self.config.few_shot_example_set()

        # If this dataset config is a string, read the corrresponding csv file
        if isinstance(seed_examples, str):
            seed_loader = AutolabelDataset(seed_examples, self.config)
            seed_examples = seed_loader.inputs

        # Check explanations are present in data if explanation_column is passed in
        if (
            self.config.explanation_column()
            and len(seed_examples) > 0
            and self.config.explanation_column() not in list(seed_examples[0].keys())
        ):
            raise ValueError(
                f"Explanation column {self.config.explanation_column()} not found in dataset.\nMake sure that explanations were generated using labeler.generate_explanations(seed_file)."
            )

        if self.example_selector is None:
            self.example_selector = ExampleSelectorFactory.initialize_selector(
                self.config,
                [safe_serialize_to_string(example) for example in seed_examples],
                dataset.df.keys().tolist(),
                cache=self.generation_cache is not None,
            )

        if self.config.label_selection():
            if self.config.task_type() != TaskType.CLASSIFICATION:
                self.console.print(
                    "Warning: label_selection only supported for classification tasks!"
                )
            else:
                self.label_selector = LabelSelector.from_examples(
                    labels=self.config.labels_list(),
                    embedding_func=PROVIDER_TO_MODEL.get(
                        self.config.embedding_provider(), DEFAULT_EMBEDDING_PROVIDER
                    )(),
                    k=self.config.label_selection_count(),
                )

        current_index = self.task_run.current_index if self.create_task else 0
        cost = 0.0
        postfix_dict = {}

        indices = range(current_index, len(dataset.inputs))

        for current_index in track_with_stats(
            indices,
            postfix_dict,
            total=len(dataset.inputs) - current_index,
            console=self.console,
        ):
            chunk = dataset.inputs[current_index]

            if self.example_selector:
                examples = self.example_selector.select_examples(
                    safe_serialize_to_string(chunk)
                )
            else:
                examples = []
            # Construct Prompt to pass to LLM
            if (
                self.config.label_selection()
                and self.config.task_type() == TaskType.CLASSIFICATION
            ):
                selected_labels = self.label_selector.select_labels(chunk["example"])
                final_prompt = self.task.construct_prompt(
                    chunk, examples, selected_labels=selected_labels
                )
            else:
                final_prompt = self.task.construct_prompt(chunk, examples)

            response = self.llm.label([final_prompt])
            for i, generations, error, latency in zip(
                range(len(response.generations)),
                response.generations,
                response.errors,
                response.latencies,
            ):
                input_tokens = self.get_num_tokens(final_prompt)
                if error is not None:
                    annotation = LLMAnnotation(
                        successfully_labeled=False,
                        label=self.task.NULL_LABEL_TOKEN,
                        raw_response="",
                        curr_sample=pickle.dumps(chunk),
                        prompt=final_prompt,
                        confidence_score=0,
                        error=error,
                        input_tokens=input_tokens,
                        cost=0,
                        latency=0,
                    )
                else:
                    annotations = []
                    for generation in generations:
                        annotation = self.task.parse_llm_response(
                            generation, chunk, final_prompt
                        )
                        annotation.confidence_prompt = (
                            self.task.construct_confidence_prompt(chunk, examples)
                        )
                        annotation.input_tokens = input_tokens
                        annotation.output_tokens = self.get_num_tokens(
                            annotation.raw_response
                        )
                        annotation.cost = sum(response.costs)
                        annotation.latency = latency

                        if self.config.confidence():
                            full_confidence_input = (
                                annotation.confidence_prompt + annotation.raw_response
                            )
                            if (
                                not self.config.confidence_chunk_column()
                                or self.get_num_tokens(full_confidence_input)
                                < self.CONFIDENCE_MAX_CONTEXT_LENGTH
                            ):
                                annotation.confidence_score = self.confidence.calculate(
                                    model_generation=annotation,
                                )
                            else:
                                key_to_chunk = self.config.confidence_chunk_column()
                                if not key_to_chunk:
                                    raise ValueError(
                                        "confidence_chunk_column must be set in the config to use confidence_chunk_size"
                                    )

                                empty_chunk = chunk.copy()
                                empty_chunk[key_to_chunk] = ""
                                empty_prompt = self.task.construct_confidence_prompt(
                                    empty_chunk, examples
                                )
                                num_tokens_empty_prompt = self.get_num_tokens(
                                    empty_prompt
                                )
                                num_tokens_per_chunk = (
                                    self.config.confidence_chunk_size()
                                    - num_tokens_empty_prompt
                                )
                                confidence_chunks = self.chunk_string(
                                    chunk[key_to_chunk], num_tokens_per_chunk
                                )

                                confidence_scores = []
                                for confidence_chunk in confidence_chunks:
                                    new_chunk = chunk.copy()
                                    new_chunk[key_to_chunk] = confidence_chunk
                                    new_prompt = self.task.construct_confidence_prompt(
                                        new_chunk, examples
                                    )
                                    annotation_dict = annotation.dict()
                                    annotation_dict["confidence_prompt"] = new_prompt
                                    confidence_scores.append(
                                        self.confidence.calculate(
                                            model_generation=LLMAnnotation(
                                                **annotation_dict
                                            ),
                                        )
                                    )

                                merge_function = MERGE_FUNCTION[
                                    self.config.confidence_merge_function()
                                ]
                                if isinstance(confidence_scores[0], dict):
                                    merged_confidence = {}
                                    for key in confidence_scores[0].keys():
                                        merged_confidence[key] = merge_function(
                                            [conf[key] for conf in confidence_scores]
                                        )
                                else:
                                    merged_confidence = merge_function(
                                        confidence_scores
                                    )
                                annotation.confidence_score = merged_confidence

                        annotations.append(annotation)
                    annotation = self.majority_annotation(annotations)

                # Save the annotation in the database
                self.save_annotation(annotation, current_index, i)

            cost += sum(response.costs)
            postfix_dict[self.COST_KEY] = f"{cost:.2f}"

            # Evaluate the task every eval_every examples
            if not skip_eval and (current_index + 1) % 100 == 0:
                llm_labels = self.get_all_annotations()
                if dataset.gt_labels:
                    eval_result = self.task.eval(
                        llm_labels,
                        dataset.gt_labels[: len(llm_labels)]
                        if isinstance(dataset.gt_labels, list)
                        else {
                            k: v[: len(llm_labels)]
                            for k, v in dataset.gt_labels.items()
                        },
                        additional_metrics=additional_metrics,
                    )

                    for m in eval_result:
                        # This is a row wise metric
                        if isinstance(m.value, list):
                            continue
                        elif m.show_running:
                            postfix_dict[m.name] = (
                                f"{m.value:.4f}"
                                if isinstance(m.value, float)
                                else m.value
                            )

            if self.create_task:
                # Update task run state
                self.task_run = self.save_task_run_state(
                    current_index=current_index + len(chunk)
                )

        llm_labels = self.get_all_annotations()
        eval_result = None
        table = {}

        # if true labels are provided, evaluate accuracy of predictions
        if not skip_eval and dataset.gt_labels:
            eval_result = self.task.eval(
                llm_labels,
                dataset.gt_labels[: len(llm_labels)]
                if isinstance(dataset.gt_labels, list)
                else {k: v[: len(llm_labels)] for k, v in dataset.gt_labels.items()},
                additional_metrics=additional_metrics,
            )
            # TODO: serialize and write to file
            for m in eval_result:
                if isinstance(m.value, list):
                    continue
                elif m.show_running:
                    table[m.name] = m.value
                else:
                    self.console.print(f"{m.name}:\n{m.value}")

        # print cost
        self.console.print(f"Actual Cost: {maybe_round(cost)}")
        print_table(table, console=self.console, default_style=METRIC_TABLE_STYLE)

        dataset.process_labels(llm_labels, eval_result)
        # Only save to csv if output_name is provided or dataset is a string
        if not output_name and isinstance(dataset, str):
            output_name = (
                dataset.rsplit(".", 1)[0] + "_labeled." + dataset.rsplit(".", 1)[1]
            )

        if output_name:
            dataset.save(output_file_name=output_name)
        return dataset

    def plan(
        self,
        dataset: AutolabelDataset,
        max_items: Optional[int] = None,
        start_index: int = 0,
    ) -> None:
        """Calculates and prints the cost of calling autolabel.run() on a given dataset

        Args:
            dataset: path to a CSV dataset
        """
        dataset = dataset.get_slice(max_items=max_items, start_index=start_index)

        if self.config.confidence() and "REFUEL_API_KEY" not in os.environ:
            raise ValueError(
                "REFUEL_API_KEY environment variable must be set to compute confidence scores. You can request an API key at https://refuel-ai.typeform.com/llm-access."
            )

        prompt_list = []
        total_cost = 0

        # Get the seed examples from the dataset config
        seed_examples = self.config.few_shot_example_set()

        # If this dataset config is a string, read the corrresponding csv file
        if isinstance(seed_examples, str):
            seed_loader = AutolabelDataset(seed_examples, self.config)
            seed_examples = seed_loader.inputs

        # Check explanations are present in data if explanation_column is passed in
        if (
            self.config.explanation_column()
            and len(seed_examples) > 0
            and self.config.explanation_column() not in list(seed_examples[0].keys())
        ):
            raise ValueError(
                f"Explanation column {self.config.explanation_column()} not found in dataset.\nMake sure that explanations were generated using labeler.generate_explanations(seed_file)."
            )

        self.example_selector = ExampleSelectorFactory.initialize_selector(
            self.config,
            [safe_serialize_to_string(example) for example in seed_examples],
            dataset.df.keys().tolist(),
            cache=self.generation_cache is not None,
        )

        if self.config.label_selection():
            if self.config.task_type() != TaskType.CLASSIFICATION:
                self.console.print(
                    "Warning: label_selection only supported for classification tasks!"
                )
            else:
                self.label_selector = LabelSelector.from_examples(
                    labels=self.config.labels_list(),
                    embedding_func=PROVIDER_TO_MODEL.get(
                        self.config.embedding_provider(), DEFAULT_EMBEDDING_PROVIDER
                    )(),
                    k=self.config.label_selection_count(),
                )

        input_limit = min(len(dataset.inputs), 100)

        for input_i in track(
            dataset.inputs[:input_limit],
            description="Generating Prompts...",
            console=self.console,
        ):
            # TODO: Check if this needs to use the example selector
            if self.example_selector:
                examples = self.example_selector.select_examples(
                    safe_serialize_to_string(input_i)
                )
            else:
                examples = []
            if (
                self.config.label_selection()
                and self.config.task_type() == TaskType.CLASSIFICATION
            ):
                selected_labels = self.label_selector.select_labels(input_i["example"])
                final_prompt = self.task.construct_prompt(
                    input_i, examples, selected_labels=selected_labels
                )
            else:
                final_prompt = self.task.construct_prompt(input_i, examples)
            prompt_list.append(final_prompt)

            # Calculate the number of tokens
            curr_cost = self.llm.get_cost(prompt=final_prompt, label="")
            total_cost += curr_cost

        total_cost = total_cost * (len(dataset.inputs) / input_limit)
        table = {
            "Total Estimated Cost": f"${maybe_round(total_cost)}",
            "Number of Examples": len(dataset.inputs),
            "Average cost per example": f"${maybe_round(total_cost / len(dataset.inputs))}",
        }
        table = {"parameter": list(table.keys()), "value": list(table.values())}

        print_table(
            table, show_header=False, console=self.console, styles=COST_TABLE_STYLES
        )
        self.console.rule("Prompt Example")
        self.console.print(f"{prompt_list[0]}", markup=False)
        self.console.rule()

    async def async_run_transform(
        self, transform: BaseTransform, dataset: AutolabelDataset
    ):
        transform_outputs = [
            transform.apply(input_dict) for input_dict in dataset.inputs
        ]

        outputs = await gather_async_tasks_with_progress(
            transform_outputs,
            description=f"Running transform {transform.name()}...",
            console=self.console,
        )
        output_df = pd.DataFrame.from_records(outputs)
        final_df = pd.concat([dataset.df, output_df], axis=1)
        dataset = AutolabelDataset(final_df, self.config)
        return dataset

    def transform(self, dataset: AutolabelDataset):
        transforms = []
        for transform_dict in self.config.transforms():
            transforms.append(
                TransformFactory.from_dict(transform_dict, cache=self.transform_cache)
            )
        for transform in transforms:
            dataset = asyncio.run(self.async_run_transform(transform, dataset))

        return dataset

    def handle_existing_task_run(
        self,
        task_run: TaskRun,
        csv_file_name: str,
        gt_labels: List[str] = None,
        additional_metrics: List[BaseMetric] = [],
    ) -> TaskRun:
        """
        Allows for continuing an existing labeling task. The user will be asked whether they wish to continue from where the run previously left off, or restart from the beginning.
        Args:
            task_run: TaskRun to retry
            csv_file_name: path to the dataset we wish to label (only used if user chooses to restart the task)
            gt_labels: If ground truth labels are provided, performance metrics will be displayed, such as label accuracy
        """
        self.console.print(
            f"There is an existing task with following details: {task_run}"
        )
        llm_labels = self.get_all_annotations()
        if gt_labels and len(llm_labels) > 0:
            pprint("Evaluating the existing task...")
            gt_labels = (
                gt_labels[: len(llm_labels)]
                if isinstance(gt_labels, list)
                else {k: v[: len(llm_labels)] for k, v in gt_labels.items()}
            )
            eval_result = self.task.eval(
                llm_labels, gt_labels, additional_metrics=additional_metrics
            )
            table = {}
            for m in eval_result:
                if isinstance(m.value, list):
                    continue
                elif m.show_running:
                    table[m.name] = m.value
                else:
                    self.console.print(f"{m.name}:\n{m.value}")

            print_table(table, console=self.console, default_style=METRIC_TABLE_STYLE)
        self.console.print(f"{task_run.current_index} examples labeled so far.")
        if not Confirm.ask("Do you want to resume the task?"):
            TaskRunModel.delete_by_id(self.db.session, task_run.id)
            self.console.print("Deleted the existing task and starting a new one...")
            task_run = self.db.create_task_run(
                csv_file_name, self.task_object.id, self.dataset_obj.id
            )
        return task_run

    def save_task_run_state(
        self, current_index: int = None, status: TaskStatus = "", error: str = ""
    ) -> TaskRun:
        """Saves the current state of the Task being performed"""
        # Save the current state of the task
        if error:
            self.task_run.error = error
        if status:
            self.task_run.status = status
        if current_index:
            self.task_run.current_index = current_index
        return TaskRunModel.update(self.db.session, self.task_run)

    def majority_annotation(
        self, annotation_list: List[LLMAnnotation]
    ) -> LLMAnnotation:
        labels = [a.label for a in annotation_list]
        counts = {}
        for ind, label in enumerate(labels):
            # Needed for named entity recognition which outputs lists instead of strings
            label = str(label)

            if label not in counts:
                counts[label] = (1, ind)
            else:
                counts[label] = (counts[label][0] + 1, counts[label][1])
        max_label = max(counts, key=lambda x: counts[x][0])
        return annotation_list[counts[max_label][1]]

    def generate_explanations(
        self,
        seed_examples: Union[str, List[Dict]],
    ) -> List[Dict]:
        """Use LLM to generate explanations for why examples are labeled the way that they are."""
        out_file = None
        if isinstance(seed_examples, str):
            out_file = seed_examples
            seed_loader = AutolabelDataset(seed_examples, self.config)
            seed_examples = seed_loader.inputs

        explanation_column = self.config.explanation_column()
        if not explanation_column:
            raise ValueError(
                "The explanation column needs to be specified in the dataset config."
            )

        for seed_example in track(
            seed_examples,
            description="Generating explanations",
            console=self.console,
        ):
            explanation_prompt = self.task.get_explanation_prompt(seed_example)
            explanation = self.llm.label([explanation_prompt])
            explanation = explanation.generations[0][0].text
            seed_example["explanation"] = str(explanation) if explanation else ""

        if out_file:
            df = pd.DataFrame.from_records(seed_examples)
            df.to_csv(out_file, index=False)

        return seed_examples

    def generate_synthetic_dataset(self) -> AutolabelDataset:
        columns = get_format_variables(self.config.example_template())
        df = pd.DataFrame(columns=columns)
        for label in track(
            self.config.labels_list(),
            description="Generating dataset",
            console=self.console,
        ):
            prompt = self.task.get_generate_dataset_prompt(label)

            result = self.llm.label([prompt])
            if result.errors[0] is not None:
                self.console.print(
                    f"Error generating rows for label {label}: {result.errors[0]}"
                )
            else:
                response = result.generations[0][0].text.strip()

                response = io.StringIO(response)
                label_df = pd.read_csv(response, sep=self.config.delimiter())
                label_df[self.config.label_column()] = label
                df = pd.concat([df, label_df], axis=0, ignore_index=True)
        return AutolabelDataset(df, self.config)

    def clear_cache(self, use_ttl: bool = True):
        """
        Clears the generation and transformation cache from autolabel.
        Args:
            use_ttl: If true, only clears the cache if the ttl has expired.
        """
        self.generation_cache.clear(use_ttl=use_ttl)
        self.transform_cache.clear(use_ttl=use_ttl)

    def save_annotation(self, annotation: LLMAnnotation, current_index: int, i: int):
        if self.create_task:
            # Store the annotation in the database
            AnnotationModel.create_from_llm_annotation(
                self.db.session,
                annotation,
                current_index + i,
                self.task_run.id,
            )
        else:
            self.all_annotations.append(annotation)

    def get_all_annotations(self):
        if self.create_task:
            db_result = AnnotationModel.get_annotations_by_task_run_id(
                self.db.session, self.task_run.id
            )
            return [pickle.loads(a.llm_annotation) for a in db_result]
        else:
            return self.all_annotations

    def get_num_tokens(self, inp: str) -> int:
        """Returns the number of tokens in the prompt"""
        return len(self.confidence_tokenizer.encode(str(inp)))

    def chunk_string(self, inp: str, chunk_size: int) -> List[str]:
        """Chunks the input string into chunks of size chunk_size"""
        tokens = self.confidence_tokenizer.encode(inp)
        chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        return [self.confidence_tokenizer.decode(chunk) for chunk in chunks]
