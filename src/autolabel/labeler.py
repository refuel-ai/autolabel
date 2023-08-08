import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import json
import asyncio
import pandas as pd
from rich import print as pprint
from rich.console import Console
from rich.prompt import Confirm

from autolabel.cache import SQLAlchemyGenerationCache, SQLAlchemyTransformCache
from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import AutolabelConfig
from autolabel.dataset import AutolabelDataset
from autolabel.data_models import AnnotationModel, TaskRunModel
from autolabel.database import StateManager
from autolabel.few_shot import ExampleSelectorFactory
from autolabel.models import BaseModel, ModelFactory
from autolabel.metrics import BaseMetric
from autolabel.transforms import BaseTransform, TransformFactory
from autolabel.schema import (
    LLMAnnotation,
    MetricResult,
    TaskRun,
    TaskStatus,
)
from autolabel.tasks import TaskFactory
from autolabel.utils import (
    maybe_round,
    print_table,
    track,
    track_with_stats,
    gather_async_tasks_with_progress,
)

console = Console()
logger = logging.getLogger(__name__)

COST_TABLE_STYLES = {
    "parameter": "magenta bold",
    "value": "green bold",
}
METRIC_TABLE_STYLE = "cyan bold"


class LabelingAgent:
    COST_KEY = "Cost in $"

    def __init__(
        self,
        config: Union[AutolabelConfig, str, dict],
        cache: Optional[bool] = True,
    ) -> None:
        self.db = StateManager()
        self.generation_cache = SQLAlchemyGenerationCache() if cache else None
        self.transform_cache = SQLAlchemyTransformCache() if cache else None

        self.config = (
            config if isinstance(config, AutolabelConfig) else AutolabelConfig(config)
        )
        self.task = TaskFactory.from_config(self.config)
        self.llm: BaseModel = ModelFactory.from_config(
            self.config, cache=self.generation_cache
        )
        self.confidence = ConfidenceCalculator(
            score_type="logprob_average", llm=self.llm
        )

    def run(
        self,
        dataset: AutolabelDataset,
        output_name: Optional[str] = None,
        eval_every: Optional[int] = 50,
        additional_metrics: Optional[List[BaseMetric]] = [],
        max_items: Optional[int] = None,
        start_index: int = 0,
    ) -> Tuple[pd.Series, pd.DataFrame, List[MetricResult]]:
        """Labels data in a given dataset. Output written to new CSV file.

        Args:
            dataset: path to CSV dataset to be annotated
            max_items: maximum items in dataset to be annotated
            output_name: custom name of output CSV file
            start_index: skips annotating [0, start_index)
        """

        dataset = dataset.get_slice(max_items=max_items, start_index=start_index)

        self.db.initialize()
        self.dataset_obj = self.db.initialize_dataset(dataset.df, self.config)
        self.task_object = self.db.initialize_task(self.config)
        if isinstance(dataset, str):
            csv_file_name = (
                output_name
                if output_name
                else f"{dataset.replace('.csv','')}_labeled.csv"
            )
        else:
            csv_file_name = f"{self.config.task_name()}_labeled.csv"

        # Initialize task run and check if it already exists
        self.task_run = self.db.get_task_run(self.task_object.id, self.dataset_obj.id)
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

        self.example_selector = ExampleSelectorFactory.initialize_selector(
            self.config,
            seed_examples,
            dataset.df.keys().tolist(),
            cache=self.generation_cache is not None,
        )

        num_failures = 0
        current_index = self.task_run.current_index
        cost = 0.0
        postfix_dict = {}

        indices = range(current_index, len(dataset.inputs))

        for current_index in track_with_stats(
            indices,
            postfix_dict,
            total=len(dataset.inputs) - current_index,
            console=console,
        ):
            chunk = dataset.inputs[current_index]

            if self.example_selector:
                examples = self.example_selector.select_examples(chunk)
            else:
                examples = []
                # Construct Prompt to pass to LLM
            final_prompt = self.task.construct_prompt(chunk, examples)

            response = self.llm.label([final_prompt])
            for i, generations, error in zip(
                range(len(response.generations)), response.generations, response.errors
            ):
                if error is not None:
                    annotation = LLMAnnotation(
                        successfully_labeled=False,
                        label=self.task.NULL_LABEL_TOKEN,
                        raw_response="",
                        curr_sample=json.dumps(chunk),
                        prompt=final_prompt,
                        confidence_score=0,
                        error=error,
                    )
                else:
                    annotations = []
                    for generation in generations:
                        annotation = self.task.parse_llm_response(
                            generation, chunk, final_prompt
                        )

                        if self.config.confidence():
                            annotation.confidence_score = self.confidence.calculate(
                                model_generation=annotation,
                                prompt=final_prompt,
                            )

                        annotations.append(annotation)
                    annotation = self.majority_annotation(annotations)

                # Store the annotation in the database
                AnnotationModel.create_from_llm_annotation(
                    self.db.session,
                    annotation,
                    current_index + i,
                    self.task_run.id,
                )

            cost += sum(response.costs)
            postfix_dict[self.COST_KEY] = f"{cost:.2f}"

            # Evaluate the task every eval_every examples
            if (current_index + 1) % eval_every == 0:
                db_result = AnnotationModel.get_annotations_by_task_run_id(
                    self.db.session, self.task_run.id
                )
                llm_labels = [LLMAnnotation(**a.llm_annotation) for a in db_result]
                if dataset.gt_labels:
                    eval_result = self.task.eval(
                        llm_labels,
                        dataset.gt_labels[: len(llm_labels)],
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

            # Update task run state
            self.task_run = self.save_task_run_state(
                current_index=current_index + len(chunk)
            )

        db_result = AnnotationModel.get_annotations_by_task_run_id(
            self.db.session, self.task_run.id
        )
        llm_labels = [LLMAnnotation(**a.llm_annotation) for a in db_result]
        eval_result = None
        # if true labels are provided, evaluate accuracy of predictions
        if dataset.gt_labels:
            eval_result = self.task.eval(
                llm_labels,
                dataset.gt_labels[: len(llm_labels)],
                additional_metrics=additional_metrics,
            )
            table = {}
            # TODO: serialize and write to file
            for m in eval_result:
                if isinstance(m.value, list):
                    continue
                elif m.show_running:
                    table[m.name] = m.value
                else:
                    print(f"{m.name}:\n{m.value}")
            print(f"Actual Cost: {maybe_round(cost)}")
            print_table(table, console=console, default_style=METRIC_TABLE_STYLE)

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
            seed_examples,
            dataset.df.keys().tolist(),
            cache=self.generation_cache is not None,
        )

        input_limit = min(len(dataset.inputs), 100)
        for input_i in track(
            dataset.inputs[:input_limit],
            description="Generating Prompts...",
            console=console,
        ):
            # TODO: Check if this needs to use the example selector
            if self.example_selector:
                examples = self.example_selector.select_examples(input_i)
            else:
                examples = []
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
        print_table(table, show_header=False, console=console, styles=COST_TABLE_STYLES)

        console.rule("Prompt Example")
        print(f"{prompt_list[0]}")
        console.rule()

    async def async_run_transform(
        self, transform: BaseTransform, dataset: AutolabelDataset
    ):
        transform_outputs = [
            transform.apply(input_dict) for input_dict in dataset.inputs
        ]
        outputs = await gather_async_tasks_with_progress(
            transform_outputs,
            description=f"Running transform {transform.name()}...",
            console=console,
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
        pprint(f"There is an existing task with following details: {task_run}")
        db_result = AnnotationModel.get_annotations_by_task_run_id(
            self.db.session, task_run.id
        )
        llm_labels = [LLMAnnotation(**a.llm_annotation) for a in db_result]
        if gt_labels and len(llm_labels) > 0:
            pprint("Evaluating the existing task...")
            gt_labels = gt_labels[: len(llm_labels)]
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
                    print(f"{m.name}:\n{m.value}")

            print_table(table, console=console, default_style=METRIC_TABLE_STYLE)
        pprint(f"{task_run.current_index} examples labeled so far.")
        if len(llm_labels) > 0:
            console.rule("Last Annotated Example")
            pprint("[bold blue]Prompt[/bold blue]: ", end="")
            print(llm_labels[-1].prompt)
            pprint("[bold blue]Annotation[/bold blue]: ", end="")
            print(llm_labels[-1].label)
            console.rule()

        if not Confirm.ask("Do you want to resume the task?"):
            TaskRunModel.delete_by_id(self.db.session, task_run.id)
            pprint("Deleted the existing task and starting a new one...")
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
            seed_examples, description="Generating explanations", console=console
        ):
            explanation_prompt = self.task.get_explanation_prompt(seed_example)
            explanation = self.llm.label([explanation_prompt])
            explanation = explanation.generations[0][0].text
            seed_example["explanation"] = str(explanation) if explanation else ""

        if out_file:
            df = pd.DataFrame.from_records(seed_examples)
            df.to_csv(out_file, index=False)

        return seed_examples

    def clear_cache(self, use_ttl: bool = True):
        """
        Clears the generation and transformation cache from autolabel.
        Args:
            use_ttl: If true, only clears the cache if the ttl has expired.
            If this is true, only the transform cache will be cleared (as the generation cache does not use ttl)
        """
        if self.generation_cache and not use_ttl:
            self.generation_cache.clear()

        if self.transform_cache:
            self.transform_cache.clear(use_ttl=use_ttl)
