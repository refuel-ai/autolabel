import sys
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from rich import print as pprint
from rich.console import Console
from rich.prompt import Confirm
import pandas as pd

from autolabel.cache import SQLAlchemyCache
from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import AutolabelConfig
from autolabel.data_models import AnnotationModel, TaskRunModel
from autolabel.database import StateManager
from autolabel.dataset_loader import DatasetLoader
from autolabel.few_shot import ExampleSelectorFactory
from autolabel.models import BaseModel, ModelFactory
from autolabel.schema import LLMAnnotation, MetricResult, TaskRun, TaskStatus
from autolabel.tasks import TaskFactory
from autolabel.utils import track, track_with_stats, print_table, maybe_round

console = Console()
logger = logging.getLogger(__name__)

COST_TABLE_STYLES = {
    "parameter": "magenta bold",
    "value": "green bold",
}
METRIC_TABLE_STYLE = "cyan bold"


class LabelingAgent:
    CHUNK_SIZE = 5
    COST_KEY = "Cost in $"

    def __init__(
        self,
        config: Union[str, Dict],
        cache: Optional[bool] = True,
    ) -> None:
        self.db = StateManager()
        self.cache = SQLAlchemyCache() if cache else None

        self.config = AutolabelConfig(config)
        self.task = TaskFactory.from_config(self.config)
        self.llm: BaseModel = ModelFactory.from_config(self.config, cache=self.cache)
        self.confidence = ConfidenceCalculator(
            score_type="logprob_average", llm=self.llm
        )

    def run(
        self,
        dataset: Union[str, pd.DataFrame],
        max_items: Optional[int] = None,
        output_name: Optional[str] = None,
        start_index: Optional[int] = 0,
        eval_every: Optional[int] = 50,
    ) -> Tuple[pd.Series, pd.DataFrame, List[MetricResult]]:
        """Labels data in a given dataset. Output written to new CSV file.

        Args:
            dataset: path to CSV dataset to be annotated
            max_items: maximum items in dataset to be annotated
            output_name: custom name of output CSV file
            start_index: skips annotating [0, start_index)
        """
        dataset_loader = DatasetLoader(dataset, self.config, max_items, start_index)

        self.db.initialize()
        self.dataset = self.db.initialize_dataset(
            dataset_loader.dat, self.config, start_index, max_items
        )
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
        self.task_run = self.db.get_task_run(self.task_object.id, self.dataset.id)
        # Resume/Delete the task if it already exists or create a new task run
        if self.task_run:
            logger.info("Task run already exists.")
            self.task_run = self.handle_existing_task_run(
                self.task_run, csv_file_name, gt_labels=dataset_loader.gt_labels
            )
        else:
            self.task_run = self.db.create_task_run(
                csv_file_name, self.task_object.id, self.dataset.id
            )

        # Get the seed examples from the dataset config
        seed_examples = self.config.few_shot_example_set()

        # If this dataset config is a string, read the corrresponding csv file
        if isinstance(seed_examples, str):
            seed_loader = DatasetLoader(seed_examples, self.config)
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
            self.config, seed_examples, dataset_loader.dat.keys().tolist()
        )

        num_failures = 0
        current_index = self.task_run.current_index
        cost = 0.0
        postfix_dict = {}

        indices = range(current_index, len(dataset_loader.inputs), self.CHUNK_SIZE)
        for current_index in track_with_stats(
            indices,
            postfix_dict,
            total=len(dataset_loader.inputs) - current_index,
            advance=self.CHUNK_SIZE,
            console=console,
        ):
            chunk = dataset_loader.inputs[
                current_index : current_index + self.CHUNK_SIZE
            ]
            final_prompts = []
            for i, input_i in enumerate(chunk):
                # Fetch few-shot seed examples
                if self.example_selector:
                    examples = self.example_selector.select_examples(input_i)
                else:
                    examples = []
                # Construct Prompt to pass to LLM
                final_prompt = self.task.construct_prompt(input_i, examples)
                final_prompts.append(final_prompt)

            # Get response from LLM
            try:
                response, curr_cost = self.llm.label(final_prompts)
            except Exception as e:
                # TODO (dhruva): We need to handle this case carefully
                # When we erorr out, we will have less elements in the llm_labels
                # than the gt_labels array, with the 1:1 mapping not being
                # maintained either. We should either remove the elements we errored
                # out on from gt_labels or add None labels to the llm_labels.
                logger.error(
                    "Error in generating response:" + repr(e), "Prompt: ", chunk
                )
                for i in range(len(chunk)):
                    annotation = LLMAnnotation(
                        successfully_labeled=False,
                        label=self.task.NULL_LABEL_TOKEN,
                        raw_response="",
                        curr_sample=chunk[i],
                        prompt=final_prompts[i],
                        confidence_score=0,
                    )
                    AnnotationModel.create_from_llm_annotation(
                        self.db.session,
                        annotation,
                        current_index + i,
                        self.task_run.id,
                    )
                num_failures += len(chunk)
                response = None

            if response is not None:
                for i in range(len(response.generations)):
                    response_item = response.generations[i]
                    annotations = []
                    for generation in response_item:
                        if self.config.confidence():
                            annotation = self.confidence.calculate(
                                model_generation=self.task.parse_llm_response(
                                    generation, chunk[i], final_prompts[i]
                                ),
                                prompt=final_prompts[i],
                            )
                        else:
                            annotation = self.task.parse_llm_response(
                                generation, chunk[i], final_prompts[i]
                            )
                        annotations.append(annotation)
                    final_annotation = self.majority_annotation(annotations)
                    AnnotationModel.create_from_llm_annotation(
                        self.db.session,
                        final_annotation,
                        current_index + i,
                        self.task_run.id,
                    )
            cost += curr_cost
            postfix_dict[self.COST_KEY] = f"{cost:.2f}"

            # Evaluate the task every eval_every examples
            if (current_index + self.CHUNK_SIZE) % eval_every == 0:
                db_result = AnnotationModel.get_annotations_by_task_run_id(
                    self.db.session, self.task_run.id
                )
                llm_labels = [LLMAnnotation(**a.llm_annotation) for a in db_result]
                if dataset_loader.gt_labels:
                    eval_result = self.task.eval(
                        llm_labels, dataset_loader.gt_labels[: len(llm_labels)]
                    )

                    for m in eval_result:
                        if not isinstance(m.value, list) or len(m.value) < 1:
                            continue
                        elif isinstance(m.value[0], float):
                            postfix_dict[m.name] = f"{m.value[0]:.4f}"
                        elif isinstance(m.value[0], int):
                            postfix_dict[m.name] = f"{m.value[0]}"
                        elif len(m.value[0]) > 0:
                            postfix_dict[m.name] = f"{m.value[0][0]:.4f}"

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
        if dataset_loader.gt_labels:
            eval_result = self.task.eval(
                llm_labels, dataset_loader.gt_labels[: len(llm_labels)]
            )
            table = {}
            # TODO: serialize and write to file
            for m in eval_result:
                if isinstance(m.value, list) and len(m.value) > 0:
                    table[m.name] = m.value
                else:
                    print(f"Metric: {m.name}: {maybe_round(m.value)}")
            print(f"Actual Cost: {maybe_round(cost)}")
            print_table(table, console=console, default_style=METRIC_TABLE_STYLE)

        # Write output to CSV
        output_df = dataset_loader.dat.copy()
        output_df[self.config.task_name() + "_llm_labeled_successfully"] = [
            l.successfully_labeled for l in llm_labels
        ]
        output_df[self.config.task_name() + "_llm_label"] = [
            l.label for l in llm_labels
        ]
        if self.config.confidence():
            output_df["llm_confidence"] = [l.confidence_score for l in llm_labels]

        # Only save to csv if output_name is provided or dataset is a string
        if not output_name and isinstance(dataset, str):
            output_name = (
                dataset.rsplit(".", 1)[0] + "_labeled." + dataset.rsplit(".", 1)[1]
            )

        if output_name:
            if output_name.endswith(".csv"):
                output_df.to_csv(
                    str(output_name),
                    sep=self.config.delimiter(),
                    header=True,
                    index=False,
                )
            elif output_name.endswith(".jsonl"):
                output_df.to_json(
                    str(output_name),
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )
            else:
                raise ValueError(f"Unsupported output file format: {output_name}")

        pprint(f"Total number of failures: {num_failures}")
        return (
            output_df[self.config.task_name() + "_llm_label"],
            output_df,
            eval_result,
        )

    def plan(
        self,
        dataset: Union[str, pd.DataFrame],
        max_items: int = None,
        start_index: int = 0,
    ) -> None:
        """Calculates and prints the cost of calling autolabel.run() on a given dataset

        Args:
            dataset: path to a CSV dataset
        """

        dataset_loader = DatasetLoader(dataset, self.config, max_items, start_index)

        prompt_list = []
        total_cost = 0

        # Get the seed examples from the dataset config
        seed_examples = self.config.few_shot_example_set()

        # If this dataset config is a string, read the corrresponding csv file
        if isinstance(seed_examples, str):
            seed_loader = DatasetLoader(seed_examples, self.config)
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
            self.config, seed_examples, dataset_loader.dat.keys().tolist()
        )

        input_limit = min(len(dataset_loader.inputs), 100)
        for input_i in track(
            dataset_loader.inputs[:input_limit],
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

        total_cost = total_cost * (len(dataset_loader.inputs) / input_limit)
        table = {
            "Total Estimated Cost": f"${maybe_round(total_cost)}",
            "Number of Examples": len(dataset_loader.inputs),
            "Average cost per example": f"${maybe_round(total_cost / len(dataset_loader.inputs))}",
        }
        table = {"parameter": list(table.keys()), "value": list(table.values())}
        print_table(table, show_header=False, console=console, styles=COST_TABLE_STYLES)

        console.rule("Prompt Example")
        print(f"{prompt_list[0]}")
        console.rule()

    def handle_existing_task_run(
        self, task_run: TaskRun, csv_file_name: str, gt_labels: List[str] = None
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
            eval_result = self.task.eval(llm_labels, gt_labels)
            table = {}
            for m in eval_result:
                if isinstance(m.value, list) and len(m.value) > 0:
                    table[m.name] = m.value
                else:
                    print(f"Metric: {m.name}: {m.value}")
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
                csv_file_name, self.task_object.id, self.dataset.id
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
            _, seed_examples, _ = DatasetLoader.read_file(seed_examples, self.config)

        explanation_column = self.config.explanation_column()
        if not explanation_column:
            raise ValueError(
                "The explanation column needs to be specified in the dataset config."
            )

        for seed_example in track(
            seed_examples, description="Generating explanations", console=console
        ):
            explanation_prompt = self.task.get_explanation_prompt(seed_example)
            explanation, _ = self.llm.label([explanation_prompt])
            explanation = explanation.generations[0][0].text
            seed_example["explanation"] = str(explanation) if explanation else ""

        if out_file:
            df = pd.DataFrame.from_records(seed_examples)
            df.to_csv(out_file, index=False)

        return seed_examples

    def clear_cache(self):
        if self.cache:
            self.cache.clear()
        else:
            logger.error("No cache to clear")
