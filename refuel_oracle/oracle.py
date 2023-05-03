from loguru import logger
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Union, Optional
import langchain
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

from refuel_oracle.confidence import ConfidenceCalculator
from refuel_oracle.llm_cache import RefuelSQLLangchainCache
from refuel_oracle.task_config import TaskConfig
from refuel_oracle.example_selector import ExampleSelector
from refuel_oracle.models import ModelConfig, ModelFactory, BaseModel
from refuel_oracle.schema import LLMAnnotation
from refuel_oracle.task_config import TaskConfig
from refuel_oracle.tasks import TaskFactory
from refuel_oracle.dataset_config import DatasetConfig
from refuel_oracle.database import Database
from refuel_oracle.schema import TaskRun, TaskStatus
from refuel_oracle.data_models import TaskRunModel, AnnotationModel


class Oracle:
    CHUNK_SIZE = 5

    def __init__(
        self,
        task_config: Union[str, Dict],
        llm_config: Optional[Union[str, Dict]] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        self.set_task_config(task_config, **kwargs)
        self.set_llm_config(llm_config, **kwargs)
        self.debug = debug
        self.dir_path = Path(Path.home() / ".refuel_oracle")
        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True)
        self.db = Database(f"sqlite:///{self.dir_path}/database.db")
        log_level = "DEBUG" if self.debug else "INFO"
        logger.remove()
        logger.add(sys.stdout, level=log_level)
        if not self.debug:
            self.set_cache()

    # TODO: all this will move to a separate input parser class
    # this is a temporary solution to quickly add this feature and unblock expts
    def _read_csv(
        self,
        csv_file: str,
        dataset_config: DatasetConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        logger.debug(f"reading the csv from: {start_index}")
        delimiter = dataset_config.get_delimiter()
        input_columns = dataset_config.get_input_columns()
        label_column = dataset_config.get_label_column()

        dat = pd.read_csv(csv_file, sep=delimiter, dtype="str")[start_index:]
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat[input_columns + [label_column]].to_dict(orient="records")
        gt_labels = None if not label_column else dat[label_column].tolist()
        return (dat, inputs, gt_labels)

    def _read_dataframe(
        self,
        df: pd.DataFrame,
        dataset_config: DatasetConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        input_columns = dataset_config.get_input_columns()
        label_column = dataset_config.get_label_column()

        dat = df[start_index:]
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat[input_columns + [label_column]].to_dict(orient="records")
        gt_labels = None if not label_column else dat[label_column].tolist()
        return (dat, inputs, gt_labels)

    def annotate(
        self,
        dataset: Union[str, pd.DataFrame],
        dataset_config: Union[str, Dict],
        max_items: int = None,
        output_name: str = None,
        start_index: int = 0,
    ) -> None:
        """Labels data in a given dataset. Output written to new CSV file.

        Args:
            dataset: path to CSV dataset to be annotated
            max_items: maximum items in dataset to be annotated
            output_name: custom name of output CSV file
            start_index: skips annotating [0, start_index)
        """
        dataset_config = self.create_dataset_config(dataset_config)
        self.task.set_dataset_config(dataset_config)
        self.db.initialize()
        self.dataset = self.db.initialize_dataset(
            dataset, dataset_config, start_index, max_items
        )
        self.task_object = self.db.initialize_task(self.task_config, self.llm_config)
        csv_file_name = (
            output_name if output_name else f"{dataset.replace('.csv','')}_labeled.csv"
        )
        if isinstance(dataset, str):
            df, inputs, gt_labels = self._read_csv(
                dataset, dataset_config, max_items, start_index
            )
        elif isinstance(dataset, pd.DataFrame):
            df, inputs, gt_labels = self._read_dataframe(
                dataset, dataset_config, max_items, start_index
            )
        # Initialize task run and check if it already exists
        self.task_run = self.db.get_task_run(self.task_object.id, self.dataset.id)
        # Resume/Delete the task if it already exists or create a new task run
        if self.task_run:
            logger.info("Task run already exists.")
            self.task_run = self.handle_existing_task_run(
                self.task_run, csv_file_name, gt_labels=gt_labels
            )
        else:
            self.task_run = self.db.create_task_run(
                csv_file_name, self.task_object.id, self.dataset.id
            )

        # Get the seed examples from the dataset config
        seed_examples = dataset_config.get_seed_examples()

        # If this dataset config is a string, read the corrresponding csv file
        if isinstance(seed_examples, str):
            _, seed_examples, _ = self._read_csv(seed_examples, dataset_config)

        if self.task_config.get_example_selector():
            self.example_selector = ExampleSelector(
                self.task_config.get_example_selector(), seed_examples
            )
        else:
            self.example_selector = None

        num_failures = 0
        current_index = self.task_run.current_index

        for current_index in tqdm(range(current_index, len(inputs), self.CHUNK_SIZE)):
            chunk = inputs[current_index : current_index + self.CHUNK_SIZE]
            final_prompts = []
            for i, input_i in enumerate(chunk):
                # Fetch few-shot seed examples
                if self.example_selector:
                    examples = self.example_selector.get_examples(input_i)
                else:
                    examples = seed_examples

                # Construct Prompt to pass to LLM
                final_prompt = self.task.construct_prompt(input_i, examples)
                final_prompts.append(final_prompt)

            # Get response from LLM
            try:
                response = self.llm.label(final_prompts)
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
                        successfully_labeled="No",
                        label=None,
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
                    generation = response_item[0]
                    if self.task_config.get_compute_confidence():
                        annotation = self.confidence.calculate(
                            model_generation=self.task.parse_llm_response(
                                generation, chunk[i], final_prompts[i]
                            ),
                            empty_response=self.task_config.get_empty_response(),
                            prompt=final_prompts[i],
                            logprobs_available=self.task_config.get_has_logprob()
                            == "True",
                        )
                    else:
                        annotation = self.task.parse_llm_response(
                            generation, chunk[i], final_prompts[i]
                        )
                    AnnotationModel.create_from_llm_annotation(
                        self.db.session,
                        annotation,
                        current_index + i,
                        self.task_run.id,
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
        if gt_labels:
            eval_result = self.task.eval(llm_labels, gt_labels)
            # TODO: serialize and write to file
            for m in eval_result:
                print(f"Metric: {m.name}: {m.value}")

        # Write output to CSV
        output_df = df.copy()
        output_df[self.task_config.get_task_name() + "_llm_labeled_successfully"] = [
            l.successfully_labeled for l in llm_labels
        ]
        output_df[self.task_config.get_task_name() + "_llm_label"] = [
            l.label for l in llm_labels
        ]
        if self.task_config.get_compute_confidence():
            output_df["llm_confidence"] = [l.confidence_score for l in llm_labels]

        # Only save to csv if output_name is provided or dataset is a string
        if output_name:
            csv_file_name = output_name
        elif isinstance(dataset, str):
            csv_file_name = f"{dataset.replace('.csv','')}_labeled.csv"
            output_df.to_csv(
                csv_file_name,
                sep=dataset_config.get_delimiter(),
                header=True,
                index=False,
            )

        print(f"Total number of failures: {num_failures}")
        return (
            output_df[self.task_config.get_task_name() + "_llm_label"],
            output_df,
            eval_result,
        )

    def plan(
        self,
        dataset: Union[str, pd.DataFrame],
        dataset_config: Union[str, Dict],
        max_items: int = None,
        start_index: int = 0,
    ):
        """Calculates and prints the cost of calling oracle.annotate() on a given dataset

        Args:
            dataset: path to a CSV dataset
        """
        dataset_config = self.create_dataset_config(dataset_config)
        self.task.set_dataset_config(dataset_config)

        if isinstance(dataset, str):
            _, inputs, _ = self._read_csv(
                dataset, dataset_config, max_items, start_index
            )
        elif isinstance(dataset, pd.DataFrame):
            _, inputs, _ = self._read_dataframe(
                dataset, dataset_config, max_items, start_index
            )

        prompt_list = []
        total_cost = 0

        # Get the seed examples from the dataset config
        seed_examples = dataset_config.get_seed_examples()

        # If this dataset config is a string, read the corrresponding csv file
        if isinstance(seed_examples, str):
            _, seed_examples, _ = self._read_csv(seed_examples, dataset_config)

        input_limit = min(len(inputs), 100)
        num_sections = max(input_limit / self.CHUNK_SIZE, 1)
        for chunk in tqdm(np.array_split(inputs[:input_limit], num_sections)):
            for i, input_i in enumerate(chunk):
                # TODO: Check if this needs to use the example selector
                final_prompt = self.task.construct_prompt(input_i, seed_examples)
                prompt_list.append(final_prompt)

                # Calculate the number of tokens
                curr_cost = self.llm.get_cost(prompt=final_prompt, label="")
                total_cost += curr_cost

        total_cost = total_cost * (len(inputs) / input_limit)
        print(f"Total Estimated Cost: ${round(total_cost, 3)}")
        print(f"Number of examples to label: {len(inputs)}")
        print(f"Average cost per example: ${round(total_cost/len(inputs), 5)}")
        print(f"\n\nA prompt example:\n\n{prompt_list[0]}")
        return

    def set_cache(self):
        # Set cache for langchain
        langchain.llm_cache = RefuelSQLLangchainCache(database_path=".langchain.db")

    def set_task_config(self, task_config: Union[str, Dict], **kwargs):
        if isinstance(task_config, str):
            self.task_config = TaskConfig.from_json(task_config, **kwargs)
        else:
            self.task_config = TaskConfig(task_config)

        self.task = TaskFactory.from_config(self.task_config)
        self.example_selector = None
        if "example_selector" in self.task_config.keys():
            self.example_selector = ExampleSelector(self.task_config)

    def set_llm_config(self, llm_config: Union[str, Dict]):
        if isinstance(llm_config, str):
            self.llm_config = ModelConfig.from_json(llm_config)
        else:
            self.llm_config = ModelConfig(llm_config)

        self.llm: BaseModel = ModelFactory.from_config(self.llm_config)
        self.confidence = ConfidenceCalculator(
            score_type="logprob_average", llm=self.llm
        )

    def create_dataset_config(self, dataset_config: Union[str, Dict]):
        if isinstance(dataset_config, str):
            dataset_config = DatasetConfig.from_json(dataset_config)
        else:
            dataset_config = DatasetConfig(dataset_config)
        return dataset_config

    def handle_existing_task_run(
        self, task_run: TaskRun, csv_file_name: str, gt_labels: List[str] = None
    ) -> TaskRun:
        print(f"There is an existing task with following details: {task_run}")
        db_result = AnnotationModel.get_annotations_by_task_run_id(
            self.db.session, task_run.id
        )
        llm_labels = [LLMAnnotation(**a.llm_annotation) for a in db_result]
        if gt_labels:
            print("Evaluating the existing task...")
            gt_labels = gt_labels[: len(llm_labels)]
            eval_result = self.task.eval(llm_labels, gt_labels)
            for m in eval_result:
                print(f"Metric: {m.name}: {m.value}")
        print(f"{len(llm_labels)} examples have been labeled so far.")
        print(f"Last annotated example - Prompt: {llm_labels[-1].prompt}")
        print(f"Annotation: {llm_labels[-1].label}")
        user_input = input("Do you want to resume it? (y/n)")
        if user_input.lower() in ["y", "yes"]:
            print("Resuming the task...")
        else:
            TaskRunModel.delete_by_id(self.db.session, task_run.id)
            print("Deleted the existing task and starting a new one...")
            task_run = self.db.create_task_run(
                csv_file_name, self.task_object.id, self.dataset.id
            )
        return task_run

    def save_task_run_state(
        self, current_index: int = None, status: TaskStatus = "", error: str = ""
    ):
        # Save the current state of the task
        if error:
            self.task_run.error = error
        if status:
            self.task_run.status = status
        if current_index:
            self.task_run.current_index = current_index
        return TaskRunModel.update(self.db.session, self.task_run)
