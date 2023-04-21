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
from refuel_oracle.utils import calculate_md5
from refuel_oracle.schema import TaskResult, TaskStatus, Task, Dataset
from refuel_oracle.data_models import (
    DatasetModel,
    TaskModel,
    TaskResultModel,
    AnnotationModel,
)


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
        self.dataset = self.initialize_dataset(dataset, max_items, start_index)
        self.task_object = self.initialize_task()
        csv_file_name = (
            output_name if output_name else f"{dataset.replace('.csv','')}_labeled.csv"
        )
        self.task_result = self.initialize_task_result(csv_file_name)

        if isinstance(dataset, str):
            df, inputs, gt_labels = self._read_csv(
                dataset, dataset_config, max_items, start_index
            )
        elif isinstance(dataset, pd.DataFrame):
            df, inputs, gt_labels = self._read_dataframe(
                dataset, dataset_config, max_items, start_index
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

        llm_labels = []
        prompt_list = []
        num_failures = 0

        num_sections = max(len(df) / self.CHUNK_SIZE, 1)
        for chunk in tqdm(np.array_split(inputs, num_sections)):
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

                prompt_list.append(final_prompt)

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
                    "Error in generating response:" + repr(e), "Prompt: ", chunk[i]
                )
                for _ in range(len(chunk)):
                    llm_labels.append(
                        LLMAnnotation(
                            successfully_labeled="No",
                            label=None,
                            raw_response="",
                            curr_sample=chunk[i],
                            promp=final_prompts[i],
                            confidence_score=0,
                        )
                    )
                num_failures += len(chunk)
                response = None

            if response is not None:
                for i in range(len(response.generations)):
                    response_item = response.generations[i]
                    generation = response_item[0]
                    if self.task_config.get_compute_confidence():
                        llm_labels.append(
                            self.confidence.calculate(
                                model_generation=self.task.parse_llm_response(
                                    generation, chunk[i], final_prompts[i]
                                ),
                                empty_response=self.task_config.get_empty_response(),
                                prompt=final_prompts[i],
                                logprobs_available=self.llm_config.get_has_logprob(),
                            )
                        )
                    else:
                        llm_labels.append(
                            self.task.parse_llm_response(
                                generation, chunk[i], final_prompts[i]
                            )
                        )
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
                curr_cost = self.llm.get_cost(prompt=final_prompt)
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

    def initialize_dataset(self, input_file, start_index, max_items):
        dataset_id = calculate_md5(open(input_file, "rb"))
        dataset_orm = DatasetModel.get_by_id(self.db.session, dataset_id)
        if dataset_orm:
            return Dataset.from_orm(dataset_orm)

        dataset = Dataset(
            id=dataset_id,
            input_file=input_file,
            start_index=start_index,
            end_index=start_index + max_items,
        )
        return Dataset.from_orm(DatasetModel.create(self.db.session, dataset))

    def initialize_task(self):
        task_id = calculate_md5(self.task_config.config)
        task_orm = TaskModel.get_by_id(self.db.session, task_id)
        if task_orm:
            return Task.from_orm(task_orm)

        task = Task(
            id=task_id,
            config=self.task_config.to_json(),
            task_type=self.task_config.get_task_type(),
            provider=self.llm_config.get_provider(),
            model_name=self.llm_config.get_model_name(),
        )
        return Task.from_orm(TaskModel.create(self.db.session, task))

    def initialize_task_result(self, output_file):
        task_result_orm = TaskResultModel.get(
            self.db.session, self.task_object.id, self.dataset.id
        )
        task_result = TaskResult.from_orm(task_result_orm) if task_result_orm else None
        logger.debug(f"existing task_result: {task_result}")
        new_task_result = TaskResult(
            task_id=self.task_object.id,
            dataset_id=self.dataset.id,
            status=TaskStatus.ACTIVE,
            current_index=self.dataset.start_index,
            output_file=output_file,
        )

        create_new_task_result = False
        # TODO: handle parallel tasks in ACTIVE state
        if task_result:
            print(f"There is an existing task with following details: {task_result}")
            if task_result.status in [
                TaskStatus.ACTIVE,
                TaskStatus.FAILURE,
            ]:
                raise Exception("There is an existing task in ACTIVE or FAILURE state")
            elif task_result.status == TaskStatus.PAUSED:
                user_input = input("Do you want to resume it? (y/n)")
                if user_input.lower() in ["y", "yes"]:
                    task_result.status = TaskStatus.ACTIVE
                    task_result_orm.update(self.db.session, task_result)
                else:
                    task_result_orm.delete(self.db.session)
                    self.db.delete_annotation_table(task_result.id)
                    create_new_task_result = True
            else:
                user_input = input("Do you want to start a new task? (y/n)")
                if user_input.lower() in ["y", "yes"]:
                    task_result_orm.delete(self.db.session)
                    self.db.delete_annotation_table(task_result.id)
                    create_new_task_result = True
                else:
                    exit(0)
        else:
            create_new_task_result = True

        if create_new_task_result:
            task_result_orm = TaskResultModel.create(self.db.session, new_task_result)
            self.db.create_annotation_table(task_result_orm.id)

        self.annotations_model = AnnotationModel.get_annotation_model(
            task_result_orm.id
        )
        self.task_result_orm = task_result_orm
        # print(Database.get_annotation_table(task_result_orm.id))
        return TaskResult.from_orm(task_result_orm)

    def save_state(self, error: str = None):
        # Save the current state of the task
        self.task_result.error = error
        self.task_result.status = TaskStatus.PAUSED
        self.task_result_orm.update(self.db.session, self.task_result)

    def test(self):
        return
