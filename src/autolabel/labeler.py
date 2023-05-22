from loguru import logger
from tqdm import tqdm
from typing import Tuple, List, Dict, Union, Optional
import langchain
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

from autolabel.confidence import ConfidenceCalculator
from autolabel.cache import SQLAlchemyCache
from autolabel.few_shot import ExampleSelectorFactory
from autolabel.models import ModelFactory, BaseModel
from autolabel.schema import LLMAnnotation
from autolabel.tasks import TaskFactory
from autolabel.database import StateManager
from autolabel.schema import TaskRun, TaskStatus
from autolabel.data_models import TaskRunModel, AnnotationModel
from autolabel.configs import ModelConfig, DatasetConfig, TaskConfig


class LabelingAgent:
    CHUNK_SIZE = 5
    COST_KEY = "Cost in $"

    def __init__(
        self,
        task_config: Union[str, Dict],
        llm_config: Optional[Union[str, Dict]] = None,
        log_level: Optional[str] = "INFO",
        cache: Optional[bool] = True,
    ) -> None:
        self.db = StateManager()
        logger.remove()
        logger.add(sys.stdout, level=log_level)

        self.cache = SQLAlchemyCache() if cache else None

        self.set_task_config(task_config)
        self.set_llm_config(llm_config)

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
        label_column = dataset_config.get_label_column()

        dat = pd.read_csv(csv_file, sep=delimiter, dtype="str")[start_index:]
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat.to_dict(orient="records")
        gt_labels = None if not label_column else dat[label_column].tolist()
        return (dat, inputs, gt_labels)

    def _read_dataframe(
        self,
        df: pd.DataFrame,
        dataset_config: DatasetConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        label_column = dataset_config.get_label_column()

        dat = df[start_index:]
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat.to_dict(orient="records")
        gt_labels = None if not label_column else dat[label_column].tolist()
        return (dat, inputs, gt_labels)

    def run(
        self,
        dataset: Union[str, pd.DataFrame],
        dataset_config: Union[str, Dict],
        max_items: Optional[int] = None,
        output_name: Optional[str] = None,
        start_index: Optional[int] = 0,
        eval_every: Optional[int] = 50,
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

        if self.task_config.use_chain_of_thought():
            out_file = None
            if isinstance(dataset_config.get_seed_examples(), str):
                out_file = dataset_config.get_seed_examples().replace(
                    ".csv", "_explanations.csv"
                )
            seed_examples = self.generate_explanations(seed_examples, out_file)

        self.example_selector = ExampleSelectorFactory.initialize_selector(
            self.task_config, seed_examples
        )

        num_failures = 0
        current_index = self.task_run.current_index
        cost = 0.0
        postfix_dict = {}

        index_tqdm = tqdm(range(current_index, len(inputs), self.CHUNK_SIZE))
        for current_index in index_tqdm:
            chunk = inputs[current_index : current_index + self.CHUNK_SIZE]
            final_prompts = []
            for i, input_i in enumerate(chunk):
                # Fetch few-shot seed examples
                examples = self.example_selector.select_examples(input_i)
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
                        successfully_labeled="No",
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
                        if self.task_config.get_compute_confidence():
                            annotation = self.confidence.calculate(
                                model_generation=self.task.parse_llm_response(
                                    generation, chunk[i], final_prompts[i]
                                ),
                                empty_response=self.task_config.get_empty_response(),
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
                if gt_labels:
                    eval_result = self.task.eval(llm_labels, gt_labels)

                    for m in eval_result:
                        if not isinstance(m.value, list) or len(m.value) < 1:
                            continue
                        elif isinstance(m.value[0], float):
                            postfix_dict[m.name] = f"{m.value[0]:.4f}"
                        elif len(m.value[0]) > 0:
                            postfix_dict[m.name] = f"{m.value[0][0]:.4f}"

            index_tqdm.set_postfix(postfix_dict)
            # Update task run state
            self.task_run = self.save_task_run_state(
                current_index=current_index + len(chunk)
            )
            index_tqdm.refresh()

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
        """Calculates and prints the cost of calling autolabel.run() on a given dataset

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

        if self.task_config.use_chain_of_thought():
            out_file = None
            if isinstance(dataset_config.get_seed_examples(), str):
                out_file = dataset_config.get_seed_examples().replace(
                    ".csv", "_explanations.csv"
                )
            seed_examples = self.generate_explanations(seed_examples, out_file)

        self.example_selector = ExampleSelectorFactory.initialize_selector(
            self.task_config, seed_examples
        )

        input_limit = min(len(inputs), 100)
        num_sections = max(input_limit / self.CHUNK_SIZE, 1)
        for chunk in tqdm(np.array_split(inputs[:input_limit], num_sections)):
            for i, input_i in enumerate(chunk):
                # TODO: Check if this needs to use the example selector
                examples = self.example_selector.select_examples(input_i)
                final_prompt = self.task.construct_prompt(input_i, examples)
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

    def set_task_config(self, task_config: Union[str, Dict]):
        self.task_config = TaskConfig(task_config)
        self.task = TaskFactory.from_config(self.task_config)

    def set_llm_config(self, llm_config: Union[str, Dict]):
        self.llm_config = ModelConfig(llm_config)
        self.llm: BaseModel = ModelFactory.from_config(
            self.llm_config, cache=self.cache
        )
        self.confidence = ConfidenceCalculator(
            score_type="logprob_average", llm=self.llm
        )

    def create_dataset_config(self, dataset_config: Union[str, Dict]):
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
        if gt_labels and len(llm_labels) > 0:
            print("Evaluating the existing task...")
            gt_labels = gt_labels[: len(llm_labels)]
            eval_result = self.task.eval(llm_labels, gt_labels)
            for m in eval_result:
                print(f"Metric: {m.name}: {m.value}")
        print(f"{len(llm_labels)} examples have been labeled so far.")
        if len(llm_labels) > 0:
            print(f"Last annotated example - Prompt: {llm_labels[-1].prompt}")
            print(f"Annotation: {llm_labels[-1].label}")

        resume = None
        while resume is None:
            user_input = input("Do you want to resume the task? (y/n)")
            if user_input.lower() in ["y", "yes"]:
                print("Resuming the task...")
                resume = True
            elif user_input.lower() in ["n", "no"]:
                resume = False

        if not resume:
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
        self, seed_examples: List[Dict], save_file: str
    ) -> List[Dict]:
        generate_explanations = False
        for seed_example in tqdm(seed_examples):
            if not seed_example.get("explanation", ""):
                if not generate_explanations:
                    logger.info(
                        "Chain of thought requires explanations for seed examples. Generating explanations for seed examples."
                    )
                    generate_explanations = True

                explanation_prompt = self.task.generate_explanation(seed_example)
                explanation, _ = self.llm.label([explanation_prompt])
                explanation = explanation.generations[0][0].text
                seed_example["explanation"] = str(explanation) if explanation else ""

        if generate_explanations and save_file:
            logger.info(
                "Generated explanations for seed examples. Saving explanations to the seed file."
            )
            df = pd.DataFrame.from_records(seed_examples)
            df.to_csv(save_file, index=False)

        return seed_examples
