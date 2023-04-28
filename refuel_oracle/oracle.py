from typing import Tuple, List, Dict, Union, Optional

import langchain
from loguru import logger
import numpy as np
import pandas as pd
from refuel_oracle.llm_cache import RefuelSQLLangchainCache
from tqdm import tqdm

from refuel_oracle.confidence import ConfidenceCalculator
from refuel_oracle.task_config import TaskConfig
from refuel_oracle.example_selector import ExampleSelector
from refuel_oracle.llm import LLMFactory, LLMProvider, LLMConfig
from refuel_oracle.schema import LLMAnnotation
from refuel_oracle.tasks import TaskFactory
from refuel_oracle.utils import calculate_cost, calculate_num_tokens
from refuel_oracle.dataset_config import DatasetConfig


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
        if not debug:
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

    def annotate(
        self,
        dataset: str,
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

        df, inputs, gt_labels = self._read_csv(
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
                response = self.llm.generate(final_prompts)
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
        if output_name:
            csv_file_name = output_name
        else:
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
        dataset: str,
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

        _, inputs, _ = self._read_csv(dataset, dataset_config, max_items, start_index)
        prompt_list = []
        total_tokens = 0

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

                if self.llm_config.get_provider() == LLMProvider.huggingface:
                    # Locally hosted Huggingface models do not have a cost per token
                    continue

                # Calculate the number of tokens
                num_tokens = calculate_num_tokens(self.llm_config, final_prompt)
                total_tokens += num_tokens
        total_cost = calculate_cost(self.llm_config, total_tokens)
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
            self.llm_config = LLMConfig.from_json(llm_config)
        else:
            self.llm_config = LLMConfig(llm_config)

        self.llm = LLMFactory.from_config(self.llm_config)
        self.confidence = ConfidenceCalculator(
            score_type="logprob_average", llm=self.llm
        )

    def create_dataset_config(self, dataset_config: Union[str, Dict]):
        if isinstance(dataset_config, str):
            dataset_config = DatasetConfig.from_json(dataset_config)
        else:
            dataset_config = DatasetConfig(dataset_config)

        return dataset_config
