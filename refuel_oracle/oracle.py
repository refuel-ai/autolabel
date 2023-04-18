from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from refuel_oracle.config import Config
from refuel_oracle.example_selector import ExampleSelector
from refuel_oracle.llm import LLMFactory
from refuel_oracle.tasks import TaskFactory
from refuel_oracle.utils import calculate_cost, calculate_num_tokens
from tqdm import tqdm

import langchain
from langchain.cache import SQLiteCache


class Oracle:
    CHUNK_SIZE = 5
    DEFAULT_SEPARATOR = ","

    def __init__(self, config: str, debug: bool = False) -> None:
        self.debug = debug
        self.config = Config.from_json(config)
        self.llm = LLMFactory.from_config(self.config)
        self.task = TaskFactory.from_config(self.config)
        self.example_selector = None
        if "example_selector" in self.config.keys():
            self.example_selector = ExampleSelector(self.config)

        if not debug:
            self.set_cache()

    # TODO: all this will move to a separate input parser class
    # this is a temporary solution to quickly add this feature and unblock expts
    def _read_csv(
        self, csv_file: str, max_items: int = None, start_index: int = 0
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        dataset_schema = self.config.get("dataset_schema", {})
        delimiter = dataset_schema.get("delimiter", self.DEFAULT_SEPARATOR)
        input_columns = dataset_schema.get("input_columns", [])
        label_column = dataset_schema.get("label_column")

        dat = pd.read_csv(csv_file, sep=delimiter, dtype="str")[start_index:]
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat[input_columns].to_dict(orient="records")
        gt_labels = None if not label_column else dat[label_column].tolist()
        return (dat, inputs, gt_labels)

    def annotate(
        self,
        dataset: str,
        max_items: int = None,
        output_name: str = None,
        start_index: int = 0,
    ) -> None:
        df, inputs, gt_labels = self._read_csv(dataset, max_items, start_index)

        if not max_items:
            max_items = len(df)

        llm_labels = []
        prompt_list = []
        total_tokens = 0

        num_sections = max(max_items / self.CHUNK_SIZE, 1)
        for chunk_id, chunk in enumerate(tqdm(np.array_split(inputs, num_sections))):
            if chunk_id % 10 == 0:
                print(
                    f"Labeling {chunk_id*self.CHUNK_SIZE+1}-{chunk_id*self.CHUNK_SIZE+(self.CHUNK_SIZE*10)}"
                )
            final_prompts = []
            for i, input_i in enumerate(chunk):
                # Fetch few-shot seed examples
                if self.example_selector:
                    examples = self.example_selector.get_examples(input_i)
                else:
                    examples = self.config.get("seed_examples", [])
                # Construct Prompt to pass to LLM
                final_prompt = self.task.construct_prompt(input_i, examples)
                final_prompts.append(final_prompt)
                num_tokens = calculate_num_tokens(self.config, final_prompt)
                total_tokens += num_tokens
                prompt_list.append(final_prompt)
            # Get response from LLM
            response = self.llm.generate(final_prompts)
            for i in range(len(response.generations)):
                response_item = response.generations[i]
                input_i = chunk[i]
                generation = response_item[0]
                llm_labels.append(self.task.parse_llm_response(generation, input_i))

        # if true labels are provided, evaluate accuracy of predictions
        if gt_labels:
            eval_result = self.task.eval(llm_labels, gt_labels)
            # TODO: serialize and write to file
            for m in eval_result:
                print(f"Metric: {m.name}: {m.value}")

        # Write output to CSV
        output_df = df.copy()
        output_df["llm_labeled_successfully"] = [
            l.successfully_labeled for l in llm_labels
        ]
        output_df["llm_label"] = [l.label for l in llm_labels]
        if output_name:
            csv_file_name = output_name
        else:
            csv_file_name = f"{dataset.replace('.csv','')}_labeled.csv"
        output_df.to_csv(
            csv_file_name,
            sep=self.DEFAULT_SEPARATOR,
            header=True,
            index=False,
        )

    def plan(
        self,
        dataset: str,
    ):
        _, inputs, _ = self._read_csv(dataset)
        prompt_list = []
        total_tokens = 0

        num_sections = max(len(inputs) / self.CHUNK_SIZE, 1)
        for chunk_id, chunk in enumerate(
            np.array_split(inputs[: len(inputs)], num_sections)
        ):
            for i, input_i in enumerate(chunk):
                # Fetch few-shot seed examples
                # TODO: Check if this needs to use the example selector
                examples = self.config.get("seed_examples", [])

                final_prompt = self.task.construct_prompt(input_i, examples)
                num_tokens = calculate_num_tokens(self.config, final_prompt)
                total_tokens += num_tokens
                prompt_list.append(final_prompt)
        total_cost = calculate_cost(self.config, total_tokens)
        print(f"Total Estimated Cost: ${round(total_cost, 3)}")
        print(f"Number of examples to label: {len(inputs)}")
        print(f"Average cost per example: ${round(total_cost/len(inputs), 5)}")
        print(f"\n\nA prompt example:\n\n{prompt_list[0]}")
        return

    def set_cache(self):
        # Set cache for langchain
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    def test(self):
        return
