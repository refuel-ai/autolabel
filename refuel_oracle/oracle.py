import pandas as pd
import numpy as np

from refuel_oracle.llm import LLMFactory
from refuel_oracle.config import Config
from refuel_oracle.tasks import TaskFactory
from refuel_oracle.utils import (
    calculate_num_tokens,
    calculate_cost
)

class Oracle:
    CHUNK_SIZE = 5

    def __init__(self, config: str, debug: bool = False) -> None:
        self.debug = debug
        self.config = Config.from_json(config)
        self.llm = LLMFactory.from_config(self.config)
        self.task = TaskFactory.from_config(self.config)

    def annotate(
        self,
        dataset: str,
        input_column: str,
        ground_truth_column: str = None,
        delimiter: str = ',',
        max_items: int = 100,
    ) -> None:
        dat = pd.read_csv(dataset, sep=delimiter)
        max_items = min(max_items, len(dat))

        input = dat[:max_items][input_column].tolist()
        truth = None if not ground_truth_column else dat[:max_items][ground_truth_column].tolist()

        llm_labels = []
        prompt_list = []
        total_tokens = 0

        num_sections = max(max_items / self.CHUNK_SIZE, 1)
        for chunk_id, chunk in enumerate(np.array_split(input, num_sections)):
            final_prompts = []
            for i, input_i in enumerate(chunk):
                if (i + 1) % 10 == 0:
                    print(f"{i+1}/{len(input)}...")

                # Fetch few-shot seed examples
                # In the future this task will be delegated to an example selector
                examples = self.config["seed_examples"]

                # Construct Prompt to pass to LLM
                final_prompt = self.task.construct_prompt(input_i, examples)
                final_prompts.append(final_prompt)
                num_tokens = calculate_num_tokens(
                    self.config,
                    final_prompt
                )
                total_tokens += num_tokens
                prompt_list.append(final_prompt)
            # Get response from LLM
            response = self.llm.generate(final_prompts)
            for response_item in response.generations:
                generation = response_item[0]
                llm_labels.append(self.task.parse_llm_response(generation))

        # if true labels are provided, evaluate accuracy of predictions
        if truth:
            eval_result = self.task.eval(llm_labels, truth)
            # TODO: serialize and write to file
            for m in eval_result:
                print(f"Metric: {m.name}: {m.value}")

        # Write output to CSV
        output_df = dat[:max_items].copy()
        output_df["llm_labeled_successfully"] = [l.successfully_labeled for l in llm_labels]
        output_df["llm_label"] = [l.label for l in llm_labels]
        output_df.to_csv(f"{dataset}_labeled.csv", sep=delimiter, header=True, index=False)

    def plan(
        self,
        dataset: str,
        input_column: str,
    ):
        dat = pd.read_csv(dataset)
        input = dat[input_column].tolist()
        prompt_list = []
        total_tokens = 0

        num_sections = max(len(input) / self.CHUNK_SIZE, 1)
        for chunk_id, chunk in enumerate(
            np.array_split(input[: len(input)], num_sections)
        ):
            for i, input_i in enumerate(chunk):
                if (i + 1) % 10 == 0:
                    print(f"{i+1}/{len(input)}...")
                # Fetch few-shot seed examples
                examples = self.config["seed_examples"]

                final_prompt = self.task.construct_prompt(input_i, examples)
                num_tokens = calculate_num_tokens(
                    self.config, final_prompt
                )
                total_tokens += num_tokens
                prompt_list.append(final_prompt)
        total_cost = calculate_cost(self.config, total_tokens)
        print(f"Total Estimated Cost: {total_cost}")
        print(f"Number of examples to label: {len(input)}")
        print(f"Average cost per example: {total_cost/len(input)}")
        print(f"\n\nFinal prompt example:\n\n{prompt_list[0]}")
        return

    def test(self):
        return
