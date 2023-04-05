import pandas as pd
import numpy as np
from typing import List
from collections import Counter

from refuel_oracle.llm import LLMFactory
from refuel_oracle.config import Config
from refuel_oracle.tasks import TaskFactory
from refuel_oracle.schema import LLMAnnotation
from refuel_oracle.utils import (
    calculate_num_tokens,
    calculate_cost
)

CHUNK_SIZE = 5

def evaluate(true_labels: List, pred_labels: List, verbose: bool = False):
    # Calculate accuracy of predictions vs provided true labels
    lower_case_true_labels = [s.lower() for s in true_labels]
    true_distribution = Counter(lower_case_true_labels)
    if verbose:
        print("Ground truth label distribution:")
        print(true_distribution)
        print("#" * 80)

    lower_case_preds = [s.lower() for s in pred_labels]
    preds_distribution = Counter(lower_case_preds)
    if verbose:
        print("Predictions distribution:")
        print(preds_distribution)
        print("#" * 80)

    correct = 0
    for i in range(len(lower_case_preds)):
        if lower_case_preds[i] == lower_case_true_labels[i]:
            correct += 1

    print(f'Final number of examples labeled "successfully": {correct}')
    print(f"Accuracy: {correct / len(lower_case_preds)}")
    print("#" * 80)

class Oracle:
    def __init__(self, config: str, debug: bool = False) -> None:
        self.debug = debug
        self.config = Config.from_json(config)
        self.llm = LLMFactory.from_config(self.config)
        self.task = TaskFactory.from_config(self.config)

    def annotate(
        self,
        dataset: str,
        input_column: str,
        output_column: str,
        output_dataset: str = "output.csv",
        ground_truth_column=None,
        n_trials: int = 1,
        max_items: int = 100,
    ) -> None:
        dat = pd.read_csv(dataset)

        input = dat[input_column].tolist()
        truth = None if not ground_truth_column else dat[ground_truth_column].tolist()

        llm_labels = []
        prompt_list = []
        total_tokens = 0

        max_items = min(max_items, len(input))
        num_sections = max(max_items / CHUNK_SIZE, 1)
        for chunk_id, chunk in enumerate(
            np.array_split(input[:max_items], num_sections)
        ):
            final_prompts = []
            for i, input_i in enumerate(chunk):
                if (i + 1) % 10 == 0:
                    print(f"{i+1}/{len(input)}...")

                # Fetch few-shot seed examples
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
            response = self.llm.generate(prompt_list)
            for prompt, response_item in zip(final_prompts, response.generations):
                generation = response_item[0]
                llm_annotation: LLMAnnotation = self.task.parse_llm_response(
                    prompt, generation)
                llm_labels.append(llm_annotation.label)

        # if true labels are provided, evaluate accuracy of predictions
        if truth:
            eval_result = self.task.eval(llm_labels, truth)
            # TODO: serialize and write to file
            for metric, val in eval_result.items():
                print(f"{metric}: {val}")

        # Write output to CSV
        if len(llm_labels) < len(dat):
            llm_labels = llm_labels + [None for i in range(len(dat) - len(llm_labels))]
        dat[output_column] = llm_labels
        dat.to_csv(output_dataset)

    def plan(
        self,
        dataset: str,
        input_column: str,
    ):
        dat = pd.read_csv(dataset)
        input = dat[input_column].tolist()
        prompt_list = []
        total_tokens = 0

        num_sections = max(len(input) / CHUNK_SIZE, 1)
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
        return

    def test(self):
        return
