from refuel_oracle.llm import LLMFactory
from refuel_oracle.config import Config
from refuel_oracle.tasks import TaskFactory
from refuel_oracle.utils import (
    calculate_num_tokens,
    calculate_cost
)

import json
from math import exp
import pandas as pd
import numpy as np
from typing import List
from collections import Counter

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


class Annotation:
    __slots__ = "data", "labels", "confidence"

    def __init__(self, data, labels, confidence) -> None:
        self.data = data
        self.labels = labels
        self.confidence = confidence

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
    ) -> Annotation:
        dat = pd.read_csv(dataset)

        input = dat[input_column].tolist()
        truth = None if not ground_truth_column else dat[ground_truth_column].tolist()

        yes_or_no = []
        llm_labels = []
        logprobs = []
        prompt_list = []
        total_tokens = 0

        max_items = min(max_items, len(input))
        num_sections = max(max_items / CHUNK_SIZE, 1)
        for chunk_id, chunk in enumerate(
            np.array_split(input[:max_items], num_sections)
        ):
            for i, input_i in enumerate(chunk):
                if (i + 1) % 10 == 0:
                    print(f"{i+1}/{len(input)}...")

                # Construct Prompt to pass to LLM
                final_prompt = self.task.construct_prompt(input_i)

                num_tokens = calculate_num_tokens(
                    self.config,
                    final_prompt
                )
                total_tokens += num_tokens
                prompt_list.append(final_prompt)
            # Get response from LLM
            response = self.llm.generate(prompt_list)
            for response_item in response.generations:
                parts = response_item[0].text
                print(parts)
                parts = json.loads(parts.strip())
                yes_or_no.append(parts["answered"])
                llm_labels.append(parts["label"])
                generation_info = response_item[0].generation_info
                logprobs.append(
                    exp(generation_info["logprobs"]["token_logprobs"][-1])
                )

        # if true labels are provided, evaluate accuracy of predictions
        if truth:
            evaluate(true_labels=truth, pred_labels=llm_labels, verbose=self.debug)

        # Write output to CSV
        if len(llm_labels) < len(dat):
            llm_labels = llm_labels + [None for i in range(len(dat) - len(llm_labels))]
        dat[output_column] = llm_labels
        dat.to_csv(output_dataset)

        return Annotation(
            data=input[:max_items], labels=llm_labels[:max_items], confidence=logprobs
        )

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
                final_prompt = self.construct_final_prompt(input_i)
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
