from refuel_oracle.llm import LLM, LLMProvider, LLMResults, OpenAI

import json
import pprint
import pandas as pd
import numpy as np
import tiktoken

CHUNK_SIZE = 5


def get_num_tokens_from_string(string: str, text_model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(text_model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class Oracle:
    def __init__(self, config: str, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        self.parse_config_json(config)
        self.llm = OpenAI(self.model_name)

    def parse_config_json(self, config: str) -> None:
        f = open(config)
        data = json.load(f)

        if self.debug:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(data)

        # TODO , add checks to ensure data is correctly formatted
        self.name = data["project_name"]
        self.task_type = data["task_type"]
        self.provider = data["provider_name"]
        self.model_name = data["model_name"]
        self.instruction = data["instruction"]
        self.labels = data["labels_list"]
        self.prompt = data["prompt"]
        self.seed_examples = data["seed_examples"]
        return

    def annotate(
        self,
        dataset: str,
        input_column: str,
        output_column: str,
        output_dataset: str = "output.csv",
        ground_truth_column=None,
        verbose: bool = False,
        n_trials: int = 1,
        max_items: int = 100,
    ):
        dat = pd.read_csv(dataset)

        input = dat[input_column].tolist()
        truth = None if not ground_truth_column else dat[ground_truth_column].tolist()

        yes_or_no = []
        llm_labels = []
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
                annotation_instruction = self.instruction
                annotation_instruction += (
                    f"\n\nDo you know the answer(YES/NO):\nCategories:\n"
                )
                for label_category in self.labels:
                    annotation_instruction += f"{label_category}\n"
                examples = self.seed_examples
                examples_string = (
                    "Some examples with their categories are provided below:\n\n"
                )
                for example in examples:
                    examples_string += f"Example:\n{example['example']}\n\nDo you know the answer(YES/NO):\n"
                    examples_string += (
                        f"{example['yes_or_no']}\nCATEGORY:\n{example['label']}\n\n"
                    )
                prompt = self.prompt
                instruction = prompt.replace("{example}", input_i)
                final_prompt = (
                    f"""{annotation_instruction}\n{examples_string}\n{instruction}"""
                )
                num_tokens = get_num_tokens_from_string(final_prompt, self.model_name)
                total_tokens += num_tokens
                prompt_list.append(final_prompt)
            # Get response from LLM
            response = self.llm.generate(prompt_list)
            for response_item in response.completions:
                parts = response_item["text"].split("\n")
                yes_or_no.append(parts[0])
                llm_labels.append(parts[-1])

        # Write output to CSV
        if len(llm_labels) < len(dat):
            llm_labels = llm_labels + [None for i in range(len(dat) - len(llm_labels))]
        dat[output_column] = llm_labels
        dat.to_csv(output_dataset)
        return

    def plan(self):
        return

    def test(self):
        return
