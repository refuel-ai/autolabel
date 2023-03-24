from refuel_oracle.llm import LLM, LLMProvider, LLMResults, OpenAI

import json
import pprint
import pandas as pd


def get_num_tokens_from_string(string: str, text_model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(text_model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class Oracle:
    def __init__(self, config: str, debug: bool = False) -> None:
        self.config = config
        self.parse_config_json(config)
        self.debug = debug
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
        self.provider = data["provider"]
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
    ):
        dat = pd.read_csv(dataset)

        input = dat[input_column].tolist()
        truth = None if not ground_truth_column else dat[ground_truth_column].tolist()

        annotation_instruction = self.instruction
        examples = self.seed_examples
        prompt = self.prompt
        final_prompt = f"""{annotation_instruction}\n{prompt}\n{examples}"""

        num_tokens = get_num_tokens_from_string(final_prompt, self.model_name)
        total_tokens, token_counter = 0, 0
        total_tokens += num_tokens
        token_counter += num_tokens

        yes_or_no = []
        llm_labels = []
        if token_counter > MAX_TOKENS_PER_REQUEST:
            # calling OpenAI and parsing the response
            response = self.llm.generate(prompt_list)
            for response_item in response.choices:
                parts = response_item.text.split("\n")
                yes_or_no.append(parts[0])
                llm_labels.append(parts[-1])

            # refreshing list
            token_counter = num_tokens
            prompt_list = [final_prompt]
        else:
            prompt_list.append(final_prompt)

        # Write output to CSV
        dat[output_column] = llm_labels
        dat.to_csv(output_dataset)
        return

    def plan(self):
        return

    def test(self):
        return
