# Before running this file run the following command in the same folder as this file:
# aws s3 cp --recursive s3://autolabel-benchmarking data

import json
from argparse import ArgumentParser
from rich.console import Console

from autolabel import LabelingAgent, AutolabelConfig, AutolabelDataset

DATASETS = [
    "civil_comments",
    "banking",
    "company",
    "conll2003",
    "craigslist",
    "ledgar",
    "lexical_relation",
    "math",
    "quoref",
    "sciq",
    "squad_v2",
    "walmart_amazon",
    "quail",
    "acronym",
    "numeric",
    "diagnosis",
    "belebele",
    "multiconer",
]

MODEL_TO_PROVIDER = {
    "gpt-3.5-turbo": "openai",
    "gpt-4": "openai",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-sonnet-20240229": "anthropic",
    "mistralai/Mistral-7B-v0.1": "vllm",
    "mistralai/Mistral-7B-Instruct-v0.1": "vllm",
    "mistralai/Mixtral-8x7B-v0.1": "mistral",
}

PROMPT_TEMPLATES = {
    "mistralai/Mistral-7B-v0.1": {
        "zero-shot": "<s> [INST] {task_guidelines}\n\n{output_guidelines}\n\nNow I want you to label the following example:\n{current_example} [/INST]",
        "few-shot": "{task_guidelines}\n\n{output_guidelines}\n\nSome examples with their output answers are provided below:\n\n{seed_examples}\n\nNow I want you to label the following example:\n{current_example}",
    },
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "zero-shot": "<s> [INST] {task_guidelines}\n\n{output_guidelines}\n\nNow I want you to label the following example:\n{current_example} [/INST]",
        "few-shot": "<s> [INST] {task_guidelines}\n\n{output_guidelines}\n\nSome examples with their output answers are provided below:\n\n{seed_examples}\n\nNow I want you to label the following example:\n{current_example} [/INST]",
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "zero-shot": "<s> [INST] {task_guidelines}\n\n{output_guidelines}\n\nNow I want you to label the following example:\n{current_example} [/INST]",
        "few-shot": "<s> [INST] {task_guidelines}\n\n{output_guidelines}\n\nSome examples with their output answers are provided below:\n\n{seed_examples}\n\nNow I want you to label the following example:\n{current_example} [/INST]",
    },
    # Yi models don't have any prompt format as they are base models https://github.com/01-ai/Yi/issues/30
}


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=100)
    args = parser.parse_args()

    eval_file_name = f"eval_{args.model}_{args.few_shot}_{args.max_items}.json"
    eval_file_name = eval_file_name.replace("/", "")
    eval_result = []
    for dataset in DATASETS:
        config = json.load(open(f"configs/{dataset}.json", "r"))
        config["model"]["name"] = args.model
        config["model"]["provider"] = MODEL_TO_PROVIDER[args.model]
        config["prompt"]["few_shot_num"] = args.few_shot
        if not args.few_shot:
            config["prompt"]["few_shot_selection"] = "fixed"

        if args.model in PROMPT_TEMPLATES:
            config["prompt"]["zero_shot_template"] = PROMPT_TEMPLATES[args.model][
                "zero-shot"
            ]
            config["prompt"]["few_shot_template"] = PROMPT_TEMPLATES[args.model][
                "few-shot"
            ]

        agent = LabelingAgent(config, console_output=False)
        ds = AutolabelDataset(f"data/{dataset}/test.csv", config=config)
        print("Benchmarking", dataset)
        new_ds = agent.run(ds, max_items=args.max_items)
        eval_result.append([x.dict() for x in agent.eval_result])
        json.dump(eval_result, open(eval_file_name, "w"))
        print(eval_result[-1])
        if config["model"]["provider"] == "vllm":
            agent.llm.destroy()

    print(eval_result)


if __name__ == "__main__":
    main()
