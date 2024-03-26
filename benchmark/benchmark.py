# Before running this file run the following command in the same folder as this file:
# aws s3 cp --recursive s3://autolabel-benchmarking data

import json
from argparse import ArgumentParser
from rich.console import Console

from autolabel import LabelingAgent, AutolabelConfig, AutolabelDataset
from autolabel.tasks import TaskFactory

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
    "gpt-4-1106-preview": "openai",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-sonnet-20240229": "anthropic",
    "mistralai/Mistral-7B-Instruct-v0.1": "vllm",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "vllm",
    "01-ai/Yi-34B-Chat": "vllm",
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
    agent = None
    for dataset in DATASETS:
        config = json.load(open(f"configs/{dataset}.json", "r"))
        config["model"]["name"] = args.model
        config["model"]["provider"] = MODEL_TO_PROVIDER[args.model]
        config['model']['params'] = {"tensor_parallel_size": 4}
        config["prompt"]["few_shot_num"] = args.few_shot
        if not args.few_shot:
            config["prompt"]["few_shot_selection"] = "fixed"

        if not agent:
            agent = LabelingAgent(config, console_output=False, use_tqdm=True)
        else:
            config = AutolabelConfig(config)
            agent.config = config
            agent.task =  TaskFactory.from_config(config)
            agent.llm.config = config
            agent.example_selector = None
        ds = AutolabelDataset(f"data/{dataset}/test.csv", config=config)
        print("Benchmarking", dataset)
        new_ds = agent.run(ds, max_items=args.max_items)
        eval_result.append([x.dict() for x in agent.eval_result])
        json.dump(eval_result, open(eval_file_name, "w"))
        print(eval_result[-1])
        # if config["model"]["provider"] == "vllm":
        #     agent.llm.destroy()

    print(eval_result)


if __name__ == "__main__":
    main()
