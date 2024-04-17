# Before running this file run the following command in the same folder as this file:
# aws s3 cp --recursive s3://autolabel-benchmarking data

import json
from argparse import ArgumentParser
from rich.console import Console
from typing import List
import json
import pickle as pkl
import ast
from sklearn.metrics import f1_score
import torch

from autolabel import LabelingAgent, AutolabelConfig, AutolabelDataset
from autolabel.tasks import TaskFactory
from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult


class F1Metric(BaseMetric):
    def compute(
        llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        def construct_binary_preds(curr_input: List[str], positive_tokens: List[str]):
            curr_token_index = 0
            binary_preds = [0 for _ in range(len(curr_input))]
            while curr_token_index < len(positive_tokens):
                for i in range(len(curr_input)):
                    if (
                        curr_input[i] in positive_tokens[curr_token_index]
                        or positive_tokens[curr_token_index] in curr_input[i]
                    ) and binary_preds[i] != 1:
                        binary_preds[i] = 1
                        curr_token_index += 1
                        break
                curr_token_index += 1
            return binary_preds

        predictions, gt = [], []
        for i in range(len(llm_labels)):
            curr_input = pkl.loads(llm_labels[i].curr_sample)["example"].split(" ")
            try:
                curr_pred = " ".join(ast.literal_eval(llm_labels[i].label)).split(" ")
                predictions.extend(construct_binary_preds(curr_input, curr_pred))
            except Exception as e:
                print(e, llm_labels[i].label)
                predictions.extend([0 for _ in range(len(curr_input))])
            curr_gt = " ".join(ast.literal_eval(gt_labels[i])).split(" ")
            gt.extend(construct_binary_preds(curr_input, curr_gt))
        return [MetricResult(name="F1", value=f1_score(gt, predictions))]


NUM_GPUS = torch.cuda.device_count()
NER_METRICS = set(["Macro:accuracy", "Macro:F1"])
# NER_DATASETS = []
DATASETS = []
# NER_DATASETS = ["acronym", "numeric", "multiconer", "quoref", "conll2003"]
# NER_DATASETS = ["quoref"]
NER_DATASETS = ["acronym"]
# DATASETS = [
#     "civil_comments",
#     "banking",
#     "company",
#     "craigslist",
#     "ledgar",
#     "lexical_relation",
#     "math",
#     "sciq",
#     "squad_v2",
#     "walmart_amazon",
#     "quail",
#     "diagnosis",
#     "belebele",
# ]
ALL_DATASETS = DATASETS + NER_DATASETS
MODEL_TO_PROVIDER = {
    "gpt-3.5-turbo": "openai",
    "gpt-4": "openai",
    "gpt-4-1106-preview": "openai",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-sonnet-20240229": "anthropic",
    "mistralai/Mistral-7B-Instruct-v0.1": "vllm",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "vllm",
    "01-ai/Yi-34B-Chat": "vllm",
    "/workspace/refuel_llm_v2_1000": "vllm",
    "/workspace/gcp_run_2000": "vllm",
}


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--few-shot", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=200)
    args = parser.parse_args()

    eval_file_name = f"eval_{args.model}_{args.few_shot}_{args.max_items}.json"
    eval_file_name = eval_file_name.replace("/", "")
    eval_result = []
    agent = None
    for dataset in ALL_DATASETS:
        config = json.load(open(f"configs/{dataset}.json", "r"))
        config["model"]["name"] = args.model
        config["model"]["provider"] = MODEL_TO_PROVIDER[args.model]
        if MODEL_TO_PROVIDER[args.model] == "vllm":
            config["model"]["params"] = {
                "tensor_parallel_size": NUM_GPUS,
                "max_tokens": 1024,
                "temperature": 0.01,
                "top_p": 0.999999999999,
            }
        config["prompt"]["few_shot_num"] = args.few_shot
        if not args.few_shot:
            config["prompt"]["few_shot_selection"] = "fixed"

        if not agent:
            agent = LabelingAgent(config, console_output=False, use_tqdm=True)
        else:
            config = AutolabelConfig(config)
            agent.config = config
            agent.task = TaskFactory.from_config(config)
            agent.llm.config = config
            agent.example_selector = None
        ds = AutolabelDataset(f"data/{dataset}/test.csv", config=config)
        print("Benchmarking", dataset)
        additional_metrics = [F1Metric] if dataset in NER_DATASETS else []
        new_ds = agent.run(
            ds, max_items=args.max_items, additional_metrics=additional_metrics
        )
        eval_result.append([x.dict() for x in agent.eval_result])
        json.dump(eval_result, open(eval_file_name, "w"))
        print(eval_result[-1])

    print(eval_result)


if __name__ == "__main__":
    main()
