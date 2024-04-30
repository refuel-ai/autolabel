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
import pylcs

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
                curr_pred = llm_labels[i].label.split(" ")
                predictions.extend(construct_binary_preds(curr_input, curr_pred))
                # print(e, llm_labels[i].label)
                # predictions.extend([0 for _ in range(len(curr_input))])
            try:
                curr_gt = " ".join(ast.literal_eval(gt_labels[i])).split(" ")
            except Exception as e:
                curr_gt = gt_labels[i].split(" ")
            gt.extend(construct_binary_preds(curr_input, curr_gt))
        return [MetricResult(name="F1", value=f1_score(gt, predictions))]

class TextSimilarity(BaseMetric):
    def compute(
        llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        def get_similarity(str_a, str_b):
            substring_lengths = pylcs.lcs_string_length(str_a, str_b)
            return substring_lengths / max(len(str_a), len(str_b)) 
        
        text_similarity = []
        for i in range(len(llm_labels)):
            text_similarity.append(get_similarity(llm_labels[i].label, gt_labels[i]))
        return [MetricResult(name="TextSimilarity", value=sum(text_similarity)/len(text_similarity))]

NUM_GPUS = torch.cuda.device_count()
# NER_DATASETS = ["favespro", "acronym", "numeric", "multiconer", "quoref", "conll2003"]
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
#     "teachfx",
#     "pathstream",
#     "goodfit",
#     "harmonic"
# ]
DATASETS = []
NER_DATASETS = ["acronym"]
ALL_DATASETS = DATASETS + NER_DATASETS
FEW_SHOT_OVERRIDES = {
    "company": 4,
    "squad_v2": 6,
    "quail": 4,
    "quoref": 2,
    "goodfit": 1,
    "harmonic": 6
}
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
    "/workspace/7m_v2_9500": "vllm",
    "/workspace/7m_v2_21000": "vllm",
    "/workspace/7m_v1_llama3_10500": "vllm",
    "NousResearch/Meta-Llama-3-70B-Instruct": "vllm",
    "/workspace/7m_v1_llama3_21500": "vllm"
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
    for index, dataset in enumerate(ALL_DATASETS):
        config = json.load(open(f"configs/{dataset}.json", "r"))
        config["model"]["name"] = args.model
        config["model"]["provider"] = MODEL_TO_PROVIDER[args.model]
        if MODEL_TO_PROVIDER[args.model] == "openai" or MODEL_TO_PROVIDER[args.model] == "vllm":
            config["model"]["compute_confidence"] = True
        if MODEL_TO_PROVIDER[args.model] == "vllm":
            config["model"]["params"] = {
                "tensor_parallel_size": NUM_GPUS,
                "max_tokens": 1024,
                "temperature": 0.05,
                "top_p": 0.999999999999,
            }
        config["prompt"]["few_shot_num"] = FEW_SHOT_OVERRIDES[dataset] if dataset in FEW_SHOT_OVERRIDES else args.few_shot
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
            if dataset in NER_DATASETS:
                agent.confidence.score_type = "logprob_average_per_key"
            else:
                agent.confidence.score_type = "logprob_average"
        ds = AutolabelDataset(f"data/{dataset}/test.csv", config=config)
        print("Benchmarking", dataset)
        additional_metrics = [F1Metric, TextSimilarity] if dataset in NER_DATASETS else []
        new_ds = agent.run(
            ds, max_items=args.max_items, additional_metrics=additional_metrics
        )
        new_ds.df.to_csv(f"outputs_{dataset}.csv")
        eval_result.append([x.dict() for x in agent.eval_result])
        json.dump(eval_result, open(eval_file_name, "w"))
        print(eval_result[-1])

    print(eval_result)


if __name__ == "__main__":
    main()
