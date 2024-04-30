import argparse
import os
import json
import pandas as pd

METRICS = {
    "classification": ["accuracy", "completion_rate", "auroc"],
    "entity_matching": ["accuracy", "completion_rate", "auroc"],
    "question_answering": ["accuracy", "f1", "auroc"],
    "named_entity_recognition": ["accuracy", "f1_strict", "auroc"],
    "attribute_extraction": [
        "Macro:accuracy",
        "Macro:F1",
        "Macro:TextSimilarity",
        "Macro:auroc",
    ],
    "multilabel_classification": ["accuracy", "f1_weighted", "auroc"],
}

DATASETS = [
    "civil_comments",
    "banking",
    "company",
    "craigslist",
    "ledgar",
    "lexical_relation",
    "math",
    "sciq",
    "squad_v2",
    "walmart_amazon",
    "quail",
    "diagnosis",
    "belebele",
    "teachfx",
    "pathstream",
    "goodfit",
    "harmonic",
    "favespro",
    "acronym",
    "numeric",
    "multiconer",
    "quoref",
    "conll2003",
    "quality",
    "qasper",
    "contract_nli",
    "naturalqa",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    args = parser.parse_args()

    # List all files starting with eval_ in the eval_dir and ends with json
    eval_files = [
        f
        for f in os.listdir(args.eval_dir)
        if f.startswith("eval_") and f.endswith(".json")
    ]
    rows = []
    header_created = False
    header = []
    for file in eval_files:
        d = json.load(open(f"{args.eval_dir}/{file}", "r"))
        row = []
        row.append("-".join(file.split(".")[0].split("/")))
        if not header_created:
            header.append("model")
        for i, dataset in enumerate(DATASETS):
            print(dataset)
            config = json.load(open(f"configs/{dataset}.json", "r"))
            metrics_to_add = METRICS[config["task_type"]]
            for metric_to_add in metrics_to_add:
                for metric in d[i]:
                    if metric["name"] == metric_to_add:
                        row.append(metric["value"])
                        if not header_created:
                            header.append(f"{dataset}_{metric_to_add}")
        header_created = True
        rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    df.to_csv(f"{args.eval_dir}/results.csv", index=False)


if __name__ == "__main__":
    main()
