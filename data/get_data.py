import os
from datasets import load_dataset
import json
import csv
import gdown
import zipfile
import random
import pandas as pd
import urllib.request


def map_label_to_string(dataset, col):
    label_dict = {k: v for k, v in enumerate(dataset.features[col].names)}

    def add_label(example):
        example["tmpcol"] = label_dict[example[col]]
        return example

    dataset = dataset.map(add_label, remove_columns=[col])
    dataset = dataset.rename_column("tmpcol", col)

    return dataset


def get_ledgar(output_folder="."):
    dataset = load_dataset("lex_glue", "ledgar")

    test_ds = dataset["test"]

    test_ds = map_label_to_string(test_ds, "label")

    test_ds = test_ds.rename_column("text", "description")
    test_ds.to_csv(f"{output_folder}/ledgar_test.csv")


def get_banking(output_folder="."):
    dataset = load_dataset("banking77")
    test_ds = dataset["test"]

    test_ds = map_label_to_string(test_ds, "label")

    test_ds = test_ds.rename_column("text", "description")
    test_ds.to_csv(f"{output_folder}/banking_test.csv")


def get_emotion(output_folder="."):
    dataset = load_dataset("emotion")
    test_ds = dataset["test"]

    test_ds = map_label_to_string(test_ds, "label")

    test_ds = test_ds.rename_column("text", "description")
    test_ds.to_csv(f"{output_folder}/emotion_test.csv")


def get_sciq(output_folder="."):
    test_ds = load_dataset("sciq")["test"]

    def process(example):
        example["answer"] = example["correct_answer"]
        la = [
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
            example["correct_answer"],
        ]
        random.shuffle(la)
        example["options"] = str(la)
        return example

    test_ds = test_ds.map(
        process,
        remove_columns=[
            "distractor3",
            "distractor1",
            "distractor2",
            "correct_answer",
            "support",
        ],
    )

    test_ds.to_csv(f"{output_folder}/sciq_test.csv")


def get_medqa(output_folder="."):
    url = "https://drive.google.com/u/0/uc?id=1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw&export=download"
    output_file = f"{output_folder}/tmp/data_clean.zip"
    gdown.download(url, output_file, quiet=False)

    # Extract the zip file
    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall(f"{output_folder}/tmp/medqa_data")

    dataset = []
    for line in open(
        f"{output_folder}/tmp/medqa_data/data_clean/questions/US/test.jsonl", "r"
    ):
        data = json.loads(line)
        data["options"] = str(list(data["options"].values()))
        del data["meta_info"]
        del data["answer_idx"]
        dataset.append(data)

    keys = dataset[0].keys()

    with open(f"{output_folder}/medqa_test.csv", "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)


def get_pubmed_qa(output_folder="."):
    test_ds = load_dataset("pubmed_qa", "pqa_labeled")["train"]

    def process(ex):
        ex["context"] = "\n".join(ex["context"]["contexts"])
        ex["answer"] = ex["final_decision"]
        ex["options"] = str(["yes", "no", "maybe"])
        return ex

    test_ds = test_ds.map(
        process, remove_columns=["pubid", "long_answer", "final_decision"]
    )
    test_ds.to_csv(f"{output_folder}/pubmed_qa_test.csv")


def get_walmart_amazon(output_folder="."):
    urllib.request.urlretrieve(
        "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/exp_data/tableA.csv",
        f"{output_folder}/tmp/tableA.csv",
    )
    urllib.request.urlretrieve(
        "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/exp_data/tableB.csv",
        f"{output_folder}/tmp/tableB.csv",
    )
    urllib.request.urlretrieve(
        "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/exp_data/test.csv",
        f"{output_folder}/tmp/test.csv",
    )

    test_ds = pd.read_csv(f"{output_folder}/tmp/test.csv")
    tableA = pd.read_csv(f"{output_folder}/tmp/tableA.csv")
    tableB = pd.read_csv(f"{output_folder}/tmp/tableB.csv")

    tableA["text"] = tableA.apply(
        lambda x: f'Title: {x["title"]}; Category: {x["category"]}; Brand: {x["brand"]}; ModelNo: {x["modelno"]}; Price: {x["price"]};',
        axis=1,
    )

    tableB["text"] = tableB.apply(
        lambda x: f'Title: {x["title"]}; Category: {x["category"]}; Brand: {x["brand"]}; ModelNo: {x["modelno"]}; Price: {x["price"]};',
        axis=1,
    )

    test_ds["entity1"] = test_ds.apply(
        lambda x: tableA.iloc[x["ltable_id"]]["text"], axis=1
    )

    test_ds["entity2"] = test_ds.apply(
        lambda x: tableB.iloc[x["rtable_id"]]["text"], axis=1
    )
    test_ds["label"] = test_ds.apply(
        lambda x: "duplicate" if x["label"] == 1 else "not duplicate", axis=1
    )

    dataset = test_ds.drop(columns=["ltable_id", "rtable_id"])
    dataset = dataset.to_dict(orient="records")
    keys = dataset[0].keys()

    with open(
        f"{output_folder}/walmart_amazon_test.csv", "w", newline=""
    ) as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)


SUPPORTED_DATASETS = {
    "ledgar": get_ledgar,
    "banking": get_banking,
    "emotion": get_emotion,
    "sciq": get_sciq,
    "medqa": get_medqa,
    "pubmed_qa": get_pubmed_qa,
    "walmart_amazon": get_walmart_amazon,
}


def get_dataset(dataset, output_folder="."):
    assert dataset in SUPPORTED_DATASETS, f"Dataset {dataset} not supported"
    if dataset == "ledgar":
        get_ledgar(output_folder)
    elif dataset == "banking":
        get_banking(output_folder)
    elif dataset == "emotion":
        get_emotion(output_folder)
    elif dataset == "sciq":
        get_sciq(output_folder)
    elif dataset == "medqa":
        get_medqa(output_folder)
    elif dataset == "pubmed_qa":
        get_pubmed_qa(output_folder)
    elif dataset == "walmart_amazon":
        get_walmart_amazon(output_folder)


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir(f"{file_dir}/tmp"):
        os.mkdir(f"{file_dir}/tmp")

    datasets = os.listdir(file_dir)
    for dataset in SUPPORTED_DATASETS:
        if f"{dataset}_test.csv" not in datasets:
            print(f"Downloading {dataset} dataset")
            get_dataset(dataset, output_folder=file_dir)
