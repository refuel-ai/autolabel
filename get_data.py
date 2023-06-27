import argparse
import urllib.request
import os

EXAMPLE_DATASETS = [
    "banking",
    "civil_comments",
    "ledgar",
    "walmart_amazon",
    "company",
    "squad_v2",
    "sciq",
    "conll2003",
    "twitter_emotion_detection",
]


def get_data(dataset_name):
    if dataset_name not in EXAMPLE_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported")
    print(f"Getting data for {dataset_name}")
    seed_url = f"https://autolabel-benchmarking.s3.us-west-2.amazonaws.com/{dataset_name}/seed.csv"
    test_url = f"https://autolabel-benchmarking.s3.us-west-2.amazonaws.com/{dataset_name}/test.csv"

    # Make directory if it doesn't exist
    os.makedirs(f"examples/{dataset_name}", exist_ok=True)

    urllib.request.urlretrieve(seed_url, f"examples/{dataset_name}/seed.csv")
    urllib.request.urlretrieve(test_url, f"examples/{dataset_name}/test.csv")


def get_all_data(args):
    if args.dataset:
        get_data(args.dataset)
    else:
        for dataset_name in EXAMPLE_DATASETS:
            get_data(dataset_name)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Get data for running autolabel")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset to get data for"
    )
    args = parser.parse_args()
    get_all_data(args)
