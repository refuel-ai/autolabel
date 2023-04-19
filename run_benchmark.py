from refuel_oracle.oracle import Oracle

SUPPORTED_DATASETS = [
    # "ledgar",
    # "banking",
    # "emotion",
    # "sciq",
    # "medqa",
    # "pubmed_qa",
    "walmart_amazon",
]


class Benchmark:
    def __init__(self):
        pass

    def run(self, **kwargs):
        for i, dataset_name in enumerate(SUPPORTED_DATASETS):
            config_path = f"examples/config_{dataset_name}_hf.json"
            print(f"Running {dataset_name} with config {config_path}")
            # Only load the llm the first time
            if i == 0:
                annotator = Oracle(config_path, **kwargs)
            else:
                annotator.set_config(config_path, load_llm=False, **kwargs)
            annotator.plan(f"data/{dataset_name}_test.csv")
            annotator.annotate(f"data/{dataset_name}_test.csv", max_items=500)


if __name__ == "__main__":
    Benchmark().run()
