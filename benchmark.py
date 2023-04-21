import time
from typing import List
from refuel_oracle.oracle import Oracle
from data.get_data import SUPPORTED_DATASETS


class Benchmark:
    def __init__(self):
        pass

    def run(self, datasets: List[str] = None, **kwargs):
        if datasets is None:
            datasets = list(SUPPORTED_DATASETS.keys())
        for i, dataset_name in enumerate(datasets):
            config_path = f"examples/config_{dataset_name}_hf.json"
            print(f"Running {dataset_name} with config {config_path}")
            # Only load the llm the first time
            if i == 0:
                annotator = Oracle(config_path, **kwargs)
            else:
                annotator.set_config(config_path, load_llm=False, **kwargs)
            annotator.plan(f"data/{dataset_name}_test.csv")
            start_time = time.time()
            annotator.annotate(f"data/{dataset_name}_test.csv", max_items=2000)
            time_taken = time.time() - start_time
            print(f"Time taken for {dataset_name}: {time_taken} seconds")


if __name__ == "__main__":
    Benchmark().run(
        datasets=["wikiann"], provider_name="anthropic", model_name="claude-v1"
    )
