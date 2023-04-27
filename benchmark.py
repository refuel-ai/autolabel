import time
from typing import List
from refuel_oracle.oracle import Oracle
from data.get_data import SUPPORTED_DATASETS, DATASET_TASK_PATH


class Benchmark:
    def __init__(self):
        pass

    def run(
        self,
        datasets: List[str] = None,
        model_config_path="examples/configs/llm_configs/gpt_3.5_turbo.json",
        max_items=100,
        **kwargs,
    ):
        if datasets is None:
            datasets = list(SUPPORTED_DATASETS.keys())
        for i, dataset_name in enumerate(datasets):
            assert (
                dataset_name in DATASET_TASK_PATH
            ), f"Dataset {dataset_name} does not have a configured task config"
            task_config_path = (
                f"examples/configs/task_configs/{DATASET_TASK_PATH[dataset_name]}"
            )
            dataset_config_path = (
                f"examples/configs/dataset_configs/{dataset_name}.json"
            )
            print(
                f"Running {dataset_name} with task config {task_config_path}, dataset config {dataset_config_path} and model config {model_config_path}"
            )
            # Only load the llm the first time
            if i == 0:
                annotator = Oracle(task_config_path, model_config_path)
            else:
                annotator.set_task_config(task_config_path, **kwargs)
            annotator.plan(f"data/{dataset_name}_test.csv", dataset_config_path)
            start_time = time.time()
            annotator.annotate(
                f"data/{dataset_name}_test.csv",
                dataset_config_path,
                max_items=max_items,
            )
            time_taken = time.time() - start_time
            print(f"Time taken for {dataset_name}: {time_taken} seconds")


if __name__ == "__main__":
    Benchmark().run(max_items=10)
