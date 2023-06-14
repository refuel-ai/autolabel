import pandas as pd
from autolabel import LabelingAgent
from autolabel.configs import AutolabelConfig
from autolabel.dataset_loader import DatasetLoader
from autolabel.few_shot import ExampleSelectorFactory


def test_num_examples(mocker):
    config = AutolabelConfig("assets/testing/config_banking.json")
    _, seed_examples, _ = DatasetLoader.read_csv(
        "assets/testing/banking_test.csv", config
    )
    print(seed_examples, "seed")
    dataset = pd.read_csv("assets/testing/banking_test.csv")
    df, inputs, _ = DatasetLoader.read_dataframe(dataset, config, len(dataset), 0)
    input_keys = df.keys().tolist()
    example_selector = ExampleSelectorFactory.initialize_selector(
        config, seed_examples, input_keys
    )
    mocker.patch(
        "langchain.prompts.example_selector.SemanticSimilarityExampleSelector.select_examples",
        return_value=[seed_example["example"] for seed_example in seed_examples],
    )
    examples = example_selector.select_examples(inputs[0])
    assert len(examples) == 10
