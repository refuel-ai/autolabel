import pickle

import pandas as pd
from autolabel.configs import AutolabelConfig
from autolabel.dataset_loader import DatasetLoader
from autolabel.few_shot import ExampleSelectorFactory
from autolabel.few_shot.vector_store import VectorStoreWrapper


def test_num_examples(mocker):
    config = AutolabelConfig("assets/testing/config_banking.json")
    _, seed_examples, _ = DatasetLoader.read_csv(
        "assets/testing/banking_test.csv", config
    )
    dataset = pd.read_csv("assets/testing/banking_test.csv")
    df, inputs, _ = DatasetLoader.read_dataframe(dataset, config, len(dataset), 0)
    input_keys = df.keys().tolist()
    example_selector = ExampleSelectorFactory.initialize_selector(
        config, seed_examples, input_keys
    )
    seed_example_texts = [seed_example["example"] for seed_example in seed_examples]
    with open("assets/testing/banking_seed_embeddings.pkl", "rb") as f:
        corpus_embeddings = pickle.load(f)
    mocker.patch(
        "autolabel.few_shot.vector_store.VectorStoreWrapper.from_texts",
        return_value=VectorStoreWrapper(
            embedding_function=None,
            corpus_embeddings=corpus_embeddings,
            texts=seed_example_texts,
            metadatas=seed_examples,
        ),
    )
    examples = example_selector.select_examples(inputs[0])
    assert len(examples) == 10


# def test_num_examples(mocker):
#     config = AutolabelConfig("assets/testing/config_banking.json")
#     _, seed_examples, _ = DatasetLoader.read_csv(
#         "assets/testing/banking_test.csv", config
#     )
#     print(seed_examples, "seed")
#     dataset = pd.read_csv("assets/testing/banking_test.csv")
#     df, inputs, _ = DatasetLoader.read_dataframe(dataset, config, len(dataset), 0)
#     input_keys = df.keys().tolist()
#     example_selector = ExampleSelectorFactory.initialize_selector(
#         config, seed_examples, input_keys
#     )
#     # mocker.patch(
#     #     "langchain.prompts.example_selector.SemanticSimilarityExampleSelector.select_examples",
#     #     return_value=[seed_example["example"] for seed_example in seed_examples],
#     # )
#     examples = example_selector.select_examples(inputs[0])
#     assert len(examples) == 10
