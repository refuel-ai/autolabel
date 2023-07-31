import json

from autolabel.configs import AutolabelConfig
from autolabel.dataset import AutolabelDataset
from autolabel.few_shot import ExampleSelectorFactory
from langchain.embeddings import HuggingFaceEmbeddings
from pytest import approx

BANKING_HF_EMBEDDINGS_CONFIG = json.load(
    open("tests/assets/banking/config_banking_hf_embeddings.json", "r")
)
BANKING_CONFIG = json.load(open("tests/assets/banking/config_banking.json", "r"))


def test_embedding_provider():
    config = AutolabelConfig(BANKING_HF_EMBEDDINGS_CONFIG)
    seed_examples = config.few_shot_example_set()
    dataset = AutolabelDataset("tests/assets/banking/test.csv", config, 5, 0)
    seed_loader = AutolabelDataset(seed_examples, config)
    seed_examples = seed_loader.inputs
    example_selector = ExampleSelectorFactory.initialize_selector(
        config, seed_examples, dataset.df.keys().tolist()
    )
    assert isinstance(
        example_selector.vectorstore._embedding_function, HuggingFaceEmbeddings
    )


def test_embedding_provider_config_exists():
    config = AutolabelConfig(BANKING_HF_EMBEDDINGS_CONFIG)
    embedding_provider = config.embedding_provider()
    assert embedding_provider == "huggingface_pipeline"


def test_embedding_provider_config_default():
    config = AutolabelConfig(BANKING_CONFIG)
    embedding_provider = config.embedding_provider()
    assert embedding_provider == "openai"
