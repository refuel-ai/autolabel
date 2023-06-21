from autolabel import LabelingAgent
from autolabel.dataset_loader import DatasetLoader
from pandas import DataFrame

csv_path = "tests/assets/banking/test.csv"
jsonl_path = "tests/assets/banking/test.jsonl"
config_path = "tests/assets/banking/config_banking.json"


def test_read_csv():
    agent = LabelingAgent(config=config_path)
    dataset_loader = DatasetLoader(csv_path, agent.config)
    # test return types
    assert isinstance(dataset_loader, DatasetLoader)
    assert isinstance(dataset_loader.dat, DataFrame)
    assert isinstance(dataset_loader.inputs, list)
    assert (
        isinstance(dataset_loader.gt_labels, list) or dataset_loader.gt_labels is None
    )
    # test reading_csv with max_items = 5, start_index = 5
    dataset_loader_max_5_index_5 = DatasetLoader(
        csv_path, agent.config, max_items=5, start_index=5
    )
    assert dataset_loader_max_5_index_5.dat.shape[0] == 5
    assert dataset_loader_max_5_index_5.dat.iloc[0].equals(dataset_loader.dat.iloc[5])
    assert len(dataset_loader_max_5_index_5.inputs) == 5
    assert len(dataset_loader_max_5_index_5.gt_labels) == 5


def test_read_dataframe():
    agent = LabelingAgent(config=config_path)
    df = DatasetLoader(csv_path, agent.config).dat
    dataset_loader = DatasetLoader(df, agent.config)
    # test return types
    assert isinstance(dataset_loader, DatasetLoader)
    assert isinstance(dataset_loader.dat, DataFrame)
    assert isinstance(dataset_loader.inputs, list)
    assert (
        isinstance(dataset_loader.gt_labels, list) or dataset_loader.gt_labels is None
    )
    # confirm data matches
    assert df.equals(dataset_loader.dat)
    # test loading data with max_items = 5, start_index = 5
    dataset_loader_max_5_index_5 = DatasetLoader(
        df, agent.config, max_items=5, start_index=5
    )
    assert dataset_loader_max_5_index_5.dat.shape[0] == 5
    assert dataset_loader_max_5_index_5.dat.iloc[0].equals(dataset_loader.dat.iloc[5])
    assert len(dataset_loader_max_5_index_5.inputs) == 5
    assert len(dataset_loader_max_5_index_5.gt_labels) == 5


def test_read_jsonl():
    agent = LabelingAgent(config=config_path)
    dataset_loader = DatasetLoader(jsonl_path, agent.config)
    # test return types
    assert isinstance(dataset_loader, DatasetLoader)
    assert isinstance(dataset_loader.dat, DataFrame)
    assert isinstance(dataset_loader.inputs, list)
    assert (
        isinstance(dataset_loader.gt_labels, list) or dataset_loader.gt_labels is None
    )
    # test reading_csv with max_items = 5, start_index = 5
    dataset_loader_max_5_index_5 = DatasetLoader(
        jsonl_path, agent.config, max_items=5, start_index=5
    )
    assert dataset_loader_max_5_index_5.dat.shape[0] == 5
    assert dataset_loader_max_5_index_5.dat.iloc[0].equals(dataset_loader.dat.iloc[5])
    assert len(dataset_loader_max_5_index_5.inputs) == 5
    assert len(dataset_loader_max_5_index_5.gt_labels) == 5
