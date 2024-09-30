from autolabel import LabelingAgent
from autolabel.dataset import AutolabelDataset
from pandas import DataFrame
from autolabel.configs import AutolabelConfig

csv_path = "tests/assets/banking/test.csv"
jsonl_path = "tests/assets/banking/test.jsonl"
config_path = "tests/assets/banking/config_banking.json"


def test_read_csv():
    agent = LabelingAgent(config=AutolabelConfig(config_path))
    dataset = AutolabelDataset(csv_path, agent.config)
    # test return types
    assert isinstance(dataset, AutolabelDataset)
    assert isinstance(dataset.df, DataFrame)
    assert isinstance(dataset.inputs, list)
    assert isinstance(dataset.gt_labels, list) or dataset.gt_labels is None
    # test reading_csv with max_items = 5, start_index = 5
    dataset_loader_max_5_index_5 = AutolabelDataset(
        csv_path, agent.config, max_items=5, start_index=5
    )
    assert dataset_loader_max_5_index_5.df.shape[0] == 5
    assert dataset_loader_max_5_index_5.df.iloc[0].equals(dataset.df.iloc[5])
    assert len(dataset_loader_max_5_index_5.inputs) == 5


def test_read_dataframe():
    agent = LabelingAgent(config=AutolabelConfig(config_path))
    df = AutolabelDataset(csv_path, agent.config).df
    dataset = AutolabelDataset(df, agent.config)
    # test return types
    assert isinstance(dataset, AutolabelDataset)
    assert isinstance(dataset.df, DataFrame)
    assert isinstance(dataset.inputs, list)
    assert isinstance(dataset.gt_labels, list) or dataset.gt_labels is None
    # confirm data matches
    assert df.equals(dataset.df)
    # test loading data with max_items = 5, start_index = 5
    dataset_loader_max_5_index_5 = AutolabelDataset(
        df, agent.config, max_items=5, start_index=5
    )
    assert dataset_loader_max_5_index_5.df.shape[0] == 5
    assert dataset_loader_max_5_index_5.df.iloc[0].equals(dataset.df.iloc[5])
    assert len(dataset_loader_max_5_index_5.inputs) == 5


def test_read_jsonl():
    agent = LabelingAgent(config=AutolabelConfig(config_path))
    dataset = AutolabelDataset(jsonl_path, agent.config)
    # test return types
    assert isinstance(dataset, AutolabelDataset)
    assert isinstance(dataset.df, DataFrame)
    assert isinstance(dataset.inputs, list)
    assert isinstance(dataset.gt_labels, list) or dataset.gt_labels is None
    # test reading_csv with max_items = 5, start_index = 5
    dataset_loader_max_5_index_5 = AutolabelDataset(
        jsonl_path, agent.config, max_items=5, start_index=5
    )
    assert dataset_loader_max_5_index_5.df.shape[0] == 5
    assert dataset_loader_max_5_index_5.df.iloc[0].equals(dataset.df.iloc[5])
    assert len(dataset_loader_max_5_index_5.inputs) == 5
