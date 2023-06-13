from autolabel import LabelingAgent
from autolabel.dataset_loader import DatasetLoader
from pandas import DataFrame


def test_read_csv():
    csv_path = "assets/banking/test.csv"
    agent = LabelingAgent(config="assets/banking/config_banking.json")
    data = DatasetLoader.read_csv(csv_path, agent.config)
    # test return types
    assert isinstance(data, tuple)
    assert isinstance(data[0], DataFrame)
    assert isinstance(data[1], list)
    assert isinstance(data[2], list) or data[2] is None
    # test reading_csv with max_items = 5, start_index = 5
    data_max_5_index_5 = DatasetLoader.read_csv(
        csv_path, agent.config, max_items=5, start_index=5
    )
    assert data_max_5_index_5[0].shape[0] == 5
    assert data_max_5_index_5[0].iloc[0].equals(data[0].iloc[5])
    assert len(data_max_5_index_5[1]) == 5
    assert len(data_max_5_index_5[2]) == 5
    return True

def test_read_dataframe():
    csv_path = "assets/banking/test.csv"
    agent = LabelingAgent(config="assets/banking/config_banking.json")
    df, _, _ = DatasetLoader.read_csv(csv_path, agent.config)
    data = DatasetLoader.read_dataframe(df, agent.config)
    # test return types
    assert isinstance(data, tuple)
    assert isinstance(data[0], DataFrame)
    assert isinstance(data[1], list)
    assert isinstance(data[2], list) or data[2] is None
    # confirm data matches
    assert df.equals(data[0])
    # test loading data with max_items = 5, start_index = 5
    data_max_5_index_5 = DatasetLoader.read_dataframe(
        df, agent.config, max_items=5, start_index=5
    )
    assert data_max_5_index_5[0].shape[0] == 5
    assert data_max_5_index_5[0].iloc[0].equals(data[0].iloc[5])
    assert len(data_max_5_index_5[1]) == 5
    assert len(data_max_5_index_5[2]) == 5
    return True

def test_read_jsonl():
    jsonl_path = "assets/banking/test.jsonl"
    agent = LabelingAgent(config="assets/banking/config_banking.json")
    data = DatasetLoader.read_jsonl(jsonl_path, agent.config)
    # test return types
    assert isinstance(data, tuple)
    assert isinstance(data[0], DataFrame)
    assert isinstance(data[1], list)
    assert isinstance(data[2], list) or data[2] is None
    # test reading_csv with max_items = 5, start_index = 5
    data_max_5_index_5 = DatasetLoader.read_jsonl(
        jsonl_path, agent.config, max_items=5, start_index=5
    )
    assert data_max_5_index_5[0].shape[0] == 5
    assert data_max_5_index_5[0].iloc[0].equals(data[0].iloc[5])
    assert len(data_max_5_index_5[1]) == 5
    assert len(data_max_5_index_5[2]) == 5
    return True

print(f'test_read_csv        :: {test_read_csv()}')
print(f'test_read_dataframe  :: {test_read_dataframe()}')
print(f'test_read_jsonl      :: {test_read_jsonl()}')
