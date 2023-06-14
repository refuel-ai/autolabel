from autolabel import LabelingAgent


def test_config():
    agent0 = LabelingAgent(config="tests/assets/banking/config_banking.json")
    config0 = agent0.config
    assert config0.label_column() == "label"
    assert config0.task_type() == "classification"
    assert config0.delimiter() == ","
    agent1 = LabelingAgent(config="tests/assets/conll2003/config_conll2003.json")
    config1 = agent1.config
    assert config1.label_column() == "CategorizedLabels"
    assert config1.text_column() == "example"
    assert config1.task_type() == "named_entity_recognition"
    agent2 = LabelingAgent(config="tests/assets/squad_v2/config_squad_v2.json")
    config2 = agent2.config
    assert config2.label_column() == "answer"
    assert config2.task_type() == "question_answering"
    assert config2.few_shot_example_set() == "seed.csv"
