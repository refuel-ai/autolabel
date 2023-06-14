from autolabel import LabelingAgent


def test_config():
    agent0 = LabelingAgent(config="tests/assets/banking/config_banking.json")
    agent1 = LabelingAgent(config="tests/assets/conll2003/config_conll2003.json")
    agent2 = LabelingAgent(config="tests/assets/squad_v2/config_squad_v2.json")
