from autolabel import LabelingAgent
from autolabel.configs import AutolabelConfig


def test_labeling_agent_config():
    agent = LabelingAgent(
        config=AutolabelConfig("tests/assets/banking/config_banking.json")
    )
    assert agent.config != None


def test_mock_example(mocker):
    agent = LabelingAgent(
        config=AutolabelConfig("tests/assets/banking/config_banking.json")
    )
    mocker.patch("autolabel.labeler.LabelingAgent.run", return_value=True)
    res = agent.run("tests/assets/banking/banking_test.csv")
    assert res == True
