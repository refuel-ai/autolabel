from autolabel import LabelingAgent
from langchain.llms.fake import FakeListLLM


def test_labeling_agent_config():
    agent = LabelingAgent(config="examples/banking/config_banking.json")
    assert agent.config != None


def test_mock_example(mocker):
    agent = LabelingAgent(config="examples/banking/config_banking.json")
    mocker.patch("autolabel.labeler.LabelingAgent.run", return_value=True)
    res = agent.run("examples/banking/test.csv")
    assert res == True
