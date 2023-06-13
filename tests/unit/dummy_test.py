from autolabel import LabelingAgent


def test_labeling_agent_config():
    agent = LabelingAgent(config="assets/testing/config_banking.json")
    assert agent.config != None


def test_mock_example(mocker):
    agent = LabelingAgent(config="assets/testing/config_banking.json")
    mocker.patch("autolabel.labeler.LabelingAgent.run", return_value=True)
    res = agent.run("assets/testing/banking_test.csv")
    assert res == True
