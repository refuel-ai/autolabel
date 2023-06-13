from autolabel import LabelingAgent


def test_labeling_agent_chunk_size():
    agent = LabelingAgent(config="examples/banking/config_banking.json")
    assert agent.config != None
