from autolabel import LabelingAgent


def test_config():
    agent0 = LabelingAgent(config="tests/assets/banking/config_banking.json")
    agent1 = LabelingAgent(
        config="tests/assets/civil_comments/config_civil_comments.json"
    )
    agent2 = LabelingAgent(config="tests/assets/company/config_company_match.json")
    agent3 = LabelingAgent(config="tests/assets/conll2003/config_conll2003.json")
    agent4 = LabelingAgent(config="tests/assets/ledgar/config_ledgar.json")
    agent5 = LabelingAgent(config="tests/assets/sciq/config_sciq.json")
    agent6 = LabelingAgent(config="tests/assets/squad_v2/config_squad_v2.json")
    agent7 = LabelingAgent(
        config="tests/assets/walmart_amazon/config_walmart_amazon.json"
    )
