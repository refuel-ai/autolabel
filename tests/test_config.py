from autolabel import LabelingAgent


def test_config():
    agent0 = LabelingAgent(config="assets/banking/config_banking.json")
    agent1 = LabelingAgent(config="assets/civil_comments/config_civil_comments.json")
    agent2 = LabelingAgent(config="assets/company/config_company_match.json")
    agent3 = LabelingAgent(config="assets/conll2003/config_conll2003.json")
    agent4 = LabelingAgent(config="assets/ledgar/config_ledgar.json")
    agent5 = LabelingAgent(config="assets/sciq/config_sciq.json")
    agent6 = LabelingAgent(config="assets/squad_v2/config_squad_v2.json")
    agent7 = LabelingAgent(config="assets/walmart_amazon/config_walmart_amazon.json")
    return True


print(f"test_config :: {test_config()}")
