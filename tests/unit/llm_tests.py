from langchain.schema import Generation, LLMResult, HumanMessage

from autolabel.models.anthropic import AnthropicLLM
from autolabel.models.openai import OpenAILLM
from autolabel.models.palm import PaLMLLM
from autolabel.models.refuel import RefuelLLM
from autolabel.configs import AutolabelConfig


################### ANTHROPIC TESTS #######################
def test_anthropic_initialization():
    model = AnthropicLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_anthropic.json")
    )


def test_anthropic_label(mocker):
    model = AnthropicLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_anthropic.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatAnthropic.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x[0].generations] == ["Answers", "Answers"]
    assert x[1] == 0.00010944


def test_anthropic_get_cost():
    model = AnthropicLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_anthropic.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == 0.03271306


def test_anthropic_return_probs():
    model = AnthropicLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_anthropic.json")
    )
    assert model.returns_token_probs() is False


################### ANTHROPIC TESTS #######################


################### OPENAI GPT 3.5 TESTS #######################
def test_gpt35_initialization():
    model = OpenAILLM(
        config=AutolabelConfig(config="assets/testing/config_banking.json")
    )


def test_gpt35_label(mocker):
    model = OpenAILLM(
        config=AutolabelConfig(config="assets/testing/config_banking.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatOpenAI.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x[0].generations] == ["Answers", "Answers"]
    assert x[1] == 1.2e-05


def test_gpt35_get_cost():
    model = OpenAILLM(
        config=AutolabelConfig(config="assets/testing/config_banking.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == 0.002006


def test_gpt35_return_probs():
    model = OpenAILLM(
        config=AutolabelConfig(config="assets/testing/config_banking.json")
    )
    assert model.returns_token_probs() is False


################### OPENAI GPT 3.5 TESTS #######################


################### OPENAI GPT 4 TESTS #######################
def test_gpt4_initialization():
    model = OpenAILLM(
        config=AutolabelConfig(config="assets/testing/config_banking_gpt4.json")
    )


def test_gpt4_label(mocker):
    model = OpenAILLM(
        config=AutolabelConfig(config="assets/testing/config_banking_gpt4.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatOpenAI.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x[0].generations] == ["Answers", "Answers"]
    assert x[1] == 0.00023999999999999998


def test_gpt4_get_cost():
    model = OpenAILLM(
        config=AutolabelConfig(config="assets/testing/config_banking_gpt4.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == 0.06009


def test_gpt4_return_probs():
    model = OpenAILLM(
        config=AutolabelConfig(config="assets/testing/config_banking_gpt4.json")
    )
    assert model.returns_token_probs() is False


################### OPENAI GPT 4 TESTS #######################


################### PALM TESTS #######################
def test_palm_initialization():
    model = PaLMLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_palm.json")
    )


def test_palm_label(mocker):
    model = PaLMLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_palm.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.llms.VertexAI.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x[0].generations] == ["Answers", "Answers"]
    assert x[1] == 9.999999999999999e-06


def test_palm_get_cost():
    model = PaLMLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_palm.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == 1.9999999999999998e-05


def test_palm_return_probs():
    model = PaLMLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_palm.json")
    )
    assert model.returns_token_probs() is False


################### PALM TESTS #######################


################### REFUEL TESTS #######################
def test_refuel_initialization():
    model = RefuelLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_refuel.json")
    )


def test_refuel_label(mocker):
    class PostRequestMockResponse:
        def __init__(self, resp):
            self.resp = resp

        def json(self):
            return {"body": self.resp}

        def raise_for_status(self):
            pass

    model = RefuelLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_refuel.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "requests.post",
        return_value=PostRequestMockResponse(resp='"Answers"'),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x[0].generations] == ["Answers", "Answers"]
    assert x[1] == 0


def test_refuel_get_cost():
    model = RefuelLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_refuel.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == 00


def test_refuel_return_probs():
    model = RefuelLLM(
        config=AutolabelConfig(config="assets/testing/config_banking_refuel.json")
    )
    assert model.returns_token_probs() is False


################### REFUEL TESTS #######################
