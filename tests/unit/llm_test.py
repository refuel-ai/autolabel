import json
from autolabel.configs import AutolabelConfig
from autolabel.models.anthropic import AnthropicLLM
from autolabel.models.openai import OpenAILLM
from autolabel.models.palm import PaLMLLM
from autolabel.models.refuel import RefuelLLM
from langchain.schema import Generation, LLMResult
from pytest import approx


################### ANTHROPIC TESTS #######################
def test_anthropic_initialization():
    model = AnthropicLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json"
        )
    )


def test_anthropic_label(mocker):
    model = AnthropicLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json"
        )
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatAnthropic.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
    assert sum(x.costs) == approx(0.00010944, rel=1e-3)


def test_anthropic_get_cost():
    model = AnthropicLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json"
        )
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.03271306, rel=1e-3)


def test_anthropic_return_probs():
    model = AnthropicLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json"
        )
    )
    assert model.returns_token_probs() is False


################### ANTHROPIC TESTS #######################


################### OPENAI GPT 3.5 TESTS #######################
def test_gpt35_initialization():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking.json")
    )


def test_gpt35_label(mocker):
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatOpenAI.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]


def test_gpt35_get_cost():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.002006, rel=1e-3)


def test_gpt35_return_probs():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking.json")
    )
    assert model.returns_token_probs() is False


################### OPENAI GPT 3.5 TESTS #######################


################### OPENAI GPT 4 TESTS #######################
def test_gpt4_initialization():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4.json")
    )


def test_gpt4_label(mocker):
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatOpenAI.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
    assert sum(x.costs) == approx(0.00023999, rel=1e-3)


def test_gpt4_get_cost():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.06009, rel=1e-3)


def test_gpt4_return_probs():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4.json")
    )
    assert model.returns_token_probs() is False


################### OPENAI GPT 4 TESTS #######################


################### PALM TESTS #######################
def test_palm_initialization(mocker):
    mocker.patch(
        "vertexai.preview.language_models.TextGenerationModel.from_pretrained",
        return_value="Test",
    )
    model = PaLMLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_palm.json")
    )


def test_palm_label(mocker):
    mocker.patch(
        "vertexai.preview.language_models.TextGenerationModel.from_pretrained",
        return_value="Test",
    )
    model = PaLMLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_palm.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.llms.VertexAI.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
    assert sum(x.costs) == approx(2.4e-05, rel=1e-3)


def test_palm_get_cost(mocker):
    mocker.patch(
        "vertexai.preview.language_models.TextGenerationModel.from_pretrained",
        return_value="Test",
    )
    model = PaLMLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_palm.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.00402, rel=1e-3)


def test_palm_return_probs(mocker):
    mocker.patch(
        "vertexai.preview.language_models.TextGenerationModel.from_pretrained",
        return_value="Test",
    )
    model = PaLMLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_palm.json")
    )
    assert model.returns_token_probs() is False


################### PALM TESTS #######################


################### REFUEL TESTS #######################
def test_refuel_initialization():
    model = RefuelLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_refuel.json")
    )


def test_refuel_label(mocker):
    class PostRequestMockResponse:
        def __init__(self, resp, status_code):
            self.resp = resp
            self.status_code = status_code

        def json(self):
            return self.resp

        def raise_for_status(self):
            pass

    model = RefuelLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_refuel.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "requests.post",
        return_value=PostRequestMockResponse(
            resp='{"generated_text": "Answers"}', status_code=200
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
    assert sum(x.costs) == 0


def test_refuel_label_non_retryable(mocker):
    class PostRequestMockResponse:
        def __init__(self, resp, status_code):
            self.resp = resp
            self.status_code = status_code
            self.text = resp

        def json(self):
            return self.resp

        def raise_for_status(self):
            pass

    model = RefuelLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_refuel.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "requests.post",
        return_value=PostRequestMockResponse(
            resp='{"error_message": "Error123"}', status_code=422
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x.generations] == ["", ""]
    for error in x.errors:
        assert "NonRetryable Error:" in error.error_message
    assert sum(x.costs) == 0


def test_refuel_label_retryable(mocker):
    class PostRequestMockResponse:
        def __init__(self, resp, status_code):
            self.resp = resp
            self.status_code = status_code
            self.text = resp

        def json(self):
            return self.resp

        def raise_for_status(self):
            pass

    model = RefuelLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_refuel.json")
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "requests.post",
        return_value=PostRequestMockResponse(
            resp='{"error_message": "Error123"}', status_code=500
        ),
    )
    x = model.label(prompts)
    assert [i[0].text for i in x.generations] == ["", ""]
    for error in x.errors:
        assert "NonRetryable Error:" not in error.error_message
    assert sum(x.costs) == 0


def test_refuel_get_cost():
    model = RefuelLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_refuel.json")
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == 0


def test_refuel_return_probs():
    model = RefuelLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_refuel.json")
    )
    assert model.returns_token_probs() is True


################### REFUEL TESTS #######################
