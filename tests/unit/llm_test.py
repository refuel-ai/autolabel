import json
from autolabel.configs import AutolabelConfig
from autolabel.models.anthropic import AnthropicLLM
from autolabel.models.openai import OpenAILLM
from autolabel.models.openai_vision import OpenAIVisionLLM
from autolabel.models.palm import PaLMLLM
from autolabel.models.replicate import ReplicateLLM
from autolabel.models.refuel import RefuelLLM
from langchain.schema import Generation, LLMResult
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pytest import approx
import pytest


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
    assert model.returns_token_probs() is True


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
    assert model.returns_token_probs() is True


################### OPENAI GPT 4 TESTS #######################


################### OPENAI GPT 4V TESTS #######################
def test_gpt4V_initialization():
    model = OpenAIVisionLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4V.json")
    )


def test_gpt4V_label(mocker):
    model = OpenAIVisionLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4V.json")
    )
    prompts = [
        json.dumps({"text": "test1", "image_url": "dummy1.jpg"}),
        json.dumps({"text": "test2", "image_url": "dummy2.jpg"}),
    ]
    model.client.chat.completions._post = lambda *args, **kargs: ChatCompletion(
        id="test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content="Answers", role="assistant"),
            )
        ],
        created=0,
        model="test",
        object="chat.completion",
    )
    x = model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
    assert sum(x.costs) == approx(0.01568, rel=1e-3)


def test_gpt4V_get_cost():
    model = OpenAIVisionLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4V.json")
    )
    example_prompt = json.dumps(
        {"text": "TestingExamplePrompt", "image_url": "dummy1.jpg"}
    )
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.01682, rel=1e-3)


def test_gpt4V_return_probs():
    model = OpenAIVisionLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4V.json")
    )
    assert model.returns_token_probs() is False


################### OPENAI GPT 4V TESTS #######################


################### REPLICATE TESTS #######################
class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


@pytest.fixture
def replicate_model(mocker):
    mocker.patch(
        "requests.get",
        return_value=MockResponse({"latest_version": {"id": "valid_id"}}, 200),
    )
    return ReplicateLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_replicate.json"
        )
    )


def test_replicate_initialization(replicate_model):
    assert isinstance(replicate_model, ReplicateLLM)


def test_replicate_invalid_model_initialization(mocker):
    config = AutolabelConfig(
        config="tests/assets/banking/config_banking_replicate.json"
    )
    mocker.patch(
        "requests.get",
        return_value=MockResponse({"error": "model not found"}, 404),
    )

    with pytest.raises(ValueError) as excinfo:
        model = ReplicateLLM(config)

    assert "Model meta/llama-2-70b-chat not found on Replicate" in str(excinfo.value)


def test_replicate_label(mocker, replicate_model):
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain_community.llms.Replicate.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts]
        ),
    )
    x = replicate_model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]


def test_replicate_get_cost(replicate_model):
    example_prompt = "TestingExamplePrompt"
    curr_cost = replicate_model.get_cost(example_prompt)
    assert curr_cost == approx(0.00275389, rel=1e-3)


################### REPLICATE TESTS #######################


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
