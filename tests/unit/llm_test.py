import json

from langchain.schema import Generation, LLMResult
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pytest import approx

from autolabel.configs import AutolabelConfig
from autolabel.models.anthropic import AnthropicLLM
from autolabel.models.azure_openai import AzureOpenAILLM
from autolabel.models.openai import OpenAILLM
from autolabel.models.openai_vision import OpenAIVisionLLM


################### ANTHROPIC TESTS #######################
def test_anthropic_initialization():
    model = AnthropicLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json",
        ),
    )


async def test_anthropic_label(mocker):
    model = AnthropicLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json",
        ),
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatAnthropic.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts],
        ),
    )
    x = await model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
    assert sum(x.costs) == approx(0.00010944, rel=1e-3)


def test_anthropic_get_cost():
    model = AnthropicLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json",
        ),
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.024024, rel=1e-3)


def test_anthropic_return_probs():
    model = AnthropicLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json",
        ),
    )
    assert model.returns_token_probs() is False


################### ANTHROPIC TESTS #######################


################### OPENAI GPT 3.5 TESTS #######################
def test_gpt35_initialization():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking.json"),
    )


async def test_gpt35_label(mocker):
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking.json"),
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatOpenAI.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts],
        ),
    )
    x = await model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]


def test_gpt35_get_cost():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking.json"),
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.00060045, rel=1e-3)


def test_gpt35_return_probs():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking.json"),
    )
    assert model.returns_token_probs() is True


################### OPENAI GPT 3.5 TESTS #######################


################### OPENAI GPT 4 TESTS #######################
def test_gpt4_initialization():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4.json"),
    )


async def test_gpt4_label(mocker):
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4.json"),
    )
    prompts = ["test1", "test2"]
    mocker.patch(
        "langchain.chat_models.ChatOpenAI.generate",
        return_value=LLMResult(
            generations=[[Generation(text="Answers")] for _ in prompts],
        ),
    )
    x = await model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
    assert sum(x.costs) == approx(0.00023999, rel=1e-3)


def test_gpt4_get_cost():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4.json"),
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.06009, rel=1e-3)


def test_gpt4_return_probs():
    model = OpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4.json"),
    )
    assert model.returns_token_probs() is True


################### OPENAI GPT 4 TESTS #######################


################### OPENAI GPT 4V TESTS #######################
def test_gpt4V_initialization():
    model = OpenAIVisionLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4V.json"),
    )


async def test_gpt4V_label(mocker):
    model = OpenAIVisionLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4V.json"),
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
            ),
        ],
        created=0,
        model="test",
        object="chat.completion",
    )
    x = await model.label(prompts)
    assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
    assert sum(x.costs) == approx(0.01568, rel=1e-3)


def test_gpt4V_get_cost():
    model = OpenAIVisionLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4V.json"),
    )
    example_prompt = json.dumps(
        {"text": "TestingExamplePrompt", "image_url": "dummy1.jpg"},
    )
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.01682, rel=1e-3)


def test_gpt4V_return_probs():
    model = OpenAIVisionLLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_gpt4V.json"),
    )
    assert model.returns_token_probs() is False


################### OPENAI GPT 4V TESTS #######################

################### AZURE OPENAI TESTS #######################
def test_azure_openai_initialization():
    model = AzureOpenAILLM(
        config=AutolabelConfig(config="tests/assets/banking/config_banking_azureopenai.json"),
    )
    assert model.model_name == "gpt-4o-mini"
