from autolabel.configs import AutolabelConfig
from autolabel.models.litellm import LiteLLM
from langchain.schema import Generation, LLMResult
from pytest import approx


################### ANTHROPIC TESTS #######################
def test_anthropic_initialization():
    model = LiteLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json"
        )
    )


# def test_anthropic_label(mocker):
#     model = LiteLLM(
#         config=AutolabelConfig(
#             config="tests/assets/banking/config_banking_anthropic.json"
#         )
#     )
#     prompts = ["test1", "test2"]
#     mocker.patch(
#         "langchain.chat_models.ChatAnthropic.generate",
#         return_value=LLMResult(
#             generations=[[Generation(text="Answers")] for _ in prompts]
#         ),
#     )
#     x = model.label(prompts)
#     print(x)
#     assert [i[0].text for i in x.generations] == ["Answers", "Answers"]
#     assert sum(x.costs) == approx(0.00010944, rel=1e-3)

# def test_anthropic_label():
#     model = LiteLLM(
#         config=AutolabelConfig(
#             config="tests/assets/banking/config_banking_anthropic.json"
#         )
#     )
#     prompts = ["test1", "test2"]
#     x = model._label(prompts)
#     print(x)

# test_anthropic_label()
model = LiteLLM(
    config=AutolabelConfig(config="tests/assets/banking/config_banking_anthropic.json")
)
print(model.get_cost("hello world"))


def test_anthropic_get_cost():
    model = LiteLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json"
        )
    )
    example_prompt = "TestingExamplePrompt"
    curr_cost = model.get_cost(example_prompt)
    assert curr_cost == approx(0.03271306, rel=1e-3)


def test_anthropic_return_probs():
    model = LiteLLM(
        config=AutolabelConfig(
            config="tests/assets/banking/config_banking_anthropic.json"
        )
    )
    assert model.returns_token_probs() is False


################### ANTHROPIC TESTS #######################
