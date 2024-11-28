import copy
import json

from langchain.schema import Generation

from autolabel.configs import AutolabelConfig
from autolabel.tasks import (
    AttributeExtractionTask,
)

BANKING_CONFIG = json.load(open("tests/assets/banking/config_banking.json"))

WALMART_AMAZON_CONFIG = json.load(
    open("tests/assets/walmart_amazon/config_walmart_amazon.json"),
)

CONLL_CONFIG = json.load(open("tests/assets/conll2003/config_conll2003.json"))

TWITTER_EMOTION_DETECTION_CONFIG = json.load(
    open(
        "tests/assets/twitter_emotion_detection/config_twitter_emotion_detection.json",
    ),
)


def test_classification_construct_prompt():
    config = AutolabelConfig(BANKING_CONFIG)
    task = AttributeExtractionTask(config=config)
    assert task.config != None

    input = {"example": "Here is an example", "label": "label-true"}
    examples = [
        {"example": "Here is a seed example", "label": "label1"},
        {"example": "Here is another seed example", "label": "label2"},
    ]
    prompt, schema = task.construct_prompt(input, examples)
    assert ",".join(BANKING_CONFIG["prompt"]["attributes"][0]["options"]) in prompt
    assert input["example"] in prompt
    assert input["label"] not in prompt
    for example in examples:
        assert example["example"] in prompt
        assert example["label"] in prompt

    new_config = copy.deepcopy(BANKING_CONFIG)
    del new_config["prompt"]["few_shot_selection"]
    new_config = AutolabelConfig(new_config)
    task = AttributeExtractionTask(config=new_config)
    prompt, schema = task.construct_prompt(input, examples)
    for example in examples:
        assert example["example"] not in prompt
        assert example["label"] not in prompt


def test_classification_no_label_column_in_input():
    config = AutolabelConfig(BANKING_CONFIG)
    task = AttributeExtractionTask(config=config)
    assert task.config != None

    input = {"example": "Here is an example"}
    examples = [
        {"example": "Here is a seed example", "label": "label1"},
        {"example": "Here is another seed example", "label": "label2"},
    ]
    prompt, schema = task.construct_prompt(input, examples)

    assert ",".join(BANKING_CONFIG["prompt"]["attributes"][0]["options"]) in prompt
    assert input["example"] in prompt
    for example in examples:
        assert example["example"] in prompt
        assert example["label"] in prompt


def test_classification_parse_llm_response():
    new_config = copy.deepcopy(BANKING_CONFIG)
    new_config["prompt"]["attributes"][0]["options"].append("label-true")
    new_config = AutolabelConfig(new_config)
    task = AttributeExtractionTask(config=new_config)

    input = {"example": "Here is an example", "label": "label-true"}
    prompt = "This is a prompt"

    label = '{"label": "label-true"}'
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == {"label": "label-true"}
    assert parsed.successfully_labeled == True
    assert parsed.raw_response == label

    label = ""
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == {}
    assert parsed.successfully_labeled == False
