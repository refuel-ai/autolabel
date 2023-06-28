import copy
import json

from autolabel.tasks import (
    ClassificationTask,
    EntityMatchingTask,
    QuestionAnsweringTask,
    MultilabelClassificationTask,
)
from autolabel.configs import AutolabelConfig
from autolabel.schema import LLMAnnotation, Metric

from langchain.schema import Generation

BANKING_CONFIG = json.load(open("tests/assets/banking/config_banking.json", "r"))

WALMART_AMAZON_CONFIG = json.load(
    open("tests/assets/walmart_amazon/config_walmart_amazon.json", "r")
)

SCIQ_CONFIG = json.load(open("tests/assets/sciq/config_sciq.json", "r"))

TWITTER_EMOTION_DETECTION_CONFIG = json.load(
    open(
        "tests/assets/twitter_emotion_detection/config_twitter_emotion_detection.json",
        "r",
    )
)


def test_classification_construct_prompt():
    config = AutolabelConfig(BANKING_CONFIG)
    task = ClassificationTask(config=config)
    assert task.config != None

    input = {"example": "Here is an example", "label": "label-true"}
    examples = [
        {"example": "Here is a seed example", "label": "label1"},
        {"example": "Here is another seed example", "label": "label2"},
    ]
    prompt = task.construct_prompt(input, examples)

    assert BANKING_CONFIG["prompt"]["output_guidelines"] in prompt
    assert "\n".join(BANKING_CONFIG["prompt"]["labels"]) in prompt
    assert input["example"] in prompt
    assert input["label"] not in prompt
    for example in examples:
        assert example["example"] in prompt
        assert example["label"] in prompt

    new_config = copy.deepcopy(BANKING_CONFIG)
    del new_config["prompt"]["few_shot_selection"]
    new_config = AutolabelConfig(new_config)
    task = ClassificationTask(config=new_config)
    prompt = task.construct_prompt(input, examples)
    for example in examples:
        assert example["example"] not in prompt
        assert example["label"] not in prompt


def test_classification_parse_llm_response():
    new_config = copy.deepcopy(BANKING_CONFIG)
    new_config["prompt"]["labels"].append("label-true")
    new_config = AutolabelConfig(new_config)
    task = ClassificationTask(config=new_config)

    input = {"example": "Here is an example", "label": "label-true"}
    prompt = "This is a prompt"

    label = "This is the thing we want to test.\nlabel-true"
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == "label-true"
    assert parsed.successfully_labeled == True
    assert parsed.raw_response == label

    label = ""
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == task.NULL_LABEL_TOKEN
    assert parsed.successfully_labeled == False


def test_classification_eval():
    config = AutolabelConfig(BANKING_CONFIG)
    task = ClassificationTask(config=config)

    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="label1",
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label3-wrong",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label4",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label5-wrong",
        ),
    ]
    gt_labels = ["label1", "label2", "label3", "label4", "label5"]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.metric_type == Metric.ACCURACY:
            assert metric.value[0] == 0.5
        elif metric.metric_type == Metric.COMPLETION_RATE:
            assert metric.value[0] == 0.8
        elif metric.metric_type == Metric.SUPPORT:
            assert metric.value[0] == 4


def test_entity_matching_construct_prompt():
    config = AutolabelConfig(WALMART_AMAZON_CONFIG)
    task = EntityMatchingTask(config=config)
    assert task.config != None

    input = {
        "Title_entity1": "lexmark extra high yield return pgm print cartridge - magenta",
        "Category_entity1": "printers",
        "Brand_entity1": "lexmark",
        "ModelNo_entity1": "c782u1mg",
        "Price_entity1": "214.88",
        "Title_entity2": "lexmark 18c1428 return program print cartridge black",
        "Category_entity2": "inkjet printer ink",
        "Brand_entity2": "lexmark",
        "ModelNo_entity2": "18c1428",
        "Price_entity2": "19.97",
        "label": "not duplicate - 1",
    }
    examples = [
        {
            "Title_entity1": "lexmark extra high yield return pgm print cartridge - magentaplus",
            "Category_entity1": "printers",
            "Brand_entity1": "lexmark",
            "ModelNo_entity1": "c782u1mg",
            "Price_entity1": "214.88",
            "Title_entity2": "lexmark 18c1428 return program print cartridge black",
            "Category_entity2": "inkjet printer ink",
            "Brand_entity2": "lexmark",
            "ModelNo_entity2": "18c1428",
            "Price_entity2": "19.97",
            "label": "not duplicate - 2",
        },
        {
            "Title_entity1": "edge tech proshot 4gb sdhc class 6 memory card",
            "Category_entity1": "usb drives",
            "Brand_entity1": "edge tech",
            "ModelNo_entity1": "pe209780",
            "Price_entity1": "10.88",
            "Title_entity2": "4gb edge proshot sdhc memory card class6",
            "Category_entity2": "computers accessories",
            "Brand_entity2": "edge",
            "ModelNo_entity2": "nan",
            "Price_entity2": "17.83",
            "label": "duplicate - 3",
        },
    ]
    prompt = task.construct_prompt(input, examples)

    assert "\n".join(WALMART_AMAZON_CONFIG["prompt"]["labels"]) in prompt
    assert input["Title_entity1"] in prompt
    assert input["Category_entity1"] in prompt
    assert input["label"] not in prompt
    for example in examples:
        assert example["Title_entity1"] in prompt
        assert example["label"] in prompt

    new_config = copy.deepcopy(WALMART_AMAZON_CONFIG)
    del new_config["prompt"]["few_shot_selection"]
    new_config = AutolabelConfig(new_config)
    task = EntityMatchingTask(config=new_config)
    prompt = task.construct_prompt(input, examples)
    for example in examples:
        assert example["Title_entity1"] not in prompt


def test_entity_matching_parse_llm_response():
    new_config = copy.deepcopy(WALMART_AMAZON_CONFIG)
    new_config = AutolabelConfig(new_config)
    new_config["prompt"]["labels"].append("not duplicate - 1")
    task = EntityMatchingTask(config=new_config)

    input = {
        "Title_entity1": "lexmark extra high yield return pgm print cartridge - magenta",
        "Category_entity1": "printers",
        "Brand_entity1": "lexmark",
        "ModelNo_entity1": "c782u1mg",
        "Price_entity1": "214.88",
        "Title_entity2": "lexmark 18c1428 return program print cartridge black",
        "Category_entity2": "inkjet printer ink",
        "Brand_entity2": "lexmark",
        "ModelNo_entity2": "18c1428",
        "Price_entity2": "19.97",
        "label": "not duplicate - 1",
    }
    prompt = "This is a prompt"

    label = "This is the thing we want to test.\nnot duplicate - 1"
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == "not duplicate - 1"
    assert parsed.successfully_labeled == True
    assert parsed.raw_response == label

    label = ""
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == task.NULL_LABEL_TOKEN
    assert parsed.successfully_labeled == False


def test_entity_matching_eval():
    config = AutolabelConfig(WALMART_AMAZON_CONFIG)
    task = EntityMatchingTask(config=config)

    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="duplicate",
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="not duplicate",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="not duplicate",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="duplicate",
        ),
    ]
    gt_labels = [
        "duplicate",
        "not duplicate",
        "not duplicate",
        "not duplicate",
        "not duplicate",
    ]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.metric_type == Metric.ACCURACY:
            assert metric.value[0] == 0.75
        elif metric.metric_type == Metric.COMPLETION_RATE:
            assert metric.value[0] == 0.8
        elif metric.metric_type == Metric.SUPPORT:
            assert metric.value[0] == 4


def question_answering_construct_prompt():
    config = AutolabelConfig(SCIQ_CONFIG)
    task = QuestionAnsweringTask(config=config)
    assert task.config != None

    input = {
        "question": "What is the capital of France?",
        "options": "[Paris, London, Berlin]",
        "answer": "Paris-label",
    }
    examples = [
        {
            "question": "What is the capital of India?",
            "options": "[Delhi, Mumbai, Bangalore]",
            "answer": "Delhi-label",
        },
        {
            "question": "What is the capital of USA?",
            "options": "[New York, Washington DC, Los Angeles]",
            "answer": "Washington DC-label",
        },
    ]
    prompt = task.construct_prompt(input, examples)

    assert input["question"] in prompt
    assert input["options"] in prompt
    assert input["answer"] not in prompt
    for example in examples:
        assert example["question"] in prompt
        assert example["options"] in prompt
        assert example["answer"] in prompt


def question_answering_parse_llm_response():
    config = AutolabelConfig(SCIQ_CONFIG)
    task = QuestionAnsweringTask(config=config)

    input = {
        "question": "What is the capital of France?",
        "options": "[Paris, London, Berlin]",
        "answer": "Paris-label",
    }
    prompt = "This is a prompt"

    label = "This is the thing we want to test.\nParis-label"
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == "Paris-label"
    assert parsed.successfully_labeled == True
    assert parsed.raw_response == label

    label = ""
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == task.NULL_LABEL_TOKEN
    assert parsed.successfully_labeled == False


def test_question_answering_eval():
    config = AutolabelConfig(SCIQ_CONFIG)
    task = QuestionAnsweringTask(config=config)

    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="Delhi",
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Paris",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Bangalore",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Delhi",
        ),
    ]
    gt_labels = [
        "Delhi",
        "Washington DC",
        "Paris",
        "Bangalore",
        "Washington DC",
    ]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.metric_type == Metric.ACCURACY:
            assert metric.value[0] == 0.75
        elif metric.metric_type == Metric.COMPLETION_RATE:
            assert metric.value[0] == 0.8
        elif metric.metric_type == Metric.SUPPORT:
            assert metric.value[0] == 4


def test_classification_labels_not_in_labels_list():
    config = AutolabelConfig(BANKING_CONFIG)
    task = ClassificationTask(config=config)

    input = {"example": "Here is an example", "label": "not-in-labels-list"}
    prompt = "This is a prompt"

    label = "This is the thing we want to test.\nnot-in-labels-list"
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == "not-in-labels-list"
    assert parsed.successfully_labeled == False
    assert parsed.raw_response == label


def test_entity_matching_label_not_in_labels_list():
    config = AutolabelConfig(WALMART_AMAZON_CONFIG)
    task = EntityMatchingTask(config=config)

    input = {
        "Title_entity1": "lexmark extra high yield return pgm print cartridge - magenta",
        "Category_entity1": "printers",
        "Brand_entity1": "lexmark",
        "ModelNo_entity1": "c782u1mg",
        "Price_entity1": "214.88",
        "Title_entity2": "lexmark 18c1428 return program print cartridge black",
        "Category_entity2": "inkjet printer ink",
        "Brand_entity2": "lexmark",
        "ModelNo_entity2": "18c1428",
        "Price_entity2": "19.97",
        "label": "not-in-labels-list",
    }
    prompt = "This is a prompt"

    label = "This is the thing we want to test.\nnot-in-labels-list"
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == "not-in-labels-list"
    assert parsed.successfully_labeled == False
    assert parsed.raw_response == label


def test_multilabel_classification_construct_prompt():
    config = AutolabelConfig(TWITTER_EMOTION_DETECTION_CONFIG)
    task = MultilabelClassificationTask(config=config)
    assert task.config != None

    input = {"example": "Here is an example", "label": "label-1, label-2"}
    examples = [
        {"example": "Here is a seed example", "label": "labela, labelb"},
        {"example": "Here is another seed example", "label": "labelc, labeld"},
    ]
    prompt = task.construct_prompt(input, examples)

    assert TWITTER_EMOTION_DETECTION_CONFIG["prompt"]["output_guidelines"] in prompt
    assert "\n".join(TWITTER_EMOTION_DETECTION_CONFIG["prompt"]["labels"]) in prompt
    assert input["example"] in prompt
    assert input["label"] not in prompt
    for example in examples:
        assert example["example"] in prompt
        assert example["label"] in prompt

    new_config = copy.deepcopy(TWITTER_EMOTION_DETECTION_CONFIG)
    del new_config["prompt"]["few_shot_selection"]
    new_config = AutolabelConfig(new_config)
    task = ClassificationTask(config=new_config)
    prompt = task.construct_prompt(input, examples)
    for example in examples:
        assert example["example"] not in prompt
        assert example["label"] not in prompt


def test_multilabel_classification_eval():
    config = AutolabelConfig(TWITTER_EMOTION_DETECTION_CONFIG)
    task = MultilabelClassificationTask(config=config)

    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="neutral",
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="sadness",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="anger, disgust",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="joy, love, trust",
        ),
    ]
    gt_labels = [
        "anger, disgust",
        "joy, optimism, trust",
        "anticipation, joy, optimism",
        "anger, disgust",
        "joy, optimism",
    ]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.metric_type == Metric.ACCURACY:
            assert metric.value[0] == 0.25
        elif metric.metric_type == Metric.F1:
            assert metric.value[0] == 0.35
        elif metric.metric_type == Metric.COMPLETION_RATE:
            assert metric.value[0] == 0.8
        elif metric.metric_type == Metric.SUPPORT:
            assert metric.value[0] == 4
