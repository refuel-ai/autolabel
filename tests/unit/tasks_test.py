import copy
import json

from autolabel.tasks import (
    ClassificationTask,
    EntityMatchingTask,
    QuestionAnsweringTask,
    MultilabelClassificationTask,
    NamedEntityRecognitionTask,
)
from autolabel.configs import AutolabelConfig
from autolabel.schema import LLMAnnotation, MetricType, LabelingError, ErrorType

from langchain.schema import Generation

BANKING_CONFIG = json.load(open("tests/assets/banking/config_banking.json", "r"))

WALMART_AMAZON_CONFIG = json.load(
    open("tests/assets/walmart_amazon/config_walmart_amazon.json", "r")
)

SCIQ_CONFIG = json.load(open("tests/assets/sciq/config_sciq.json", "r"))

CONLL_CONFIG = json.load(open("tests/assets/conll2003/config_conll2003.json", "r"))

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


def test_classification_no_label_column_in_input():
    config = AutolabelConfig(BANKING_CONFIG)
    task = ClassificationTask(config=config)
    assert task.config != None

    input = {"example": "Here is an example"}
    examples = [
        {"example": "Here is a seed example", "label": "label1"},
        {"example": "Here is another seed example", "label": "label2"},
    ]
    prompt = task.construct_prompt(input, examples)

    assert BANKING_CONFIG["prompt"]["output_guidelines"] in prompt
    assert "\n".join(BANKING_CONFIG["prompt"]["labels"]) in prompt
    assert input["example"] in prompt
    for example in examples:
        assert example["example"] in prompt
        assert example["label"] in prompt


def test_classification_no_label_column_in_config():
    new_config = copy.deepcopy(BANKING_CONFIG)
    new_config["dataset"]["label_column"] = None
    config = AutolabelConfig(new_config)
    task = ClassificationTask(config=config)
    assert task.config != None

    input = {"example": "Here is an example"}
    examples = [
        {"example": "Here is a seed example", "label": "label1"},
        {"example": "Here is another seed example", "label": "label2"},
    ]
    prompt = task.construct_prompt(input, examples)

    assert BANKING_CONFIG["prompt"]["output_guidelines"] in prompt
    assert "\n".join(BANKING_CONFIG["prompt"]["labels"]) in prompt
    assert input["example"] in prompt
    for example in examples:
        assert example["example"] in prompt
        assert example["label"] in prompt


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
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
            error=LabelingError(
                error_type=ErrorType.LLM_PROVIDER_ERROR,
                error_message="No label provided",
            ),
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label3-wrong",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label4",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label5-wrong",
            error=None,
        ),
    ]
    gt_labels = ["label1", "label2", "label3", "label4", "label5"]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.name == MetricType.ACCURACY:
            assert metric.value == 0.5
        elif metric.name == MetricType.COMPLETION_RATE:
            assert metric.value == 0.8
        elif metric.name == MetricType.SUPPORT:
            assert metric.value == 5


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
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
            error=LabelingError(
                error_type=ErrorType.LLM_PROVIDER_ERROR,
                error_message="No label provided",
            ),
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="not duplicate",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="not duplicate",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="duplicate",
            error=None,
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
        if metric.name == MetricType.ACCURACY:
            assert metric.value == 0.75
        elif metric.name == MetricType.COMPLETION_RATE:
            assert metric.value == 0.8
        elif metric.name == MetricType.SUPPORT:
            assert metric.value == 5


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
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
            error=LabelingError(
                error_type=ErrorType.LLM_PROVIDER_ERROR,
                error_message="No label provided",
            ),
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Paris",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Bangalore",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Delhi",
            error=None,
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
        if metric.name == MetricType.ACCURACY:
            assert metric.value == 0.75
        elif metric.name == MetricType.COMPLETION_RATE:
            assert metric.value == 0.8
        elif metric.name == MetricType.SUPPORT:
            assert metric.value == 5


def ner_construct_prompt():
    config = AutolabelConfig(CONLL_CONFIG)
    task = NamedEntityRecognitionTask(config=config)
    assert task.config != None

    input = {
        "example": "The role of the 70,000 mainly Kurdish village guards who fight Kurdistan Workers Party ( PKK ) guerrillas in the southeast has been questioned recently after media allegations that many of them are involved in common crime .",
        "IndividualLabels": '[{"Description": "Miscellaneous", "Text": "Kurdish"}, {"Description": "Organization", "Text": "Kurdistan Workers Party"}, {"Description": "Organization", "Text": "PKK"}]',
        "CategorizedLabels": '{"Location": [], "Organization": ["Kurdistan Workers Party", "PKK"], "Person": [], "Miscellaneous": ["Kurdish"]}',
    }
    examples = [
        {
            "example": "The head of the region 's main pro-state militia is at the centre of a security scandal that has shaken the government .",
            "IndividualLabels": "[]",
            "CategorizedLabels": '{"Location": [], "Organization": [], "Person": [], "Miscellaneous": []}',
        },
        {
            "example": "More than 21,000 people have been killed in the 12-year-old conflict between Turkish security forces and the PKK , fighting for Kurdish autonomy or independence .",
            "IndividualLabels": '[{"Description": "Miscellaneous", "Text": "Turkish"}, {"Description": "Organization", "Text": "PKK"}, {"Description": "Miscellaneous", "Text": "Kurdish"}]',
            "CategorizedLabels": '{"Location": [], "Organization": ["PKK"], "Person": [], "Miscellaneous": ["Turkish", "Kurdish"]}',
        },
    ]
    prompt = task.construct_prompt(input, examples)

    assert input["example"] in prompt
    assert task._json_to_llm_format(input["CategorizedLabels"]) in prompt
    for example in examples:
        assert example["example"] in prompt
        assert task._json_to_llm_format(example["CategorizedLabels"]) in prompt


def ner_parse_llm_response():
    config = AutolabelConfig(SCIQ_CONFIG)
    task = NamedEntityRecognitionTask(config=config)

    input = {
        "example": "The role of the 70,000 mainly Kurdish village guards who fight Kurdistan Workers Party ( PKK ) guerrillas in the southeast has been questioned recently after media allegations that many of them are involved in common crime .",
        "IndividualLabels": '[{"Description": "Miscellaneous", "Text": "Kurdish"}, {"Description": "Organization", "Text": "Kurdistan Workers Party"}, {"Description": "Organization", "Text": "PKK"}]',
        "CategorizedLabels": '{"Location": [], "Organization": ["Kurdistan Workers Party", "PKK"], "Person": [], "Miscellaneous": ["Kurdish"]}',
    }
    prompt = "This is a prompt"

    label = task._json_to_llm_format(input["CategorizedLabels"])
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == input["CategorizedLabels"]
    assert parsed.successfully_labeled == True
    assert parsed.raw_response == label
    assert parsed.error is None

    label = ""
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == task.NULL_LABEL_TOKEN
    assert parsed.successfully_labeled == False
    assert parsed.error is not None


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

    input = {"example": "Here is an example", "labels": "label-1, label-2"}
    examples = [
        {"example": "Here is a seed example", "labels": "labela, labelb"},
        {"example": "Here is another seed example", "labels": "labelc, labeld"},
    ]
    prompt = task.construct_prompt(input, examples)

    assert TWITTER_EMOTION_DETECTION_CONFIG["prompt"]["output_guidelines"] in prompt
    assert "\n".join(TWITTER_EMOTION_DETECTION_CONFIG["prompt"]["labels"]) in prompt
    assert input["example"] in prompt
    assert input["labels"] not in prompt
    for example in examples:
        assert example["example"] in prompt
        assert example["labels"] in prompt

    new_config = copy.deepcopy(TWITTER_EMOTION_DETECTION_CONFIG)
    del new_config["prompt"]["few_shot_selection"]
    new_config = AutolabelConfig(new_config)
    task = ClassificationTask(config=new_config)
    prompt = task.construct_prompt(input, examples)
    for example in examples:
        assert example["example"] not in prompt
        assert example["labels"] not in prompt


def test_multilabel_classification_eval():
    config = AutolabelConfig(TWITTER_EMOTION_DETECTION_CONFIG)
    task = MultilabelClassificationTask(config=config)

    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="neutral",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
            error=LabelingError(
                error_type=ErrorType.LLM_PROVIDER_ERROR,
                error_message="No label provided",
            ),
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="sadness",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="anger, disgust",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="joy, love, trust",
            error=None,
        ),
    ]
    gt_labels = [
        "anger, disgust",
        "joy, optimism, trust",
        "anticipation, joy, sadness",
        "anger, disgust",
        "joy, optimism",
    ]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.name == MetricType.ACCURACY:
            assert metric.value == 0.25
        elif metric.name == MetricType.F1_MACRO:
            assert metric.value == 0.25
        elif metric.name == MetricType.F1_WEIGHTED:
            assert metric.value == 5 / 9
        elif metric.name == MetricType.COMPLETION_RATE:
            assert metric.value == 0.8
        elif metric.name == MetricType.SUPPORT:
            assert metric.value == 5


def custom_metric_test():
    config = AutolabelConfig(TWITTER_EMOTION_DETECTION_CONFIG)
    task = MultilabelClassificationTask(config=config)
    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="neutral",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
            error=LabelingError(
                error_type=ErrorType.LLM_PROVIDER_ERROR,
                error_message="No label provided",
            ),
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="sadness",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="anger, disgust",
            error=None,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="joy, love, trust",
            error=None,
        ),
    ]

    gt_labels = [
        "anger, disgust",
        "joy, optimism, trust",
        "anticipation, joy, sadness",
        "anger, disgust",
        "joy, optimism",
    ]

    from autolabel.metrics import BaseMetric
    from autolabel.schema import MetricResult

    class NewMetric(BaseMetric):
        def compute(self, llm_labels, gt_labels):
            return [MetricResult(name="new_metric", value=0.25)]

    eval = task.eval(llm_labels, gt_labels, additional_metrics=[NewMetric()])

    for metric in eval:
        if metric.name == "new_metric":
            assert metric.value == 0.25
