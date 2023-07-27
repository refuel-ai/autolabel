"""Test Configuration"""

import pytest
from jsonschema import validate, exceptions
from autolabel import LabelingAgent
from autolabel.configs.schema import schema


CONFIG_SAMPLE_DICT = {
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {"label_column": "label", "delimiter": ","},
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "compute_confidence": True,
    },
    "prompt": {
        "task_guidelines": "You are an expert at identifying toxic comments",
        "labels": ["toxic", "not toxic"],
        "example_template": "Input: {example}\nOutput: {label}",
    },
}


def test_config():
    """Test configurations"""
    agent0 = LabelingAgent(config="tests/assets/banking/config_banking.json")
    config0 = agent0.config
    assert config0.label_column() == "label"
    assert config0.task_type() == "classification"
    assert config0.delimiter() == ","

    agent1 = LabelingAgent(config="tests/assets/conll2003/config_conll2003.json")
    config1 = agent1.config
    assert config1.label_column() == "CategorizedLabels"
    assert config1.text_column() == "example"
    assert config1.task_type() == "named_entity_recognition"

    agent2 = LabelingAgent(config="tests/assets/squad_v2/config_squad_v2.json")
    config2 = agent2.config
    assert config2.label_column() == "answer"
    assert config2.task_type() == "question_answering"
    assert config2.few_shot_example_set() == "seed.csv"


def test_schema_validation():
    """Validate config

    This test case tests several validation scenarios.
    """

    # Case 1:
    # ----------
    # Here we do not probide the task_guideline field which is required.1
    config_dict_copy = CONFIG_SAMPLE_DICT.copy()
    config_dict_copy["prompt"] = {
        "labels": ["toxic", "not toxic"],
        "example_template": "Input: {example}\nOutput: {label}",
    }
    with pytest.raises(
        exceptions.ValidationError, match=r"'task_guidelines' is a required property"
    ):
        validate(
            instance=config_dict_copy,
            schema=schema,
        )

    # Case 2:
    # ----------
    # Here we miss spell labelsss, since `prompt` property takes no more additinoal
    # keys other than label, task_guidelines, example_template, typos are considered
    # as errors.
    config_dict_copy = CONFIG_SAMPLE_DICT.copy()
    config_dict_copy["prompt"] = {
        # We make a type with task_guidelines and example_template
        "task_guidelines": "You are an expert at identifying toxic comments",
        "labelsss": ["toxic", "not toxic"],
        "example_template": "Input: {example}\nOutput: {label}",
    }
    with pytest.raises(
        exceptions.ValidationError,
        match="Additional properties are not allowed",
    ):
        validate(
            instance=config_dict_copy,
            schema=schema,
        )

    # Case 3:
    # ----------
    # Here `not_supported_task_type` is not a supported task type
    # classification, "entity_matching", "named_entity_recognition", "question_answering",
    config_dict_copy = CONFIG_SAMPLE_DICT.copy()
    config_dict_copy["task_type"] = "not_supported_task_type"
    with pytest.raises(
        exceptions.ValidationError,
        match=r"not_supported_task_type",
    ):
        validate(
            instance=config_dict_copy,
            schema=schema,
        )

    # Case 4:
    # ----------
    # Test when one of the required field is not provided
    config_dict_copy = CONFIG_SAMPLE_DICT.copy()
    del config_dict_copy["model"]
    with pytest.raises(
        exceptions.ValidationError,
        match=r"'model' is a required property",
    ):
        validate(
            instance=config_dict_copy,
            schema=schema,
        )

    # Case 5:
    # ----------
    # Test when schema validation passes
    validate(
        instance=CONFIG_SAMPLE_DICT,
        schema=schema,
    )
