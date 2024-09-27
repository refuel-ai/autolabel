"""Test Configuration"""

import pytest
from jsonschema import validate, exceptions
from autolabel import LabelingAgent
from autolabel.configs.schema import schema
from autolabel.configs import AutolabelConfig


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

    # Case 3:
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

    # Case 4:
    # ----------
    # Test when schema validation passes
    validate(
        instance=CONFIG_SAMPLE_DICT,
        schema=schema,
    )
