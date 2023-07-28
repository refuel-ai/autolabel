"""Test Validation"""
import pytest
from autolabel.configs import AutolabelConfig
from autolabel.dataset.validation import TaskDataValidation

CLASSIFICATION_CONFIG_SAMPLE_DICT = {
    "task_name": "LegalProvisionsClassification",
    "task_type": "classification",
    "dataset": {"label_column": "label", "delimiter": ","},
    "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
    "prompt": {
        "task_guidelines": "You are an expert at understanding legal contracts. Your job is to correctly classify legal provisions in contracts into one of the following categories.\nCategories:{labels}\n",
        "labels": [
            "Agreements",
            "Argument",
        ],
        "example_template": "Example: {example}\nOutput: {label}",
        "few_shot_examples": "seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 4,
    },
}

NER_CONFIG_SAMPLE_DICT = {
    "task_name": "PersonLocationOrgMiscNER",
    "task_type": "named_entity_recognition",
    "dataset": {
        "label_column": "CategorizedLabels",
        "text_column": "example",
        "delimiter": ",",
    },
    "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
    "prompt": {
        "task_guidelines": "You are an expert at extracting Person, Organization, Location, and Miscellaneous entities from text. Your job is to extract named entities mentioned in text, and classify them into one of the following categories.\nCategories:\n{labels}\n ",
        "labels": [
            "Location",
        ],
        "example_template": "Example: {example}\nOutput:\n{CategorizedLabels}",
        "few_shot_examples": "seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 5,
    },
}

EM_CONFIG_SAMPLE_DICT = {
    "task_name": "ProductCatalogEntityMatch",
    "task_type": "entity_matching",
    "dataset": {"label_column": "label", "delimiter": ","},
    "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
    "prompt": {
        "task_guidelines": "You are an expert at identifying duplicate products from online product catalogs.\nYou will be given information about two product entities, and your job is to tell if they are the same (duplicate) or different (not duplicate). Your answer must be from one of the following options:\n{labels}",
        "labels": ["duplicate", "not duplicate"],
        "example_template": "Title of entity1: {Title_entity1}; \nDuplicate or not: {label}",
        "few_shot_selection": "fixed",
        "few_shot_num": 2,
    },
}


def test_validate_classification_task():
    """Test Validate classification"""
    data = [
        {"question": "s", "answer": "ee"},  # Wrong column names
        {"example": "s", "label": 1},  # int value not accepted
        {"example": "s", "label": "['w']"},  # w, d are incorrect labels
        {
            "example": "s",
            "label": "['Agreements', 'Random']",
        },  # Random not valid label
        {"example": "s", "label": "['Agreements', 'Argument']"},  # Correct
        {"example": "s", "label": "['Agreements']"},  # Correct
    ]
    expected_output = [
        {
            "loc": "example",
            "msg": "field required",
            "row_num": 0,
            "type": "value_error.missing",
        },
        {
            "loc": "label",
            "msg": "field required",
            "row_num": 0,
            "type": "value_error.missing",
        },
        {
            "loc": "label",
            "msg": "str type expected",
            "row_num": 1,
            "type": "type_error.str",
        },
        {
            "loc": "__root__",
            "msg": "labels: '{'w'}' not in prompt/labels provided in config ",
            "row_num": 2,
            "type": "value_error",
        },
        {
            "loc": "__root__",
            "msg": "labels: '{'Random'}' not in prompt/labels provided in config ",
            "row_num": 3,
            "type": "value_error",
        },
    ]
    data_validation = TaskDataValidation(
        config=AutolabelConfig(CLASSIFICATION_CONFIG_SAMPLE_DICT)
    )
    error_table = data_validation.validate(data=data)

    for exp_out, err_out in zip(expected_output, error_table):
        assert exp_out == err_out


def test_validate_ner_task():
    """Test Validate NamedEntityRecognition"""
    data = [
        {
            # Miscellaneous is not a valid label mentioned in NER_CONFIG_SAMPLE_DICT
            "example": "example1",
            "CategorizedLabels": '{"Location": ["Okla"], "Miscellaneous": []}',
        },
        {
            # Not a valid Json
            "example": "example2",
            "CategorizedLabels": '{"Location":["Texas"], ""Miscellaneous"": []}',
        },
        {
            # label is not the correct column name
            "example": "example3",
            "label": '{"Location":["Texas"], "Miscellaneous": ["USDA", "PPAS"]}',
        },
        {
            # Correct
            "example": "example2",
            "CategorizedLabels": '{"Location":["Texas"]}',
        },
    ]
    expected_output = [
        {
            "loc": "__root__",
            "msg": "labels: '{'Miscellaneous'}' not in prompt/labels provided in config ",
            "row_num": 0,
            "type": "value_error",
        },
        {
            "loc": "__root__",
            "msg": "Expecting ':' delimiter: line 1 column 26 (char 25)",
            "row_num": 1,
            "type": "value_error.jsondecode",
        },
        {
            "loc": "CategorizedLabels",
            "msg": "field required",
            "row_num": 2,
            "type": "value_error.missing",
        },
    ]

    data_validation = TaskDataValidation(config=AutolabelConfig(NER_CONFIG_SAMPLE_DICT))

    error_table = data_validation.validate(data=data)

    for exp_out, err_out in zip(expected_output, error_table):
        assert exp_out == err_out


def test_validate_EM_task():
    """Test Validate NamedEntityRecognition"""
    data = [
        {"Title_entity1": "example1", "label": "duplicate"},
        {"Title_entity1": "example2", "label": "not duplicate"},
        {"ErrorColumn": "example2", "label": '{"Location":["Texas"]}'},
        {"Title_entity1": "example2", "label": "duplicate not duplicate"},
    ]

    expected_output = [
        {
            "row_num": 2,
            "loc": "Title_entity1",
            "msg": "field required",
            "type": "value_error.missing",
        },
        {
            "row_num": 3,
            "loc": "__root__",
            "msg": "labels: 'duplicate not duplicate' not in prompt/labels provided in config ",
            "type": "value_error",
        },
    ]

    data_validation = TaskDataValidation(config=AutolabelConfig(EM_CONFIG_SAMPLE_DICT))

    error_table = data_validation.validate(data=data)

    for exp_out, err_out in zip(expected_output, error_table):
        assert exp_out == err_out


def test_columns():
    """Test Validate NamedEntityRecognition"""
    data_validation = TaskDataValidation(config=AutolabelConfig(NER_CONFIG_SAMPLE_DICT))

    with pytest.raises(
        AssertionError, match=r"columns={'example'} missing in seed.csv file"
    ):
        data_validation.validate_dataset_columns(
            dataset_columns=["input", "CategorizedLabels"]
        )
