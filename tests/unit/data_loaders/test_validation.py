"""Test Validation"""
from autolabel.configs import AutolabelConfig
from autolabel.data_loaders.validation import TaskDataValidation

CLASSIFICATION_CONFIG_SAMPLE_DICT = {
    "task_name": "LegalProvisionsClassification",
    "task_type": "classification",
    "dataset": {"label_column": "label", "delimiter": ","},
    "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
    "prompt": {
        "task_guidelines": "You are an expert at understanding legal contracts. Your job is to correctly classify legal provisions in contracts into one of the following categories.\nCategories:{labels}\n",
        "labels": [
            "Agreements",
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


def test_validate_classification_task():
    """Test Validate classification"""
    config = AutolabelConfig(CLASSIFICATION_CONFIG_SAMPLE_DICT)
    expected_output = expected_output = [
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
            "loc": "example",
            "msg": "str type expected",
            "row_num": 3,
            "type": "type_error.str",
        },
        {
            "loc": "label",
            "msg": "str type expected",
            "row_num": 3,
            "type": "type_error.str",
        },
    ]
    data = [
        {"question": "s", "answer": "ee"},
        {"example": "s", "label": 1},
        {"example": "s", "label": "['w', 'd']"},
        {"example": 1234, "label": 443},
        {"example": "qo", "label": "wiw"},
    ]
    data_validation = TaskDataValidation(
        task_type=config.task_type(),
        label_column=config.label_column(),
        example_template=config.example_template(),
    )
    error_table = data_validation.validate(data=data)

    for exp_out, err_out in zip(expected_output, error_table):
        assert exp_out == err_out


def test_validate_ner_task():
    """Test Validate NamedEntityRecognition"""
    config = AutolabelConfig(NER_CONFIG_SAMPLE_DICT)

    expected_output = [
        {
            "loc": "__root__",
            "msg": "Expecting ':' delimiter: line 1 column 26 (char 25)",
            "row_num": 1,
            "type": "value_error.jsondecode",
        },
        {
            "loc": "CategorizedLabels",
            "msg": "field required",
            "row_num": 3,
            "type": "value_error.missing",
        },
    ]

    data = [
        {
            "example": "example1",
            "CategorizedLabels": '{"Location": ["Okla"], "Miscellaneous": []}',
        },
        {
            "example": "example2",
            "CategorizedLabels": '{"Location":["Texas"], ""Miscellaneous"": []}',
        },
        {
            "example": "example2",
            "CategorizedLabels": '{"Location":["Texas"], "Miscellaneous": ["USDA", "PPAS"]}',
        },
        {
            "example": "example3",
            "label": '{"Location":["Texas"], "Miscellaneous": ["USDA", "PPAS"]}',
        },
    ]

    data_validation = TaskDataValidation(
        task_type=config.task_type(),
        label_column=config.label_column(),
        example_template=config.example_template(),
    )

    error_table = data_validation.validate(data=data)
    for exp_out, err_out in zip(expected_output, error_table):
        assert exp_out == err_out


def test_columns():
    import pytest

    """Test Validate NamedEntityRecognition"""
    config = AutolabelConfig(NER_CONFIG_SAMPLE_DICT)

    data_validation = TaskDataValidation(
        task_type=config.task_type(),
        label_column=config.label_column(),
        example_template=config.example_template(),
    )

    with pytest.raises(
        AssertionError, match=r"columns={'example'} missing in config.example_template"
    ):
        data_validation.validate_dataset_columns(
            dataset_columns=["input", "CategorizedLabels"]
        )


if __name__ == "__main__":
    test_validate_classification_task()
