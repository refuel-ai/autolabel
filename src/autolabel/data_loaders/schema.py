"""Data and Schema Validation"""

import re
import json

from typing import Dict, List, Union
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from autolabel.configs import AutolabelConfig
from pydantic import BaseModel, create_model, ValidationError, root_validator
from pydantic.types import StrictStr


@dataclass
class NERTaskValidate:
    label_column: str

    @staticmethod
    def validate(value: str):
        if value.startswith("{") and value.endswith("}"):
            try:
                _ = json.loads(value)
            except JSONDecodeError:
                raise
        else:
            raise


@dataclass
class ClassificationTaskValidate:
    label_column: str

    @staticmethod
    def validate(value: str):
        if value.startswith("[") and value.endswith("]"):
            try:
                value_list = eval(value)
                if not isinstance(value_list, list):
                    raise
            except SyntaxError:
                raise


@dataclass
class EMTaskValidate:
    label_column: str


@dataclass
class QATaskValidate:
    label_column: str


TaskTypeValidate = Union[
    NERTaskValidate,
    ClassificationTaskValidate,
    EMTaskValidate,
    QATaskValidate,
]


@dataclass
class DataValidationTasks:
    classification: TaskTypeValidate = ClassificationTaskValidate
    named_entity_recognition: TaskTypeValidate = NERTaskValidate
    entity_matching: TaskTypeValidate = EMTaskValidate
    question_answering: TaskTypeValidate = QATaskValidate


def data_validation_and_schema_check(schema: Dict, validation_task: TaskTypeValidate):
    Model = create_model("Model", **schema)

    class SchemaCheck(BaseModel):
        @classmethod
        def validate(cls, data):
            model = Model(**data)
            try:
                cls(**model.dict())
            except ValidationError as e:
                raise e

        @root_validator(pre=True, allow_reuse=True)
        def check_fields(cls, values):
            try:
                label_column_value = values[validation_task.label_column]
                validation_task.validate(label_column_value)
            except ValidationError as e:
                raise e

    return SchemaCheck


def fetch_expected_columns(
    example_template: str, pattern: str = r"\{([^}]*)\}"
) -> List:
    """_summary_

    Args:
        example_template (str): _description_
        pattern (str, optional): _description_. Defaults to r"\{([^}]*)}".

    Returns:
        List: _description_
    """
    column_name_lists = []
    for text in example_template.split("\n"):
        matches = re.findall(pattern, text)
        column_name_lists += matches
    return column_name_lists


@dataclass
class TaskDataValidation:
    task_type: str
    label_column: str
    example_template: str

    @property
    def expected_columns(self) -> List:
        return fetch_expected_columns(self.example_template)

    @property
    def schema(self) -> Dict:
        return {col: (StrictStr, ...) for col in self.expected_columns}

    @property
    def validation_task(
        self,
    ) -> TaskTypeValidate:
        validation_task = DataValidationTasks.__dict__[self.task_type]
        validation_task.label_column = self.label_column
        return validation_task

    def validate(self, data: List[dict]) -> List[Dict]:
        validation = data_validation_and_schema_check(
            self.schema,
            self.validation_task,
        )
        error_table = []
        for index, item in enumerate(data):
            try:
                validation.validate(item)
            except ValidationError as e:
                error_table += [{index: err} for err in e.errors()]
        return error_table


def test_validate_classification_task():
    """Test Validate classification"""
    config = AutolabelConfig(
        "/Users/sardhendu.mishra/workspace/research/autolabel/examples/ledgar/config_ledgar.json"
    )
    expected_output = [
        {
            0: {
                "loc": ("example",),
                "msg": "field required",
                "type": "value_error.missing",
            }
        },
        {
            0: {
                "loc": ("label",),
                "msg": "field required",
                "type": "value_error.missing",
            }
        },
        {1: {"loc": ("label",), "msg": "str type expected", "type": "type_error.str"}},
        {
            3: {
                "loc": ("example",),
                "msg": "str type expected",
                "type": "type_error.str",
            }
        },
        {3: {"loc": ("label",), "msg": "str type expected", "type": "type_error.str"}},
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
    config = AutolabelConfig(
        "/Users/sardhendu.mishra/workspace/research/autolabel/examples/conll2003/config_conll2003.json"
    )

    expected_output = [
        {
            1: {
                "ctx": {
                    "colno": 26,
                    "doc": '{"Location":["Texas"], ""Miscellaneous"": []}',
                    "lineno": 1,
                    "msg": "Expecting ':' delimiter",
                    "pos": 25,
                },
                "loc": ("__root__",),
                "msg": "Expecting ':' delimiter: line 1 column 26 (char 25)",
                "type": "value_error.jsondecode",
            }
        },
        {
            3: {
                "loc": ("CategorizedLabels",),
                "msg": "field required",
                "type": "value_error.missing",
            }
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


if __name__ == "__main__":
    test_validate_ner_task()
