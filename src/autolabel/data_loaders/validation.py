"""Data and Schema Validation"""

import re
import json

from typing import Dict, List, Union
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from pydantic import BaseModel, create_model, ValidationError, root_validator
from pydantic.types import StrictStr


@dataclass
class NERTaskValidate:
    """Validate NER Task

    The label column can either be a string or a json
    """

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
    """Validate Classification Task

    The label column can either be a string or a string of list
    """

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
    """Validate Entity Matching Task

    As of now we assume that the input label_column is a string
    """

    label_column: str


@dataclass
class QATaskValidate:
    """Validate Question Answering Task

    As of now we assume that the input label_column is a string
    """

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


def fetch_expected_columns(
    example_template: str, pattern: str = r"\{([^}]*)\}"
) -> List:
    """Fetch desired columns fro example template

    Args:
        example_template (str): example template from config file
        pattern (str, optional): regex pattern. Defaults to r"\{([^}]*)}".

    Returns:
        List: Returns list of columns
    """
    column_name_lists = []
    for text in example_template.split("\n"):
        matches = re.findall(pattern, text)
        column_name_lists += matches
    return column_name_lists


def data_validation_and_schema_check(schema: Dict, validation_task: TaskTypeValidate):
    """Validate data and schema

    Args:
        schema (Dict): input schema
        validation_task (TaskTypeValidate): validation task

    Raises:
        e: Validation error if the inputs are not string
        e: Validation error if validation_task fails

    Returns:
        DataValidation: Pydantic Model for validation
    """
    Model = create_model("Model", **schema)

    class DataValidation(BaseModel):
        """Data Validation"""

        @classmethod
        def validate(cls, data):
            """Valdiate data types"""
            model = Model(**data)
            try:
                cls(**model.dict())
            except ValidationError as e:
                raise e

        @root_validator(pre=True, allow_reuse=True)
        def check_fields(cls, values):
            """Validate data format"""
            try:
                label_column_value = values[validation_task.label_column]
                validation_task.validate(label_column_value)
            except ValidationError as e:
                raise e

    return DataValidation


class TaskDataValidation:
    """Task Validation"""

    def __init__(
        self,
        task_type: str,
        label_column: str,
        example_template: str,
    ):
        self.__expected_columns = fetch_expected_columns(example_template)
        self.__schema = {col: (StrictStr, ...) for col in self.__expected_columns}
        self.__validation_task = DataValidationTasks.__dict__[task_type]
        self.__validation_task.label_column = label_column

    @property
    def expected_columns(self) -> List:
        """Fetch expected columns"""
        return self.__expected_columns

    @property
    def schema(self) -> Dict:
        """Fecth Schema"""
        return self.__schema

    @property
    def validation_task(
        self,
    ) -> TaskTypeValidate:
        """Fetch validation task"""
        return self.__validation_task

    def validate(self, data: List[dict]) -> List[Dict]:
        """Validate Data"""
        validation = data_validation_and_schema_check(
            self.schema,
            self.validation_task,
        )
        error_messages = []
        for index, item in enumerate(data):
            try:
                validation.validate(item)
            except ValidationError as e:
                for err in e.errors():
                    field = ".".join(err["loc"])
                    error_messages += [
                        {
                            "row_num": index,
                            "loc": field,
                            "msg": err["msg"],
                            "type": err["type"],
                        }
                    ]
        return error_messages

    def validate_dataset_columns(self, dataset_columns: List):
        """Validate columns

        Valiate if the columns mentioned in example_template dataset are correct
        and are contined within the columns of the dataset(seed.csv)
        """
        missing_columns = set(self.expected_columns) - set(dataset_columns)

        assert (
            len(missing_columns) == 0
        ), f"columns={missing_columns} missing in config.example_template"
