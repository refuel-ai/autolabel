"""Data and Schema Validation"""

import re
import json
from functools import cached_property
from typing import Dict, List, Union, Optional
from json.decoder import JSONDecodeError
from pydantic import BaseModel, create_model, ValidationError, root_validator
from pydantic.types import StrictStr
from autolabel.configs import AutolabelConfig


class NERTaskValidate(BaseModel):
    """Validate NER Task

    The label column can either be a string or a json
    """

    label_column: str
    labels_set: set  # A NER Task should have a unique set of labels in config

    def validate(self, value: str):
        """Validate NER

        A NER label can only be a dictionary
        """
        # TODO: This can be made better
        if value.startswith("{") and value.endswith("}"):
            try:
                seed_labels = json.loads(value)
                unmatched_label = set(seed_labels.keys()) - self.labels_set
                if len(unmatched_label) != 0:
                    raise ValueError(
                        f"labels: {unmatched_label} do not match promt/labels provided in config "
                    )
            except JSONDecodeError:
                raise
        else:
            raise


class ClassificationTaskValidate(BaseModel):
    """Validate Classification Task

    The label column can either be a string or a string of list
    """

    label_column: str
    labels_set: set  # A classification Task should have a unique set of labels in config

    def validate(self, value: str):
        """Validate classification

        A classification label(ground_truth) could either be a list or string
        """
        # TODO: This can be made better
        if value.startswith("[") and value.endswith("]"):
            try:
                seed_labels = eval(value)
                if not isinstance(seed_labels, list):
                    raise
                unmatched_label = set(seed_labels) - self.labels_set
                if len(unmatched_label) != 0:
                    raise ValueError(
                        f"labels: {unmatched_label} do not match promt/labels provided in config "
                    )
            except SyntaxError:
                raise
        else:
            if value not in self.labels_set:
                raise ValueError(
                    f"labels: {value} do not match promt/labels provided in config "
                )


class EMTaskValidate(BaseModel):
    """Validate Entity Matching Task

    As of now we assume that the input label_column is a string
    """

    label_column: str
    labels_set: set  # An EntityMatching Task should have a unique set of labels in config

    def validate(self, value: str):
        if value not in self.labels_set:
            raise ValueError(
                f"labels: {value} do not match promt/labels provided in config "
            )


class QATaskValidate(BaseModel):
    """Validate Question Answering Task

    As of now we assume that the input label_column is a string
    """

    label_column: str
    labels_set: Optional[
        set
    ]  # A QA task may or may not have a unique set of label list


TaskTypeValidate = Union[
    NERTaskValidate,
    ClassificationTaskValidate,
    EMTaskValidate,
    QATaskValidate,
]


class DataValidationTasks(BaseModel):
    classification: TaskTypeValidate = ClassificationTaskValidate
    named_entity_recognition: TaskTypeValidate = NERTaskValidate
    entity_matching: TaskTypeValidate = EMTaskValidate
    question_answering: TaskTypeValidate = QATaskValidate


class TaskDataValidation:
    """Task Validation"""

    def __init__(self, config: AutolabelConfig):
        """Task Validation

        Args:
            config: AutolabelConfig = User passed parsed configuration
        """
        # the type of task, classification, named_entity_recognition, etc.., "config/task_type"
        task_type: str = config.task_type()
        # the label column as specified in config, "config/dataset/label_column"
        label_column: str = config.label_column()
        # list of valid labels provided in config "config/prompt/labels"
        labels_list: Optional[List] = config.labels_list()
        # example template from config "config/prompt/example_template"
        self.example_template: str = config.example_template()
        # Regex pattern to extract expected column from onfig.example_template()
        self.excepted_column_template_pattern: str = r"\{([^}]*)\}"

        # self.__expected_columns = self.fetch_expected_columns(example_template)
        self.__schema = {col: (StrictStr, ...) for col in self.expected_columns}

        self.__validation_task = DataValidationTasks.__dict__[task_type](
            label_column=label_column, labels_set=set(labels_list)
        )
        self.__data_validation = self.data_validation_and_schema_check(
            self.__validation_task
        )

    @cached_property
    def expected_columns(self) -> List:
        """Fetch expected columns"""
        column_name_lists = []
        for text in self.example_template.split("\n"):
            matches = re.findall(self.excepted_column_template_pattern, text)
            column_name_lists += matches
        return column_name_lists

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

    def data_validation_and_schema_check(self, validation_task: BaseModel):
        """Validate data format and datatype

        Args:
            validation_task (TaskTypeValidate): validation task

        Raises:
            e: Validation error if the inputs are not string
            e: Validation error if validation_task fails

        Returns:
            DataValidation: Pydantic Model for validation
        """
        Model = create_model("Model", **self.__schema)

        class DataValidation(BaseModel):
            """Data Validation"""

            # We define validate as a classmethod such that a dynamic `data` can be passed
            # iteratively to the validate method using `DataValidation.validate`
            @classmethod
            def validate(cls, data):
                """Valdiate data types"""
                model = Model(**data)
                try:
                    # We perform the normal pydantic validation here
                    # This checks both the Schema and also calls check_fields
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

    def validate(self, data: List[dict]) -> List[Dict]:
        """Validate Data"""
        error_messages = []
        for index, item in enumerate(data):
            try:
                self.__data_validation.validate(item)
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
