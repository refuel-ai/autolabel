from typing import Dict, List, Union

from jsonschema import validate

from .base import BaseConfig


class TaskChainConfig(BaseConfig):
    """Class to parse and store configs for Task Chain"""

    # Top-level config keys
    TASK_NAME_KEY = "task_name"
    TASK_TYPE_KEY = "task_type"
    SUBTASKS_KEY = "subtasks"

    def __init__(self, config: Union[str, Dict], validate: bool = True) -> None:
        super().__init__(config, validate=validate)

    def _validate(self) -> bool:
        """Returns true if the config settings are valid"""
        from autolabel.configs.schema import schema

        for subtask in self.subtasks():
            validate(
                instance=subtask,
                schema=schema,
            )
        return True

    def task_name(self) -> str:
        return self.config.get(self.TASK_NAME_KEY, None)

    def task_type(self) -> str:
        """Returns the type of task we have configured the labeler to perform (e.g. Classification, Question Answering)"""
        return self.config.get(self.TASK_TYPE_KEY, None)

    def subtasks(self) -> List[Dict]:
        """Returns the subtasks that are part of the task chain"""
        return self.config.get(self.SUBTASKS_KEY, [])
