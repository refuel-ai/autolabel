from .base import BaseTask
from .classification import ClassificationTask

from refuel_oracle.config import Config

TASK_TYPE_TO_IMPLEMENTATION = {
    "binary_classification": ClassificationTask,
    "multiclass_classification": ClassificationTask
}

class TaskFactory:

    @staticmethod
    def from_config(config: Config) -> BaseTask:
        task_type = config.get_task_type()
        task_cls = TASK_TYPE_TO_IMPLEMENTATION[task_type]
        return task_cls(config)