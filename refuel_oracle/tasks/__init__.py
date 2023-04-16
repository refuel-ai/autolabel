from loguru import logger
from refuel_oracle.config import Config

from .base import BaseTask
from .classification import ClassificationTask
from .completion import CompletionTask
from .entity_recognition import EntityRecognitionTask

TASK_TYPE_TO_IMPLEMENTATION = {
    "classification": ClassificationTask,
    "entity_recognition": EntityRecognitionTask,
    "completion": CompletionTask,
}


class TaskFactory:
    @staticmethod
    def from_config(config: Config) -> BaseTask:
        task_type = config.get_task_type()
        if task_type not in TASK_TYPE_TO_IMPLEMENTATION:
            logger.error(
                f"Task type {task_type} is not in the list of supported tasks: {TASK_TYPE_TO_IMPLEMENTATION.keys()}"
            )
            return None
        task_cls = TASK_TYPE_TO_IMPLEMENTATION[task_type]
        return task_cls(config)
