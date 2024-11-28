import logging
from typing import Dict

from autolabel.configs import AutolabelConfig
from autolabel.schema import TaskType

from .attribute_extraction import AttributeExtractionTask
from .base import BaseTask

TASK_TYPE_TO_IMPLEMENTATION: Dict[TaskType, BaseTask] = {
    TaskType.ATTRIBUTE_EXTRACTION: AttributeExtractionTask,
}

logger = logging.getLogger(__name__)


class TaskFactory:
    @staticmethod
    def from_config(config: AutolabelConfig) -> BaseTask:
        try:
            task_type = TaskType(config.task_type())
            task_cls = TASK_TYPE_TO_IMPLEMENTATION[task_type]
            return task_cls(config)
        except ValueError as _:
            logger.error(
                f"{config.task_type()} is not in the list of supported tasks: {TASK_TYPE_TO_IMPLEMENTATION.keys()}",
            )
            return None
