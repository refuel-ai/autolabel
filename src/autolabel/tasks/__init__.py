from typing import Dict
import logging

from .base import BaseTask
from .classification import ClassificationTask
from .entity_matching import EntityMatchingTask
from .question_answering import QuestionAnsweringTask
from .named_entity_recognition import NamedEntityRecognitionTask

from autolabel.configs import AutolabelConfig
from autolabel.schema import TaskType

TASK_TYPE_TO_IMPLEMENTATION: Dict[TaskType, BaseTask] = {
    TaskType.CLASSIFICATION: ClassificationTask,
    TaskType.NAMED_ENTITY_RECOGNITION: NamedEntityRecognitionTask,
    TaskType.QUESTION_ANSWERING: QuestionAnsweringTask,
    TaskType.ENTITY_MATCHING: EntityMatchingTask,
}

logger = logging.getLogger(__name__)


class TaskFactory:
    @staticmethod
    def from_config(config: AutolabelConfig) -> BaseTask:
        try:
            task_type = TaskType(config.task_type())
            task_cls = TASK_TYPE_TO_IMPLEMENTATION[task_type]
            return task_cls(config)
        except ValueError as e:
            logger.error(
                f"{config.task_type()} is not in the list of supported tasks: {TASK_TYPE_TO_IMPLEMENTATION.keys()}"
            )
            return None
