import logging
from typing import Dict

from autolabel.configs import AutolabelConfig
from autolabel.schema import TaskType

# need to be first
from autolabel.tasks.base import BaseTask

from .attribute_extraction import AttributeExtractionTask
from .classification import ClassificationTask
from .entity_matching import EntityMatchingTask
from .hierarchical_classification import HierarchicalClassificationTask
from .multilabel_classification import MultilabelClassificationTask
from .named_entity_recognition import NamedEntityRecognitionTask
from .question_answering import QuestionAnsweringTask

TASK_TYPE_TO_IMPLEMENTATION: Dict[TaskType, BaseTask] = {
    TaskType.CLASSIFICATION: ClassificationTask,
    TaskType.NAMED_ENTITY_RECOGNITION: NamedEntityRecognitionTask,
    TaskType.QUESTION_ANSWERING: QuestionAnsweringTask,
    TaskType.ENTITY_MATCHING: EntityMatchingTask,
    TaskType.MULTILABEL_CLASSIFICATION: MultilabelClassificationTask,
    TaskType.HIERARCHICAL_CLASSIFICATION: HierarchicalClassificationTask,
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
        except ValueError as e:
            logger.error(
                f"{config.task_type()} is not in the list of supported tasks: {TASK_TYPE_TO_IMPLEMENTATION.keys()}"
            )
            return None
