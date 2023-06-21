from .base import Base
import uuid
import logging
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import json

from autolabel.schema import TaskRun

logger = logging.getLogger(__name__)


class TaskRunModel(Base):
    __tablename__ = "task_runs"

    id = Column(
        Integer,
        default=lambda: uuid.uuid4().int >> (128 - 32),
        primary_key=True,
    )
    task_id = Column(String(32), ForeignKey("tasks.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    dataset_id = Column(String(32), ForeignKey("datasets.id"))
    current_index = Column(Integer)
    error = Column(String(256))
    metrics = Column(Text)
    output_file = Column(String(50))
    status = Column(String(50))
    task = relationship("TaskModel", back_populates="task_runs")
    dataset = relationship("DatasetModel", back_populates="task_runs")
    annotations = relationship("AnnotationModel", back_populates="task_runs")

    def __repr__(self):
        return f"<TaskRunModel(id={self.id}, created_at={str(self.created_at)}, task_id={self.task_id}, dataset_id={self.dataset_id}, output_file={self.output_file}, current_index={self.current_index}, status={self.status}, error={self.error}, metrics={self.metrics})"

    @classmethod
    def create(cls, db, task_run: BaseModel):
        logger.debug(f"creating new task: {task_run}")
        db_object = cls(**task_run.dict())
        db.add(db_object)
        db.flush()
        db.refresh(db_object)
        logger.debug(f"created new task: {db_object}")
        return db_object

    @classmethod
    def get(cls, db, task_id: str, dataset_id: str):
        return (
            db.query(cls)
            .filter(cls.task_id == task_id, cls.dataset_id == dataset_id)
            .first()
        )

    @classmethod
    def from_pydantic(cls, task_run: BaseModel):
        return cls(**json.loads(task_run.json()))

    @classmethod
    def update(cls, db, task_run: BaseModel):
        task_run_id = task_run.id
        task_run_orm = db.query(cls).filter(cls.id == task_run_id).first()
        logger.debug(f"updating task_run: {task_run}")
        for key, value in task_run.dict().items():
            setattr(task_run_orm, key, value)
        logger.debug(f"task_run updated: {task_run}")
        return TaskRun.from_orm(task_run_orm)

    @classmethod
    def delete_by_id(cls, db, id: int):
        db.query(cls).filter(cls.id == id).delete()

    def delete(self, db):
        db.delete(self)
        db.flush()
