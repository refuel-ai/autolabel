from .base import Base
import uuid
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import json
from refuel_oracle.schema import TaskResult


class TaskResultModel(Base):
    __tablename__ = "task_results"

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
    task = relationship("TaskModel", back_populates="task_results")
    dataset = relationship("DatasetModel", back_populates="task_results")
    annotations = relationship("AnnotationModel", back_populates="task_results")

    def __repr__(self):
        return f"<TaskResultModel(id={self.id}, created_at={self.created_at}, task_id={self.task_id}, dataset_id={self.dataset_id}, output_file={self.output_file}, current_index={self.current_index}, status={self.status}, error={self.error}, metrics={self.metrics})"

    @classmethod
    def create(cls, db, task_result: BaseModel):
        logger.debug(f"creating new task: {task_result}")
        db_object = cls(**json.loads(task_result.json()))
        db.add(db_object)
        db_object = db.query(cls).order_by(cls.id.desc()).first()
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
    def from_pydantic(cls, task_result: BaseModel):
        return cls(**json.loads(task_result.json()))

    @classmethod
    def update(cls, db, task_result: BaseModel):
        task_result_id = task_result.id
        task_result_orm = db.query(cls).filter(cls.id == task_result_id).first()
        logger.debug(f"updating task_result: {task_result}")
        for key, value in task_result.dict().items():
            setattr(task_result_orm, key, value)
        logger.debug(f"task_result updated: {task_result}")
        return TaskResult.from_orm(task_result_orm)

    @classmethod
    def delete_by_id(cls, db, id: int):
        db.query(cls).filter(cls.id == id).delete()

    def delete(self, db):
        db.delete(self)
        db.flush()
