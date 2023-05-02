from .base import Base
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
import json


class TaskResultModel(Base):
    __tablename__ = "task_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(32), ForeignKey("tasks.id"))
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
        return f"<TaskResultModel(id={self.id}, task_id={self.task_id}, dataset_id={self.dataset_id}, output_file={self.output_file}, current_index={self.current_index}, status={self.status}, error={self.error}, metrics={self.metrics})"

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

    def update(self, db, task_result: BaseModel):
        logger.debug(f"updating task_result: {task_result}")
        for key, value in json.loads(task_result.json()).items():
            setattr(self, key, value)
        db.flush()
        logger.debug(f"task_result updated: {self}")
        return self

    @classmethod
    def delete_by_id(cls, db, id: int):
        db.query(cls).filter(cls.id == id).delete()

    def delete(self, db):
        db.delete(self)
        db.flush()
