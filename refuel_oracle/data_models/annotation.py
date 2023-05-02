from .base import Base
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship
import json


class AnnotationModel(Base):
    __abstract__ = True  # this line is necessary

    id = Column(Integer, primary_key=True, autoincrement=True)
    index = Column(Integer)
    llm_annotation = Column(JSON)
    task_result_id = Column(Integer, ForeignKey("task_results.id"))
    task_results = relationship("TaskResultModel", back_populates="annotations")
    ANNOTATION_MODEL_CLASSES = {}

    def __repr__(self):
        return f"<AnnotationModel(id={self.id}, index={self.index}, annotation={self.llm_annotation})"

    @classmethod
    def create(cls, db, annotation: BaseModel):
        logger.debug(f"creating new annotation: {annotation}")
        db_object = cls(**json.loads(annotation.json()))
        db.add(db_object)
        db_object = db.query(cls).order_by(cls.id.desc()).first()
        logger.debug(f"created new annotation: {db_object}")
        return db_object

    @classmethod
    def from_pydantic(cls, annotation: BaseModel):
        return cls(**json.loads(annotation.json()))

    def delete(self, db):
        db.delete(self)
