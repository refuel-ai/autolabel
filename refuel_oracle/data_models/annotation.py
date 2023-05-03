from .base import Base
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column, Integer, ForeignKey, JSON, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import json

from refuel_oracle.schema import LLMAnnotation


class AnnotationModel(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
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
    def create_from_llm_annotation(
        cls, db, llm_annotation: LLMAnnotation, index: int, task_result_id: int
    ):
        db_object = cls(
            llm_annotation=json.loads(llm_annotation.json()),
            index=index,
            task_result_id=task_result_id,
        )
        db.add(db_object)
        db_object = db.query(cls).order_by(cls.id.desc()).first()
        logger.debug(f"created new annotation: {db_object}")
        return db_object

    @classmethod
    def get_annotations_by_task_result_id(cls, db, task_result_id: int):
        annotations = (
            db.query(cls)
            .filter(cls.task_result_id == task_result_id)
            .order_by(cls.index)
            .distinct(cls.index)
            .all()
        )
        filtered_annotations = []
        ids = {}
        for annotation in annotations:
            if annotation.index not in ids:
                ids[annotation.index] = True
                filtered_annotations.append(annotation)
        return filtered_annotations

    @classmethod
    def from_pydantic(cls, annotation: BaseModel):
        return cls(**json.loads(annotation.json()))

    def delete(self, db):
        db.delete(self)
