from .base import Base
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from langchain.schema import Generation
import json

from autolabel.schema import CacheEntry


class CacheEntryModel(Base):
    """an SQLAlchemy based Cache system for storing and retriving CacheEntries"""

    __tablename__ = "generation_cache"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))
    prompt = Column(Text)
    model_params = Column(Text)
    generations = Column(JSON)

    def __repr__(self):
        return f"<Cache(model_name={self.model_name},prompt={self.prompt},model_params={self.model_params},generations={self.generations})>"

    @classmethod
    def get(cls, db, cache_entry: CacheEntry):
        looked_up_entry = (
            db.query(cls)
            .filter(
                cls.model_name == cache_entry.model_name,
                cls.prompt == cache_entry.prompt,
                cls.model_params == cache_entry.model_params,
            )
            .first()
        )

        if not looked_up_entry:
            return None

        generations = json.loads(looked_up_entry.generations)["generations"]
        generations = [Generation(**gen) for gen in generations]

        entry = CacheEntry(
            model_name=looked_up_entry.model_name,
            prompt=looked_up_entry.prompt,
            model_params=looked_up_entry.model_params,
            generations=generations,
        )
        return entry

    @classmethod
    def insert(cls, db, cache_entry: BaseModel):
        generations = {"generations": [gen.dict() for gen in cache_entry.generations]}
        db_object = cls(
            model_name=cache_entry.model_name,
            prompt=cache_entry.prompt,
            model_params=cache_entry.model_params,
            generations=json.dumps(generations),
        )
        db.add(db_object)
        return cache_entry

    @classmethod
    def clear(cls, db):
        db.query(cls).delete()
