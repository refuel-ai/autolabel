from .base import Base
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from langchain.schema import Generation
import json

from autolabel.schema import CacheEntry


class CacheEntryModel(Base):
    __tablename__ = "generation_cache"

    model_name = Column(String(50))
    prompt = Column(Text)
    model_params_string = Column(Text)
    generations = Column(JSON)

    def __repr__(self):
        return f"<Cache(model_name={self.model_name},prompt={self.prompt},max_tokens={self.max_tokens})>"

    @classmethod
    def get(cls, db, cache_entry: CacheEntry):
        looked_up_entry = (
            db.query(cls)
            .filter(
                cls.model_name == cache_entry.model_name
                and cls.prompt == cache_entry.prompt
                and cls.model_params_string == cache_entry.model_params_string
            )
            .first()
        )

        generations = [Generation(**gen) for gen in looked_up_entry["generations"]]

        entry = CacheEntry(
            model_name=looked_up_entry.model_name,
            prompt=looked_up_entry.prompt,
            model_params_string=looked_up_entry.model_params_string,
            generations=generations,
        )
        return entry

    @classmethod
    def insert(cls, db, cache_entry: BaseModel):
        generations = {"generations": [gen.dict() for gen in cache_entry.generations]}
        db_object = cls(
            model_name=cache_entry.model_name,
            prompt=cache_entry.prompt,
            model_params_string=cache_entry.model_params_string,
            generations=json.dumps(generations),
        )
        db.add(db_object)
        return cache_entry
