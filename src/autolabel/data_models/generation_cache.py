from .base import Base
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from langchain.schema import Generation, ChatGeneration
import json
import time

from autolabel.schema import GenerationCacheEntry


class GenerationCacheEntryModel(Base):
    """an SQLAlchemy based Cache system for storing and retriving CacheEntries"""

    __tablename__ = "generation_cache"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))
    prompt = Column(Text)
    model_params = Column(Text)
    generations = Column(JSON)
    creation_time_ms = Column(Integer)
    ttl_ms = Column(Integer)

    def __repr__(self):
        return f"<Cache(model_name={self.model_name},prompt={self.prompt},model_params={self.model_params},generations={self.generations})>"

    @classmethod
    def get(cls, db, cache_entry: GenerationCacheEntry):
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
        generations = [
            Generation(**gen) if gen["type"] == "Generation" else ChatGeneration(**gen)
            for gen in generations
        ]

        entry = GenerationCacheEntry(
            model_name=looked_up_entry.model_name,
            prompt=looked_up_entry.prompt,
            model_params=looked_up_entry.model_params,
            generations=generations,
            creation_time_ms=looked_up_entry.creation_time_ms,
            ttl_ms=looked_up_entry.ttl_ms,
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
            creation_time_ms=int(time.time() * 1000),
            ttl_ms=cache_entry.ttl_ms,
        )
        db.add(db_object)
        db.commit()
        return cache_entry

    @classmethod
    def clear(cls, db, use_ttl: bool = True) -> None:
        if use_ttl:
            current_time_ms = int(time.time() * 1000)
            db.query(cls).filter(
                current_time_ms - cls.creation_time_ms > cls.ttl_ms
            ).delete()
        else:
            db.query(cls).delete()
        db.commit()
