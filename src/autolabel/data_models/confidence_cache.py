from .base import Base
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Text, JSON
import json
import time

from autolabel.schema import ConfidenceCacheEntry


class ConfidenceCacheEntryModel(Base):
    """an SQLAlchemy based Cache system for storing and retriving CacheEntries"""

    __tablename__ = "confidence_cache"

    id = Column(Integer, primary_key=True)
    prompt = Column(Text)
    raw_response = Column(Text)
    logprobs = Column(JSON)
    score_type = Column(String)
    creation_time_ms = Column(Integer)
    ttl_ms = Column(Integer)

    def __repr__(self):
        return f"<Cache(prompt={self.prompt},raw_response={self.raw_response},logprobs={self.logprobs})>"

    @classmethod
    def get(cls, db, cache_entry: ConfidenceCacheEntry):
        looked_up_entry = (
            db.query(cls)
            .filter(
                cls.prompt == cache_entry.prompt,
                cls.raw_response == cache_entry.raw_response,
                cls.score_type == cache_entry.score_type,
            )
            .first()
        )

        if not looked_up_entry:
            return None

        entry = ConfidenceCacheEntry(
            prompt=looked_up_entry.prompt,
            raw_response=looked_up_entry.raw_response,
            logprobs=json.loads(looked_up_entry.logprobs),
            score_type=looked_up_entry.score_type,
            creation_time_ms=looked_up_entry.creation_time_ms,
            ttl_ms=looked_up_entry.ttl_ms,
        )
        return entry

    @classmethod
    def insert(cls, db, cache_entry: BaseModel):
        db_object = cls(
            prompt=cache_entry.prompt,
            raw_response=cache_entry.raw_response,
            logprobs=json.dumps(cache_entry.logprobs),
            score_type=cache_entry.score_type,
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
