import pickle
import time

from sqlalchemy import TEXT, Column, Integer, String

from autolabel.transforms.schema import TransformCacheEntry

from .base import Base


class TransformCacheEntryModel(Base):

    """an SQLAlchemy based Cache system for storing and retriving CacheEntries"""

    __tablename__ = "transform_cache"

    id = Column(String, primary_key=True)
    transform_name = Column(String(50))
    transform_params = Column(TEXT)
    input = Column(TEXT)
    output = Column(TEXT)
    creation_time_ms = Column(Integer)
    ttl_ms = Column(Integer)

    def __repr__(self):
        return f"<TransformCache(id={self.id},transform_name={self.transform_name},transform_params={self.transform_params},input={self.input},output={self.output})>"

    @classmethod
    def get(cls, db, cache_entry: TransformCacheEntry) -> TransformCacheEntry:
        id = cache_entry.get_id()
        looked_up_entry = db.query(cls).filter(cls.id == id).first()

        if not looked_up_entry:
            return None

        entry = TransformCacheEntry(
            transform_name=looked_up_entry.transform_name,
            transform_params=pickle.loads(looked_up_entry.transform_params),
            input=pickle.loads(looked_up_entry.input),
            output=pickle.loads(looked_up_entry.output),
            creation_time_ms=looked_up_entry.creation_time_ms,
            ttl_ms=looked_up_entry.ttl_ms,
        )
        return entry

    @classmethod
    def insert(cls, db, cache_entry: TransformCacheEntry) -> None:
        db_object = cls(
            id=cache_entry.get_id(),
            transform_name=cache_entry.transform_name,
            transform_params=pickle.dumps(cache_entry.transform_params),
            input=pickle.dumps(cache_entry.input),
            output=pickle.dumps(cache_entry.output),
            creation_time_ms=int(time.time() * 1000),
            ttl_ms=cache_entry.ttl_ms,
        )
        db.merge(db_object)
        db.commit()
        return db_object

    @classmethod
    def clear(cls, db, use_ttl: bool = True) -> None:
        if use_ttl:
            current_time_ms = int(time.time() * 1000)
            db.query(cls).filter(
                current_time_ms - cls.creation_time_ms > cls.ttl_ms,
            ).delete()
        else:
            db.query(cls).delete()
        db.commit()
