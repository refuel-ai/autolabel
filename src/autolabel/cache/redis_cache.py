from autolabel.cache import BaseCache
from langchain.schema import Generation
from typing import Any, List
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class RedisCache(BaseCache):
    """A cache system implemented with Redis"""

    def __init__(self, endpoint: str, db: int = 0):
        self.endpoint = endpoint
        self.db = db

    def initialize(self):
        try:
            from redis import Redis

            self.redis = Redis.from_url(self.endpoint, db=self.db)
        except ImportError:
            raise ImportError(
                "redis is required to use the Redis Cache. Please install it with the following command: pip install redis"
            )

    def lookup(self, entry: Any) -> List[Generation]:
        """Retrieves an entry from the Cache. Returns an empty list [] if not found.
        Args:
            entry: Entry we wish to retrieve from the Cache
        Returns:
            result: Deserialized cache entry. None if entry not found.
        """
        redis_key = entry.get_id()
        if self.redis.exists(redis_key):
            logger.debug("Cache hit")
            generations = entry.deserialize_output(
                self.redis.get(redis_key).decode("utf-8")
            )
            return generations

        logger.debug("Cache miss")
        return None

    def update(self, entry: Any) -> None:
        """Inserts the provided entry into the Cache, overriding it if it already exists
        Args:
            entry: Entry we wish to put into the Cache
        """
        redis_key = entry.get_id()
        redis_value = entry.get_serialized_output()
        with self.redis.pipeline() as pipe:
            pipe.set(redis_key, redis_value)
            pipe.expire(redis_key, entry.ttl_ms // 1000)
            pipe.execute()

    def clear(self) -> None:
        """Clears the entire Cache"""
        self.redis.flushdb()
