from sqlalchemy.orm import sessionmaker
from typing import Dict, Any
from autolabel.schema import TransformCacheEntry
from autolabel.database import create_db_engine
from autolabel.data_models import Base
from typing import Optional
from autolabel.data_models import TransformCacheEntryModel
from .base import BaseCache
import logging

logger = logging.getLogger(__name__)


class SQLAlchemyTransformCache(BaseCache):
    """
    A cache system implemented with SQL Alchemy for storing the output of transforms.
    This cache system is used to avoid re-computing the output of transforms that have already been computed.
    This currently stores the input and the outputs of the transform.
    Caching is based on the transform name, params and input.
    """

    def __init__(self):
        self.engine = create_db_engine()
        self.base = Base
        self.session = None
        self.initialize()

    def initialize(self):
        self.base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def lookup(self, entry: TransformCacheEntry) -> Optional[Dict[str, Any]]:
        """Retrieves an entry from the Cache. Returns None if not found.
        Args:
            entry: TransformCacheEntry we wish to retrieve from the Cache
        Returns:
            result: The output of the transform for this input. None if not found.
        """
        cache_entry = TransformCacheEntryModel.get(self.session, entry)
        if cache_entry is None:
            logger.debug("Cache miss")
            return None

        logger.debug("Cache hit")
        return cache_entry.output

    def update(self, entry: TransformCacheEntry) -> None:
        """Inserts the provided TransformCacheEntry into the Cache, overriding it if it already exists
        Args:
            entry: TransformCacheEntry we wish to put into the Cache
        """
        TransformCacheEntryModel.insert(self.session, entry)

    def clear(self, use_ttl: bool = True) -> None:
        """Clears the entire Cache based on ttl"""
        TransformCacheEntryModel.clear(self.session, use_ttl=use_ttl)
