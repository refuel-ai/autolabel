from typing import Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine

DB_ENGINE = None
DB_PATH = ".oracle.db"


def create_db_engine(db_path: Optional[str] = DB_PATH) -> Engine:
    global DB_ENGINE
    if DB_ENGINE is None:
        DB_ENGINE = create_engine(f"sqlite:///{db_path}")
    return DB_ENGINE
