from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import ROOT, get_settings


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


@lru_cache
def get_pool() -> ConnectionPool:
    pool = ConnectionPool(
        get_settings().database_url,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row},
        configure=_configure,
        open=True,
    )
    return pool


@contextmanager
def connection():
    with get_pool().connection() as conn:
        yield conn


def migrate() -> list[str]:
    applied = []
    # розширення треба створити до register_vector, тому окреме з'єднання без пулу
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        for path in sorted(Path(ROOT / "migrations").glob("*.sql")):
            conn.execute(path.read_text())
            applied.append(path.name)
    return applied
