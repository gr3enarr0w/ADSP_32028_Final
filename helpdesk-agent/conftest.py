"""
Root pytest configuration — test isolation for SQLite and Postgres.

SQLite:  each test that needs isolation uses
         `monkeypatch.setattr("db.DB_PATH", str(tmp_path / "test.db"))`.
         No global fixture needed.

Postgres: when DATABASE_URL is set we redirect all tests to a dedicated
          `<dbname>_test` database and wrap each test in a SAVEPOINT so
          teardown can roll back every write.

This avoids the O(N*tables) TRUNCATE overhead per test and keeps all test
data out of the shared Postgres database.
"""

import os
from urllib.parse import urlsplit, urlunsplit

import pytest

_ORIG_DATABASE_URL = os.getenv("DATABASE_URL", "")


def _test_database_url(database_url: str) -> str:
    """Return the sibling <dbname>_test URL, preserving query parameters."""
    parsed = urlsplit(database_url)
    path = parsed.path.rstrip("/")
    if "/" not in path:
        raise ValueError(f"Invalid DATABASE_URL path: {database_url!r}")
    base_path, dbname = path.rsplit("/", 1)
    return urlunsplit(parsed._replace(path=f"{base_path}/{dbname}_test"))

# Redirect to the test database BEFORE db.py is imported by any test module.
# conftest.py is loaded by pytest before test collection, so this runs before
# any `import db` in test files.
if _ORIG_DATABASE_URL:
    os.environ["DATABASE_URL"] = _test_database_url(_ORIG_DATABASE_URL)


# ── Session fixture: create + initialise the test database once ─────────────

@pytest.fixture(scope="session", autouse=True)
def _postgres_test_db():
    """Create <dbname>_test and run init_db() once per test session."""
    if not _ORIG_DATABASE_URL:
        return  # SQLite path — no-op

    import psycopg2
    from urllib.parse import urlparse

    test_url = os.environ["DATABASE_URL"]
    parsed = urlparse(test_url)
    test_dbname = parsed.path.lstrip("/")

    admin_conn = psycopg2.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        dbname="postgres",
    )
    admin_conn.autocommit = True
    cur = admin_conn.cursor()
    try:
        cur.execute(f'CREATE DATABASE "{test_dbname}"')
    except psycopg2.errors.DuplicateDatabase:
        pass
    finally:
        admin_conn.close()

    # Drop and recreate the public schema each session so schema changes
    # (new columns, type changes like REAL→DOUBLE PRECISION) are always applied.
    schema_conn = psycopg2.connect(test_url)
    schema_conn.autocommit = True
    schema_conn.cursor().execute(
        "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
    )
    schema_conn.close()

    from db import init_db
    init_db()


# ── Session fixture: shared connection for transactional rollback ─────────────

class _NoCommitConn:
    """Proxy over _PgConn that suppresses commit() and close().

    Passed to every test via monkeypatched db.get_db so the outer transaction
    is never prematurely ended by test code that uses get_db_conn().
    """

    def __init__(self, wrapped):
        self._w = wrapped

    def commit(self):
        pass

    def close(self):
        pass

    def rollback(self):
        pass

    def execute(self, *args, **kwargs):
        return self._w.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._w.executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self._w.executescript(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._w, name)


@pytest.fixture(scope="session")
def _pg_shared_conn(_postgres_test_db):
    """One long-lived psycopg2 connection for SAVEPOINT-based test isolation."""
    if not _ORIG_DATABASE_URL:
        yield None
        return

    import psycopg2
    from db import _PgConn

    raw = psycopg2.connect(os.environ["DATABASE_URL"])
    raw.autocommit = False
    conn = _PgConn(raw)
    yield conn
    raw.rollback()
    raw.close()


@pytest.fixture(autouse=True)
def _postgres_isolate(_pg_shared_conn, monkeypatch):
    """Wrap each Postgres test in a SAVEPOINT and roll it back at teardown."""
    if not _ORIG_DATABASE_URL:
        yield
        return

    import db as _db

    proxy = _NoCommitConn(_pg_shared_conn)
    monkeypatch.setattr(_db, "get_db", lambda: proxy)

    _pg_shared_conn.execute("SAVEPOINT _test_sp")
    yield
    _pg_shared_conn.execute("ROLLBACK TO SAVEPOINT _test_sp")
