"""Tests for database context manager commit/rollback behavior."""

import sqlite3
import pytest
from db import get_db_conn, init_db


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path, monkeypatch):
    """Force SQLite for each test, overriding any Postgres proxy from conftest."""
    import db as _db

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(_db, "DATABASE_URL", None)
    monkeypatch.setattr(_db, "DB_PATH", db_path)

    def _sqlite_get_db():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    monkeypatch.setattr(_db, "get_db", _sqlite_get_db)
    init_db()
    yield db_path


def test_commit_on_success():
    """Context manager commits on clean exit."""
    with get_db_conn() as conn:
        conn.execute("INSERT INTO tickets (ticket_key, summary) VALUES (?, ?)",
                     ("TEST-1", "Test ticket"))

    with get_db_conn() as conn:
        row = conn.execute("SELECT summary FROM tickets WHERE ticket_key = 'TEST-1'").fetchone()
    assert row is not None
    assert row["summary"] == "Test ticket"


def test_rollback_on_exception():
    """Context manager rolls back on exception."""
    with pytest.raises(ValueError):
        with get_db_conn() as conn:
            conn.execute("INSERT INTO tickets (ticket_key, summary) VALUES (?, ?)",
                         ("TEST-2", "Should be rolled back"))
            raise ValueError("test error")

    with get_db_conn() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_key = 'TEST-2'").fetchone()
    assert row is None


def test_connection_closed():
    """Context manager always closes the connection."""
    with get_db_conn() as conn:
        conn.execute("SELECT 1")

    # Connection should be closed — executing should raise
    with pytest.raises(Exception):
        conn.execute("SELECT 1")
