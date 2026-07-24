"""Tests for reporting data builders."""

import sqlite3

import pytest

from db import get_db_conn, init_db
from reporting.reports import build_dashboard_data, build_ticket_summary


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    """Force SQLite for each test, overriding any Postgres proxy from conftest.

    Also patches reporting.reports.get_db since that module binds get_db
    locally at import time and won't pick up a db.get_db monkeypatch.
    """
    import db as _db
    import reporting.reports as _reports

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(_db, "DATABASE_URL", None)
    monkeypatch.setattr(_db, "DB_PATH", db_path)

    def _sqlite_get_db():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    monkeypatch.setattr(_db, "get_db", _sqlite_get_db)
    monkeypatch.setattr(_reports, "get_db", _sqlite_get_db)
    init_db()
    yield


def _insert_ticket(conn, ticket_key: str):
    conn.execute(
        "INSERT INTO tickets (ticket_key, summary) VALUES (?, ?)",
        (ticket_key, f"Summary for {ticket_key}"),
    )


def _insert_classification(
    conn,
    ticket_key: str,
    category: str,
    issue_type: str,
    confidence: float,
    affect_version: str | None = None,
):
    conn.execute(
        """
        INSERT INTO ticket_classifications
            (ticket_key, category, issue_type, confidence, affect_version)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ticket_key, category, issue_type, confidence, affect_version),
    )


def _insert_csat(conn, ticket_key: str, score: int, comment: str = "Helpful"):
    conn.execute(
        """
        INSERT INTO ticket_csat (ticket_key, csat_score, csat_comment, submitted_at)
        VALUES (?, ?, ?, ?)
        """,
        (ticket_key, score, comment, "2026-06-02T12:00:00+00:00"),
    )


def test_build_ticket_summary_includes_csat_metrics():
    with get_db_conn() as conn:
        _insert_ticket(conn, "TEST-1")
        _insert_ticket(conn, "TEST-2")
        _insert_ticket(conn, "TEST-3")
        _insert_classification(conn, "TEST-1", "access", "password_reset", 0.9, "Live 1.0")
        _insert_classification(conn, "TEST-2", "access", "password_reset", 0.7, "Live 1.0")
        _insert_classification(conn, "TEST-3", "billing", "invoice_help", 0.8, "Live 1.0")
        _insert_csat(conn, "TEST-1", 5)
        _insert_csat(conn, "TEST-3", 2)

    summary = build_ticket_summary(version="Live 1.0")

    assert summary["total_classified"] == 3
    assert summary["total_csat_responses"] == 2

    access = summary["categories"]["access"]
    assert access["total"] == 2
    assert access["csat_responses"] == 1
    assert access["issue_types"][0]["issue_type"] == "password_reset"
    assert access["issue_types"][0]["csat_responses"] == 1
    assert access["issue_types"][0]["avg_csat"] == 5.0

    billing = summary["categories"]["billing"]
    assert billing["csat_responses"] == 1
    assert billing["issue_types"][0]["avg_csat"] == 2.0


def test_build_dashboard_data_counts_csat_rows():
    with get_db_conn() as conn:
        _insert_ticket(conn, "TEST-10")
        _insert_classification(conn, "TEST-10", "access", "password_reset", 0.95)
        _insert_csat(conn, "TEST-10", 4)

    stats = build_dashboard_data()

    assert stats["tickets"] == 1
    assert stats["classifications"] == 1
    assert stats["csat_responses"] == 1
