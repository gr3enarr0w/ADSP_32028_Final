"""Tests for ticket ingestion — is_cloud / is_uat_only flag logic."""

import pytest
from unittest.mock import patch
from db import init_db, get_db_conn
from ingest.tickets import _is_uat_only, _parse_ticket_cloud, _upsert_ticket

CUTOVER = "2026-03-16"


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("db.DB_PATH", db_path):
        init_db()
        yield db_path


# --- _is_uat_only ---

@pytest.mark.parametrize("created_at,expected", [
    ("2026-03-15T23:59:59+00:00", 1),  # one second before cutover
    ("2026-03-16T00:00:00+00:00", 0),  # exactly at cutover
    ("2026-03-17T12:00:00+00:00", 0),  # post-cutover
    ("2026-01-01T00:00:00+00:00", 1),  # well before cutover
    (None, 0),                          # missing date
    ("not-a-date", 0),                  # malformed date
])
def test_is_uat_only(created_at, expected):
    with patch("ingest.tickets.CLOUD_CUTOVER_DATE", CUTOVER):
        assert _is_uat_only(created_at) == expected


# --- _parse_ticket_cloud ---

def _make_issue(key="<PROJECT_KEY>-1", created="2026-03-10T10:00:00+00:00"):
    return {
        "key": key,
        "fields": {
            "summary": "Test ticket",
            "description": None,
            "status": {"name": "Open"},
            "resolution": None,
            "versions": [],
            "components": [],
            "reporter": {"accountId": "u1", "emailAddress": "u@example.com"},
            "assignee": None,
            "created": created,
            "resolutiondate": None,
            "updated": created,
            "customfield_10010": None,
            "issuelinks": [],
        },
    }


def test_parse_sets_is_cloud():
    with patch("ingest.tickets.CLOUD_CUTOVER_DATE", CUTOVER):
        ticket = _parse_ticket_cloud(_make_issue())
    assert ticket["is_cloud"] == 1


def test_parse_uat_only_pre_cutover():
    with patch("ingest.tickets.CLOUD_CUTOVER_DATE", CUTOVER):
        ticket = _parse_ticket_cloud(_make_issue(created="2026-03-10T10:00:00+00:00"))
    assert ticket["is_uat_only"] == 1


def test_parse_not_uat_only_post_cutover():
    with patch("ingest.tickets.CLOUD_CUTOVER_DATE", CUTOVER):
        ticket = _parse_ticket_cloud(_make_issue(created="2026-04-01T10:00:00+00:00"))
    assert ticket["is_uat_only"] == 0


# --- _upsert_ticket round-trip ---

def test_upsert_persists_flags(use_memory_db):
    with patch("ingest.tickets.CLOUD_CUTOVER_DATE", CUTOVER), \
         patch("db.DB_PATH", use_memory_db):
        ticket = _parse_ticket_cloud(_make_issue(created="2026-03-10T10:00:00+00:00"))
        with get_db_conn() as conn:
            _upsert_ticket(conn, ticket)

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT is_cloud, is_uat_only FROM tickets WHERE ticket_key = '<PROJECT_KEY>-1'"
            ).fetchone()

    assert row["is_cloud"] == 1
    assert row["is_uat_only"] == 1


def test_upsert_updates_flags_on_conflict(use_memory_db):
    """ON CONFLICT should update is_cloud and is_uat_only."""
    with patch("ingest.tickets.CLOUD_CUTOVER_DATE", CUTOVER), \
         patch("db.DB_PATH", use_memory_db):
        # Insert with old values (simulate pre-migration row with defaults)
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, is_cloud, is_uat_only) VALUES (?, ?, 0, 0)",
                ("<PROJECT_KEY>-1", "old"),
            )

        ticket = _parse_ticket_cloud(_make_issue(created="2026-04-01T10:00:00+00:00"))
        with get_db_conn() as conn:
            _upsert_ticket(conn, ticket)

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT is_cloud, is_uat_only FROM tickets WHERE ticket_key = '<PROJECT_KEY>-1'"
            ).fetchone()

    assert row["is_cloud"] == 1
    assert row["is_uat_only"] == 0
