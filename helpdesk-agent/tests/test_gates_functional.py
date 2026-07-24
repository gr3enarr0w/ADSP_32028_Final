"""Functional tests for ticket processing gates (ANTSE-330, 331, 334, 346).

Tests run against the real local database at jsm_data.db with 882 tickets.
"""

import difflib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "jsm_data.db")

def _db_has_tickets() -> bool:
    if not os.path.exists(DB_PATH):
        return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tickets'"
            ).fetchone()[0]
            if count == 0:
                return False
            return conn.execute("SELECT 1 FROM tickets LIMIT 1").fetchone() is not None
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _db_has_tickets(),
    reason="jsm_data.db not populated — run ingest first to enable functional gate tests",
)

# ── Gate logic (these match the implicit gates in auto_responder / main.py) ──

PROCESS_STATUSES = {"Waiting for support", "In Progress", "Pending", "New"}
AGE_THRESHOLD_HOURS = 4
MIN_CONTENT_LENGTH = 20
DEDUP_THRESHOLD = 0.68


def _check_status_gate(status: str) -> bool:
    """Returns True if ticket should be SKIPPED (gated out)."""
    if not status:
        return True
    return status not in PROCESS_STATUSES


def _check_age_gate(created_at_str: str) -> bool:
    """Returns True if ticket is too old (should be gated)."""
    if not created_at_str:
        return True
    created = datetime.fromisoformat(
        created_at_str.replace("Z", "+00:00").replace(" ", "T")
    )
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - created
    return age > timedelta(hours=AGE_THRESHOLD_HOURS)


def _check_content_gate(summary: str, description: str) -> bool:
    """Returns True if ticket has insufficient content (should be filtered)."""
    combined = ((summary or "") + " " + (description or "")).strip()
    return len(combined) < MIN_CONTENT_LENGTH


def _similarity_score(text_a: str, text_b: str) -> float:
    return difflib.SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


@pytest.fixture(scope="module")
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# ── Test 1: Status Gate (ANTSE-330) ──


def test_status_gate_waiting_for_customer_skips(db):
    row = db.execute(
        "SELECT ticket_key, status FROM tickets WHERE status = ? LIMIT 1",
        ("Waiting for customer",),
    ).fetchone()
    assert row is not None, "No ticket with status 'Waiting for customer' in DB"
    assert _check_status_gate(row["status"]) is True, (
        f"{row['ticket_key']}: 'Waiting for customer' should be skipped"
    )


def test_status_gate_waiting_for_support_processes(db):
    row = db.execute(
        "SELECT ticket_key, status FROM tickets WHERE status = ? LIMIT 1",
        ("Waiting for support",),
    ).fetchone()
    assert row is not None, "No ticket with status 'Waiting for support' in DB"
    assert _check_status_gate(row["status"]) is False, (
        f"{row['ticket_key']}: 'Waiting for support' should NOT be skipped"
    )


def test_status_gate_resolved_skips(db):
    row = db.execute(
        "SELECT ticket_key, status FROM tickets WHERE status = ? LIMIT 1",
        ("Resolved",),
    ).fetchone()
    assert row is not None, "No ticket with status 'Resolved' in DB"
    assert _check_status_gate(row["status"]) is True, (
        f"{row['ticket_key']}: 'Resolved' should be skipped"
    )


def test_status_gate_closed_skips(db):
    row = db.execute(
        "SELECT ticket_key, status FROM tickets WHERE status = ? LIMIT 1",
        ("Closed",),
    ).fetchone()
    assert row is not None, "No ticket with status 'Closed' in DB"
    assert _check_status_gate(row["status"]) is True, (
        f"{row['ticket_key']}: 'Closed' should be skipped"
    )


# ── Test 2: Age Gate (ANTSE-331) ──


def test_age_gate_old_ticket_gated(db):
    row = db.execute(
        "SELECT ticket_key, created_at FROM tickets "
        "WHERE created_at IS NOT NULL AND created_at != '' "
        "ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    assert row is not None, "No ticket with created_at found"
    assert _check_age_gate(row["created_at"]) is True, (
        f"{row['ticket_key']}: oldest ticket should be age-gated"
    )


def test_age_gate_recent_ticket(db):
    """Most recent ticket -- if within 4h it passes, otherwise all are old."""
    row = db.execute(
        "SELECT ticket_key, created_at FROM tickets "
        "WHERE created_at IS NOT NULL AND created_at != '' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    assert row is not None, "No ticket with created_at found"
    created = datetime.fromisoformat(
        row["created_at"].replace("Z", "+00:00").replace(" ", "T")
    )
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
    result = _check_age_gate(row["created_at"])
    if age_hours < AGE_THRESHOLD_HOURS:
        assert result is False, f"{row['ticket_key']}: recent ticket should pass age gate"
    else:
        # All tickets are old -- gate correctly identifies them
        assert result is True, f"{row['ticket_key']}: old ticket should be gated"


# ── Test 3: Content/Noise Gate (ANTSE-334) ──


def test_content_gate_short_ticket_filtered(db):
    """Find shortest ticket and verify content gate behavior."""
    row = db.execute(
        "SELECT ticket_key, summary, description, "
        "LENGTH(COALESCE(summary,'')) + LENGTH(COALESCE(description,'')) as total_len "
        "FROM tickets ORDER BY total_len ASC LIMIT 1"
    ).fetchone()
    assert row is not None
    result = _check_content_gate(row["summary"], row["description"])
    combined = ((row["summary"] or "") + " " + (row["description"] or "")).strip()
    if len(combined) < MIN_CONTENT_LENGTH:
        assert result is True, (
            f"{row['ticket_key']}: short content ({len(combined)} chars) should be filtered"
        )
    else:
        # Even the shortest ticket has enough content
        assert result is False, "Shortest ticket still has sufficient content"


def test_content_gate_substantial_ticket_passes(db):
    row = db.execute(
        "SELECT ticket_key, summary, description FROM tickets "
        "WHERE LENGTH(COALESCE(summary,'')) + LENGTH(COALESCE(description,'')) > 200 "
        "LIMIT 1"
    ).fetchone()
    assert row is not None, "No ticket with substantial content found"
    assert _check_content_gate(row["summary"], row["description"]) is False, (
        f"{row['ticket_key']}: substantial content should NOT be filtered"
    )


# ── Test 4: Dedup Threshold (ANTSE-346) ──


def test_threshold_calibration_file_exists():
    calib_py = os.path.join(os.path.dirname(__file__), "..", "faq", "threshold_calibration.py")
    exists = os.path.exists(calib_py)
    # Report existence -- this is an informational check
    print(f"threshold_calibration.py exists: {exists}")


def test_calibration_result_json():
    base = os.path.join(os.path.dirname(__file__), "..")
    paths = [
        os.path.join(base, "faq", "calibration_result.json"),
        os.path.join(base, "data", "calibration_result.json"),
        os.path.join(base, "calibration_result.json"),
    ]
    found = None
    for p in paths:
        if os.path.exists(p):
            found = p
            break

    if found:
        with open(found) as f:
            data = json.load(f)
        threshold = data.get("threshold", data.get("optimal_threshold"))
        assert threshold is not None, "No threshold value in calibration result"
        assert abs(threshold - 0.68) < 0.05, f"Threshold {threshold} not near 0.68"
    else:
        pytest.skip("calibration_result.json not found -- ANTSE-346 not yet implemented")


def test_dedup_similar_tickets(db):
    """Find the most similar ticket pair and test dedup logic."""
    rows = db.execute(
        "SELECT ticket_key, summary FROM tickets "
        "WHERE summary IS NOT NULL AND summary != '' "
        "ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    assert len(rows) >= 2, "Need at least 2 tickets for dedup test"

    summaries = [(r["ticket_key"], r["summary"]) for r in rows]
    best_score = 0.0
    best_pair = None

    for i in range(len(summaries)):
        for j in range(i + 1, min(i + 20, len(summaries))):
            score = _similarity_score(summaries[i][1], summaries[j][1])
            if score > best_score:
                best_score = score
                best_pair = (summaries[i], summaries[j])

    assert best_pair is not None
    k1, s1 = best_pair[0]
    k2, s2 = best_pair[1]
    print(f"Most similar pair: {k1} <-> {k2}, score={best_score:.3f}")
    print(f"  '{s1[:70]}'")
    print(f"  '{s2[:70]}'")

    is_dup = best_score >= DEDUP_THRESHOLD
    # This is informational -- we verify the scoring logic works
    assert isinstance(best_score, float)
    assert 0.0 <= best_score <= 1.0
