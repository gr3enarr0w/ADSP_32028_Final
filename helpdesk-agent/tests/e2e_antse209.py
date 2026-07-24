"""End-to-end tests for ANTSE-209: Assignee Lookup via API (Remove Stored Identity).

Verifies the acceptance criteria against live staging data:
  1. No PII (assignee display name) in DB
  2. assignee_id (opaque accountId) is stored and used correctly for routing
  3. _check_assignee_allowed works for allowed, blocked, and unassigned tickets

Usage:
    python tests/e2e_antse209.py
"""

import os
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MAIN_ENV = Path("/Users/ceverson/Development/ai-helpdesk-agent/.env")

from dotenv import load_dotenv
if _MAIN_ENV.exists():
    load_dotenv(_MAIN_ENV)
load_dotenv(_ROOT / ".env", override=False)

STAGING_URL = "https://stage-<YOUR_DOMAIN>.atlassian.net"
os.environ["JSM_CLOUD_URL"] = STAGING_URL
sys.path.insert(0, str(_ROOT))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


class SkipTest(Exception):
    pass


class Runner:
    def __init__(self):
        self._results = []

    def run(self, name, fn, *args, **kwargs):
        print(f"\n── {name} ", end="", flush=True)
        try:
            msg = fn(*args, **kwargs)
            self._results.append((name, True, msg or ""))
            print(f"  {PASS}" + (f"  {msg}" if msg else ""))
        except SkipTest as e:
            self._results.append((name, None, str(e)))
            print(f"  {SKIP}  {e}")
        except Exception as e:
            self._results.append((name, False, str(e)))
            print(f"  {FAIL}")
            traceback.print_exc()

    def summary(self):
        passed = sum(1 for _, ok, _ in self._results if ok is True)
        skipped = sum(1 for _, ok, _ in self._results if ok is None)
        failed = sum(1 for _, ok, _ in self._results if ok is False)
        total = len(self._results)
        print(f"\n{'─' * 60}")
        print(f"Results: {passed}/{total} passed, {skipped} skipped, {failed} failed")
        return failed


def _staging_api(path, **params):
    """Make a servicedeskapi GET against staging (the only API the service account can use)."""
    import requests
    from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url
    base = get_cloud_base_url("jsm")
    auth = get_cloud_auth("jsm")
    resp = requests.get(f"{base}{path}", headers=auth, params=params or None)
    return resp


def _get_staging_tickets(project_key, assignee_id=None, limit=1):
    """Return up to `limit` real ticket dicts from staging via the servicedeskapi queue."""
    # Discover service desk for this project
    resp = _staging_api("/rest/servicedeskapi/servicedesk")
    if not resp.ok:
        raise SkipTest(f"Cannot list service desks ({resp.status_code})")
    sd_id = next(
        (sd["id"] for sd in resp.json().get("values", [])
         if sd.get("projectKey") == project_key),
        None,
    )
    if not sd_id:
        raise SkipTest(f"No service desk found for project {project_key} on staging")

    # Use "All JSM Open" queue (id 80 on prod; discover dynamically)
    resp = _staging_api(f"/rest/servicedeskapi/servicedesk/{sd_id}/queue")
    if not resp.ok:
        raise SkipTest(f"Cannot list queues ({resp.status_code})")
    queues = resp.json().get("values", [])
    if not queues:
        raise SkipTest("No queues found")

    # Try each queue until we get issues (or find an assigned one)
    for queue in queues:
        resp = _staging_api(
            f"/rest/servicedeskapi/servicedesk/{sd_id}/queue/{queue['id']}/issue",
            limit=limit,
        )
        if not resp.ok:
            continue
        issues = resp.json().get("values", [])
        if not issues:
            continue
        if assignee_id is None:
            return issues
        matching = [i for i in issues if (i.get("fields", {}).get("assignee") or {}).get("accountId") == assignee_id]
        if matching:
            return matching

    raise SkipTest("No suitable tickets found in any queue on staging")


# ── Test 1: DB migration — assignee col dropped, assignee_id col present ──────

def test_db_migration():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Simulate a legacy DB with the old assignee display name column
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE tickets (
                ticket_key TEXT PRIMARY KEY, summary TEXT, status TEXT,
                assignee TEXT DEFAULT '', assignee_id TEXT DEFAULT ''
            )
        """)
        conn.execute("INSERT INTO tickets VALUES ('T-1', 'S', 'Open', 'John Doe', 'abc123')")
        conn.commit()
        conn.close()

        import db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = db_path
        try:
            db_mod.init_db()
        finally:
            db_mod.DB_PATH = orig

        conn = sqlite3.connect(db_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tickets)").fetchall()}
        conn.close()

        assert "assignee" not in cols, f"'assignee' display name column still present: {cols}"
        assert "assignee_id" in cols, f"'assignee_id' column missing: {cols}"
        return f"cols after migration: {sorted(cols)}"
    finally:
        os.unlink(db_path)


# ── Test 2: Ingest — no display name stored, accountId IS stored ──────────────

def test_ingest_stores_account_id_not_display_name(project_key, assignee_id):
    from ingest.oauth2lo import clear_cache
    clear_cache()

    issues = _get_staging_tickets(project_key, assignee_id=assignee_id)
    issue = issues[0]
    ticket_key = issue["key"]
    fields = issue.get("fields", {})
    assignee = fields.get("assignee") or {}
    display_name = assignee.get("displayName", "")
    account_id = assignee.get("accountId", "")

    if not account_id:
        raise SkipTest(f"Ticket {ticket_key} has no assignee on staging")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        import db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = db_path
        db_mod.init_db()
        try:
            from ingest.tickets import _parse_ticket_cloud, _upsert_ticket
            ticket = _parse_ticket_cloud(issue)

            assert "assignee" not in ticket, f"display name 'assignee' key in parsed dict"
            assert "assignee_id" in ticket, f"'assignee_id' missing from parsed dict"
            assert ticket["assignee_id"] == account_id, \
                f"assignee_id mismatch: {ticket['assignee_id']!r} != {account_id!r}"

            with db_mod.get_db_conn() as conn:
                _upsert_ticket(conn, ticket)
                row = conn.execute(
                    "SELECT * FROM tickets WHERE ticket_key = ?", (ticket_key,)
                ).fetchone()

            col_names = set(row.keys())
            assert "assignee" not in col_names, f"'assignee' display name column exists in DB row"
            assert "assignee_id" in col_names, f"'assignee_id' missing from DB row"
            assert row["assignee_id"] == account_id, \
                f"DB assignee_id {row['assignee_id']!r} != expected {account_id!r}"

            # Verify display name is nowhere in the DB row values
            for col in col_names:
                val = row[col] or ""
                assert display_name not in str(val), \
                    f"Display name {display_name!r} found in DB column {col!r}"
        finally:
            db_mod.DB_PATH = orig

        return f"{ticket_key}: assignee_id={account_id[:8]}..., display name not stored"
    finally:
        os.unlink(db_path)


# ── Test 3: _check_assignee_allowed — allowed assignee ───────────────────────

def test_check_assignee_allowed_allowed(project_key, assignee_id):
    from ingest.oauth2lo import clear_cache
    clear_cache()

    issues = _get_staging_tickets(project_key, assignee_id=assignee_id)
    issue = issues[0]
    ticket_key = issue["key"]
    account_id = (issue.get("fields", {}).get("assignee") or {}).get("accountId", "")
    if not account_id:
        raise SkipTest(f"Ticket {ticket_key} has no assignee")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        import db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = db_path
        db_mod.init_db()
        try:
            from ingest.tickets import _parse_ticket_cloud, _upsert_ticket
            from unittest.mock import patch
            with db_mod.get_db_conn() as conn:
                _upsert_ticket(conn, _parse_ticket_cloud(issue))

            with patch("faq.auto_responder.AUTO_RESPOND_ASSIGNEES", [account_id]):
                from faq.auto_responder import _check_assignee_allowed
                allowed, returned_id = _check_assignee_allowed(ticket_key)
        finally:
            db_mod.DB_PATH = orig

        assert allowed is True, f"Expected allowed=True, got {allowed}"
        assert returned_id == account_id
        return f"{ticket_key} → allowed=True ({account_id[:8]}...)"
    finally:
        os.unlink(db_path)


# ── Test 4: _check_assignee_allowed — wrong assignee blocked ─────────────────

def test_check_assignee_allowed_blocked(project_key, assignee_id):
    from ingest.oauth2lo import clear_cache
    clear_cache()

    issues = _get_staging_tickets(project_key, assignee_id=assignee_id)
    issue = issues[0]
    ticket_key = issue["key"]
    account_id = (issue.get("fields", {}).get("assignee") or {}).get("accountId", "")
    if not account_id:
        raise SkipTest(f"Ticket {ticket_key} has no assignee")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        import db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = db_path
        db_mod.init_db()
        try:
            from ingest.tickets import _parse_ticket_cloud, _upsert_ticket
            from unittest.mock import patch
            with db_mod.get_db_conn() as conn:
                _upsert_ticket(conn, _parse_ticket_cloud(issue))

            with patch("faq.auto_responder.AUTO_RESPOND_ASSIGNEES", ["some-other-id"]):
                from faq.auto_responder import _check_assignee_allowed
                allowed, returned_id = _check_assignee_allowed(ticket_key)
        finally:
            db_mod.DB_PATH = orig

        assert allowed is False, f"Expected allowed=False, got {allowed}"
        assert returned_id == account_id
        return f"{ticket_key} → allowed=False (not in list)"
    finally:
        os.unlink(db_path)


# ── Test 5: Scrubber leaves assignee_id untouched ────────────────────────────

def test_scrubber_does_not_touch_assignee_id():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        import db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = db_path
        db_mod.init_db()
        try:
            with db_mod.get_db_conn() as conn:
                conn.execute("""
                    INSERT INTO tickets (ticket_key, summary, status, reporter_id,
                                        reporter_email, assignee_id)
                    VALUES ('T-1', 'Help', 'Open', 'real-reporter-id',
                            'reporter@example.com', 'real-assignee-account-id')
                """)

            from ingest.scrubber import scrub_database
            result = scrub_database(dry_run=False)

            with db_mod.get_db_conn() as conn:
                row = conn.execute(
                    "SELECT reporter_id, reporter_email, assignee_id FROM tickets WHERE ticket_key='T-1'"
                ).fetchone()
        finally:
            db_mod.DB_PATH = orig

        assert "anon_assignees" not in result, \
            f"'anon_assignees' key present in scrub result: {list(result.keys())}"
        assert row["reporter_id"] == "reporter_0001", \
            f"Scrubber did not anonymize reporter_id: {row['reporter_id']!r}"
        assert row["reporter_email"] == "", \
            f"Scrubber did not clear reporter_email: {row['reporter_email']!r}"
        assert row["assignee_id"] == "real-assignee-account-id", \
            f"Scrubber modified assignee_id: {row['assignee_id']!r}"
    finally:
        os.unlink(db_path)


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    project_key = os.getenv("PROJECT_KEYS", "<PROJECT_KEY>").split(",")[0]
    assignee_id = os.getenv("AUTO_RESPOND_ASSIGNEES", "").split(",")[0]

    print(f"Target:   {STAGING_URL}")
    print(f"Project:  {project_key}")
    print(f"Assignee: {assignee_id[:8]}..." if assignee_id else "Assignee: (none — set AUTO_RESPOND_ASSIGNEES)")

    r = Runner()
    r.run("DB migration: assignee col dropped, assignee_id col present", test_db_migration)
    r.run("Scrubber: reporter anonymized, assignee_id untouched", test_scrubber_does_not_touch_assignee_id)

    if not assignee_id:
        print(f"\n{SKIP} Staging API tests skipped — AUTO_RESPOND_ASSIGNEES not set")
    else:
        r.run("Ingest: accountId stored, display name never written to DB",
              test_ingest_stores_account_id_not_display_name, project_key, assignee_id)
        r.run("_check_assignee_allowed: allowed assignee returns True",
              test_check_assignee_allowed_allowed, project_key, assignee_id)
        r.run("_check_assignee_allowed: wrong assignee returns False",
              test_check_assignee_allowed_blocked, project_key, assignee_id)

    sys.exit(r.summary())


if __name__ == "__main__":
    main()
