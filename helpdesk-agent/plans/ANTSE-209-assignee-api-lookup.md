# ANTSE-209: Assignee Lookup via API (Remove Stored Identity)

## Context
The PII scrubber anonymizes assignee names stored in the DB, causing inconsistency when the auto-responder tries to use them. Fix: never store assignee in the DB — always look up via Jira API at query time.

## Implementation

### 1. Modify `ingest/tickets.py`
- `_parse_ticket_cloud` (~line 86): stop extracting `assignee` and `assignee_id` fields
- `_upsert_ticket` (~line 119): stop writing assignee columns (set to empty string)

### 2. Modify `ingest/scrubber.py`
- Remove `_build_identity_map` assignee section (lines 38-43)
- Remove `anon_assignees` anonymization loop (lines 172-176)

### 3. Modify `faq/auto_responder.py`
- `_check_assignee_allowed` (~line 419): replace DB lookup with live Jira API call
  - `GET /rest/api/3/issue/{ticket_key}?fields=assignee`
  - Extract `accountId` from response
  - Compare against `AUTO_RESPOND_ASSIGNEES`
- Update `_has_agent_response` (~line 376) callers accordingly

### 4. Cleanup migration in `db.py`
- Add to `init_db()`:
  ```python
  conn.execute("UPDATE tickets SET assignee = '', assignee_id = '' WHERE assignee != '' OR assignee_id != ''")
  ```

## Files
- Modify: `ingest/tickets.py`, `ingest/scrubber.py`, `faq/auto_responder.py`, `db.py`

## Dependencies
None.

## Verification
- Pipeline runs end-to-end
- Auto-responder drafts correctly (assignee check works via API)
- No PII in DB: `SELECT DISTINCT assignee FROM tickets WHERE assignee != ''` returns empty
- `scrub_database(dry_run=True)` shows no assignee-related scrubbing
- Confirm OAuth token has `read:jira-work` scope (already used elsewhere)
