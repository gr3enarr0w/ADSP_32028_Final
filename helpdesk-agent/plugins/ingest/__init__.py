"""Ingest plugin — ticket fetching, comment fetching, and PII scrubbing.

Re-exports public API from ingest.* modules so new code can import from
plugins.ingest while old code continues to work.
"""

import logging

from plugins._protocol import BasePlugin

# -- Re-exports from ingest.tickets --
from ingest.tickets import (
    fetch_tickets_cloud,
    fetch_comments_cloud,
    api_get,
)

# -- Re-exports from ingest.scrubber --
from ingest.scrubber import (
    scrub_pii,
    audit_pii,
    scrub_database,
)

# -- Re-exports from ingest.slack --
from ingest.slack import (
    store_signal,
    get_signal_summary,
    get_gap_signals,
    check_resolved_signals,
    get_resolved_threads,
    CHANNELS,
)

# -- Re-exports from ingest.links --
from ingest.links import parse_issue_links

log = logging.getLogger(__name__)

__all__ = [
    "fetch_tickets_cloud",
    "fetch_comments_cloud",
    "api_get",
    "scrub_pii",
    "audit_pii",
    "scrub_database",
    "store_signal",
    "get_signal_summary",
    "get_gap_signals",
    "check_resolved_signals",
    "get_resolved_threads",
    "CHANNELS",
    "parse_issue_links",
    "plugin",
]


class IngestPlugin(BasePlugin):
    """Fetch JSM tickets/comments and scrub PII."""

    name = "ingest"

    def on_schedule(self) -> None:
        from config import JSM_PROJECTS
        log.info("[ingest] scheduled run — fetching tickets + comments")
        for project_key in JSM_PROJECTS:
            _, keys = fetch_tickets_cloud(project_key)
            if keys:
                fetch_comments_cloud(keys)
        scrub_database()
        log.info("[ingest] scheduled run complete")


plugin = IngestPlugin()
