"""Shared FAQ source tracking."""

import hashlib
from datetime import datetime, timezone

from db import get_db_conn


def upsert_faq_source(source_type: str, source_id: str, title: str, content: str):
    """Track a FAQ source with content hash for change detection."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    with get_db_conn() as conn:
        conn.execute("""
            INSERT INTO faq_sources (source_type, source_id, title, content_hash, last_fetched)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_type, source_id) DO UPDATE SET
                title = excluded.title, content_hash = excluded.content_hash,
                last_fetched = excluded.last_fetched
        """, (source_type, source_id, title, content_hash, datetime.now(timezone.utc).isoformat()))
