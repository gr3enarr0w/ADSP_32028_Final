"""Slack signal ingestion via Slack MCP tools.

Channels:
    forum-atlassian-cloud (C08PVUD4S1W)
    forum-ants-tools (C0500LDQTJ7)

Requires Slack MCP to be configured globally at ~/.claude/.mcp.json.
This module provides helper functions for processing Slack messages
that are fetched via the MCP get_channel_history tool.
"""

import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# Channel IDs for target channels
CHANNELS = {
    "forum-atlassian-cloud": "C08PVUD4S1W",
    "forum-ants-tools": "C0500LDQTJ7",
}


def store_signal(conn, channel, thread_ts, message_text, user_id,
                 signal_type, sentiment=None, topic=None,
                 related_tickets=None, ticket_key=None,
                 is_resolved=False, thread_replies=None):
    """Store a classified Slack signal in the database."""
    conn.execute("""
        INSERT OR IGNORE INTO slack_signals
            (channel, thread_ts, message_text, user_id, signal_type,
             sentiment, topic, related_tickets, ticket_key, ingested_at,
             is_resolved, thread_replies)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        channel,
        thread_ts,
        message_text,
        user_id,
        signal_type,
        sentiment,
        topic,
        json.dumps(related_tickets or []),
        ticket_key,
        datetime.now(timezone.utc).isoformat(),
        int(is_resolved),
        json.dumps(thread_replies) if thread_replies else None,
    ))


def get_signal_summary(conn):
    """Get summary of stored signals by type and channel."""
    rows = conn.execute("""
        SELECT channel, signal_type, COUNT(*) as cnt, sentiment
        FROM slack_signals
        GROUP BY channel, signal_type, sentiment
        ORDER BY cnt DESC
    """).fetchall()
    return [dict(r) for r in rows]


def get_gap_signals(conn, limit=20):
    """Get top signals for gap analysis integration (weight 0.3)."""
    rows = conn.execute("""
        SELECT topic, signal_type, COUNT(*) as occurrences, sentiment
        FROM slack_signals
        WHERE topic IS NOT NULL
        GROUP BY topic, signal_type
        ORDER BY occurrences DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def check_resolved_signals():
    """Scan Slack signals for ✅ reactions and fetch thread replies.

    For each unresolved signal, checks the Slack API for reactions.
    If ✅ (white_check_mark or heavy_check_mark) is found, marks as
    resolved and fetches thread replies.
    """
    import requests
    from config import SLACK_BOT_TOKEN, SLACK_XOXC_TOKEN, SLACK_XOXD_TOKEN
    from db import get_db_conn

    # Use bot token if available, otherwise fall back to MCP session tokens
    token = SLACK_BOT_TOKEN or SLACK_XOXC_TOKEN
    if not token:
        log.debug("No Slack token configured — skipping resolved signal check")
        return 0

    headers = {"Authorization": f"Bearer {token}"}
    if SLACK_XOXD_TOKEN and not SLACK_BOT_TOKEN:
        headers["Cookie"] = f"d={SLACK_XOXD_TOKEN}"
    resolved_emojis = {"white_check_mark", "heavy_check_mark", "ballot_box_with_check"}

    # Map channel names to IDs
    channel_id_map = {v: v for v in CHANNELS.values()}  # ID → ID
    channel_id_map.update({name: cid for name, cid in CHANNELS.items()})  # name → ID

    with get_db_conn() as conn:
        # Get unresolved signals that have a thread_ts (can check reactions)
        pending = conn.execute("""
            SELECT id, channel, thread_ts
            FROM slack_signals
            WHERE (is_resolved = 0 OR is_resolved IS NULL)
              AND thread_ts IS NOT NULL
        """).fetchall()

    if not pending:
        return 0

    log.info("Checking %d Slack signals for ✅ reactions", len(pending))
    resolved_count = 0

    for row in pending:
        try:
            # Resolve channel name to ID
            channel_id = channel_id_map.get(row["channel"], row["channel"])

            # Check reactions on the message
            resp = requests.get(
                "https://slack.com/api/reactions.get",
                headers=headers,
                params={"channel": channel_id, "timestamp": row["thread_ts"]},
                timeout=10,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            if not data.get("ok"):
                continue

            message = data.get("message", {})
            reactions = message.get("reactions", [])
            has_checkmark = any(
                r["name"] in resolved_emojis for r in reactions
            )

            if not has_checkmark:
                continue

            # Fetch thread replies
            replies_resp = requests.get(
                "https://slack.com/api/conversations.replies",
                headers=headers,
                params={
                    "channel": channel_id,
                    "ts": row["thread_ts"],
                    "limit": 20,
                },
                timeout=10,
            )

            thread_texts = []
            if replies_resp.status_code == 200:
                replies_data = replies_resp.json()
                if replies_data.get("ok"):
                    for msg in replies_data.get("messages", [])[1:]:  # skip the parent
                        text = msg.get("text", "").strip()
                        if text:
                            thread_texts.append(text[:500])

            # Update signal as resolved
            with get_db_conn() as conn:
                conn.execute(
                    """UPDATE slack_signals
                       SET is_resolved = 1,
                           signal_type = 'resolved',
                           thread_replies = ?
                       WHERE id = ?""",
                    (json.dumps(thread_texts) if thread_texts else None, row["id"]),
                )
            resolved_count += 1

        except Exception as e:
            log.debug("Could not check reactions for signal %d: %s", row["id"], e)

    if resolved_count:
        log.info("Marked %d Slack signals as resolved", resolved_count)
    return resolved_count


def get_resolved_threads(limit=20):
    """Get resolved Slack threads with their Q&A content for FAQ enrichment."""
    from db import get_db_conn

    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT message_text, thread_replies, topic, channel
            FROM slack_signals
            WHERE is_resolved = 1
              AND thread_replies IS NOT NULL
            ORDER BY ingested_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

    results = []
    for r in rows:
        replies = json.loads(r["thread_replies"]) if r["thread_replies"] else []
        results.append({
            "question": r["message_text"],
            "answers": replies,
            "topic": r["topic"],
            "channel": r["channel"],
        })
    return results
