"""Rising cluster OpsGenie alerts (ANTSE-319)."""

from __future__ import annotations

import logging
from typing import Any

from config import GEMINI_MODEL_ANALYSIS
from core.genai import get_genai_client
from plugins.alerting.clustering import DeltaResult
from plugins.alerting.opsgenie import OpsGenieClient

log = logging.getLogger(__name__)

_LABEL_PROMPT = """You label Jira Service Management ticket clusters for on-call alerting.

Given a preliminary cluster label and representative ticket summaries, return ONE concise
topic label (max 12 words) that describes the emerging issue. Use title case for major words.
Return only the label text — no quotes, bullets, or explanation.

Preliminary label: {initial_label}

Ticket summaries:
{summaries}
"""


def filter_alert_deltas(
    deltas: list[DeltaResult],
    velocity_threshold: float,
) -> list[DeltaResult]:
    """Return deltas that qualify for an alert (new or fast-growing)."""
    return [
        delta
        for delta in deltas
        if delta.is_new or delta.growth_rate > velocity_threshold
    ]


def lookup_summaries(conn, ticket_keys: list[str], *, limit: int = 8) -> list[str]:
    """Fetch ticket summaries from the tickets table for representative examples."""
    if not ticket_keys:
        return []
    keys = ticket_keys[:limit]
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"""
        SELECT ticket_key, summary
        FROM tickets
        WHERE ticket_key IN ({placeholders})
        ORDER BY created_at DESC
        """,
        keys,
    ).fetchall()
    by_key = {row["ticket_key"]: (row["summary"] or "").strip() for row in rows}
    summaries: list[str] = []
    for key in keys:
        text = by_key.get(key, "")
        if text:
            summaries.append(f"- {key}: {text}")
    return summaries


def enrich_topic_label(initial_label: str, summaries: list[str]) -> str:
    """Refine the cluster label with Gemini using representative ticket summaries."""
    if not summaries:
        return initial_label

    prompt = _LABEL_PROMPT.format(
        initial_label=initial_label,
        summaries="\n".join(summaries),
    )
    try:
        from google.genai import types as genai_types
        client = get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL_ANALYSIS,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                http_options=genai_types.HttpOptions(timeout=90_000),
            ),
        )
        label = (response.text or "").strip().splitlines()[0].strip().strip('"').rstrip(".")
        return label or initial_label
    except Exception as exc:
        log.warning("[cluster_alert] Gemini label enrichment failed: %s", exc)
        return initial_label


def _format_velocity(delta: DeltaResult) -> str:
    if delta.is_new:
        return "new cluster"
    pct = round(delta.growth_rate * 100)
    return f"+{pct}% growth"


def build_cluster_alert(
    delta: DeltaResult,
    topic_label: str,
    summaries: list[str],
    *,
    run_date: str,
) -> tuple[str, str, str, dict[str, str]]:
    """Build OpsGenie message, description, alias, and details for a cluster delta."""
    velocity = _format_velocity(delta)
    example_keys = ", ".join(delta.ticket_keys[:8])
    message = f"Rising topic cluster: {topic_label} ({delta.size} tickets, {velocity})"
    if len(message) > 130:
        message = message[:127] + "..."

    summary_block = "\n".join(summaries) if summaries else "(no summaries found)"
    description = (
        f"Topic: {topic_label}\n"
        f"Cluster ID: {delta.cluster_id}\n"
        f"Ticket count: {delta.size}\n"
        f"Velocity: {velocity} (growth_rate={delta.growth_rate})\n"
        f"Run date: {run_date}\n"
        f"Example keys: {example_keys or 'n/a'}\n\n"
        f"Representative tickets:\n{summary_block}"
    )
    alias = f"ai-helpdesk-cluster-{delta.cluster_id}-{run_date}"
    details = {
        "cluster_id": str(delta.cluster_id),
        "topic_label": topic_label,
        "ticket_count": str(delta.size),
        "growth_rate": str(delta.growth_rate),
        "is_new": str(delta.is_new),
        "example_keys": example_keys,
        "run_date": run_date,
    }
    return message, description, alias, details


def process_cluster_alerts(
    conn,
    deltas: list[DeltaResult],
    *,
    velocity_threshold: float,
    run_date: str,
    cfg: dict[str, Any],
) -> int:
    """Send OpsGenie alerts for qualifying cluster deltas. Returns alert count."""
    if not cfg.get("cluster_alerts_enabled", True):
        log.debug("[cluster_alert] cluster alerts disabled in config")
        return 0
    if not cfg.get("opsgenie_enabled", True):
        log.debug("[cluster_alert] OpsGenie disabled in config")
        return 0

    client = OpsGenieClient.from_config(cfg)
    if not client.enabled:
        log.warning("[cluster_alert] OPSGENIE_API_KEY not set — skipping alerts")
        return 0

    priority = str(cfg.get("opsgenie_priority", "P3"))
    tags = ["ai-helpdesk", "topic-cluster"]
    alert_deltas = filter_alert_deltas(deltas, velocity_threshold)
    if not alert_deltas:
        return 0

    sent = 0
    for delta in alert_deltas:
        summaries = lookup_summaries(conn, delta.ticket_keys)
        topic_label = enrich_topic_label(delta.label, summaries)
        message, description, alias, details = build_cluster_alert(
            delta,
            topic_label,
            summaries,
            run_date=run_date,
        )
        if client.create_alert(
            message=message,
            description=description,
            alias=alias,
            priority=priority,
            tags=tags,
            details=details,
        ):
            sent += 1
    return sent
