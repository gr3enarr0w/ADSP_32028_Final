"""Volume and content anomaly OpsGenie alerts — ANTSE-315, ANTSE-449."""

from __future__ import annotations

from datetime import datetime, timezone

from plugins.alerting import opsgenie
from plugins.alerting.anomaly import AnomalyResult, CategoryAnomalyResult


def _priority_for_zscore(zscore: float) -> str:
    if zscore > 10:
        return "P1"
    if zscore > 5:
        return "P2"
    return "P3"


def _top_categories(breakdown: dict[str, int], limit: int = 3) -> list[tuple[str, int]]:
    return sorted(breakdown.items(), key=lambda item: item[1], reverse=True)[:limit]


def _build_description(result: AnomalyResult) -> str:
    lines = [
        f"Segment: {result.segment}",
        f"Ticket count: {result.ticket_count}",
        f"Z-score: {result.zscore:.1f}",
        f"Rolling mean: {result.rolling_mean:.4f}",
        f"Rolling std: {result.rolling_std:.4f}",
    ]
    top = _top_categories(result.category_breakdown)
    if top:
        lines.append("Top categories:")
        for category, count in top:
            lines.append(f"  - {category}: {count}")
    return "\n".join(lines)


def fire_anomaly_alert(result: AnomalyResult) -> bool:
    """Fire an OpsGenie alert for a volume anomaly."""
    now = datetime.now(timezone.utc)
    alias = f"anomaly-{result.segment}-{now.strftime('%Y%m%dT%H')}"
    message = (
        f"Volume anomaly: {result.ticket_count} tickets in {result.segment} "
        f"(z={result.zscore:.1f})"
    )
    return opsgenie.post_alert(
        message=message,
        alias=alias,
        description=_build_description(result),
        priority=_priority_for_zscore(result.zscore),
        tags=["alerting", "anomaly", "volume", "ai-helpdesk"],
    )


def _build_content_anomaly_description(result: CategoryAnomalyResult) -> str:
    """Build a human-readable OpsGenie description for a content anomaly."""
    kind = "novel category" if result.is_novel else "category spike"
    lines = [
        f"Type: {kind}",
        f"Category: {result.category}",
        f"Segment: {result.segment}",
        f"Recent count: {result.recent_count}",
        f"Z-score: {result.zscore:.2f}",
        f"Rolling mean: {result.rolling_mean:.4f}",
        f"Rolling std: {result.rolling_std:.4f}",
        f"Run date: {result.run_date}",
    ]
    return "\n".join(lines)


def _content_anomaly_priority(result: CategoryAnomalyResult) -> str:
    """P2 for category spike, P3 for novel category (ANTSE-449 spec)."""
    return "P2" if result.is_spike else "P3"


def fire_content_anomaly_alert(result: CategoryAnomalyResult) -> bool:
    """Fire an OpsGenie alert for a content anomaly (category spike or novel category).

    Alias format: ``content-anomaly-{category}-{date}``
    Priority: P2 for spikes, P3 for novel categories.

    Args:
        result: A CategoryAnomalyResult describing the anomaly.

    Returns:
        True if the alert was accepted by OpsGenie, False otherwise.
    """
    safe_category = result.category.lower().replace(" ", "_").replace("/", "_")
    alias = f"content-anomaly-{safe_category}-{result.run_date}"
    kind = "spike" if result.is_spike else "novel"
    message = (
        f"Content anomaly [{kind}]: {result.category} "
        f"({result.recent_count} tickets, z={result.zscore:.1f})"
    )
    if len(message) > 130:
        message = message[:127] + "..."

    return opsgenie.post_alert(
        message=message,
        alias=alias,
        description=_build_content_anomaly_description(result),
        priority=_content_anomaly_priority(result),
        tags=["alerting", "anomaly", "content", "ai-helpdesk"],
    )


def fire_content_anomaly_alerts(
    results: list[CategoryAnomalyResult],
) -> int:
    """Fire OpsGenie alerts for a list of content anomaly results.

    Args:
        results: CategoryAnomalyResult items from CategoryAnomalyDetector.

    Returns:
        Number of alerts successfully sent.
    """
    sent = 0
    for result in results:
        if fire_content_anomaly_alert(result):
            sent += 1
    return sent
