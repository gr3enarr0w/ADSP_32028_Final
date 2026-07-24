"""Isolation Forest anomaly detector (ANTSE-316).

Activates automatically when the control chart's live false-positive rate
(measured from persisted control-chart scores) exceeds a configurable threshold.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sklearn.ensemble import IsolationForest

from plugins.alerting.anomaly import (
    BASELINE_EXCLUDE_DATES,
    AnomalyResult,
    _parse_datetime,
    segment_key,
)

log = logging.getLogger(__name__)

MIN_FPR_ROWS = 48
CONTROL_CHART_SCORES_TABLE = "anomaly_scores_control_chart"


def _is_business_hours(hour: int) -> int:
    return 1 if 8 <= hour <= 17 else 0


def _ticket_features(created_at: datetime) -> list[float]:
    utc = created_at.astimezone(timezone.utc)
    return [float(utc.hour), float(utc.weekday()), float(_is_business_hours(utc.hour))]


def _measure_fpr(conn, window_days: int = 7) -> float:
    """Measure empirical FPR from recent persisted control-chart scores."""
    _ensure_control_chart_scores_table(conn)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    rows = conn.execute(
        f"""
        SELECT is_anomaly, zscore FROM {CONTROL_CHART_SCORES_TABLE}
        WHERE scored_at >= ?
        """,
        (cutoff,),
    ).fetchall()

    total = len(rows)
    if total < MIN_FPR_ROWS:
        return 0.0

    false_positives = sum(
        1 for row in rows if int(row["is_anomaly"]) == 1 and float(row["zscore"]) < 10
    )
    return false_positives / total


def _ensure_control_chart_scores_table(conn) -> None:
    """Create sidecar storage that tracks control-chart scores every cycle."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CONTROL_CHART_SCORES_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scored_at TEXT NOT NULL,
            segment TEXT,
            window_minutes INTEGER,
            ticket_count INTEGER,
            is_anomaly INTEGER DEFAULT 0,
            zscore REAL,
            category_breakdown TEXT
        )
        """
    )


def store_control_chart_score(
    conn,
    result: AnomalyResult,
    *,
    window_minutes: int = 60,
    scored_at: datetime | None = None,
) -> None:
    """Persist control-chart score history used for IF activation FPR checks."""
    _ensure_control_chart_scores_table(conn)
    conn.execute(
        f"""
        INSERT INTO {CONTROL_CHART_SCORES_TABLE}
            (scored_at, segment, window_minutes, ticket_count, is_anomaly, zscore, category_breakdown)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (scored_at or datetime.now(timezone.utc)).isoformat(),
            result.segment,
            window_minutes,
            result.ticket_count,
            int(result.is_anomaly),
            result.zscore,
            json.dumps(result.category_breakdown),
        ),
    )


class IsolationForestDetector:
    """Ticket-pattern anomaly detector; auto-activates on high control-chart FPR."""

    def __init__(
        self,
        contamination: float = 0.05,
        fpr_threshold: float = 0.05,
        fpr_window_days: int = 7,
    ):
        self.contamination = contamination
        self.fpr_threshold = fpr_threshold
        self.fpr_window_days = fpr_window_days
        self._model: IsolationForest | None = None

    def is_activated(self, conn) -> bool:
        fpr = _measure_fpr(conn, window_days=self.fpr_window_days)
        log.info(
            "[alerting] isolation forest FPR=%.4f threshold=%.4f (window=%dd, min_rows=%d)",
            fpr,
            self.fpr_threshold,
            self.fpr_window_days,
            MIN_FPR_ROWS,
        )
        return fpr > self.fpr_threshold

    def fit(self, conn, *, window_days: int = 90, reference_time: datetime | None = None) -> bool:
        """Train on per-ticket temporal features from the last window_days."""
        ref = reference_time or datetime.now(timezone.utc)
        cutoff = ref - timedelta(days=window_days)
        cutoff_text = cutoff.isoformat()

        rows = conn.execute(
            """
            SELECT created_at FROM tickets
            WHERE created_at IS NOT NULL
              AND created_at >= ?
              AND substr(created_at, 1, 10) NOT IN ({exclude})
            ORDER BY created_at
            """.format(exclude=",".join("?" * len(BASELINE_EXCLUDE_DATES))),
            (cutoff_text, *BASELINE_EXCLUDE_DATES),
        ).fetchall()

        features: list[list[float]] = []
        for row in rows:
            created_at = _parse_datetime(row["created_at"])
            if created_at is None:
                continue
            features.append(_ticket_features(created_at))

        if not features:
            self._model = None
            log.warning("[alerting] isolation forest fit skipped — no training tickets")
            return False

        self._model = IsolationForest(contamination=self.contamination, random_state=42)
        self._model.fit(features)
        log.info("[alerting] isolation forest fitted on %d tickets", len(features))
        return True

    def score(
        self,
        conn,
        *,
        window_minutes: int = 60,
        reference_time: datetime | None = None,
    ) -> AnomalyResult:
        """Score the current window using aggregated ticket features."""
        ref = reference_time or datetime.now(timezone.utc)
        seg = segment_key(ref)
        window_start = ref - timedelta(minutes=window_minutes)

        ticket_rows = conn.execute(
            """
            SELECT created_at FROM tickets
            WHERE created_at > ? AND created_at <= ?
            """,
            (window_start.isoformat(), ref.isoformat()),
        ).fetchall()

        ticket_count = len(ticket_rows)

        breakdown_rows = conn.execute(
            """
            SELECT COALESCE(c.category, 'Unknown') AS category, COUNT(*) AS cnt
            FROM tickets t
            LEFT JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
            WHERE t.created_at > ? AND t.created_at <= ?
            GROUP BY COALESCE(c.category, 'Unknown')
            """,
            (window_start.isoformat(), ref.isoformat()),
        ).fetchall()
        category_breakdown = {row["category"]: int(row["cnt"]) for row in breakdown_rows}

        if self._model is None:
            return AnomalyResult(
                is_anomaly=False,
                zscore=0.0,
                category_breakdown=category_breakdown,
                segment=seg,
                ticket_count=ticket_count,
            )

        window_features = self._aggregate_window_features(ticket_rows, ref)
        prediction = int(self._model.predict([window_features])[0])
        decision = float(self._model.decision_function([window_features])[0])
        is_anomaly = prediction == -1
        zscore = round(-decision, 4)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            zscore=zscore,
            category_breakdown=category_breakdown,
            segment=seg,
            ticket_count=ticket_count,
        )

    @staticmethod
    def _aggregate_window_features(ticket_rows, ref: datetime) -> list[float]:
        """Mean per-ticket features across the window; fall back to ref time if empty."""
        parsed = [_parse_datetime(row["created_at"]) for row in ticket_rows]
        timestamps = [ts for ts in parsed if ts is not None]
        if not timestamps:
            return _ticket_features(ref)

        n = len(timestamps)
        sums = [0.0, 0.0, 0.0]
        for ts in timestamps:
            feat = _ticket_features(ts)
            for i in range(3):
                sums[i] += feat[i]
        return [s / n for s in sums]


def validate_labeled_windows(
    conn,
    detector: IsolationForestDetector,
    *,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Measure IF TPR/FPR on the same labeled windows used by control chart."""
    from plugins.alerting.anomaly import _sample_normal_windows

    ref = reference_time or datetime.now(timezone.utc)
    detector.fit(conn, reference_time=ref)

    anomaly_hits = 0
    peak_times = (
        ("2026-03-16", "T15:00:00+00:00"),
        ("2026-01-06", "T11:00:00+00:00"),
    )
    for day, peak_time in peak_times:
        eval_ref = datetime.fromisoformat(f"{day}{peak_time}")
        result = detector.score(conn, window_minutes=60, reference_time=eval_ref)
        if result.is_anomaly:
            anomaly_hits += 1

    normal_windows = _sample_normal_windows(conn, count=20)
    false_positives = 0
    for eval_ref in normal_windows:
        result = detector.score(conn, window_minutes=60, reference_time=eval_ref)
        if result.is_anomaly:
            false_positives += 1

    tpr = anomaly_hits / len(peak_times)
    fpr = false_positives / max(len(normal_windows), 1)
    return {
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "anomaly_hits": anomaly_hits,
        "false_positives": false_positives,
        "normal_windows_sampled": len(normal_windows),
    }


def compare_with_control_chart(
    conn,
    if_detector: IsolationForestDetector,
    control_chart_detector,
    *,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Compare IF and control-chart quality on identical labeled windows."""
    from plugins.alerting.anomaly import validate_labeled_windows as validate_control_chart

    control = validate_control_chart(conn, control_chart_detector)
    if_metrics = validate_labeled_windows(conn, if_detector, reference_time=reference_time)

    if if_metrics["tpr"] >= control["tpr"] and if_metrics["fpr"] <= control["fpr"]:
        recommended = "isolation_forest"
    else:
        recommended = "control_chart"

    return {
        "control_chart": control,
        "isolation_forest": if_metrics,
        "recommended_detector": recommended,
    }
