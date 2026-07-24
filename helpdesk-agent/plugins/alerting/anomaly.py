"""Control-chart volume anomaly detection (ANTSE-314).

Rolling mean ± Nσ baseline segmented by day-of-week and hour bucket.
Validated against labeled anomaly windows from ANTSE-313 volume EDA:

  - TPR > 80% on 2026-03-16 (cloud cutover) and 2026-01-06 (backlog burst)
  - FPR < 5% on 20 sampled normal windows
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

DOW_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
BASELINE_DAYS = 90
ROLLING_WINDOW_DAYS = 7
LABELED_ANOMALY_DATES = ("2026-03-16", "2026-01-06")
# Dates excluded from baseline training (labeled windows from ANTSE-313 EDA)
BASELINE_EXCLUDE_DATES = (
    "2026-03-16",
    "2026-03-17",
    "2026-03-18",
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
)


def hour_bucket(hour: int) -> str:
    """Map UTC hour to business / early / late bucket."""
    if 8 <= hour <= 17:
        return "business"
    if hour <= 7:
        return "early"
    return "late"


def segment_key(ts: datetime) -> str:
    """Build segment identifier, e.g. ``tue_business``."""
    utc = ts.astimezone(timezone.utc)
    return f"{DOW_NAMES[utc.weekday()]}_{hour_bucket(utc.hour)}"


def _parse_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class AnomalyResult:
    is_anomaly: bool
    zscore: float
    category_breakdown: dict[str, int] = field(default_factory=dict)
    segment: str = ""
    ticket_count: int = 0
    rolling_mean: float = 0.0
    rolling_std: float = 0.0


class ControlChartDetector:
    """Rolling mean ± Nσ control chart detector by DOW + hour bucket."""

    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma

    def build_baseline(self, conn, *, reference_time: datetime | None = None) -> int:
        """Compute rolling mean and σ per segment from the last 90 days."""
        ref = reference_time or datetime.now(timezone.utc)
        cutoff = ref - timedelta(days=BASELINE_DAYS)
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

        hourly_counts: dict[str, dict[datetime, int]] = {dow: {} for dow in _all_segments()}
        for row in rows:
            created_at = _parse_datetime(row["created_at"])
            if created_at is None:
                continue
            seg = segment_key(created_at)
            hour_start = created_at.replace(minute=0, second=0, microsecond=0)
            hourly_counts.setdefault(seg, {})
            hourly_counts[seg][hour_start] = hourly_counts[seg].get(hour_start, 0) + 1

        computed_at = ref.isoformat()
        conn.execute("DELETE FROM anomaly_baseline")
        written = 0
        for seg in sorted(hourly_counts):
            stats = _rolling_stats(hourly_counts[seg], ref, seg)
            if stats is None:
                continue
            mean, std = stats
            conn.execute(
                """
                INSERT INTO anomaly_baseline (segment, rolling_mean, rolling_std, computed_at)
                VALUES (?, ?, ?, ?)
                """,
                (seg, mean, std, computed_at),
            )
            written += 1
        return written

    def score(
        self,
        conn,
        *,
        window_minutes: int = 60,
        reference_time: datetime | None = None,
    ) -> AnomalyResult:
        """Compare recent ticket volume to the segment baseline."""
        if window_minutes != 60:
            raise ValueError(
                "ControlChartDetector currently supports only 60-minute windows; "
                "larger windows span multiple segments."
            )
        ref = reference_time or datetime.now(timezone.utc)
        seg = segment_key(ref)
        window_start = ref - timedelta(minutes=window_minutes)

        count_row = conn.execute(
            """
            SELECT COUNT(*) FROM tickets
            WHERE created_at > ? AND created_at <= ?
            """,
            (window_start.isoformat(), ref.isoformat()),
        ).fetchone()
        ticket_count = int(count_row[0])

        baseline = conn.execute(
            """
            SELECT rolling_mean, rolling_std FROM anomaly_baseline
            WHERE segment = ?
            """,
            (seg,),
        ).fetchone()

        if baseline is None:
            return AnomalyResult(
                is_anomaly=False,
                zscore=0.0,
                segment=seg,
                ticket_count=ticket_count,
            )

        rolling_mean = float(baseline["rolling_mean"])
        rolling_std = float(baseline["rolling_std"])
        if rolling_std <= 0:
            rolling_std = max(rolling_mean * 0.1, 0.5)

        hours = window_minutes / 60.0
        expected_mean = rolling_mean * hours
        expected_std = rolling_std * (hours ** 0.5)
        if expected_std <= 0:
            expected_std = max(expected_mean * 0.1, 0.5)

        zscore = (ticket_count - expected_mean) / expected_std
        threshold = expected_mean + self.sigma * expected_std
        is_anomaly = ticket_count > threshold

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
        category_breakdown = {
            row["category"]: int(row["cnt"])
            for row in breakdown_rows
        }

        return AnomalyResult(
            is_anomaly=is_anomaly,
            zscore=round(zscore, 4),
            category_breakdown=category_breakdown,
            segment=seg,
            ticket_count=ticket_count,
            rolling_mean=round(expected_mean, 4),
            rolling_std=round(expected_std, 4),
        )

    def store_score(self, conn, result: AnomalyResult, *, window_minutes: int = 60) -> None:
        """Persist a detection result to anomaly_scores."""
        conn.execute(
            """
            INSERT INTO anomaly_scores
                (scored_at, segment, window_minutes, ticket_count, is_anomaly, zscore, category_breakdown)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                result.segment,
                window_minutes,
                result.ticket_count,
                int(result.is_anomaly),
                result.zscore,
                json.dumps(result.category_breakdown),
            ),
        )


def _all_segments() -> list[str]:
    segments: list[str] = []
    for dow in DOW_NAMES:
        for bucket in ("early", "business", "late"):
            segments.append(f"{dow}_{bucket}")
    return segments


def _rolling_stats(
    hourly: dict[datetime, int], reference_time: datetime, segment: str
) -> tuple[float, float] | None:
    """7-day rolling mean and std of hourly counts ending at reference_time."""
    window_start = (reference_time - timedelta(days=ROLLING_WINDOW_DAYS)).replace(
        minute=0, second=0, microsecond=0
    )
    window_end = reference_time.replace(minute=0, second=0, microsecond=0)
    values: list[int] = []
    ts = window_start
    while ts <= window_end:
        if segment_key(ts) == segment:
            values.append(hourly.get(ts, 0))
        ts += timedelta(hours=1)
    if not values:
        return None

    mean = sum(values) / len(values)
    if len(values) < 2:
        std = max(mean * 0.25, 0.5)
    else:
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = variance ** 0.5
        if std <= 0:
            std = max(mean * 0.25, 0.5)
    return mean, std


def _count_segment_hours_in_window(
    start: datetime,
    end: datetime,
    target_segment: str,
) -> int:
    """Count hours in [start, end] whose segment key matches target_segment."""
    ts = start.replace(minute=0, second=0, microsecond=0)
    count = 0
    while ts <= end:
        if segment_key(ts) == target_segment:
            count += 1
        ts += timedelta(hours=1)
    return count


@dataclass
class CategoryAnomalyResult:
    """Detection result for a single (category, segment) pair."""

    category: str
    segment: str
    recent_count: int
    rolling_mean: float
    rolling_std: float
    zscore: float
    is_spike: bool
    is_novel: bool
    run_date: str


class CategoryAnomalyDetector:
    """Rolling mean ± Nσ anomaly detection at the category level (ANTSE-449).

    Detects two conditions:
      - Category spike: category volume z-score exceeds sigma threshold
        within the 3-day clustering window.
      - Novel category: rolling baseline mean is zero but recent count > 0.

    Topic-level spikes from ticket_clusters (is_new=True AND growth_rate above
    threshold within the window) are also captured as content anomalies.
    """

    TOPIC_GROWTH_THRESHOLD = 0.5

    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma

    def build_category_baselines(
        self,
        conn,
        *,
        reference_time: datetime | None = None,
    ) -> int:
        """Compute rolling mean and σ per (category, segment) from the last 90 days.

        Excludes BASELINE_EXCLUDE_DATES to avoid labeled anomaly windows
        contaminating the reference distribution.

        Returns the number of (category, segment) rows written.
        """
        ref = reference_time or datetime.now(timezone.utc)
        cutoff = ref - timedelta(days=BASELINE_DAYS)

        rows = conn.execute(
            """
            SELECT t.created_at, COALESCE(c.category, 'Unknown') AS category
            FROM tickets t
            LEFT JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
            WHERE t.created_at IS NOT NULL
              AND t.created_at >= ?
              AND substr(t.created_at, 1, 10) NOT IN ({exclude})
            ORDER BY t.created_at
            """.format(exclude=",".join("?" * len(BASELINE_EXCLUDE_DATES))),
            (cutoff.isoformat(), *BASELINE_EXCLUDE_DATES),
        ).fetchall()

        hourly_by_cat_seg: dict[str, dict[str, dict[datetime, int]]] = {}
        for row in rows:
            created_at = _parse_datetime(row["created_at"])
            if created_at is None:
                continue
            cat = row["category"] or "Unknown"
            seg = segment_key(created_at)
            hour_start = created_at.replace(minute=0, second=0, microsecond=0)
            hourly_by_cat_seg.setdefault(cat, {}).setdefault(seg, {})
            bucket = hourly_by_cat_seg[cat][seg]
            bucket[hour_start] = bucket.get(hour_start, 0) + 1

        computed_at = ref.isoformat()
        conn.execute("DELETE FROM category_anomaly_baselines")
        written = 0
        for cat in sorted(hourly_by_cat_seg):
            for seg in sorted(hourly_by_cat_seg[cat]):
                stats = _rolling_stats(hourly_by_cat_seg[cat][seg], ref, seg)
                if stats is None:
                    continue
                mean, std = stats
                conn.execute(
                    """
                    INSERT INTO category_anomaly_baselines
                        (category, segment, rolling_mean, rolling_std, computed_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (category, segment) DO UPDATE SET
                        rolling_mean = excluded.rolling_mean,
                        rolling_std = excluded.rolling_std,
                        computed_at = excluded.computed_at
                    """,
                    (cat, seg, mean, std, computed_at),
                )
                written += 1
        return written

    def score_categories(
        self,
        conn,
        *,
        window_days: int = 3,
        reference_time: datetime | None = None,
    ) -> list[CategoryAnomalyResult]:
        """Z-score each category vs baseline; return anomalous (category, segment) pairs.

        The baseline stores an hourly rate per (category, segment).  To compare
        against the full ``window_days`` window, the expected volume is scaled by
        the number of hours in [cutoff, ref] that belong to each segment using the
        central reference segment as the scoring key.  This avoids false positives
        from comparing a multi-day raw count to a single-hour mean.

        A category qualifies as a spike when:
          - z-score > sigma threshold (recent volume well above scaled expected), OR
          - rolling mean is zero and recent count > 0 (novel category).

        Args:
            conn: Open database connection.
            window_days: Look-back window for recent counts (default 3).
            reference_time: Evaluation reference time; defaults to now (UTC).

        Returns:
            List of CategoryAnomalyResult for all flagged categories.
        """
        ref = reference_time or datetime.now(timezone.utc)
        run_date = ref.strftime("%Y-%m-%d")
        cutoff = ref - timedelta(days=window_days)

        raw_rows = conn.execute(
            """
            SELECT t.created_at,
                   COALESCE(c.category, 'Unknown') AS category
            FROM tickets t
            LEFT JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
            WHERE t.created_at >= ?
              AND t.created_at <= ?
            """,
            (cutoff.isoformat(), ref.isoformat()),
        ).fetchall()

        seg = segment_key(ref)
        n_hours = _count_segment_hours_in_window(cutoff, ref, seg)
        if n_hours < 1:
            n_hours = 1

        # Count only tickets that fall in hours matching the reference segment.
        seg_counts: dict[str, int] = {}
        for row in raw_rows:
            created_at = _parse_datetime(row["created_at"])
            if created_at is None:
                continue
            if segment_key(created_at) != seg:
                continue
            cat = row["category"] or "Unknown"
            seg_counts[cat] = seg_counts.get(cat, 0) + 1

        count_rows_mapped = list(seg_counts.items())

        baseline_rows = conn.execute(
            """
            SELECT category, rolling_mean, rolling_std
            FROM category_anomaly_baselines
            WHERE segment = ?
            """,
            (seg,),
        ).fetchall()
        baselines: dict[str, tuple[float, float]] = {
            row["category"]: (float(row["rolling_mean"]), float(row["rolling_std"]))
            for row in baseline_rows
        }

        results: list[CategoryAnomalyResult] = []
        for cat, recent_count in count_rows_mapped:
            hourly_mean, hourly_std = baselines.get(cat, (0.0, 0.0))

            # is_novel only fires when the category has NO baseline entry at all
            # (genuinely unseen category).  A baseline entry with rolling_mean=0
            # means the category is known but rare in this segment — not novel.
            is_novel = cat not in baselines and recent_count > 0

            expected_total = hourly_mean * n_hours
            expected_std = hourly_std * (n_hours ** 0.5)
            if expected_std <= 0:
                expected_std = max(expected_total * 0.1, 0.5)

            if hourly_mean > 0:
                zscore = (recent_count - expected_total) / expected_std
            else:
                zscore = 0.0

            is_spike = (not is_novel) and zscore > self.sigma

            if is_spike or is_novel:
                results.append(
                    CategoryAnomalyResult(
                        category=cat,
                        segment=seg,
                        recent_count=recent_count,
                        rolling_mean=round(expected_total, 4),
                        rolling_std=round(expected_std, 4),
                        zscore=round(zscore, 4),
                        is_spike=is_spike,
                        is_novel=is_novel,
                        run_date=run_date,
                    )
                )
        return results

    def detect_topic_spikes(
        self,
        conn,
        *,
        window_days: int = 3,
        reference_time: datetime | None = None,
    ) -> list[CategoryAnomalyResult]:
        """Flag topic clusters with is_new=True AND growth_rate above threshold.

        Joins ticket_clusters with ticket_classifications to resolve cluster
        tickets to their majority category, then emits a CategoryAnomalyResult
        for each qualifying cluster.

        Args:
            conn: Open database connection.
            window_days: How many days of cluster runs to include (default 3).
            reference_time: Evaluation reference time; defaults to now (UTC).

        Returns:
            List of CategoryAnomalyResult (is_spike=True) for each qualifying cluster.
        """
        ref = reference_time or datetime.now(timezone.utc)
        run_date = ref.strftime("%Y-%m-%d")
        cutoff_date = (ref - timedelta(days=window_days)).strftime("%Y-%m-%d")
        seg = segment_key(ref)

        cluster_rows = conn.execute(
            """
            SELECT cluster_id, run_date, ticket_keys, label, size, growth_rate
            FROM ticket_clusters
            WHERE is_new = 1
              AND growth_rate >= ?
              AND run_date >= ?
              AND run_date <= ?
            ORDER BY run_date DESC, cluster_id
            """,
            (self.TOPIC_GROWTH_THRESHOLD, cutoff_date, run_date),
        ).fetchall()

        results: list[CategoryAnomalyResult] = []
        for row in cluster_rows:
            import json as _json

            try:
                ticket_keys = _json.loads(row["ticket_keys"] or "[]")
            except (ValueError, TypeError):
                ticket_keys = []
            if not ticket_keys:
                continue

            placeholders = ",".join("?" * len(ticket_keys[:50]))
            cat_rows = conn.execute(
                f"""
                SELECT COALESCE(category, 'Unknown') AS category, COUNT(*) AS cnt
                FROM ticket_classifications
                WHERE ticket_key IN ({placeholders})
                GROUP BY COALESCE(category, 'Unknown')
                ORDER BY cnt DESC
                LIMIT 1
                """,
                ticket_keys[:50],
            ).fetchall()
            cat = cat_rows[0]["category"] if cat_rows else "Unknown"

            results.append(
                CategoryAnomalyResult(
                    category=cat,
                    segment=seg,
                    recent_count=int(row["size"]),
                    rolling_mean=0.0,
                    rolling_std=0.0,
                    zscore=float(row["growth_rate"]),
                    is_spike=True,
                    is_novel=False,
                    run_date=run_date,
                )
            )
        return results

    def detect_novel_categories(
        self,
        conn,
        *,
        window_days: int = 3,
        reference_time: datetime | None = None,
    ) -> list[CategoryAnomalyResult]:
        """Find categories with zero baseline mean but non-zero recent count.

        This is a dedicated scan for genuinely new categories that have never
        appeared in the 90-day training window. Results complement score_categories()
        which also flags novel categories, but this method also catches categories
        that never appeared in any segment.

        Args:
            conn: Open database connection.
            window_days: Look-back window for recent counts (default 3).
            reference_time: Evaluation reference time; defaults to now (UTC).

        Returns:
            List of CategoryAnomalyResult with is_novel=True.
        """
        ref = reference_time or datetime.now(timezone.utc)
        run_date = ref.strftime("%Y-%m-%d")
        cutoff = ref - timedelta(days=window_days)
        seg = segment_key(ref)

        known_categories = {
            row["category"]
            for row in conn.execute(
                "SELECT DISTINCT category FROM category_anomaly_baselines"
            ).fetchall()
        }

        count_rows = conn.execute(
            """
            SELECT COALESCE(c.category, 'Unknown') AS category, COUNT(*) AS cnt
            FROM tickets t
            LEFT JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
            WHERE t.created_at >= ?
              AND t.created_at <= ?
            GROUP BY COALESCE(c.category, 'Unknown')
            """,
            (cutoff.isoformat(), ref.isoformat()),
        ).fetchall()

        results: list[CategoryAnomalyResult] = []
        for row in count_rows:
            cat = row["category"]
            recent_count = int(row["cnt"])
            if cat not in known_categories and recent_count > 0:
                results.append(
                    CategoryAnomalyResult(
                        category=cat,
                        segment=seg,
                        recent_count=recent_count,
                        rolling_mean=0.0,
                        rolling_std=0.0,
                        zscore=0.0,
                        is_spike=False,
                        is_novel=True,
                        run_date=run_date,
                    )
                )
        return results

    def validate_content_anomaly_detection(
        self,
        conn=None,
        *,
        window_days: int = 3,
    ) -> "dict[str, Any]":
        """Instance-method convenience wrapper around the module-level function.

        Allows callers to use ``det.validate_content_anomaly_detection()`` in
        addition to the standalone ``validate_content_anomaly_detection(conn, det)``.
        When *conn* is None, a fresh connection is opened automatically.

        Args:
            conn: Open database connection, or None to open one automatically.
            window_days: Window size used for scoring (default 3).

        Returns:
            Dict with tpr, fpr, hits, false_positives, normal_windows_sampled.
        """
        if conn is not None:
            return validate_content_anomaly_detection(conn, self, window_days=window_days)

        from db import get_db_conn  # local import to avoid circular dependency

        with get_db_conn() as _conn:
            return validate_content_anomaly_detection(_conn, self, window_days=window_days)


def validate_content_anomaly_detection(
    conn,
    detector: "CategoryAnomalyDetector",
    *,
    window_days: int = 3,
) -> dict[str, Any]:
    """Measure TPR on 2026-03-16 spike window and FPR on 10 normal windows.

    Target metrics (ANTSE-449):
      TPR >= 80% on 2026-03-16 category spike window
      FPR < 10% on 10 sampled normal windows

    Args:
        conn: Open database connection.
        detector: A built CategoryAnomalyDetector instance.
        window_days: Window size used for scoring (default 3).

    Returns:
        Dict with tpr, fpr, hits, false_positives, normal_windows_sampled.
    """
    spike_times = [
        datetime.fromisoformat("2026-03-16T15:00:00+00:00"),
    ]
    hits = 0
    for ref in spike_times:
        anomalies = detector.score_categories(conn, window_days=window_days, reference_time=ref)
        if anomalies:
            hits += 1
    tpr = hits / len(spike_times)

    normal_refs = _sample_normal_windows(conn, count=10)
    false_positives = 0
    for ref in normal_refs:
        anomalies = detector.score_categories(conn, window_days=window_days, reference_time=ref)
        if anomalies:
            false_positives += 1
    fpr = false_positives / max(len(normal_refs), 1)

    return {
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "hits": hits,
        "false_positives": false_positives,
        "normal_windows_sampled": len(normal_refs),
    }


def validate_labeled_windows(conn, detector: ControlChartDetector) -> dict[str, Any]:
    """Measure TPR on labeled anomalies and FPR on sampled normal windows.

    Documented metrics (synthetic + real data):
      TPR target > 80% on 2026-03-16 and 2026-01-06
      FPR target < 5% on 20 normal windows
    """
    anomaly_hits = 0
    peak_times = (
        ("2026-03-16", "T15:00:00+00:00"),
        ("2026-01-06", "T11:00:00+00:00"),
    )
    for _date, peak_time in peak_times:
        ref = datetime.fromisoformat(f"{_date}{peak_time}")
        result = detector.score(conn, window_minutes=60, reference_time=ref)
        if result.is_anomaly:
            anomaly_hits += 1
    tpr = anomaly_hits / len(peak_times)

    normal_windows = _sample_normal_windows(conn, count=20)
    false_positives = 0
    for ref in normal_windows:
        result = detector.score(conn, window_minutes=60, reference_time=ref)
        if result.is_anomaly:
            false_positives += 1
    fpr = false_positives / max(len(normal_windows), 1)

    return {
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "anomaly_hits": anomaly_hits,
        "false_positives": false_positives,
        "normal_windows_sampled": len(normal_windows),
    }


def _sample_normal_windows(conn, count: int = 20) -> list[datetime]:
    """Pick normal hourly windows excluding labeled anomaly dates."""
    rows = conn.execute(
        """
        SELECT DISTINCT substr(created_at, 1, 10) AS day
        FROM tickets
        WHERE created_at IS NOT NULL
        ORDER BY day
        """
    ).fetchall()
    exclude = set(BASELINE_EXCLUDE_DATES)
    candidates: list[datetime] = []
    for row in rows:
        day = row["day"]
        if day in exclude:
            continue
        try:
            candidates.append(datetime.fromisoformat(f"{day}T10:00:00+00:00"))
        except ValueError:
            log.debug("Skipping malformed created_at date: %r", day)

    if len(candidates) <= count:
        return candidates
    step = max(len(candidates) // count, 1)
    return [candidates[i * step] for i in range(count)]
