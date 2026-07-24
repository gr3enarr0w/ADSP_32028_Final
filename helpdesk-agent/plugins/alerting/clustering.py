"""Daily TF-IDF topic clustering with HDBSCAN and optional K-means ensemble (ANTSE-320).

Clusters recent tickets, computes delta vs previous run, and writes to
ticket_clusters. Incident runs (3-day) use TOPIC_RULES coherence; trend runs
(30-day) use mean intra-cluster cosine similarity (ANTSE-318/320).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from analysis.clustering_eda import SPECIAL_TERMS, TOPIC_RULES, parse_datetime

log = logging.getLogger(__name__)

SILHOUETTE_WEIGHT = 0.4
COHERENCE_WEIGHT = 0.6
TREND_SILHOUETTE_WEIGHT = 0.5
TREND_COHERENCE_WEIGHT = 0.5
INCIDENT_COMPOSITE_THRESHOLD = 0.3
TREND_COMPOSITE_THRESHOLD = 0.25
TREND_CLUSTER_INTRA_THRESHOLD = 0.2
APPROACH_TIEBREAK = ("ensemble", "hdbscan", "kmeans")
_VALID_RUN_TYPES: frozenset[str] = frozenset(
    {"incident", "trend", "daily", "weekly", "biweekly", "monthly", "bimonthly", "quarterly"}
)

# All six clustering windows supported by ANTSE-452.
# ``coherence_mode`` selects evaluation: ``"topic"`` uses TOPIC_RULES (short
# windows where rule coverage is high); ``"intra"`` uses mean intra-cluster
# cosine similarity (longer windows with more diverse ticket populations).
WINDOW_CONFIGS: list[dict] = [
    {"name": "daily",     "window_days": 1,  "coherence_mode": "topic"},
    {"name": "weekly",    "window_days": 7,  "coherence_mode": "topic"},
    {"name": "biweekly",  "window_days": 14, "coherence_mode": "topic"},
    {"name": "monthly",   "window_days": 30, "coherence_mode": "intra"},
    {"name": "bimonthly", "window_days": 60, "coherence_mode": "intra"},
    {"name": "quarterly", "window_days": 90, "coherence_mode": "intra"},
]

# Map coherence_mode aliases to the window_type strings recognised by cluster().
_COHERENCE_MODE_TO_WINDOW_TYPE: dict[str, str] = {
    "topic": "incident",
    "intra": "trend",
}


@dataclass
class ApproachMetrics:
    """Measured quality for one clustering approach on a fixed ticket window."""

    approach: str
    silhouette: float | None
    noise_fraction: float
    cluster_count: int
    coherent_count: int
    incoherent_count: int
    coherence_ratio: float
    composite_score: float
    valid: bool = True


@dataclass
class ClusterResult:
    cluster_id: int
    ticket_keys: list[str]
    label: str
    size: int
    top_terms: list[tuple[str, float]] = field(default_factory=list)
    approach: str = ""


@dataclass
class DeltaResult:
    cluster_id: int
    label: str
    size: int
    is_new: bool
    growth_rate: float
    ticket_keys: list[str] = field(default_factory=list)


class TopicClusterer:
    """TF-IDF topic clusterer (HDBSCAN with optional K-means ensemble) and delta detection."""

    def __init__(self, velocity_threshold: float = 0.3):
        self.velocity_threshold = velocity_threshold

    def cluster(
        self,
        conn,
        *,
        window_days: int = 3,
        k: int = 10,
        min_cluster_size: int = 3,
        clustering_approach: str = "auto",
        window_type: str = "incident",
    ) -> list[ClusterResult]:
        """Cluster tickets from the last ``window_days`` with non-empty descriptions.

        ``window_type`` selects evaluation: ``incident`` (TOPIC_RULES, 3-day bar 0.3)
        or ``trend`` (intra-cluster cosine similarity, 30-day bar 0.25).
        """
        coherence_mode = _coherence_mode_for_window(window_type)
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        rows = conn.execute(
            """
            SELECT ticket_key, summary, description, created_at
            FROM tickets
            WHERE description IS NOT NULL
              AND TRIM(description) <> ''
              AND created_at >= ?
            ORDER BY created_at DESC
            """,
            (cutoff.isoformat(),),
        ).fetchall()

        tickets: list[dict] = []
        for row in rows:
            created_at = parse_datetime(row["created_at"])
            if created_at is None:
                continue
            summary = row["summary"] or ""
            description = row["description"] or ""
            text = f"{summary.strip()} {description.strip()}".strip()
            if not text:
                continue
            tickets.append(
                {
                    "ticket_key": row["ticket_key"],
                    "text": text,
                    "summary": summary.strip(),
                }
            )

        if len(tickets) < 2:
            log.warning("[clustering] only %d tickets — need at least 2", len(tickets))
            return []

        texts = [t["text"] for t in tickets]
        vectorizer, matrix = _vectorize(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        approach_cfg = clustering_approach.strip().lower()

        benchmarks, labels_by_approach = benchmark_approaches(
            tickets,
            matrix,
            feature_names,
            k=k,
            min_cluster_size=min_cluster_size,
            coherence_mode=coherence_mode,
        )
        _log_benchmark_scores(benchmarks, ticket_count=len(tickets))

        if approach_cfg == "auto":
            selected_approach = pick_best_approach(benchmarks)
            if selected_approach is None:
                return []
            final_labels = labels_by_approach[selected_approach]
        else:
            selected_approach = _select_approach(approach_cfg)
            final_labels = labels_by_approach.get(selected_approach)
            if final_labels is None:
                return []

        selected_metrics = benchmarks[selected_approach]
        sil_display = (
            f"{selected_metrics.silhouette:.4f}"
            if selected_metrics.silhouette is not None
            else "n/a"
        )
        log.info(
            "[clustering] window_type=%s approach=%s silhouette=%s composite=%.4f "
            "coherence=%.2f noise_frac=%.2f clusters=%d tickets=%d",
            window_type,
            selected_approach,
            sil_display,
            selected_metrics.composite_score,
            selected_metrics.coherence_ratio,
            selected_metrics.noise_fraction,
            selected_metrics.cluster_count,
            len(tickets),
        )

        return _build_cluster_results(
            tickets,
            final_labels,
            matrix,
            feature_names,
            approach=selected_approach,
        )

    def compute_delta(
        self,
        current: list[ClusterResult],
        previous: list[ClusterResult],
    ) -> list[DeltaResult]:
        """Identify new and growing clusters vs the previous run."""
        deltas: list[DeltaResult] = []
        for cluster in current:
            current_keys = set(cluster.ticket_keys)
            best_overlap = 0
            best_prev_size = 0
            for prev in previous:
                overlap = len(current_keys & set(prev.ticket_keys))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_prev_size = prev.size

            is_new = best_overlap == 0
            if is_new:
                growth_rate = 1.0
            elif best_prev_size > 0:
                growth_rate = (cluster.size - best_prev_size) / best_prev_size
            else:
                growth_rate = 0.0

            deltas.append(
                DeltaResult(
                    cluster_id=cluster.cluster_id,
                    label=cluster.label,
                    size=cluster.size,
                    is_new=is_new,
                    growth_rate=round(growth_rate, 4),
                    ticket_keys=cluster.ticket_keys,
                )
            )
        return deltas

    def write_clusters(
        self,
        conn,
        run_date: str,
        deltas: list[DeltaResult],
        *,
        run_type: str = "incident",
        window_days: int = 0,
    ) -> int:
        """Persist cluster run to ticket_clusters.

        ``window_days`` records the ticket look-back window for this run so
        callers can distinguish same-day runs across different windows.
        """
        if run_type not in _VALID_RUN_TYPES:
            raise ValueError(
                f"Invalid run_type {run_type!r}; expected one of {sorted(_VALID_RUN_TYPES)}"
            )
        conn.execute(
            "DELETE FROM ticket_clusters WHERE run_date = ? AND run_type = ?",
            (run_date, run_type),
        )
        written = 0
        for delta in deltas:
            # Keep is_new semantics strict; growth is conveyed by growth_rate.
            conn.execute(
                """
                INSERT INTO ticket_clusters
                    (cluster_id, run_date, run_type, window_days, ticket_keys, label, size, is_new, growth_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delta.cluster_id,
                    run_date,
                    run_type,
                    window_days,
                    json.dumps(delta.ticket_keys),
                    delta.label,
                    delta.size,
                    int(delta.is_new),
                    delta.growth_rate,
                ),
            )
            written += 1
        return written

    def load_previous_clusters(
        self,
        conn,
        before_date: str,
        *,
        run_type: str = "incident",
    ) -> list[ClusterResult]:
        """Load the most recent cluster run before ``before_date`` for ``run_type``."""
        row = conn.execute(
            """
            SELECT run_date FROM ticket_clusters
            WHERE run_date < ? AND run_type = ?
            ORDER BY run_date DESC
            LIMIT 1
            """,
            (before_date, run_type),
        ).fetchone()
        if row is None:
            return []

        prev_date = row["run_date"]
        rows = conn.execute(
            """
            SELECT cluster_id, ticket_keys, label, size
            FROM ticket_clusters
            WHERE run_date = ? AND run_type = ?
            ORDER BY cluster_id
            """,
            (prev_date, run_type),
        ).fetchall()

        results: list[ClusterResult] = []
        for r in rows:
            keys = json.loads(r["ticket_keys"])
            results.append(
                ClusterResult(
                    cluster_id=int(r["cluster_id"]),
                    ticket_keys=keys,
                    label=r["label"],
                    size=int(r["size"]),
                )
            )
        return results


def _normalize_text(text: str) -> str:
    """Normalize JSM-specific terms before TF-IDF vectorization."""
    normalized = text
    for source, replacement in sorted(SPECIAL_TERMS.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(source, replacement)
        normalized = normalized.replace(source.lower(), replacement)
    return normalized


def _vectorize(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        min_df=2,
        preprocessor=_normalize_text,
    )
    matrix = vectorizer.fit_transform(texts)
    matrix = Normalizer(copy=False).fit_transform(matrix)
    return vectorizer, matrix


def _fit_kmeans(matrix, k: int):
    from sklearn.cluster import KMeans

    if matrix.shape[0] <= k:
        return None
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(matrix)
    return model


def _fit_hdbscan(matrix, min_cluster_size: int):
    import hdbscan

    if matrix.shape[0] < min_cluster_size:
        return None, None
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = model.fit_predict(matrix)
    return model, np.asarray(labels)


def _silhouette_score(matrix, labels: np.ndarray) -> float | None:
    from sklearn.metrics import silhouette_score

    mask = labels != -1
    if int(mask.sum()) < 2:
        return None
    filtered_labels = labels[mask]
    if len(set(filtered_labels)) < 2:
        return None
    filtered_matrix = matrix[mask]
    return float(silhouette_score(filtered_matrix, filtered_labels))


def _noise_fraction(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(np.sum(labels == -1)) / len(labels)


def _composite_score(silhouette: float | None, coherence_ratio: float) -> float:
    """Incident-window composite: silhouette×0.4 + TOPIC_RULES coherence ratio×0.6."""
    sil = silhouette if silhouette is not None else 0.0
    return sil * SILHOUETTE_WEIGHT + coherence_ratio * COHERENCE_WEIGHT


def _trend_composite_score(silhouette: float | None, intra_similarity: float) -> float:
    """Trend-window composite: silhouette×0.5 + mean intra-cluster similarity×0.5."""
    sil = silhouette if silhouette is not None else 0.0
    return sil * TREND_SILHOUETTE_WEIGHT + intra_similarity * TREND_COHERENCE_WEIGHT


def _coherence_mode_for_window(window_type: str) -> str:
    """Return the coherence evaluation mode for a given window type.

    ``"trend"`` → ``"intra_similarity"``; any other value → ``"topic_rules"``.
    """
    return "intra_similarity" if window_type.strip().lower() == "trend" else "topic_rules"


def _cluster_coherence_intra(matrix, member_indices: np.ndarray) -> float:
    """Mean pairwise cosine similarity among cluster members (trend coherence proxy)."""
    n = len(member_indices)
    if n < 2:
        return 1.0
    subset = matrix[member_indices]

    # Vectorized pairwise cosine similarity (matrix is already L2-normalized)
    sim_matrix = subset @ subset.T
    if hasattr(sim_matrix, "toarray"):
        sim_matrix = sim_matrix.toarray()
    else:
        sim_matrix = np.asarray(sim_matrix)

    total_sum = sim_matrix.sum()
    diag_sum = np.trace(sim_matrix)

    upper_sum = (total_sum - diag_sum) / 2.0
    count = n * (n - 1) / 2.0

    return float(upper_sum / count) if count > 0 else 0.0


def _cluster_is_coherent(top_terms: list[tuple[str, float]], summaries: list[str]) -> bool:
    terms = [term for term, _ in top_terms]
    matches = [
        (score, label)
        for keywords, label in TOPIC_RULES
        if (score := _topic_matches(terms, summaries, keywords)) > 0
    ]
    if not matches:
        return False
    matches.sort(key=lambda item: item[0], reverse=True)
    top_score = matches[0][0]
    competing = [label for score, label in matches if score >= max(1, top_score - 1)]
    return len(set(competing)) == 1


def _labels_valid(labels: np.ndarray) -> bool:
    assigned = labels[labels != -1]
    return len(assigned) >= 2 and len(set(assigned)) >= 2


def _evaluate_labels(
    labels: np.ndarray,
    tickets: list[dict],
    matrix,
    feature_names: np.ndarray,
    *,
    approach: str,
    coherence_mode: str = "topic_rules",
) -> ApproachMetrics:
    """Score a labeling with silhouette and coherence (topic rules or intra similarity)."""
    clusters = _build_cluster_results(
        tickets,
        labels,
        matrix,
        feature_names,
        approach=approach,
    )
    summary_by_key = {ticket["ticket_key"]: ticket["summary"] for ticket in tickets}
    cluster_count = len(clusters)
    coherent_count = 0
    incoherent_count = 0
    mode = coherence_mode.strip().lower()

    if mode == "intra_similarity":
        intra_scores: list[float] = []
        for cluster in clusters:
            member_indices = np.where(labels == cluster.cluster_id)[0]
            intra = _cluster_coherence_intra(matrix, member_indices)
            intra_scores.append(intra)
            if intra >= TREND_CLUSTER_INTRA_THRESHOLD:
                coherent_count += 1
            else:
                incoherent_count += 1
        coherence_ratio = float(np.mean(intra_scores)) if intra_scores else 0.0
    else:
        for cluster in clusters:
            summaries = [summary_by_key.get(key, "") for key in cluster.ticket_keys[:10]]
            if _cluster_is_coherent(cluster.top_terms, summaries):
                coherent_count += 1
            else:
                incoherent_count += 1
        coherence_ratio = coherent_count / cluster_count if cluster_count else 0.0

    silhouette = _silhouette_score(matrix, labels)
    noise_fraction = _noise_fraction(labels) if approach != "kmeans" else 0.0
    valid = _labels_valid(labels) and cluster_count > 0
    if valid:
        if mode == "intra_similarity":
            composite = _trend_composite_score(silhouette, coherence_ratio)
        else:
            composite = _composite_score(silhouette, coherence_ratio)
    else:
        composite = float("-inf")

    return ApproachMetrics(
        approach=approach,
        silhouette=silhouette,
        noise_fraction=noise_fraction,
        cluster_count=cluster_count,
        coherent_count=coherent_count,
        incoherent_count=incoherent_count,
        coherence_ratio=round(coherence_ratio, 4),
        composite_score=round(composite, 4) if composite != float("-inf") else composite,
        valid=valid,
    )


def benchmark_approaches(
    tickets: list[dict],
    matrix,
    feature_names: np.ndarray,
    *,
    k: int,
    min_cluster_size: int,
    coherence_mode: str = "topic_rules",
) -> tuple[dict[str, ApproachMetrics], dict[str, np.ndarray]]:
    """Run kmeans, hdbscan, and ensemble on the same matrix; return metrics and labels."""
    metrics: dict[str, ApproachMetrics] = {}
    labels_by_approach: dict[str, np.ndarray] = {}

    kmeans_model = _fit_kmeans(matrix, k)
    if kmeans_model is not None:
        kmeans_labels = np.asarray(kmeans_model.labels_)
        labels_by_approach["kmeans"] = kmeans_labels
        metrics["kmeans"] = _evaluate_labels(
            kmeans_labels,
            tickets,
            matrix,
            feature_names,
            approach="kmeans",
            coherence_mode=coherence_mode,
        )

    _, hdbscan_labels = _fit_hdbscan(matrix, min_cluster_size)
    if hdbscan_labels is not None:
        labels_by_approach["hdbscan"] = hdbscan_labels
        metrics["hdbscan"] = _evaluate_labels(
            hdbscan_labels,
            tickets,
            matrix,
            feature_names,
            approach="hdbscan",
            coherence_mode=coherence_mode,
        )

    if hdbscan_labels is not None and kmeans_model is not None:
        ensemble_labels = _ensemble_labels(hdbscan_labels, kmeans_model.labels_)
        labels_by_approach["ensemble"] = ensemble_labels
        metrics["ensemble"] = _evaluate_labels(
            ensemble_labels,
            tickets,
            matrix,
            feature_names,
            approach="ensemble",
            coherence_mode=coherence_mode,
        )

    return metrics, labels_by_approach


def pick_best_approach(benchmarks: dict[str, ApproachMetrics]) -> str | None:
    """Select the approach with the highest measured composite score."""
    candidates = [metrics for metrics in benchmarks.values() if metrics.valid]
    if not candidates:
        return None
    best_score = max(metrics.composite_score for metrics in candidates)
    winners = [metrics.approach for metrics in candidates if metrics.composite_score == best_score]
    for approach in APPROACH_TIEBREAK:
        if approach in winners:
            return approach
    return winners[0]


def _select_approach(approach_cfg: str) -> str:
    """Map a forced config value to an approach name."""
    normalized = approach_cfg.strip().lower()
    if normalized in ("hdbscan", "kmeans", "ensemble"):
        return normalized
    return "ensemble"


def _log_benchmark_scores(benchmarks: dict[str, ApproachMetrics], *, ticket_count: int) -> None:
    parts = []
    for approach in ("kmeans", "hdbscan", "ensemble"):
        metrics = benchmarks.get(approach)
        if metrics is None:
            parts.append(f"{approach}=n/a")
            continue
        sil = f"{metrics.silhouette:.4f}" if metrics.silhouette is not None else "n/a"
        parts.append(
            f"{approach}(sil={sil},coh={metrics.coherence_ratio:.2f},"
            f"noise={metrics.noise_fraction:.2f},score={metrics.composite_score:.4f})"
        )
    log.info("[clustering] benchmark ticket_count=%d %s", ticket_count, " ".join(parts))


def _ensemble_labels(hdbscan_labels: np.ndarray, kmeans_labels: np.ndarray) -> np.ndarray:
    combined = kmeans_labels.copy()
    non_noise = hdbscan_labels != -1
    combined[non_noise] = hdbscan_labels[non_noise]
    return combined


def _mean_center(matrix, member_indices: np.ndarray) -> np.ndarray:
    subset = matrix[member_indices]
    center = np.asarray(subset.mean(axis=0)).ravel()
    return center


def _build_cluster_results(
    tickets: list[dict],
    labels: np.ndarray,
    matrix,
    feature_names: np.ndarray,
    *,
    approach: str,
) -> list[ClusterResult]:
    results: list[ClusterResult] = []
    cluster_ids = sorted({int(label) for label in labels if label != -1})

    for cluster_id in cluster_ids:
        member_indices = np.where(labels == cluster_id)[0]
        if len(member_indices) == 0:
            continue
        center = _mean_center(matrix, member_indices)
        top_idx = np.argsort(center)[::-1][:12]
        top_terms = [
            (str(feature_names[idx]), float(center[idx]))
            for idx in top_idx
            if center[idx] > 0
        ]
        member_keys = [tickets[idx]["ticket_key"] for idx in member_indices]
        summaries = [tickets[idx]["summary"] for idx in member_indices[:10]]
        label = _assign_label(top_terms, summaries)
        results.append(
            ClusterResult(
                cluster_id=cluster_id,
                ticket_keys=member_keys,
                label=label,
                size=len(member_keys),
                top_terms=top_terms,
                approach=approach,
            )
        )
    return results


def _topic_matches(terms: list[str], summaries: list[str], keywords: tuple[str, ...]) -> int:
    haystack = " ".join(terms + summaries).lower()
    return sum(1 for keyword in keywords if keyword in haystack)


def _assign_label(top_terms: list[tuple[str, float]], summaries: list[str]) -> str:
    terms = [term for term, _ in top_terms]
    matches = [
        (score, label)
        for keywords, label in TOPIC_RULES
        if (score := _topic_matches(terms, summaries, keywords)) > 0
    ]
    matches.sort(key=lambda item: item[0], reverse=True)
    if matches:
        return matches[0][1]
    return "mixed / uncategorized"


def run_all_windows(
    conn,
    *,
    k: int = 10,
    min_cluster_size: int = 3,
    clustering_approach: str = "auto",
) -> dict[str, int]:
    """Run all six WINDOW_CONFIGS and persist results to ticket_clusters.

    Returns a mapping of ``window_name → rows_written`` for each window that
    produced at least one cluster.  Windows that return no tickets or no
    clusters are skipped and recorded with a value of 0.

    The 90-day (quarterly) window additionally logs the top 3 clusters by size
    as a data-ready summary for future Confluence export.
    """
    from datetime import date

    clusterer = TopicClusterer()
    run_date = date.today().isoformat()
    results: dict[str, int] = {}

    for cfg in WINDOW_CONFIGS:
        name: str = cfg["name"]
        window_days: int = cfg["window_days"]
        coherence_mode: str = cfg["coherence_mode"]
        # Map our coherence_mode shorthand to the window_type expected by cluster()
        window_type = _COHERENCE_MODE_TO_WINDOW_TYPE.get(coherence_mode, "incident")

        log.info(
            "[clustering] running %s window (window_days=%d, coherence_mode=%s)",
            name,
            window_days,
            coherence_mode,
        )

        try:
            current = clusterer.cluster(
                conn,
                window_days=window_days,
                k=k,
                min_cluster_size=min_cluster_size,
                clustering_approach=clustering_approach,
                window_type=window_type,
            )
        except Exception:
            log.exception("[clustering] %s window failed — skipping", name)
            results[name] = 0
            continue

        if not current:
            log.warning("[clustering] %s window produced no clusters — skipping", name)
            results[name] = 0
            continue

        previous = clusterer.load_previous_clusters(conn, run_date, run_type=name)
        deltas = clusterer.compute_delta(current, previous)
        written = clusterer.write_clusters(
            conn,
            run_date,
            deltas,
            run_type=name,
            window_days=window_days,
        )
        results[name] = written
        log.info("[clustering] %s window complete — %d clusters written", name, written)

        # Quarterly window: emit top-3 cluster summary for future Confluence export.
        if name == "quarterly" and deltas:
            top3 = sorted(deltas, key=lambda d: d.size, reverse=True)[:3]
            summary_parts = [
                f"{d.label!r} (n={d.size})" for d in top3
            ]
            log.info("[quarterly] Top clusters: %s", ", ".join(summary_parts))

    return results
