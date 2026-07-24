"""ANN few-shot retrieval for responder prompt generation.

CSAT-weighted prioritization (ANTSE-325).

CV results — update by running:  python -m analysis.fewshot_csat_eval
(placeholder shown below; corpus was empty at initial deploy time)

  Strategy   mean_sim ± std   coverage   gini    vs baseline (Welch p)
  baseline   —                —          —       —
  a          —                —          —       —
  b          —                —          —       —
  c          —                —          —       —

Current default: fewshot_csat_strategy = a (linear z-score).
Change pipeline.yaml or run with --apply-winner to update after eval.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import ClassVar

from core.pipeline import get_plugin_config
from db import get_db_conn
from services.embedding import embed_batch, embed_text
from services.vector_store import query_similar, store_embedding

log = logging.getLogger(__name__)

MAX_CSAT_SCORE = 5.0
MIN_CSAT_SAMPLES = 30
VALID_STRATEGIES = frozenset({"baseline", "a", "b", "c"})


def _trim(text: str | None, limit: int) -> str:
    return (text or "").strip()[:limit]


def gini_coefficient(labels: list[str]) -> float:
    """Gini coefficient for category distribution (0 = uniform, 1 = concentrated)."""
    if not labels:
        return 0.0
    counts = sorted(Counter(labels).values())
    n = len(counts)
    total = sum(counts)
    if n <= 1 or total == 0:
        return 0.0
    weighted = sum((2 * idx - n - 1) * count for idx, count in enumerate(counts, start=1))
    return weighted / (n * total)


def _compose_draft_example(row) -> str:
    summary = _trim(row["summary"], 300)
    description = _trim(row["description"], 900)
    draft = _trim(row["draft_customer_response"], 900)
    actual = _trim(row["actual_response"], 900)
    response_type = _trim(row["response_type"], 40)

    parts = ["[APPROVED DRAFT EXAMPLE]"]
    if response_type:
        parts.append(f"Response Type: {response_type}")
    if summary:
        parts.append(f"Ticket Summary: {summary}")
    if description:
        parts.append(f"Ticket Description: {description}")
    if draft:
        parts.append(f"AI Draft: {draft}")
    if actual:
        parts.append(f"Agent Final: {actual}")
    return "\n".join(parts)


def _compose_organic_example(row) -> str:
    summary = _trim(row["summary"], 300)
    description = _trim(row["description"], 900)
    response = _trim(row["agent_response"], 1200)
    response_type = _trim(row["response_type"], 40)

    parts = ["[ORGANIC RESOLVED EXAMPLE]"]
    if response_type:
        parts.append(f"Response Type: {response_type}")
    if summary:
        parts.append(f"Ticket Summary: {summary}")
    if description:
        parts.append(f"Ticket Description: {description}")
    if response:
        parts.append(f"Agent Response: {response}")
    return "\n".join(parts)


def _load_csat_weights(conn) -> dict[str, dict]:
    """Load latest CSAT correlation stats keyed by category."""
    rows = conn.execute(
        """
        SELECT category, mean_csat, std_csat, acceptance_rate, n_samples
        FROM category_csat_correlations
        WHERE run_date = (SELECT MAX(run_date) FROM category_csat_correlations)
        """
    ).fetchall()
    parsed: dict[str, dict] = {}
    for row in rows:
        category = row["category"]
        if not category:
            continue
        try:
            mean_csat = float(row["mean_csat"])
            std_csat = float(row["std_csat"] if row["std_csat"] is not None else 0.0)
            acceptance_rate = float(row["acceptance_rate"])
            n_samples = int(row["n_samples"])
        except (TypeError, ValueError):
            log.debug("Skipping malformed CSAT row for category %s", category)
            continue

        parsed[category] = {
            "mean_csat": mean_csat,
            "std_csat": std_csat,
            "acceptance_rate": acceptance_rate,
            "n_samples": n_samples,
        }
    return parsed


def _get_example_categories(conn, example_ids: list[str]) -> dict[str, str | None]:
    """Map example_id → ticket classification category."""
    if not example_ids:
        return {}

    placeholders = ",".join("?" for _ in example_ids)
    rows = conn.execute(
        f"""
        SELECT fe.example_id, tc.category
        FROM few_shot_examples fe
        LEFT JOIN ticket_classifications tc ON fe.ticket_key = tc.ticket_key
        WHERE fe.example_id IN ({placeholders})
        """,
        example_ids,
    ).fetchall()
    return {row["example_id"]: row["category"] for row in rows}


def _compute_global_stats(csat_weights: dict[str, dict]) -> dict[str, float]:
    means = [entry["mean_csat"] for entry in csat_weights.values()]
    if not means:
        return {"mean": 0.0, "std": 1.0, "max_csat": MAX_CSAT_SCORE}
    global_mean = sum(means) / len(means)
    variance = sum((value - global_mean) ** 2 for value in means) / len(means)
    global_std = math.sqrt(variance) if variance > 0 else 1.0
    return {"mean": global_mean, "std": global_std, "max_csat": MAX_CSAT_SCORE}


def _csat_entry_usable(entry: dict | None) -> bool:
    return bool(entry and entry.get("n_samples", 0) >= MIN_CSAT_SAMPLES)


def _apply_weight(
    ann_score: float,
    strategy: str,
    csat_data: dict | None,
    global_stats: dict[str, float],
    alpha: float,
) -> float:
    """Apply CSAT weighting strategy; fall back to ann_score when data is insufficient."""
    if strategy == "baseline" or not _csat_entry_usable(csat_data):
        return ann_score

    mean_csat = float(csat_data["mean_csat"])
    max_csat = global_stats["max_csat"]

    if strategy == "a":
        global_mean = global_stats["mean"]
        global_std = global_stats["std"]
        csat_z = (mean_csat - global_mean) / global_std
        csat_z = max(-1.0, min(1.0, csat_z))
        return ann_score * (1.0 + alpha * csat_z)

    if strategy == "b":
        mean_norm = mean_csat / max_csat
        return ann_score * math.log(1.0 + mean_norm)

    if strategy == "c":
        mean_norm = mean_csat / max_csat
        acceptance = float(csat_data["acceptance_rate"])
        return ann_score * (acceptance * mean_norm)

    return ann_score


class ANNFewShotIndex:
    """ANN-backed few-shot corpus for responder draft prompts."""

    entity_type: ClassVar[str] = "few_shot"
    _last_rebuild_date: ClassVar[str | None] = None
    _csat_cache: ClassVar[dict[str, dict] | None] = None
    _csat_cache_date: ClassVar[str | None] = None
    _global_csat_stats: ClassVar[dict[str, float] | None] = None
    _csat_alpha: ClassVar[float | None] = None

    @classmethod
    def _today(cls) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    @classmethod
    def _resolve_strategy(cls, strategy: str) -> str:
        if strategy == "auto":
            cfg = get_plugin_config("responder")
            strategy = str(cfg.get("fewshot_csat_strategy", "baseline")).lower()
            if strategy == "auto":
                strategy = "baseline"
        if strategy not in VALID_STRATEGIES:
            log.warning("Unknown few-shot CSAT strategy %r — using baseline", strategy)
            return "baseline"
        return strategy

    @classmethod
    def _load_csat_context(cls, conn) -> tuple[dict[str, dict], dict[str, float], float]:
        today = cls._today()
        if cls._csat_cache is not None and cls._csat_cache_date == today:
            return cls._csat_cache, cls._global_csat_stats or _compute_global_stats({}), cls._csat_alpha or 0.5

        cfg = get_plugin_config("responder")
        alpha = float(cfg.get("csat_weight", 0.5))
        csat_weights = _load_csat_weights(conn)
        global_stats = _compute_global_stats(csat_weights)
        cls._csat_cache = csat_weights
        cls._global_csat_stats = global_stats
        cls._csat_alpha = alpha
        cls._csat_cache_date = today
        return csat_weights, global_stats, alpha

    @classmethod
    def is_empty(cls) -> bool:
        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM few_shot_examples WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
        return row is None

    @classmethod
    def build(cls, db=None) -> int:
        """Rebuild the few-shot ANN corpus from approved examples."""
        if db is None:
            with get_db_conn() as conn:
                return cls._build_on_conn(conn)
        return cls._build_on_conn(db)

    @classmethod
    def _build_on_conn(cls, conn) -> int:
        draft_rows = conn.execute(
            """
            SELECT
                f.ticket_key,
                f.draft_comment_id,
                f.response_type,
                f.draft_customer_response,
                f.actual_response,
                t.summary,
                t.description
            FROM ai_draft_feedback f
            LEFT JOIN tickets t ON t.ticket_key = f.ticket_key
            WHERE f.actual_response IS NOT NULL
              AND (
                    f.agent_feedback IN ('both_good', 'customer_good', 'steps_good')
                    OR (
                        f.feedback_category IN ('as_is', 'lightly_edited')
                        AND f.similarity_score >= 0.75
                    )
              )
              AND (
                    f.agent_feedback IS NULL
                    OR f.agent_feedback NOT IN ('both_bad', 'wrong_type', 'needs_info')
              )
            ORDER BY f.captured_at DESC, f.id DESC
            """
        ).fetchall()

        organic_rows = conn.execute(
            """
            SELECT
                e.ticket_key,
                e.response_type,
                e.agent_response,
                t.summary,
                t.description
            FROM response_examples e
            LEFT JOIN tickets t ON t.ticket_key = e.ticket_key
            ORDER BY e.harvested_at DESC, e.id DESC
            """
        ).fetchall()

        records: list[dict[str, str]] = []
        for row in draft_rows:
            source_key = f"{row['ticket_key']}:{row['draft_comment_id']}"
            records.append(
                {
                    "example_id": f"ai_draft_feedback:{source_key}",
                    "source_table": "ai_draft_feedback",
                    "source_key": source_key,
                    "ticket_key": row["ticket_key"],
                    "response_type": row["response_type"],
                    "example_text": _compose_draft_example(row),
                }
            )

        for row in organic_rows:
            source_key = row["ticket_key"]
            records.append(
                {
                    "example_id": f"response_examples:{source_key}",
                    "source_table": "response_examples",
                    "source_key": source_key,
                    "ticket_key": row["ticket_key"],
                    "response_type": row["response_type"],
                    "example_text": _compose_organic_example(row),
                }
            )

        if not records:
            conn.execute("DELETE FROM few_shot_examples")
            cls._last_rebuild_date = cls._today()
            log.info("Few-shot ANN index rebuilt with 0 examples")
            return 0

        vectors = embed_batch([rec["example_text"] for rec in records], task_type="document")

        try:
            conn.execute("DELETE FROM few_shot_examples")
            conn.executemany(
                """
                INSERT INTO few_shot_examples (
                    example_id, source_table, source_key, ticket_key,
                    response_type, example_text
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(example_id) DO UPDATE SET
                    source_table = excluded.source_table,
                    source_key = excluded.source_key,
                    ticket_key = excluded.ticket_key,
                    response_type = excluded.response_type,
                    example_text = excluded.example_text
                """,
                [
                    (
                        rec["example_id"],
                        rec["source_table"],
                        rec["source_key"],
                        rec["ticket_key"],
                        rec["response_type"],
                        rec["example_text"],
                    )
                    for rec in records
                ],
            )

            for rec, vector in zip(records, vectors):
                store_embedding(rec["example_id"], vector, cls.entity_type, conn=conn)
        except Exception as exc:
            log.error(
                "Few-shot index rebuild failed mid-write — index may be empty until next rebuild: %s",
                exc,
            )
            raise

        cls._last_rebuild_date = cls._today()
        log.info("Few-shot ANN index rebuilt with %d examples", len(records))
        return len(records)

    @classmethod
    def rebuild_daily(cls, db=None) -> int:
        today = cls._today()
        if cls._last_rebuild_date == today:
            return 0
        return cls.build(db=db)

    @classmethod
    def _retrieve_scored(
        cls,
        ticket_text: str,
        k: int = 5,
        similarity_floor: float = 0.5,
        *,
        strategy: str = "auto",
        exclude_ticket_keys: set[str] | None = None,
    ) -> list[dict]:
        """Return scored few-shot hits with category metadata for eval and ranking."""
        if not ticket_text.strip():
            return []

        resolved_strategy = cls._resolve_strategy(strategy)
        query_vec = embed_text(ticket_text, task_type="query")
        candidate_k = max(k * 5, 25)
        raw_hits = query_similar(query_vec, k=candidate_k, entity_type=cls.entity_type)
        if not raw_hits:
            return []

        example_ids = [example_id for example_id, _ in raw_hits]
        placeholders = ",".join("?" for _ in example_ids)

        with get_db_conn() as conn:
            csat_weights, global_stats, alpha = cls._load_csat_context(conn)
            category_by_id = _get_example_categories(conn, example_ids)
            rows = conn.execute(
                f"""
                SELECT example_id, example_text, ticket_key
                FROM few_shot_examples
                WHERE example_id IN ({placeholders})
                """,
                example_ids,
            ).fetchall()

        row_by_id = {row["example_id"]: row for row in rows}
        fallback_logged = False
        scored: list[dict] = []

        for example_id, ann_score in raw_hits:
            row = row_by_id.get(example_id)
            if not row:
                continue
            if exclude_ticket_keys and row["ticket_key"] in exclude_ticket_keys:
                continue
            if ann_score < similarity_floor:
                continue

            category = category_by_id.get(example_id)
            csat_entry = csat_weights.get(category) if category else None
            if resolved_strategy != "baseline" and not _csat_entry_usable(csat_entry):
                if not fallback_logged:
                    log.info(
                        "CSAT weighting fallback to unweighted ann_score for strategy %s",
                        resolved_strategy,
                    )
                    fallback_logged = True

            final_score = _apply_weight(
                ann_score,
                resolved_strategy,
                csat_entry,
                global_stats,
                alpha,
            )
            scored.append(
                {
                    "example_id": example_id,
                    "example_text": row["example_text"],
                    "ticket_key": row["ticket_key"],
                    "ann_score": ann_score,
                    "final_score": final_score,
                    "category": category,
                }
            )

        scored.sort(key=lambda item: item["final_score"], reverse=True)
        return scored[:k]

    @classmethod
    def retrieve(
        cls,
        ticket_text: str,
        k: int = 5,
        similarity_floor: float = 0.5,
        *,
        strategy: str = "auto",
        exclude_ticket_keys: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return the nearest few-shot examples for the incoming ticket."""
        scored = cls._retrieve_scored(
            ticket_text,
            k=k,
            similarity_floor=similarity_floor,
            strategy=strategy,
            exclude_ticket_keys=exclude_ticket_keys,
        )
        return [(item["example_text"], item["final_score"]) for item in scored]

    @classmethod
    def retrieve_with_categories(
        cls,
        ticket_text: str,
        k: int = 5,
        similarity_floor: float = 0.5,
        *,
        strategy: str = "auto",
        exclude_ticket_keys: set[str] | None = None,
    ) -> list[dict]:
        """Return scored examples including category labels for distribution tracking."""
        return cls._retrieve_scored(
            ticket_text,
            k=k,
            similarity_floor=similarity_floor,
            strategy=strategy,
            exclude_ticket_keys=exclude_ticket_keys,
        )
