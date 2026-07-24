from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from db import get_db_conn
from plugins.responder.eval_set import get_eval_queries
from plugins.responder.lookup import _legacy_lookup, lookup, rank_in_matches

log = logging.getLogger(__name__)


def evaluate_mrr(lookup_func, queries: list[tuple[str, str, str]], k: int = 10) -> float:
    if not queries:
        return 0.0
    total = 0.0
    for query, source_type, doc_id in queries:
        rank = rank_in_matches(lookup_func(query), source_type, doc_id)
        if rank and rank <= k:
            total += 1.0 / rank
    return total / len(queries)


def run_latency_test(lookup_func, sample_size: int = 500) -> tuple[float, float]:
    with get_db_conn() as conn:
        rows = conn.execute(
            "SELECT summary FROM tickets WHERE summary IS NOT NULL AND TRIM(summary) != '' LIMIT 2000"
        ).fetchall()

    queries = [row["summary"] for row in rows]
    if not queries:
        return 0.0, 0.0
    sampled = random.sample(queries, min(sample_size, len(queries)))

    latencies = []
    for query in sampled:
        start = time.perf_counter()
        lookup_func(query)
        latencies.append((time.perf_counter() - start) * 1000.0)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    return p50, p95


@dataclass(frozen=True)
class DraftingSimilarityResult:
    mean_score: float | None
    sample_count: int
    assumption: str


def compare_drafting_similarity(limit: int = 50) -> DraftingSimilarityResult:
    """Compare drafting quality proxy via existing similarity capture data.

    Assumption: this environment may not have live LLM credentials to safely execute
    _lookup_and_draft() end-to-end without side effects. We therefore use historical
    ai_draft_feedback.similarity_score for held-out resolved tickets as the proxy.
    """
    with get_db_conn() as conn:
        rows = conn.execute(
            """
            SELECT f.similarity_score
            FROM ai_draft_feedback f
            JOIN tickets t ON t.ticket_key = f.ticket_key
            WHERE t.resolution IS NOT NULL
              AND f.similarity_score IS NOT NULL
            ORDER BY f.captured_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    if not rows:
        return DraftingSimilarityResult(
            mean_score=None,
            sample_count=0,
            assumption=(
                "No historical similarity_score rows found; drafting comparison "
                "requires credentials + captured feedback data."
            ),
        )

    scores = [float(row["similarity_score"]) for row in rows]
    return DraftingSimilarityResult(
        mean_score=sum(scores) / len(scores),
        sample_count=len(scores),
        assumption=(
            "Used historical ai_draft_feedback similarity as a held-out proxy "
            "instead of live _lookup_and_draft execution."
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from plugins.responder import bm25, dense_retrieval

    print("Building indices...")
    bm25.build()
    dense_retrieval.build()
    print("Indices built.")

    queries = get_eval_queries(min_queries=200, max_queries=300)
    print(f"Loaded {len(queries)} eval queries")

    old_mrr = evaluate_mrr(_legacy_lookup, queries)
    new_mrr = evaluate_mrr(lookup, queries)
    print(f"Old MRR@10: {old_mrr:.3f}")
    print(f"New MRR@10: {new_mrr:.3f}")

    p50_ms, p95_ms = run_latency_test(lookup, sample_size=500)
    print(f"Latency p50: {p50_ms:.0f} ms, p95: {p95_ms:.0f} ms")

    draft_cmp = compare_drafting_similarity(limit=50)
    if draft_cmp.mean_score is None:
        print("Drafting similarity (50 held-out): unavailable")
    else:
        print(
            f"Drafting similarity (50 held-out, mean similarity_score): "
            f"{draft_cmp.mean_score:.3f} from n={draft_cmp.sample_count}"
        )
    print(f"Drafting assumption: {draft_cmp.assumption}")
