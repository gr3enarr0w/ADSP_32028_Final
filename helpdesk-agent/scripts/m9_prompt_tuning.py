"""M9 prompt tuning A/B experiment — 5-fold stratified CV on ai_draft_feedback.

Tests the configured prompt variants against the full ground-truth corpus.
Stratifies folds by response_type. Reports mean ± std judge score,
response_type accuracy, draft length p50/p95, Gemini latency p95,
and Welch's t-test vs baseline.

Usage:
    PYTHONPATH=. DATABASE_URL=postgresql://... \\
        GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/service_account.json \\
        python -m scripts.m9_prompt_tuning

    python -m scripts.m9_prompt_tuning --workers 30 --apply-winner
    python -m scripts.m9_prompt_tuning --dry-run
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import date

from core.pipeline import load_pipeline_config
from db import init_db
from plugins.responder.ann_fewshot import ANNFewShotIndex
from plugins.responder.drafting import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    PROMPT_VARIANTS,
    apply_prompt_tuning_results_to_module,
    build_stratified_folds,
    evaluate_prompt_variants,
    format_prompt_tuning_results_comment,
    load_ground_truth_corpus,
    select_winning_variant,
)

log = logging.getLogger(__name__)


def _print_results_table(results) -> None:
    print(format_prompt_tuning_results_comment(
        results,
        winner=select_winning_variant(results),
        evaluated_at=date.today().isoformat(),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 prompt tuning 5-fold CV experiment")
    parser.add_argument("--folds", type=int, default=CV_N_SPLITS)
    parser.add_argument("--seed", type=int, default=CV_RANDOM_STATE)
    parser.add_argument("--workers", type=int, default=30, help="Concurrent Gemini threads")
    parser.add_argument("--limit", type=int, help="Limit corpus size for quick testing")
    parser.add_argument("--checkpoint-file", help="Override checkpoint path for CV resume")
    parser.add_argument("--apply-winner", action="store_true", help="Update drafting.py with winner")
    parser.add_argument("--dry-run", action="store_true", help="Validate corpus/folds only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_pipeline_config()
    init_db()

    corpus = load_ground_truth_corpus()
    if args.limit:
        corpus = corpus[:args.limit]
    log.info("Loaded %d labeled rows from ai_draft_feedback", len(corpus))
    if not corpus:
        print(
            "Ground truth corpus is empty. Populate ai_draft_feedback with "
            "actual_response and similarity_score, then re-run."
        )
        return

    labels = Counter(row["response_type"] for row in corpus)
    log.info("response_type distribution: %s", dict(labels))

    try:
        folds = build_stratified_folds(corpus, n_splits=args.folds, random_state=args.seed)
    except ValueError as exc:
        print(f"Cannot build stratified folds: {exc}")
        return

    log.info(
        "Stratified %d-fold CV (random_state=%d); test sizes per fold: %s",
        args.folds,
        args.seed,
        [len(test_idx) for _, test_idx in folds],
    )

    if args.dry_run:
        print(
            f"Dry run OK: {len(corpus)} rows, {args.folds} stratified folds, "
            f"{len(PROMPT_VARIANTS)} variants."
        )
        return

    if ANNFewShotIndex.is_empty():
        log.info("Building few-shot ANN index before evaluation")
        ANNFewShotIndex.build()

    from services.embedding import embed_text as _embed

    log.info("Warming up embedding model...")
    _embed("warmup", task_type="query")
    log.info("Embedding model ready.")

    log.info(
        "Starting CV: ~%d Gemini calls (%d test rows × %d variants across folds)",
        len(corpus) * len(PROMPT_VARIANTS),
        len(corpus),
        len(PROMPT_VARIANTS),
    )

    results = evaluate_prompt_variants(
        corpus,
        n_splits=args.folds,
        random_state=args.seed,
        workers=args.workers,
        checkpoint_file=args.checkpoint_file,
    )
    _print_results_table(results)

    winner = select_winning_variant(results)
    if args.apply_winner:
        apply_prompt_tuning_results_to_module(
            results,
            winner=winner,
            evaluated_at=date.today().isoformat(),
        )
        log.info("Updated drafting.py — WINNING_PROMPT_VARIANT=%s", winner)


if __name__ == "__main__":
    main()
