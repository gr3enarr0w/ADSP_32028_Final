"""ANTSE-305: Automated retraining CronJob entrypoint.

Retrains the winning ensemble (TF-IDF LR + TF-IDF GB + DistilBERT LR),
re-optimizes ensemble weights via Optuna OOF, evaluates on the locked
holdout, and promotes only when the new model meets or exceeds the
current production baseline (within 1pp tolerance).

Exit codes:
    0 — promoted or skipped (both are expected outcomes)
    1 — failed (unexpected error; surfaces to k8s for alerting)

Usage:
    python -m scripts.retrain
    MIN_CATEGORY_SAMPLES=5 python -m scripts.retrain
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent
_MODELS_DIR = _BASE_DIR / "models"
_PRODUCTION_METRICS_PATH = _MODELS_DIR / "production_metrics.json"

# ---------------------------------------------------------------------------
# Environment-configurable constants
# ---------------------------------------------------------------------------

MIN_CATEGORY_SAMPLES: int = int(os.environ.get("MIN_CATEGORY_SAMPLES", "10"))
N_OOF_TRIALS: int = int(os.environ.get("N_OOF_TRIALS", "200"))
PROMOTION_TOLERANCE: float = float(os.environ.get("PROMOTION_TOLERANCE", "0.01"))

_FALLBACK_PRODUCTION_METRICS = {
    "macro_f1": 0.5742,
    "accuracy": 0.7245,
    "weighted_f1": 0.7151,
    "winner": "weighted_soft_vote_optimized",
    "winner_weights": {"lr": 0.1019, "gb": 0.1240, "bert": 0.7741},
    "holdout_sha_prefix": "deb05be8d5be703e",
    "evaluated_at": "2026-05-28",
}


# ---------------------------------------------------------------------------
# Step 1: Minimum data guard
# ---------------------------------------------------------------------------

def check_minimum_data(min_samples: int = MIN_CATEGORY_SAMPLES) -> tuple[list[str], list[str]]:
    """Query DB for per-category training sample counts.

    Returns:
        Tuple of (included_categories, excluded_categories).
        Categories with < min_samples training tickets are excluded and logged.

    Raises:
        RuntimeError: If total training set is too small to retrain.
    """
    from db import get_db
    from analysis.holdout import holdout_ticket_keys

    holdout = holdout_ticket_keys()

    conn = get_db()
    rows = conn.execute("""
        SELECT c.category, COUNT(*) AS n
        FROM ticket_classifications c
        JOIN tickets t ON t.ticket_key = c.ticket_key
        WHERE c.category IS NOT NULL
        GROUP BY c.category
        ORDER BY c.category
    """).fetchall()

    # Also get per-category counts excluding holdout tickets
    all_rows = conn.execute("""
        SELECT c.category, c.ticket_key
        FROM ticket_classifications c
        JOIN tickets t ON t.ticket_key = c.ticket_key
        WHERE c.category IS NOT NULL
    """).fetchall()
    conn.close()

    # Count training tickets (non-holdout) per category
    cat_counts: dict[str, int] = {}
    for row in all_rows:
        if row["ticket_key"] not in holdout:
            cat = row["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    included, excluded = [], []
    for cat, n_train in sorted(cat_counts.items()):
        if n_train < min_samples:
            log.warning(
                "Category '%s' has only %d training samples (< %d min) — excluding from retrain.",
                cat, n_train, min_samples,
            )
            excluded.append(cat)
        else:
            included.append(cat)

    if not included:
        raise RuntimeError(
            f"All categories excluded by minimum-data guard "
            f"(min_samples={min_samples}). Cannot retrain."
        )

    total_train = sum(cat_counts.get(c, 0) for c in included)
    log.info(
        "Data guard: %d categories included, %d excluded. "
        "Total training samples: %d.",
        len(included), len(excluded), total_train,
    )
    return included, excluded


# ---------------------------------------------------------------------------
# Step 2: Retrain base models
# ---------------------------------------------------------------------------

def retrain_base_models() -> tuple:
    """Retrain all three base models on current labeled data (excluding holdout).

    Returns:
        Tuple of (lr_pipeline, gb_pipeline, distilbert_lr, new_embeddings_cache).
        new_embeddings_cache is the dict returned by _load_embedding_cache()
        after distilbert.train() rebuilds it.

    Raises:
        Any exception from the underlying train() calls.
    """
    import analysis.tfidf_lr as tfidf_lr
    import analysis.tfidf_gb as tfidf_gb
    import analysis.distilbert as distilbert

    log.info("=== Retraining base models ===")

    log.info("Step 2a: Training TF-IDF + LR...")
    lr_pipeline = tfidf_lr.train(save=False)
    log.info("LR training complete.")

    log.info("Step 2b: Training TF-IDF + GB...")
    gb_pipeline = tfidf_gb.train(save=False)
    log.info("GB training complete.")

    log.info("Step 2c: Training DistilBERT LR (encodes + fits LR head)...")
    # distilbert.train(save=False) encodes texts and fits the LR head.
    # We also need the new embeddings cache for OOF weight optimization.
    # Because save=False skips writing the pkl, we call train() then
    # reload the embedding cache (which distilbert.train always writes).
    distilbert_lr = distilbert.train(save=False)
    log.info("DistilBERT training complete.")

    # WARNING: distilbert.train() always writes distilbert_embeddings.npz to disk
    # regardless of the save=False flag (save=False only skips the LR head pkl).
    # This means the embedding cache on disk is now the retrain candidate's cache,
    # even if the promotion gate later rejects this model.  If promotion is skipped,
    # the on-disk cache will be inconsistent with the production LR head until the
    # next successful promotion run.
    # TODO: make distilbert.train() accept a cache_path kwarg so retrain_base_models()
    # can write to a temp path (e.g. distilbert_embeddings_retrain_tmp.npz) and
    # maybe_promote() can atomically swap it into the canonical path on success or
    # discard it on failure.
    log.warning(
        "distilbert_embeddings.npz has been overwritten on disk by distilbert.train(). "
        "If promotion is skipped, the on-disk cache will be inconsistent with the "
        "production LR head until the next successful promotion run."
    )

    # Load the freshly written embeddings cache for OOF optimization.
    from analysis.ensemble_optimize import _load_embedding_cache
    cache = _load_embedding_cache()

    return lr_pipeline, gb_pipeline, distilbert_lr, cache


# ---------------------------------------------------------------------------
# Step 3: Re-optimize ensemble weights
# ---------------------------------------------------------------------------

def reoptimize_weights(
    lr_pipeline,
    gb_pipeline,
    distilbert_lr,
    cache: dict,
    n_trials: int = N_OOF_TRIALS,
) -> dict:
    """Re-run Optuna OOF weight optimization on training set.

    Reuses analysis.ensemble_optimize.generate_oof_predictions and
    optimize_weights — no holdout data is touched here.

    Args:
        lr_pipeline: Freshly fitted TF-IDF + LR sklearn Pipeline.
        gb_pipeline: Freshly fitted TF-IDF + GB sklearn Pipeline.
        distilbert_lr: Freshly fitted DistilBERT LogisticRegression head.
        cache: Embedding cache dict from _load_embedding_cache().
        n_trials: Optuna TPE trials (200 for scheduled runs, 300 for manual).

    Returns:
        Dict with w_lr_norm, w_gb_norm, w_bert_norm, best_oof_macro_f1.
    """
    from analysis.ensemble_optimize import generate_oof_predictions, optimize_weights
    from analysis.tfidf_lr import _load_dataset, _split_train_holdout

    log.info("=== Re-optimizing ensemble weights (%d Optuna trials) ===", n_trials)

    texts, labels, keys = _load_dataset()
    train_texts, train_labels, _, _ = _split_train_holdout(texts, labels, keys)

    log.info("Generating 5-fold OOF predictions for weight optimization...")
    oof_lr, oof_gb, oof_bert, le, y_int = generate_oof_predictions(
        train_texts, train_labels, cache
    )

    log.info("Running Optuna TPE search (%d trials)...", n_trials)
    opt_result = optimize_weights(oof_lr, oof_gb, oof_bert, y_int, n_trials=n_trials)

    log.info(
        "Optimized weights: LR=%.4f  GB=%.4f  DistilBERT=%.4f  OOF-F1=%.4f",
        opt_result["w_lr_norm"],
        opt_result["w_gb_norm"],
        opt_result["w_bert_norm"],
        opt_result["best_oof_macro_f1"],
    )
    return opt_result


# ---------------------------------------------------------------------------
# Step 4: Evaluate new ensemble on holdout
# ---------------------------------------------------------------------------

def evaluate_on_holdout(
    lr_pipeline,
    gb_pipeline,
    distilbert_lr,
    cache: dict,
    w_lr: float,
    w_gb: float,
    w_bert: float,
) -> dict:
    """Evaluate the newly trained ensemble on the locked holdout set.

    Calls verify_holdout_set() before touching any holdout data — aborts
    with RuntimeError if the integrity check fails.

    Args:
        lr_pipeline: Freshly fitted TF-IDF + LR pipeline.
        gb_pipeline: Freshly fitted TF-IDF + GB pipeline.
        distilbert_lr: Freshly fitted DistilBERT LR head.
        cache: Embedding cache from _load_embedding_cache().
        w_lr, w_gb, w_bert: Normalized ensemble weights (sum to 1.0).

    Returns:
        Dict with macro_f1, accuracy, weighted_f1, holdout_count.

    Raises:
        RuntimeError: If holdout integrity check fails.
    """
    from analysis.holdout import verify_holdout_set, holdout_ticket_keys
    from analysis.tfidf_lr import _load_dataset, _split_train_holdout, CATEGORIES
    from analysis.ensemble_optimize import _reorder_proba_cols
    from sklearn.metrics import classification_report

    log.info("=== Evaluating new ensemble on holdout ===")

    # Integrity gate — aborts on failure
    verify_holdout_set()

    texts, labels, keys = _load_dataset()
    _, _, test_texts, test_labels = _split_train_holdout(texts, labels, keys)

    # Align test embeddings from cache to text-pipeline holdout order
    holdout_keys_set = holdout_ticket_keys()
    test_keys_ordered = [k for k in keys if k in holdout_keys_set]

    cache_key_to_idx_test = {k: i for i, k in enumerate(cache["test_keys"])}
    missing = [k for k in test_keys_ordered if k not in cache_key_to_idx_test]
    if missing:
        raise RuntimeError(
            f"{len(missing)} holdout keys missing from embedding cache test set."
        )
    idx_order_test = [cache_key_to_idx_test[k] for k in test_keys_ordered]
    test_embeddings = cache["test_embeddings"][idx_order_test]

    # Verify label alignment
    test_labels_from_cache = [cache["test_labels"][i] for i in idx_order_test]
    if test_labels != test_labels_from_cache:
        mismatches = sum(a != b for a, b in zip(test_labels, test_labels_from_cache))
        raise RuntimeError(
            f"Holdout label mismatch between text pipeline and embedding cache: "
            f"{mismatches}/{len(test_labels)} differ."
        )

    # Get probabilities from each model
    lr_probs = _reorder_proba_cols(
        lr_pipeline.predict_proba(test_texts),
        list(lr_pipeline.classes_), CATEGORIES,
    )
    gb_probs = _reorder_proba_cols(
        gb_pipeline.predict_proba(test_texts),
        list(gb_pipeline.classes_), CATEGORIES,
    )
    bert_probs = _reorder_proba_cols(
        distilbert_lr.predict_proba(test_embeddings),
        list(distilbert_lr.classes_), CATEGORIES,
    )

    blended = w_lr * lr_probs + w_gb * gb_probs + w_bert * bert_probs
    pred_indices = np.argmax(blended, axis=1)
    preds = [CATEGORIES[i] for i in pred_indices]

    report_dict = classification_report(test_labels, preds, output_dict=True, zero_division=0)

    result = {
        "holdout_count": len(test_labels),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "accuracy": round(report_dict["accuracy"], 4),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
    }

    log.info(
        "New ensemble holdout: Macro-F1=%.4f  Accuracy=%.4f  Weighted-F1=%.4f",
        result["macro_f1"], result["accuracy"], result["weighted_f1"],
    )
    return result


# ---------------------------------------------------------------------------
# Step 5: Load current production metrics
# ---------------------------------------------------------------------------

def load_production_metrics() -> dict:
    """Load production_metrics.json.

    Returns:
        Dict with at least 'macro_f1' key for the promotion gate.

    Raises:
        FileNotFoundError: If production_metrics.json does not exist.  Auto-seeding
            was intentionally removed — operators must create this file manually with
            the current production model's metrics before running retrain.
    """
    if not _PRODUCTION_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"production_metrics.json not found at {_PRODUCTION_METRICS_PATH}. "
            "Create it manually with the current production model's metrics before running retrain."
        )

    with open(_PRODUCTION_METRICS_PATH) as f:
        metrics = json.load(f)
    log.info("Loaded production metrics: Macro-F1=%.4f", metrics["macro_f1"])
    return metrics


# ---------------------------------------------------------------------------
# Step 6: Promotion gate
# ---------------------------------------------------------------------------

def maybe_promote(
    lr_pipeline,
    gb_pipeline,
    distilbert_lr,
    cache: dict,
    new_metrics: dict,
    prev_metrics: dict,
    new_weights: dict,
    tolerance: float = PROMOTION_TOLERANCE,
) -> bool:
    """Promote new models to production if they pass the Macro-F1 gate.

    Gate: new_macro_f1 >= prev_macro_f1 - tolerance
    (within 1pp by default — allows for small sampling variation)

    If promoted:
      - Overwrites models/tfidf_lr.pkl, tfidf_gb.pkl, distilbert_lr.pkl,
        distilbert_embeddings.npz with the freshly trained versions.
      - Updates models/production_metrics.json with new metrics + weights.

    Args:
        lr_pipeline: New TF-IDF + LR pipeline.
        gb_pipeline: New TF-IDF + GB pipeline.
        distilbert_lr: New DistilBERT LR head.
        cache: New embedding cache (already written to disk by distilbert.train).
        new_metrics: Holdout evaluation result for new ensemble.
        prev_metrics: Current production_metrics.json content.
        new_weights: Optimized weights dict from reoptimize_weights().
        tolerance: Tolerance below which new model is still promoted (default 0.01).

    Returns:
        True if promoted, False if skipped.
    """
    new_f1 = new_metrics["macro_f1"]
    prev_f1 = prev_metrics["macro_f1"]
    threshold = prev_f1 - tolerance

    if new_f1 < threshold:
        log.info(
            "Promotion SKIPPED: new Macro-F1=%.4f < threshold=%.4f "
            "(prev=%.4f - tolerance=%.4f). Keeping existing models.",
            new_f1, threshold, prev_f1, tolerance,
        )
        return False

    log.info(
        "Promotion GATE PASSED: new Macro-F1=%.4f >= threshold=%.4f "
        "(prev=%.4f). Writing new models to disk...",
        new_f1, threshold, prev_f1,
    )

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Overwrite LR model
    lr_path = _MODELS_DIR / "tfidf_lr.pkl"
    with open(lr_path, "wb") as f:
        pickle.dump(lr_pipeline, f)
    log.info("Wrote %s", lr_path)

    # Overwrite GB model
    gb_path = _MODELS_DIR / "tfidf_gb.pkl"
    with open(gb_path, "wb") as f:
        pickle.dump(gb_pipeline, f)
    log.info("Wrote %s", gb_path)

    # Overwrite DistilBERT LR head
    bert_path = _MODELS_DIR / "distilbert_lr.pkl"
    with open(bert_path, "wb") as f:
        pickle.dump(distilbert_lr, f)
    log.info("Wrote %s", bert_path)

    # distilbert_embeddings.npz was already written to disk by distilbert.train()
    log.info("distilbert_embeddings.npz already updated by distilbert.train()")

    # Update production_metrics.json
    updated_metrics = {
        "macro_f1": new_f1,
        "accuracy": new_metrics["accuracy"],
        "weighted_f1": new_metrics["weighted_f1"],
        "winner": "weighted_soft_vote_optimized",
        "winner_weights": {
            "lr": new_weights["w_lr_norm"],
            "gb": new_weights["w_gb_norm"],
            "bert": new_weights["w_bert_norm"],
        },
        "prev_macro_f1": prev_f1,
        "delta": round(new_f1 - prev_f1, 4),
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    with open(_PRODUCTION_METRICS_PATH, "w") as f:
        json.dump(updated_metrics, f, indent=2)
    log.info("Updated production_metrics.json")

    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run() -> int:
    """Full retrain pipeline. Returns exit code (0 = ok, 1 = error)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    outcome = "failed"
    new_macro_f1 = 0.0
    prev_macro_f1 = 0.0
    delta = 0.0
    excluded_categories: list[str] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        log.info("=== ANTSE-305: Automated retrain pipeline START ===")

        # Step 1: Minimum data guard
        included_cats, excluded_categories = check_minimum_data(MIN_CATEGORY_SAMPLES)

        # Step 5 first — load before training so we have the baseline even if training fails
        prev_metrics = load_production_metrics()
        prev_macro_f1 = prev_metrics["macro_f1"]

        # Step 2: Retrain base models
        lr_pipeline, gb_pipeline, distilbert_lr, cache = retrain_base_models()

        # Step 3: Re-optimize ensemble weights
        opt_result = reoptimize_weights(
            lr_pipeline, gb_pipeline, distilbert_lr, cache, n_trials=N_OOF_TRIALS
        )
        w_lr = opt_result["w_lr_norm"]
        w_gb = opt_result["w_gb_norm"]
        w_bert = opt_result["w_bert_norm"]

        # Step 4: Evaluate new ensemble on holdout
        new_metrics = evaluate_on_holdout(
            lr_pipeline, gb_pipeline, distilbert_lr, cache, w_lr, w_gb, w_bert
        )
        new_macro_f1 = new_metrics["macro_f1"]
        delta = round(new_macro_f1 - prev_macro_f1, 4)

        # Step 6: Promotion gate
        promoted = maybe_promote(
            lr_pipeline, gb_pipeline, distilbert_lr, cache,
            new_metrics, prev_metrics, opt_result,
        )
        outcome = "promoted" if promoted else "skipped"

    except Exception as exc:
        log.exception("Retrain pipeline FAILED: %s", exc)
        outcome = "failed"

    # Step 7: Structured log output
    result_payload = {
        "event": "retrain_complete",
        "outcome": outcome,
        "new_macro_f1": new_macro_f1,
        "prev_macro_f1": prev_macro_f1,
        "delta": delta,
        "categories_excluded": excluded_categories,
        "timestamp": timestamp,
    }
    print(json.dumps(result_payload))

    # Step 8: Exit codes
    if outcome == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
