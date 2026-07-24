"""Data-driven threshold calibration for FAQ dedup.

Replaces hardcoded DEDUP_COSINE_THRESHOLD with a threshold computed
automatically from labeled calibration pairs.  Core idea: sweep
thresholds 0.0-1.0, pick the one that maximizes F1.

Usage as module:
    from faq.threshold_calibration import calibrate_threshold, load_calibration_result

Usage as CLI:
    python -m faq.threshold_calibration              # run calibration, print report
    python -m faq.threshold_calibration --save        # also write JSON result file
    python -m faq.threshold_calibration --platt       # include Platt scaling
"""

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

CALIBRATION_RESULT_PATH = Path(__file__).resolve().parent / "calibration_result.json"


@dataclass
class CalibrationResult:
    optimal_threshold: float
    f1_at_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    score_gap: float
    dup_score_stats: dict[str, float]  # keys: min, mean, max
    distinct_score_stats: dict[str, float]  # keys: min, mean, max
    method: str
    model_id: str
    n_pairs: int
    timestamp: str
    platt_slope: float | None = None
    platt_intercept: float | None = None
    platt_threshold: float | None = None
    train_f1: float | None = None
    test_f1: float | None = None
    n_train: int | None = None
    n_test: int | None = None
    threshold_std: float | None = None
    f1_cv_mean: float | None = None
    f1_cv_std: float | None = None
    cv_seeds: int | None = None
    cv_folds: int | None = None
    stability_verdict: str | None = None
    eval_note: str | None = None


def calibrate_threshold(
    pairs: list[tuple[str, str, str]],
    embed_fn: Callable[[str], list[float]],
    *,
    method: str = "f1_optimal",
    model_id: str = "unknown",
    zero_fp: bool = False,
    preprocess: Callable[[str], str] | None = None,
    allow_overlap: bool = False,
) -> CalibrationResult:
    """Find the F1-optimal cosine threshold from labeled pairs.

    Args:
        pairs: List of (text_a, text_b, label) where label is "duplicate" or "distinct".
        embed_fn: Function that maps text -> list[float] embedding.
        method: "f1_optimal" (default) or "platt" for Platt scaling.
        model_id: Identifier for the embedding model used.
        zero_fp: If True, maximize F1 subject to zero false positives (precision=1.0).
            If no threshold achieves FP=0, the sweep returns the best threshold found
            with FP=0 constraint — which may have F1=0 if no such threshold exists.
        preprocess: Optional text preprocessor applied before embedding
            (e.g. dedup._normalize_for_embedding).
        allow_overlap: If True, skip the score-gap safety check even when duplicate
            and distinct score distributions overlap. Use only when testing with
            synthetic data where overlap is expected.

    Raises:
        ValueError: If no pairs produce a valid embedding score — either
            ``pairs`` is empty or all pairs yielded empty embedding vectors.

    Returns:
        CalibrationResult with the optimal threshold and diagnostics.
    """
    scores: list[float] = []
    labels: list[int] = []

    for text_a, text_b, label in pairs:
        if preprocess:
            text_a = preprocess(text_a)
            text_b = preprocess(text_b)
        vec_a = np.array(embed_fn(text_a))
        vec_b = np.array(embed_fn(text_b))
        if vec_a.size == 0 or vec_b.size == 0:
            continue
        sim = float(np.dot(vec_a, vec_b))
        scores.append(sim)
        labels.append(1 if label == "duplicate" else 0)

    if not scores:
        raise ValueError(
            f"calibrate_threshold: no valid pairs scored — "
            f"{len(pairs)} pairs provided but all produced empty embeddings. "
            f"Check that embed_fn is working and returning non-empty vectors."
        )

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    dup_scores = scores_arr[labels_arr == 1]
    dist_scores = scores_arr[labels_arr == 0]

    dup_stats = {
        "min": float(dup_scores.min()) if len(dup_scores) else 0.0,
        "mean": float(dup_scores.mean()) if len(dup_scores) else 0.0,
        "max": float(dup_scores.max()) if len(dup_scores) else 0.0,
    }
    distinct_stats = {
        "min": float(dist_scores.min()) if len(dist_scores) else 0.0,
        "mean": float(dist_scores.mean()) if len(dist_scores) else 0.0,
        "max": float(dist_scores.max()) if len(dist_scores) else 0.0,
    }

    score_gap = dup_stats["min"] - distinct_stats["max"]

    if not allow_overlap:
        _check_gap_safety(score_gap, model_id)

    if method == "platt":
        return _calibrate_platt(
            scores_arr, labels_arr, dup_stats, distinct_stats,
            score_gap, model_id, len(pairs),
        )

    # F1-optimal sweep
    best_threshold = 0.0
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0

    # 101 steps = 0.01 granularity (0.00, 0.01, ..., 1.00).
    # Sufficient for this dataset (score gap = 0.16); finer granularity via np.linspace(0, 1, 1001)
    # if future datasets have near-overlapping distributions.
    for t_int in range(0, 101):
        t = t_int / 100.0
        tp = int(np.sum((scores_arr >= t) & (labels_arr == 1)))
        fp = int(np.sum((scores_arr >= t) & (labels_arr == 0)))
        fn = int(np.sum((scores_arr < t) & (labels_arr == 1)))

        if zero_fp and fp > 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_p = precision
            best_r = recall

    return CalibrationResult(
        optimal_threshold=best_threshold,
        f1_at_threshold=best_f1,
        precision_at_threshold=best_p,
        recall_at_threshold=best_r,
        score_gap=score_gap,
        dup_score_stats=dup_stats,
        distinct_score_stats=distinct_stats,
        method="f1_optimal",
        model_id=model_id,
        n_pairs=len(pairs),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _calibrate_platt(
    scores: np.ndarray,
    labels: np.ndarray,
    dup_stats: dict,
    distinct_stats: dict,
    score_gap: float,
    model_id: str,
    n_pairs: int,
) -> CalibrationResult:
    """Platt scaling: fit logistic regression on scores -> P(duplicate)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for Platt scaling calibration. "
            "Install it with: pip install scikit-learn"
        ) from exc

    X = scores.reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X, labels)

    slope = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])

    probs = clf.predict_proba(X)[:, 1]
    platt_threshold = 0.5

    preds = (probs >= platt_threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Derive the raw cosine threshold equivalent: where P(dup) = 0.5
    raw_threshold = float(np.clip(-intercept / slope, 0.0, 1.0)) if abs(slope) > 1e-6 else 0.5

    return CalibrationResult(
        optimal_threshold=raw_threshold,
        f1_at_threshold=f1,
        precision_at_threshold=precision,
        recall_at_threshold=recall,
        score_gap=score_gap,
        dup_score_stats=dup_stats,
        distinct_score_stats=distinct_stats,
        method="platt",
        model_id=model_id,
        n_pairs=n_pairs,
        timestamp=datetime.now(timezone.utc).isoformat(),
        platt_slope=slope,
        platt_intercept=intercept,
        platt_threshold=platt_threshold,
    )


def _check_gap_safety(score_gap: float, model_id: str) -> None:
    if score_gap < 0.0:
        raise ValueError(
            f"Score overlap detected (gap={score_gap:.4f}) for model '{model_id}'. "
            f"Duplicate and distinct score distributions overlap — threshold is unreliable. "
            f"The embedding model cannot cleanly separate duplicates from distinct pairs."
        )
    if score_gap < 0.05:
        warnings.warn(
            f"Narrow score gap ({score_gap:.4f}) for model '{model_id}'. "
            f"Duplicate and distinct scores nearly overlap — threshold may be fragile.",
            UserWarning,
            stacklevel=3,
        )


def save_calibration_result(result: CalibrationResult, path: Path | None = None) -> Path:
    out = path or CALIBRATION_RESULT_PATH
    out.write_text(json.dumps(asdict(result), indent=2) + "\n")
    log.info("Calibration result saved to %s", out)
    log.info("Commit %s to version-control the updated threshold for all environments.", out.name)
    return out


def load_calibration_result(path: Path | None = None) -> CalibrationResult | None:
    p = path or CALIBRATION_RESULT_PATH
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return CalibrationResult(**data)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        log.warning("Failed to load calibration result from %s: %s", p, exc)
        return None


def _print_report(result: CalibrationResult) -> None:
    d = result.dup_score_stats
    s = result.distinct_score_stats
    print(
        "\n"
        "=================================================================\n"
        f"  THRESHOLD CALIBRATION REPORT\n"
        "=================================================================\n"
        f"  Model:           {result.model_id}\n"
        f"  Method:          {result.method}\n"
        f"  Pairs:           {result.n_pairs}\n"
        f"  Timestamp:       {result.timestamp}\n"
        "\n"
        f"  Optimal threshold:  {result.optimal_threshold:.4f}\n"
        f"  F1:                 {result.f1_at_threshold:.4f}\n"
        f"  Precision:          {result.precision_at_threshold:.4f}\n"
        f"  Recall:             {result.recall_at_threshold:.4f}\n"
        "\n"
        f"  Score gap:          {result.score_gap:.4f}\n"
        f"  Dup scores:         min={d['min']:.4f}  mean={d['mean']:.4f}  max={d['max']:.4f}\n"
        f"  Distinct scores:    min={s['min']:.4f}  mean={s['mean']:.4f}  max={s['max']:.4f}\n"
    )
    if result.platt_slope is not None:
        print(
            f"  Platt slope:        {result.platt_slope:.4f}\n"
            f"  Platt intercept:    {result.platt_intercept:.4f}\n"
            f"  Platt P(dup)>=0.5 at cosine: {result.optimal_threshold:.4f}\n"
        )
    print("=================================================================\n")


def _run_cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run threshold calibration against labeled pairs"
    )
    parser.add_argument("--save", action="store_true", help="Write result to JSON file")
    parser.add_argument("--output", type=str, default=None, help="Custom output path")
    parser.add_argument("--platt", action="store_true", help="Use Platt scaling method")
    parser.add_argument("--model-id", type=str, default=None, help="Override model ID")
    args = parser.parse_args()

    from faq.calibration_pairs import LABELED_PAIRS
    from faq.dedup import _normalize_for_embedding
    from services.embedding import EMBEDDING_MODEL, embed_text

    model_id = args.model_id or EMBEDDING_MODEL
    method = "platt" if args.platt else "f1_optimal"

    print(f"Running calibration ({method}) with {len(LABELED_PAIRS)} pairs...")
    result = calibrate_threshold(
        LABELED_PAIRS, embed_text, method=method, model_id=model_id,
        preprocess=_normalize_for_embedding,
    )

    _print_report(result)

    if args.save:
        out = Path(args.output) if args.output else None
        path = save_calibration_result(result, out)
        print(f"Result saved to {path}")


if __name__ == "__main__":
    _run_cli()
