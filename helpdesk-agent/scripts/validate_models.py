"""Model validation script — production scalability proof.

Runs four validation experiments and writes a machine-readable report.

Experiments
-----------
1. Learning curves  — F1 at 10–80 % of training data (skipped with --quick)
2. 5-fold stratified CV — mean ± std Macro-F1 for LR, GB, DistilBERT, Ensemble
3. Full holdout evaluation — per-class P/R/F1/support for every model
4. Confidence calibration — ensemble decile table on the holdout set

Usage
-----
    python -m scripts.validate_models           # full run
    python -m scripts.validate_models --quick   # skip learning curves
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Bootstrap — make the repo root importable regardless of cwd
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Deferred imports from analysis package (after sys.path is set)
from analysis.tfidf_lr import CATEGORIES, _load_dataset, _split_train_holdout  # noqa: E402
from analysis.holdout import holdout_ticket_keys, validation_ticket_keys, verify_holdout_set  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODELS_DIR = os.path.join(_REPO_ROOT, "models")
LR_PATH = os.path.join(_MODELS_DIR, "tfidf_lr.pkl")
GB_PATH = os.path.join(_MODELS_DIR, "tfidf_gb.pkl")
DISTILBERT_PATH = os.path.join(_MODELS_DIR, "distilbert_lr.pkl")
EMBEDDINGS_PATH = os.path.join(_MODELS_DIR, "distilbert_embeddings.npz")
PRODUCTION_METRICS_PATH = os.path.join(_MODELS_DIR, "production_metrics.json")
HOLDOUT_JSON = os.path.join(_REPO_ROOT, "holdout_set.json")
REPORT_PATH = os.path.join(_MODELS_DIR, "validation_report.json")

# ---------------------------------------------------------------------------
# TF-IDF pipeline factory — identical hyperparameters to tfidf_lr.train()
# ---------------------------------------------------------------------------


def _build_lr_pipeline() -> Pipeline:
    """Return a fresh unfitted TF-IDF + LR pipeline matching production config."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )),
    ])


def _build_gb_pipeline() -> Pipeline:
    """Return a fresh unfitted TF-IDF + LightGBM pipeline matching production config."""
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("gb", LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_fitted_models() -> dict[str, Any]:
    """Load all three fitted models from disk. Raises if any are missing."""
    missing = [p for p in (LR_PATH, GB_PATH, DISTILBERT_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}. "
            "Train all models before running validation."
        )
    with open(LR_PATH, "rb") as f:
        lr = pickle.load(f)
    with open(GB_PATH, "rb") as f:
        gb = pickle.load(f)
    with open(DISTILBERT_PATH, "rb") as f:
        bert_lr = pickle.load(f)
    return {"lr": lr, "gb": gb, "bert_lr": bert_lr}


def _load_embedding_cache() -> dict[str, Any]:
    """Load distilbert_embeddings.npz. Returns dict of arrays and label lists."""
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embedding cache not found at {EMBEDDINGS_PATH}. "
            "Run `python -m analysis.distilbert train` first."
        )
    cache = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    result = {
        "train_embeddings": cache["train_embeddings"],
        "train_labels": cache["train_labels"].tolist(),
        "train_keys": cache["train_keys"].tolist(),
        "test_embeddings": cache["test_embeddings"],
        "test_labels": cache["test_labels"].tolist(),
        "test_keys": cache["test_keys"].tolist(),
    }
    if "val_embeddings" in cache:
        result["val_embeddings"] = cache["val_embeddings"]
        result["val_labels"] = cache["val_labels"].tolist()
        result["val_keys"] = cache["val_keys"].tolist()
    return result


def _load_ensemble_weights() -> dict[str, float]:
    """Load winner_weights from production_metrics.json."""
    if not os.path.exists(PRODUCTION_METRICS_PATH):
        log.warning("production_metrics.json not found — using equal weights 1/3 each")
        return {"lr": 1 / 3, "gb": 1 / 3, "bert": 1 / 3}
    with open(PRODUCTION_METRICS_PATH) as f:
        metrics = json.load(f)
    weights = metrics["winner_weights"]
    total = weights["lr"] + weights["gb"] + weights["bert"]
    # Normalise in case of floating-point drift
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Probability alignment helper (mirrors ensemble_optimize._reorder_proba_cols)
# ---------------------------------------------------------------------------


def _reorder_proba(proba: np.ndarray, model_classes: list, target_order: list) -> np.ndarray:
    """Reorder predict_proba columns to match target_order (fills 0 for missing classes)."""
    cls_to_idx = {c: i for i, c in enumerate(model_classes)}
    out = np.zeros((proba.shape[0], len(target_order)), dtype=np.float64)
    for dest_col, cat in enumerate(target_order):
        src_col = cls_to_idx.get(cat)
        if src_col is not None:
            out[:, dest_col] = proba[:, src_col]
    return out


# ---------------------------------------------------------------------------
# Aligned embedding lookup for holdout tickets
# ---------------------------------------------------------------------------


def _align_test_embeddings(
    cache: dict[str, Any],
    test_keys_ordered: list[str],
    test_labels_text: list[str],
) -> np.ndarray:
    """Return embedding matrix aligned to test_keys_ordered, verifying label match."""
    cache_key_to_idx = {k: i for i, k in enumerate(cache["test_keys"])}
    missing = [k for k in test_keys_ordered if k not in cache_key_to_idx]
    if missing:
        raise RuntimeError(
            f"{len(missing)} holdout keys missing from embedding cache test set "
            f"(e.g. {missing[:3]}). Re-run `python -m analysis.distilbert train`."
        )
    idx_order = [cache_key_to_idx[k] for k in test_keys_ordered]
    test_emb = cache["test_embeddings"][idx_order]
    cached_labels = [cache["test_labels"][i] for i in idx_order]
    mismatches = sum(a != b for a, b in zip(test_labels_text, cached_labels))
    if mismatches:
        raise RuntimeError(
            f"Holdout label mismatch: {mismatches}/{len(test_labels_text)} "
            "differ between text pipeline and embedding cache. "
            "Re-run `python -m analysis.distilbert train`."
        )
    return test_emb


def _align_train_embeddings(
    cache: dict[str, Any],
    train_labels: list[str],
) -> np.ndarray:
    """Return training embeddings aligned to text-pipeline order via key lookup."""
    holdout_keys_set = holdout_ticket_keys()
    val_keys_set = validation_ticket_keys()
    _, all_labels, all_keys = _load_dataset()
    train_keys_ordered = [
        k for k, lb in zip(all_keys, all_labels)
        if k not in holdout_keys_set and k not in val_keys_set
    ]
    cache_key_to_idx = {k: i for i, k in enumerate(cache["train_keys"])}
    missing = [k for k in train_keys_ordered if k not in cache_key_to_idx]
    if missing:
        raise RuntimeError(
            f"{len(missing)} training keys missing from embedding cache "
            f"(e.g. {missing[:3]}). Re-run `python -m analysis.distilbert train`."
        )
    idx_order = [cache_key_to_idx[k] for k in train_keys_ordered]
    train_emb = cache["train_embeddings"][idx_order]
    cached_labels = [cache["train_labels"][i] for i in idx_order]
    mismatches = sum(a != b for a, b in zip(train_labels, cached_labels))
    if mismatches:
        raise RuntimeError(
            f"Training label mismatch: {mismatches}/{len(train_labels)} differ. "
            "Re-run `python -m analysis.distilbert train`."
        )
    return train_emb


# ---------------------------------------------------------------------------
# Ensemble probability prediction
# ---------------------------------------------------------------------------


def _ensemble_proba(
    models: dict[str, Any],
    weights: dict[str, float],
    test_texts: list[str],
    test_emb: np.ndarray,
) -> np.ndarray:
    """Return weighted-average probability matrix (n_samples, n_categories)."""
    lr_probs = _reorder_proba(
        models["lr"].predict_proba(test_texts),
        list(models["lr"].classes_),
        CATEGORIES,
    )
    gb_probs = _reorder_proba(
        models["gb"].predict_proba(test_texts),
        list(models["gb"].classes_),
        CATEGORIES,
    )
    bert_probs = _reorder_proba(
        models["bert_lr"].predict_proba(test_emb),
        list(models["bert_lr"].classes_),
        CATEGORIES,
    )
    blended = (
        weights["lr"] * lr_probs
        + weights["gb"] * gb_probs
        + weights["bert"] * bert_probs
    )
    return blended


def _proba_to_preds(proba: np.ndarray) -> list[str]:
    """Argmax to category label."""
    return [CATEGORIES[i] for i in np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# Section 1: Learning curves
# ---------------------------------------------------------------------------


def run_learning_curves(
    train_texts: list[str],
    train_labels: list[str],
    train_emb: np.ndarray,
    test_texts: list[str],
    test_labels: list[str],
    test_emb: np.ndarray,
    weights: dict[str, float],
    fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
) -> dict[str, Any]:
    """Train each model on subsample fractions and evaluate on the fixed holdout.

    Args:
        train_texts: Full training text corpus.
        train_labels: Corresponding labels.
        train_emb: Full training embeddings, aligned to train_texts order.
        test_texts: Fixed holdout texts.
        test_labels: Holdout labels.
        test_emb: Holdout embeddings.
        weights: Ensemble weights {"lr", "gb", "bert"}.
        fractions: Training data fractions to evaluate.

    Returns:
        Dict mapping fraction string -> {"lr", "gb", "bert", "ensemble"} F1 scores.
    """
    log.info("=== Learning Curves ===")
    results: dict[str, dict[str, float]] = {}

    for frac in fractions:
        t0 = time.time()
        pct_label = f"{int(frac * 100)}%"
        log.info("  Training at %s of data...", pct_label)

        # Subsample using StratifiedShuffleSplit to preserve class distribution
        if frac >= 1.0:
            idx = list(range(len(train_labels)))
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=42)
            idx, _ = next(sss.split(train_texts, train_labels))

        sub_texts = [train_texts[i] for i in idx]
        sub_labels = [train_labels[i] for i in idx]
        sub_emb = train_emb[idx]

        # --- LR ---
        lr_pipe = _build_lr_pipeline()
        lr_pipe.fit(sub_texts, sub_labels)
        lr_preds = lr_pipe.predict(test_texts)
        lr_f1 = f1_score(test_labels, lr_preds, average="macro", zero_division=0)

        # --- GB ---
        gb_pipe = _build_gb_pipeline()
        gb_pipe.fit(sub_texts, sub_labels)
        gb_preds = gb_pipe.predict(test_texts)
        gb_f1 = f1_score(test_labels, gb_preds, average="macro", zero_division=0)

        # --- DistilBERT LR head retrained on embedding subslice ---
        bert_clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs"
        )
        bert_clf.fit(sub_emb, sub_labels)
        bert_preds = bert_clf.predict(test_emb)
        bert_f1 = f1_score(test_labels, bert_preds, average="macro", zero_division=0)

        # --- Ensemble ---
        lr_probs = _reorder_proba(
            lr_pipe.predict_proba(test_texts), list(lr_pipe.classes_), CATEGORIES
        )
        gb_probs = _reorder_proba(
            gb_pipe.predict_proba(test_texts), list(gb_pipe.classes_), CATEGORIES
        )
        bert_probs = _reorder_proba(
            bert_clf.predict_proba(test_emb), list(bert_clf.classes_), CATEGORIES
        )
        ensemble_proba = (
            weights["lr"] * lr_probs
            + weights["gb"] * gb_probs
            + weights["bert"] * bert_probs
        )
        ens_preds = _proba_to_preds(ensemble_proba)
        ens_f1 = f1_score(test_labels, ens_preds, average="macro", zero_division=0)

        results[pct_label] = {
            "lr": round(lr_f1, 4),
            "gb": round(gb_f1, 4),
            "bert": round(bert_f1, 4),
            "ensemble": round(ens_f1, 4),
            "n_train": len(sub_labels),
        }
        elapsed = time.time() - t0
        log.info(
            "    %s  n=%d  LR=%.3f  GB=%.3f  BERT=%.3f  Ens=%.3f  (%.1fs)",
            pct_label, len(sub_labels), lr_f1, gb_f1, bert_f1, ens_f1, elapsed,
        )

    return results


def _print_learning_curves(curves: dict[str, dict[str, float]]) -> None:
    print("\n=== LEARNING CURVES ===")
    print(f"{'Train%':<8} {'LR-F1':>8} {'GB-F1':>8} {'BERT-F1':>8} {'Ens-F1':>10} {'n_train':>8}")
    print("-" * 54)
    for pct, m in sorted(curves.items(), key=lambda x: int(x[0].rstrip("%"))):
        print(
            f"{pct:<8} {m['lr']:>8.3f} {m['gb']:>8.3f} {m['bert']:>8.3f}"
            f" {m['ensemble']:>10.3f} {m['n_train']:>8}"
        )


# ---------------------------------------------------------------------------
# Section 2: 5-fold stratified CV
# ---------------------------------------------------------------------------


def run_cross_validation(
    train_texts: list[str],
    train_labels: list[str],
    train_emb: np.ndarray,
    weights: dict[str, float],
    n_splits: int = 5,
) -> dict[str, dict[str, float]]:
    """Run 5-fold stratified CV on full training set for all four models.

    Args:
        train_texts: Full training text corpus.
        train_labels: Corresponding labels.
        train_emb: Full training embeddings aligned to train_texts order.
        weights: Ensemble weights.
        n_splits: Number of CV folds.

    Returns:
        Dict keyed by model name with "macro_f1_mean", "macro_f1_std",
        "accuracy_mean", "accuracy_std".
    """
    from sklearn.model_selection import cross_validate as sk_cv

    log.info("=== 5-Fold Stratified CV ===")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: dict[str, dict[str, float]] = {}

    # --- LR ---
    log.info("  CV for TF-IDF + LR...")
    lr_cv = sk_cv(
        _build_lr_pipeline(), train_texts, train_labels, cv=cv,
        scoring=["f1_macro", "accuracy"], return_train_score=True,
    )
    results["lr"] = {
        "macro_f1_mean": round(float(lr_cv["test_f1_macro"].mean()), 4),
        "macro_f1_std": round(float(lr_cv["test_f1_macro"].std()), 4),
        "accuracy_mean": round(float(lr_cv["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(lr_cv["test_accuracy"].std()), 4),
        "train_macro_f1_mean": round(float(lr_cv["train_f1_macro"].mean()), 4),
    }

    # --- GB ---
    log.info("  CV for TF-IDF + GB...")
    gb_cv = sk_cv(
        _build_gb_pipeline(), train_texts, train_labels, cv=cv,
        scoring=["f1_macro", "accuracy"], return_train_score=True,
    )
    results["gb"] = {
        "macro_f1_mean": round(float(gb_cv["test_f1_macro"].mean()), 4),
        "macro_f1_std": round(float(gb_cv["test_f1_macro"].std()), 4),
        "accuracy_mean": round(float(gb_cv["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(gb_cv["test_accuracy"].std()), 4),
        "train_macro_f1_mean": round(float(gb_cv["train_f1_macro"].mean()), 4),
    }

    # --- DistilBERT LR head on cached embeddings ---
    log.info("  CV for DistilBERT embeddings + LR head...")
    bert_clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs"
    )
    bert_cv = sk_cv(
        bert_clf, train_emb, train_labels, cv=cv,
        scoring=["f1_macro", "accuracy"], return_train_score=True,
    )
    results["bert"] = {
        "macro_f1_mean": round(float(bert_cv["test_f1_macro"].mean()), 4),
        "macro_f1_std": round(float(bert_cv["test_f1_macro"].std()), 4),
        "accuracy_mean": round(float(bert_cv["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(bert_cv["test_accuracy"].std()), 4),
        "train_macro_f1_mean": round(float(bert_cv["train_f1_macro"].mean()), 4),
    }

    # --- Ensemble via fold-by-fold soft vote ---
    # We compute OOF ensemble predictions manually so the CV estimate is honest.
    log.info("  CV for Ensemble (fold-by-fold OOF soft vote)...")
    train_texts_list = list(train_texts)
    oof_true: list[str] = []
    oof_pred: list[str] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(train_texts_list, train_labels)):
        fold_train_X = [train_texts_list[i] for i in tr_idx]
        fold_train_y = [train_labels[i] for i in tr_idx]
        fold_val_X = [train_texts_list[i] for i in val_idx]
        fold_val_y = [train_labels[i] for i in val_idx]
        fold_train_emb = train_emb[tr_idx]
        fold_val_emb = train_emb[val_idx]

        fold_lr = _build_lr_pipeline()
        fold_lr.fit(fold_train_X, fold_train_y)

        fold_gb = _build_gb_pipeline()
        fold_gb.fit(fold_train_X, fold_train_y)

        fold_bert = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs"
        )
        fold_bert.fit(fold_train_emb, fold_train_y)

        lr_probs = _reorder_proba(
            fold_lr.predict_proba(fold_val_X), list(fold_lr.classes_), CATEGORIES
        )
        gb_probs = _reorder_proba(
            fold_gb.predict_proba(fold_val_X), list(fold_gb.classes_), CATEGORIES
        )
        bert_probs = _reorder_proba(
            fold_bert.predict_proba(fold_val_emb), list(fold_bert.classes_), CATEGORIES
        )
        blended = (
            weights["lr"] * lr_probs
            + weights["gb"] * gb_probs
            + weights["bert"] * bert_probs
        )
        fold_preds = _proba_to_preds(blended)
        oof_true.extend(fold_val_y)
        oof_pred.extend(fold_preds)
        log.info("    Fold %d/%d complete", fold_idx + 1, n_splits)

    # Compute per-fold F1 from OOF slices for mean±std reporting
    fold_f1s = []
    fold_accs = []
    train_texts_list_len = len(train_texts_list)
    fold_boundaries = list(cv.split(train_texts_list, train_labels))
    ptr = 0
    for tr_idx, val_idx in fold_boundaries:
        n = len(val_idx)
        fold_true = oof_true[ptr:ptr + n]
        fold_pred_slice = oof_pred[ptr:ptr + n]
        fold_f1s.append(f1_score(fold_true, fold_pred_slice, average="macro", zero_division=0))
        fold_accs.append(
            sum(t == p for t, p in zip(fold_true, fold_pred_slice)) / n
        )
        ptr += n

    results["ensemble"] = {
        "macro_f1_mean": round(float(np.mean(fold_f1s)), 4),
        "macro_f1_std": round(float(np.std(fold_f1s)), 4),
        "accuracy_mean": round(float(np.mean(fold_accs)), 4),
        "accuracy_std": round(float(np.std(fold_accs)), 4),
        "train_macro_f1_mean": None,  # not meaningful for stacked ensemble
        "oof_macro_f1": round(
            f1_score(oof_true, oof_pred, average="macro", zero_division=0), 4
        ),
    }

    return results


def _print_cv_results(cv_results: dict[str, dict[str, float]]) -> None:
    MODEL_LABELS = {"lr": "LR", "gb": "GB", "bert": "DistilBERT", "ensemble": "Ensemble"}
    print("\n=== 5-FOLD CV (full training set) ===")
    print(f"{'Model':<14} {'Macro-F1 (mean ± std)':>24} {'Accuracy (mean ± std)':>24}")
    print("-" * 64)
    for key in ("lr", "gb", "bert", "ensemble"):
        m = cv_results[key]
        f1_str = f"{m['macro_f1_mean']:.3f} ± {m['macro_f1_std']:.3f}"
        acc_str = f"{m['accuracy_mean']:.3f} ± {m['accuracy_std']:.3f}"
        overfit_note = ""
        if m.get("train_macro_f1_mean") is not None:
            gap = m["train_macro_f1_mean"] - m["macro_f1_mean"]
            overfit_note = f"  (train={m['train_macro_f1_mean']:.3f}, gap={gap:+.3f})"
        print(f"{MODEL_LABELS[key]:<14} {f1_str:>24} {acc_str:>24}{overfit_note}")


# ---------------------------------------------------------------------------
# Section 3: Full holdout evaluation
# ---------------------------------------------------------------------------


def run_holdout_evaluation(
    models: dict[str, Any],
    weights: dict[str, float],
    test_texts: list[str],
    test_labels: list[str],
    test_emb: np.ndarray,
) -> dict[str, Any]:
    """Evaluate all models on the fixed holdout set.

    Args:
        models: Dict with "lr", "gb", "bert_lr" fitted model objects.
        weights: Ensemble weights.
        test_texts: Holdout texts.
        test_labels: True holdout labels.
        test_emb: Holdout embeddings.

    Returns:
        Dict keyed by model name with per-class P/R/F1/support and overall stats.
    """
    log.info("=== Holdout Evaluation ===")
    results: dict[str, Any] = {}

    def _eval(preds: list[str], name: str) -> dict[str, Any]:
        report = classification_report(test_labels, preds, output_dict=True, zero_division=0)
        return {
            "holdout_count": len(test_labels),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "accuracy": round(report["accuracy"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            "per_class": {
                cat: {
                    "precision": round(report.get(cat, {}).get("precision", 0.0), 4),
                    "recall": round(report.get(cat, {}).get("recall", 0.0), 4),
                    "f1": round(report.get(cat, {}).get("f1-score", 0.0), 4),
                    "support": int(report.get(cat, {}).get("support", 0)),
                }
                for cat in CATEGORIES
            },
        }

    # LR
    log.info("  Evaluating LR on holdout...")
    results["lr"] = _eval(models["lr"].predict(test_texts), "LR")

    # GB
    log.info("  Evaluating GB on holdout...")
    results["gb"] = _eval(models["gb"].predict(test_texts), "GB")

    # DistilBERT
    log.info("  Evaluating DistilBERT on holdout...")
    results["bert"] = _eval(models["bert_lr"].predict(test_emb), "DistilBERT")

    # Ensemble
    log.info("  Evaluating Ensemble on holdout...")
    ens_proba = _ensemble_proba(models, weights, test_texts, test_emb)
    ens_preds = _proba_to_preds(ens_proba)
    results["ensemble"] = _eval(ens_preds, "Ensemble")

    return results


def _print_holdout_results(holdout_results: dict[str, Any]) -> None:
    MODEL_LABELS = {"lr": "LR", "gb": "GB", "bert": "DistilBERT", "ensemble": "Ensemble"}
    n = holdout_results["ensemble"]["holdout_count"]
    print(f"\n=== HOLDOUT EVALUATION (n={n}) ===")

    for model_key in ("lr", "gb", "bert", "ensemble"):
        label = MODEL_LABELS[model_key]
        m = holdout_results[model_key]
        print(f"\n--- {label} ---")
        print(f"  Macro-F1={m['macro_f1']:.4f}  Accuracy={m['accuracy']:.4f}"
              f"  Weighted-F1={m['weighted_f1']:.4f}")
        print(f"  {'Category':<20} {'P':>6} {'R':>6} {'F1':>6} {'n':>5}")
        print("  " + "-" * 43)
        for cat in CATEGORIES:
            c = m["per_class"][cat]
            flag = "  *" if c["f1"] < 0.5 and c["support"] > 0 else ""
            print(
                f"  {cat:<20} {c['precision']:>6.3f} {c['recall']:>6.3f}"
                f" {c['f1']:>6.3f} {c['support']:>5}{flag}"
            )


# ---------------------------------------------------------------------------
# Section 4: Confidence calibration
# ---------------------------------------------------------------------------


def run_calibration_analysis(
    models: dict[str, Any],
    weights: dict[str, float],
    test_texts: list[str],
    test_labels: list[str],
    test_emb: np.ndarray,
    n_deciles: int = 10,
) -> dict[str, Any]:
    """Compute ensemble confidence decile table on holdout.

    Args:
        models: Fitted models.
        weights: Ensemble weights.
        test_texts: Holdout texts.
        test_labels: True holdout labels.
        test_emb: Holdout embeddings.
        n_deciles: Number of equal-frequency confidence bins.

    Returns:
        Dict with "deciles" list (each with confidence_range, n_tickets,
        pct_correct) and summary stats.
    """
    log.info("=== Confidence Calibration ===")

    proba = _ensemble_proba(models, weights, test_texts, test_emb)
    max_confidence = np.max(proba, axis=1)  # ensemble's winning-class probability
    preds = _proba_to_preds(proba)
    correct = np.array([p == t for p, t in zip(preds, test_labels)])

    # Sort by confidence for decile binning
    sorted_idx = np.argsort(max_confidence)
    conf_sorted = max_confidence[sorted_idx]
    correct_sorted = correct[sorted_idx]

    n = len(test_labels)
    bin_size = n // n_deciles
    deciles = []

    for d in range(n_deciles):
        start = d * bin_size
        end = (d + 1) * bin_size if d < n_deciles - 1 else n
        bin_conf = conf_sorted[start:end]
        bin_correct = correct_sorted[start:end]
        deciles.append({
            "decile": d + 1,
            "conf_min": round(float(bin_conf.min()), 4),
            "conf_max": round(float(bin_conf.max()), 4),
            "conf_mean": round(float(bin_conf.mean()), 4),
            "n_tickets": int(end - start),
            "n_correct": int(bin_correct.sum()),
            "pct_correct": round(float(bin_correct.mean()) * 100, 1),
        })

    # Overall calibration stats
    overall_accuracy = round(float(correct.mean()), 4)
    # Expected calibration error (ECE): mean |confidence - accuracy| per bin
    ece = float(
        np.mean([abs(d["conf_mean"] - d["pct_correct"] / 100) for d in deciles])
    )

    result = {
        "overall_accuracy": overall_accuracy,
        "mean_confidence": round(float(max_confidence.mean()), 4),
        "ece": round(ece, 4),
        "n_tickets": n,
        "deciles": deciles,
    }

    return result


def _print_calibration(calib: dict[str, Any]) -> None:
    print("\n=== CONFIDENCE CALIBRATION (Ensemble on holdout) ===")
    print(f"Overall accuracy : {calib['overall_accuracy']:.4f}")
    print(f"Mean confidence  : {calib['mean_confidence']:.4f}")
    print(f"ECE (calibration error) : {calib['ece']:.4f}")
    print()
    print(f"{'Decile':>7} {'Conf range':>18} {'n':>5} {'% correct':>10}")
    print("-" * 46)
    for d in calib["deciles"]:
        conf_range = f"[{d['conf_min']:.3f}, {d['conf_max']:.3f}]"
        print(
            f"{d['decile']:>7} {conf_range:>18} {d['n_tickets']:>5}"
            f" {d['pct_correct']:>9.1f}%"
        )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main(quick: bool = False) -> dict[str, Any]:
    """Run all validation experiments and return structured results.

    Args:
        quick: If True, skip learning curves (faster run for CI/pre-PR checks).

    Returns:
        Full validation report dict (also written to models/validation_report.json).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t_start = time.time()

    log.info("Starting model validation  [quick=%s]", quick)

    # --- Integrity gate ---
    verify_holdout_set()

    # --- Load data ---
    log.info("Loading dataset...")
    texts, labels, keys = _load_dataset()
    train_texts, train_labels, test_texts, test_labels = _split_train_holdout(
        texts, labels, keys
    )
    log.info("Training set: %d  Holdout: %d", len(train_texts), len(test_labels))

    # --- Load models and cache ---
    log.info("Loading fitted models...")
    models = _load_fitted_models()

    log.info("Loading embedding cache...")
    cache = _load_embedding_cache()

    # Align embeddings to text-pipeline ordering
    log.info("Aligning training embeddings...")
    train_emb = _align_train_embeddings(cache, train_labels)

    holdout_keys_set = holdout_ticket_keys()
    test_keys_ordered = [k for k in keys if k in holdout_keys_set]
    log.info("Aligning holdout embeddings...")
    test_emb = _align_test_embeddings(cache, test_keys_ordered, test_labels)

    weights = _load_ensemble_weights()
    log.info("Ensemble weights: LR=%.4f  GB=%.4f  BERT=%.4f",
             weights["lr"], weights["gb"], weights["bert"])

    # --- Section 1: Learning curves ---
    if quick:
        log.info("--quick flag set: skipping learning curves")
        learning_curves: dict[str, Any] = {"skipped": True, "reason": "--quick flag"}
    else:
        learning_curves = run_learning_curves(
            train_texts, train_labels, train_emb,
            test_texts, test_labels, test_emb,
            weights,
        )
        _print_learning_curves(learning_curves)

    # --- Section 2: 5-fold CV ---
    cv_results = run_cross_validation(
        train_texts, train_labels, train_emb, weights
    )
    _print_cv_results(cv_results)

    # --- Section 3: Holdout evaluation ---
    holdout_results = run_holdout_evaluation(
        models, weights, test_texts, test_labels, test_emb
    )
    _print_holdout_results(holdout_results)

    # --- Section 4: Calibration ---
    calib_results = run_calibration_analysis(
        models, weights, test_texts, test_labels, test_emb
    )
    _print_calibration(calib_results)

    # --- Assemble report ---
    elapsed = round(time.time() - t_start, 1)
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "quick_mode": quick,
            "elapsed_seconds": elapsed,
            "training_set_size": len(train_texts),
            "holdout_set_size": len(test_labels),
            "ensemble_weights": weights,
        },
        "learning_curves": learning_curves,
        "cross_validation": cv_results,
        "holdout_evaluation": holdout_results,
        "confidence_calibration": calib_results,
    }

    os.makedirs(_MODELS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Validation report written to %s  (%.1fs total)", REPORT_PATH, elapsed)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate production classification models (scalability proof)."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip learning curves — run CV + holdout + calibration only.",
    )
    args = parser.parse_args()
    main(quick=args.quick)
