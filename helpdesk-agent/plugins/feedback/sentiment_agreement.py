"""CSAT–sentiment agreement metrics and trigger logic (ANTSE-326).

Pure functions used by analysis/sentiment_agreement_eval.py and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

JOIN_QUERY = """
    SELECT tc.ticket_key, tc.category, tc.sentiment_score, tc.sentiment_intensity,
           t.csat_score
    FROM ticket_classifications tc
    JOIN ticket_csat t ON t.ticket_key = tc.ticket_key
    WHERE t.csat_score IS NOT NULL
      AND tc.sentiment_score IS NOT NULL
      AND t.csat_score != 3
"""

AGREEMENT_THRESHOLD = 0.70
STABILITY_STD_THRESHOLD = 0.05
CV_SPLITS = 5
CV_RANDOM_STATE = 42


class TriggerOutcome(str, Enum):
    NO = "no"
    DEFERRED = "deferred"
    YES = "yes"


@dataclass(frozen=True)
class TriggerDecision:
    outcome: TriggerOutcome
    mean: float
    std: float
    message: str


@dataclass(frozen=True)
class ConfusionStats:
    tp: int
    fp: int
    tn: int
    fn: int
    false_negative_rate: float


def is_binary_agree(csat_score: int, sentiment_intensity: str) -> bool:
    """True when CSAT ground truth aligns with predicted intensity bucket."""
    if csat_score >= 4 and sentiment_intensity == "low":
        return True
    if csat_score <= 2 and sentiment_intensity == "high":
        return True
    return False


def compute_agreement_rate(
    csat_scores: Sequence[int],
    intensities: Sequence[str],
) -> float:
    """Fraction of rows where CSAT and sentiment_intensity agree (binary)."""
    if not csat_scores:
        return 0.0
    agrees = [
        is_binary_agree(int(c), str(i))
        for c, i in zip(csat_scores, intensities)
    ]
    return sum(agrees) / len(agrees)


def csat_ground_truth_positive(csat_score: int) -> bool:
    """CSAT satisfied (distress low): score >= 4."""
    return int(csat_score) >= 4


def intensity_predicted_positive(sentiment_intensity: str) -> bool:
    """Model predicts low distress."""
    return str(sentiment_intensity).lower() == "low"


def confusion_matrix_stats(
    csat_scores: Sequence[int],
    intensities: Sequence[str],
) -> ConfusionStats:
    """2×2 confusion: predicted low/high vs CSAT satisfied (≥4) / distressed (≤2)."""
    tp = fp = tn = fn = 0
    for csat, intensity in zip(csat_scores, intensities):
        actual_pos = csat_ground_truth_positive(int(csat))
        pred_pos = intensity_predicted_positive(intensity)
        if actual_pos and pred_pos:
            tp += 1
        elif actual_pos and not pred_pos:
            fn += 1
        elif not actual_pos and pred_pos:
            fp += 1
        else:
            tn += 1
    # Distress missed: predicted low when CSAT ≤ 2
    actual_distressed = sum(1 for c in csat_scores if int(c) <= 2)
    fn_distress = sum(
        1
        for c, i in zip(csat_scores, intensities)
        if int(c) <= 2 and intensity_predicted_positive(i)
    )
    fnr = fn_distress / actual_distressed if actual_distressed else 0.0
    return ConfusionStats(tp=tp, fp=fp, tn=tn, fn=fn, false_negative_rate=fnr)


def stratified_cv_fold_rates(
    categories: Sequence[str],
    csat_scores: Sequence[int],
    intensities: Sequence[str],
    *,
    n_splits: int = CV_SPLITS,
    random_state: int = CV_RANDOM_STATE,
) -> list[float]:
    """Per-fold binary agreement rates using stratified K-fold on category."""
    test_indices = stratified_fold_test_indices(
        categories,
        n_splits=n_splits,
        random_state=random_state,
    )
    csat_scores = list(csat_scores)
    intensities = list(intensities)
    rates: list[float] = []
    for test_idx in test_indices:
        test_csat = [csat_scores[i] for i in test_idx]
        test_int = [intensities[i] for i in test_idx]
        rates.append(compute_agreement_rate(test_csat, test_int))
    return rates


def stratified_fold_test_indices(
    categories: Sequence[str],
    *,
    n_splits: int = CV_SPLITS,
    random_state: int = CV_RANDOM_STATE,
) -> list[list[int]]:
    """Return held-out indices per fold for category-stratified CV."""
    categories = list(categories)
    if len(categories) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} samples for {n_splits}-fold CV, got {len(categories)}"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(len(categories))
    folds: list[list[int]] = []
    for _train_idx, test_idx in skf.split(indices, categories):
        folds.append(test_idx.tolist())
    return folds


def spearman_csat_sentiment(
    csat_scores: Sequence[int | float],
    sentiment_scores: Sequence[float],
) -> tuple[float, float]:
    """Spearman correlation; returns (rho, pvalue). Empty input → (nan, nan)."""
    if not csat_scores:
        return float("nan"), float("nan")
    rho, pval = spearmanr(list(csat_scores), list(sentiment_scores))
    return float(rho), float(pval)


def evaluate_trigger(
    fold_rates: Sequence[float],
    *,
    threshold: float = AGREEMENT_THRESHOLD,
    std_threshold: float = STABILITY_STD_THRESHOLD,
) -> TriggerDecision:
    """Map CV fold agreement rates to NO / DEFERRED / YES."""
    rates = list(fold_rates)
    if not rates:
        return TriggerDecision(
            outcome=TriggerOutcome.DEFERRED,
            mean=0.0,
            std=0.0,
            message="TRIGGER: DEFERRED — no evaluation folds",
        )

    mean = float(np.mean(rates))
    std = float(np.std(rates, ddof=0))

    if mean >= threshold:
        return TriggerDecision(
            outcome=TriggerOutcome.NO,
            mean=mean,
            std=std,
            message=f"TRIGGER: NO — pre-trained model meets threshold (mean={mean:.1%})",
        )
    if std >= std_threshold:
        return TriggerDecision(
            outcome=TriggerOutcome.DEFERRED,
            mean=mean,
            std=std,
            message=f"TRIGGER: DEFERRED — unstable (std={std:.1%}), collect more data",
        )
    return TriggerDecision(
        outcome=TriggerOutcome.YES,
        mean=mean,
        std=std,
        message="TRIGGER: YES — proceeding to adaptation strategies",
    )


def csat_to_training_label(csat_score: int) -> str:
    """Map CSAT 1–5 to negative/neutral/positive matching the 3-class base model.

    Score 1–2 → negative, 3 → neutral, 4–5 → positive.
    Aligns with cardiffnlp/twitter-roberta-base-sentiment-latest architecture.
    """
    score = int(csat_score)
    if score >= 4:
        return "positive"
    if score == 3:
        return "neutral"
    return "negative"
