"""Tests for CSAT–sentiment agreement and trigger logic (ANTSE-326)."""

from collections import Counter
import pytest

from plugins.feedback.sentiment_agreement import (
    AGREEMENT_THRESHOLD,
    STABILITY_STD_THRESHOLD,
    TriggerOutcome,
    compute_agreement_rate,
    confusion_matrix_stats,
    evaluate_trigger,
    is_binary_agree,
    stratified_fold_test_indices,
    stratified_cv_fold_rates,
)


class TestBinaryAgreement:
    def test_satisfied_low_agrees(self):
        assert is_binary_agree(5, "low") is True
        assert is_binary_agree(4, "low") is True

    def test_distressed_high_agrees(self):
        assert is_binary_agree(1, "high") is True
        assert is_binary_agree(2, "high") is True

    def test_mismatch_disagrees(self):
        assert is_binary_agree(5, "high") is False
        assert is_binary_agree(1, "low") is False
        assert is_binary_agree(4, "medium") is False

    def test_compute_agreement_rate(self):
        csat = [5, 4, 1, 2, 5]
        intensity = ["low", "low", "high", "high", "high"]
        # 4/5 agree
        assert compute_agreement_rate(csat, intensity) == pytest.approx(0.8)


class TestStratifiedCvFolds:
    def test_five_folds_stratified_by_category(self):
        n = 50
        categories = ["Access"] * 20 + ["Config"] * 15 + ["HowTo"] * 15
        csat = [5, 1] * 25
        intensities = ["low", "high"] * 25
        assert len(csat) == n
        rates = stratified_cv_fold_rates(categories, csat, intensities)
        assert len(rates) == 5
        assert all(0.0 <= r <= 1.0 for r in rates)
        folds = stratified_fold_test_indices(categories)
        assert len(folds) == 5
        expected_fold_sizes = {"Access": 4, "Config": 3, "HowTo": 3}
        for fold in folds:
            fold_counts = Counter(categories[i] for i in fold)
            assert dict(fold_counts) == expected_fold_sizes

    def test_requires_minimum_samples(self):
        with pytest.raises(ValueError, match="at least"):
            stratified_cv_fold_rates(["A"], [5], ["low"], n_splits=5)


class TestTriggerDecision:
    def test_no_when_mean_meets_threshold(self):
        rates = [0.72, 0.71, 0.73, 0.70, 0.74]
        d = evaluate_trigger(rates, threshold=AGREEMENT_THRESHOLD, std_threshold=STABILITY_STD_THRESHOLD)
        assert d.outcome == TriggerOutcome.NO
        assert "TRIGGER: NO" in d.message
        assert d.mean >= AGREEMENT_THRESHOLD

    def test_deferred_when_unstable(self):
        rates = [0.50, 0.68, 0.55, 0.62, 0.48]
        d = evaluate_trigger(rates, threshold=AGREEMENT_THRESHOLD, std_threshold=STABILITY_STD_THRESHOLD)
        assert d.outcome == TriggerOutcome.DEFERRED
        assert "DEFERRED" in d.message
        assert d.std >= STABILITY_STD_THRESHOLD

    def test_yes_when_below_threshold_and_stable(self):
        rates = [0.62, 0.64, 0.63, 0.61, 0.65]
        d = evaluate_trigger(rates, threshold=AGREEMENT_THRESHOLD, std_threshold=STABILITY_STD_THRESHOLD)
        assert d.outcome == TriggerOutcome.YES
        assert "TRIGGER: YES" in d.message
        assert d.mean < AGREEMENT_THRESHOLD
        assert d.std < STABILITY_STD_THRESHOLD

    def test_empty_rates_deferred(self):
        d = evaluate_trigger([])
        assert d.outcome == TriggerOutcome.DEFERRED


class TestConfusionMatrix:
    def test_false_negative_rate_distress_missed(self):
        # Distress missed: predicted low when CSAT ≤ 2 → 2 of 3 distressed
        csat = [1, 2, 2, 5, 4]
        intensity = ["low", "low", "high", "low", "high"]
        stats = confusion_matrix_stats(csat, intensity)
        assert stats.fp == 2
        assert stats.false_negative_rate == pytest.approx(2 / 3)
