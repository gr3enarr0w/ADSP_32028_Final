"""Tests for scripts/retrain.py — ANTSE-305 automated retraining pipeline.

Covers:
- Minimum data guard: categories below threshold are excluded
- Promotion gate: promotes when new model is better or within tolerance
- Promotion gate: skips when new model is worse beyond tolerance
- Holdout integrity failure aborts the pipeline with a RuntimeError
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lr_pipeline(classes=None):
    """Return a minimal mock sklearn Pipeline with predict_proba."""
    if classes is None:
        classes = ["Access", "Configuration", "Other"]
    mock = MagicMock()
    mock.classes_ = classes
    n = len(classes)
    mock.predict_proba.return_value = np.ones((5, n)) / n
    return mock


def _make_distilbert_lr(classes=None):
    """Return a minimal mock LogisticRegression with predict_proba."""
    if classes is None:
        classes = ["Access", "Configuration", "Other"]
    mock = MagicMock()
    mock.classes_ = classes
    n = len(classes)
    mock.predict_proba.return_value = np.ones((5, n)) / n
    return mock


# ---------------------------------------------------------------------------
# Test 1: Minimum data guard excludes small categories
# ---------------------------------------------------------------------------

class TestMinimumDataGuard:
    """check_minimum_data() must exclude categories below the threshold."""

    def test_minimum_data_guard_excludes_small_categories(self):
        """Categories with fewer than min_samples training tickets are excluded."""
        # Simulate DB rows: category counts in the training set
        # "Performance" and "Other" have only 2 and 3 samples — below threshold of 10
        fake_all_rows = []
        # Access: 20 training tickets
        for i in range(20):
            fake_all_rows.append({"ticket_key": f"ACC-{i}", "category": "Access"})
        # Configuration: 15 training tickets
        for i in range(15):
            fake_all_rows.append({"ticket_key": f"CONF-{i}", "category": "Configuration"})
        # Performance: 2 training tickets (should be excluded at threshold=10)
        for i in range(2):
            fake_all_rows.append({"ticket_key": f"PERF-{i}", "category": "Performance"})
        # Other: 3 training tickets (should be excluded at threshold=10)
        for i in range(3):
            fake_all_rows.append({"ticket_key": f"OTH-{i}", "category": "Other"})

        # Holdout set does not contain any of these keys
        fake_holdout = set()

        fake_conn = MagicMock()
        fake_conn.execute.return_value.fetchall.return_value = [
            {"ticket_key": r["ticket_key"], "category": r["category"]}
            for r in fake_all_rows
        ]

        with patch("analysis.holdout.holdout_ticket_keys", return_value=fake_holdout), \
             patch("db.get_db", return_value=fake_conn):
            from scripts.retrain import check_minimum_data
            included, excluded = check_minimum_data(min_samples=10)

        assert "Access" in included
        assert "Configuration" in included
        assert "Performance" in excluded, "Performance (2 samples) must be excluded"
        assert "Other" in excluded, "Other (3 samples) must be excluded"

    def test_minimum_data_guard_all_excluded_raises(self):
        """RuntimeError when all categories fall below the threshold."""
        fake_all_rows = [
            {"ticket_key": "ACC-0", "category": "Access"},  # only 1 sample
        ]
        fake_holdout = set()

        fake_conn = MagicMock()
        fake_conn.execute.return_value.fetchall.return_value = [
            {"ticket_key": r["ticket_key"], "category": r["category"]}
            for r in fake_all_rows
        ]

        with patch("analysis.holdout.holdout_ticket_keys", return_value=fake_holdout), \
             patch("db.get_db", return_value=fake_conn):
            from scripts.retrain import check_minimum_data
            with pytest.raises(RuntimeError, match="All categories excluded"):
                check_minimum_data(min_samples=10)


# ---------------------------------------------------------------------------
# Test 2: Promotion gate promotes when new model is better
# ---------------------------------------------------------------------------

class TestPromotionGate:
    """maybe_promote() must write models when the gate passes."""

    def _run_maybe_promote(self, new_f1: float, prev_f1: float, tolerance: float = 0.01):
        """Helper that runs maybe_promote with mocked I/O and returns (promoted, written_files)."""
        lr_pipeline = _make_lr_pipeline()
        gb_pipeline = _make_lr_pipeline()
        distilbert_lr = _make_distilbert_lr()
        cache = {
            "test_keys": [],
            "test_labels": [],
            "test_embeddings": np.array([]),
        }
        new_metrics = {"macro_f1": new_f1, "accuracy": 0.75, "weighted_f1": 0.72}
        prev_metrics = {"macro_f1": prev_f1}
        new_weights = {"w_lr_norm": 0.10, "w_gb_norm": 0.12, "w_bert_norm": 0.78}

        written_files = []

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)

            def fake_open(path, mode="r", *args, **kwargs):
                """Track written paths."""
                import builtins
                if "w" in mode or "b" in mode and "w" in mode:
                    written_files.append(str(path))
                return builtins.open(path, mode, *args, **kwargs)

            # Patch _MODELS_DIR and the production metrics path
            with patch("scripts.retrain._MODELS_DIR", models_dir), \
                 patch("scripts.retrain._PRODUCTION_METRICS_PATH", models_dir / "production_metrics.json"), \
                 patch("pickle.dump") as mock_pickle_dump:

                from scripts.retrain import maybe_promote
                promoted = maybe_promote(
                    lr_pipeline, gb_pipeline, distilbert_lr, cache,
                    new_metrics, prev_metrics, new_weights,
                    tolerance=tolerance,
                )

        return promoted, mock_pickle_dump.call_count

    def test_promotion_gate_promotes_when_better(self):
        """New model with higher Macro-F1 must be promoted."""
        promoted, pickle_calls = self._run_maybe_promote(
            new_f1=0.60, prev_f1=0.5742
        )
        assert promoted is True, "Should promote when new_f1 > prev_f1"
        # Should have called pickle.dump 3 times (lr, gb, distilbert)
        assert pickle_calls == 3, f"Expected 3 pickle.dump calls, got {pickle_calls}"

    def test_promotion_gate_promotes_when_within_tolerance(self):
        """New model within 1pp below prev should still be promoted."""
        # new=0.5660 vs prev=0.5742 → delta=-0.0082, within 0.01 tolerance
        promoted, pickle_calls = self._run_maybe_promote(
            new_f1=0.5660, prev_f1=0.5742, tolerance=0.01
        )
        assert promoted is True, "Should promote when within tolerance band"
        assert pickle_calls == 3

    def test_promotion_gate_skips_when_worse_beyond_tolerance(self):
        """New model more than 1pp below prev must NOT be promoted."""
        # new=0.5600 vs prev=0.5742 → delta=-0.0142, exceeds 0.01 tolerance
        promoted, pickle_calls = self._run_maybe_promote(
            new_f1=0.5600, prev_f1=0.5742, tolerance=0.01
        )
        assert promoted is False, "Should skip when new_f1 < prev_f1 - tolerance"
        assert pickle_calls == 0, "Should not write any models when skipped"

    def test_promotion_gate_at_exact_threshold(self):
        """New model exactly at the tolerance boundary must be promoted."""
        # new=0.5642 vs prev=0.5742, tolerance=0.01 → threshold=0.5642 → new >= threshold
        promoted, pickle_calls = self._run_maybe_promote(
            new_f1=0.5642, prev_f1=0.5742, tolerance=0.01
        )
        assert promoted is True, "Should promote when exactly at threshold"


# ---------------------------------------------------------------------------
# Test 3: Holdout integrity failure aborts
# ---------------------------------------------------------------------------

class TestHoldoutIntegrity:
    """evaluate_on_holdout() must abort when verify_holdout_set() raises."""

    def test_holdout_integrity_failure_aborts(self):
        """RuntimeError from verify_holdout_set propagates and aborts evaluation."""
        lr_pipeline = _make_lr_pipeline()
        gb_pipeline = _make_lr_pipeline()
        distilbert_lr = _make_distilbert_lr()
        cache = {
            "test_keys": ["T-1", "T-2"],
            "test_labels": ["Access", "Configuration"],
            "test_embeddings": np.zeros((2, 384)),
        }

        with patch(
            "analysis.holdout.verify_holdout_set",
            side_effect=RuntimeError("Holdout set integrity check FAILED"),
        ):
            from scripts.retrain import evaluate_on_holdout
            with pytest.raises(RuntimeError, match="Holdout set integrity check FAILED"):
                evaluate_on_holdout(
                    lr_pipeline, gb_pipeline, distilbert_lr, cache,
                    w_lr=0.10, w_gb=0.12, w_bert=0.78,
                )

    def test_holdout_integrity_success_proceeds(self):
        """When verify_holdout_set passes, evaluation runs to completion."""
        from analysis.tfidf_lr import CATEGORIES

        n_holdout = 5
        n_cats = len(CATEGORIES)

        lr_pipeline = _make_lr_pipeline(classes=CATEGORIES)
        lr_pipeline.predict_proba.return_value = np.ones((n_holdout, n_cats)) / n_cats

        gb_pipeline = _make_lr_pipeline(classes=CATEGORIES)
        gb_pipeline.predict_proba.return_value = np.ones((n_holdout, n_cats)) / n_cats

        distilbert_lr = _make_distilbert_lr(classes=CATEGORIES)
        distilbert_lr.predict_proba.return_value = np.ones((n_holdout, n_cats)) / n_cats

        test_keys = [f"T-{i}" for i in range(n_holdout)]
        test_labels = ["Access", "Configuration", "Other", "Access", "Configuration"]
        test_embeddings = np.zeros((n_holdout, 384))

        cache = {
            "test_keys": test_keys,
            "test_labels": test_labels,
            "test_embeddings": test_embeddings,
        }

        with patch("analysis.holdout.verify_holdout_set", return_value=True), \
             patch("analysis.holdout.holdout_ticket_keys", return_value=set(test_keys)), \
             patch("analysis.tfidf_lr._load_dataset", return_value=(
                 ["text"] * n_holdout,
                 test_labels,
                 test_keys,
             )), \
             patch("analysis.tfidf_lr._split_train_holdout", return_value=(
                 [], [], ["text"] * n_holdout, test_labels,
             )):

            from scripts.retrain import evaluate_on_holdout
            result = evaluate_on_holdout(
                lr_pipeline, gb_pipeline, distilbert_lr, cache,
                w_lr=0.10, w_gb=0.12, w_bert=0.78,
            )

        assert "macro_f1" in result
        assert "accuracy" in result
        assert result["holdout_count"] == n_holdout
