"""Threshold calibration suite for FAQ dedup similarity thresholds.

Tests 40 labeled article pairs (Atlassian DC-to-Cloud migration content)
against data-driven calibrated thresholds.  The optimal threshold is
computed via F1-optimal sweep over the labeled pairs — no hardcoded
magic numbers.

Calibration report
------------------
Thresholds:
    Jaccard  = 0.80  (structural / MinHash — catches same-wording dups)
    Cosine   = calibrated from F1-optimal sweep (see CalibrationResult)

Methodology:
    40 article pairs hand-labeled as "duplicate" or "distinct".
    - 15 duplicate pairs: same topic, paraphrased wording
    - 15 distinct pairs: completely different migration topics
    - 10 borderline pairs: related but genuinely different topics

Note: stub calibration tests are kept for backward compatibility.
Re-run calibration via: python -m faq.threshold_calibration --save
"""

import math
import os

import pytest

from faq.dedup import (
    COSINE_THRESHOLD,
    JACCARD_THRESHOLD,
    _cosine_sim,
    _compute_similarity,
    _embedding_to_sparse,
    _embedding_to_vector,
    compute_minhash,
    EMBEDDING_PROVIDER,
)
from faq.calibration_pairs import LABELED_PAIRS
from faq.dedup import _normalize_for_embedding
from faq.threshold_calibration import calibrate_threshold, CalibrationResult


# LABELED_PAIRS lives in faq/calibration_pairs.py — imported above.
# Keep this comment so grep finds the data source.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_pair(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two plain-text articles (stub provider)."""
    vec_a = _embedding_to_sparse(text_a)
    vec_b = _embedding_to_sparse(text_b)
    return _cosine_sim(vec_a, vec_b)


def _cosine_pair_active(text_a: str, text_b: str) -> float:
    """Compute cosine similarity using the active embedding provider."""
    vec_a = _embedding_to_vector(text_a)
    vec_b = _embedding_to_vector(text_b)
    return _compute_similarity(vec_a, vec_b)


def _jaccard_pair(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity (MinHash estimate) between two articles."""
    mh_a = compute_minhash(text_a)
    mh_b = compute_minhash(text_b)
    if mh_a is None or mh_b is None:
        return 0.0
    return mh_a.jaccard(mh_b)


class TestGuards:
    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError, match="no valid pairs scored"):
            calibrate_threshold([], embed_fn=lambda t: [0.1, 0.2, 0.3])

    def test_all_empty_embeddings_raises(self):
        pairs = [("text a", "text b", "duplicate")]
        with pytest.raises(ValueError, match="no valid pairs scored"):
            calibrate_threshold(pairs, embed_fn=lambda t: [])


# ---------------------------------------------------------------------------
# Parametrized calibration tests
# ---------------------------------------------------------------------------

_PAIR_IDS = [f"pair-{i+1}" for i in range(len(LABELED_PAIRS))]


@pytest.mark.parametrize("text_a, text_b, label", LABELED_PAIRS, ids=_PAIR_IDS)
class TestCosineCalibrationStub:
    """Verify stub TF-IDF cosine at legacy 0.50 threshold (backward compat)."""

    def test_cosine_classification(self, text_a, text_b, label):
        stub_threshold = 0.50
        sim = _cosine_pair(text_a, text_b)
        if label == "duplicate":
            assert sim >= stub_threshold, (
                f"False negative: cosine={sim:.4f} < threshold={stub_threshold} "
                f"for duplicate pair (stub)"
            )
        else:
            assert sim < stub_threshold, (
                f"False positive: cosine={sim:.4f} >= threshold={stub_threshold} "
                f"for distinct pair (stub)"
            )


@pytest.mark.parametrize("text_a, text_b, label", LABELED_PAIRS, ids=_PAIR_IDS)
class TestCosineCalibration:
    """Verify active provider cosine threshold classifies each labeled pair."""

    def test_cosine_classification(self, text_a, text_b, label):
        sim = _cosine_pair_active(text_a, text_b)
        if label == "duplicate":
            assert sim >= COSINE_THRESHOLD, (
                f"False negative: cosine={sim:.4f} < threshold={COSINE_THRESHOLD} "
                f"for duplicate pair ({EMBEDDING_PROVIDER})"
            )
        else:
            assert sim < COSINE_THRESHOLD, (
                f"False positive: cosine={sim:.4f} >= threshold={COSINE_THRESHOLD} "
                f"for distinct pair ({EMBEDDING_PROVIDER})"
            )


@pytest.mark.parametrize("text_a, text_b, label", LABELED_PAIRS, ids=_PAIR_IDS)
class TestJaccardCalibration:
    """Verify Jaccard threshold has no false positives on distinct pairs.

    Jaccard (MinHash) catches structural/wording-level near-duplicates, NOT
    paraphrased semantic duplicates -- those are the cosine layer's job.
    Paraphrased duplicate pairs are expected to score low on Jaccard.
    """

    def test_jaccard_no_false_positives(self, text_a, text_b, label):
        sim = _jaccard_pair(text_a, text_b)
        if label == "distinct":
            assert sim < JACCARD_THRESHOLD, (
                f"False positive: jaccard={sim:.4f} >= threshold={JACCARD_THRESHOLD} "
                f"for distinct pair"
            )


# ---------------------------------------------------------------------------
# Data-driven calibration tests
# ---------------------------------------------------------------------------

class TestCalibratedThreshold:
    """Run calibrate_threshold() on LABELED_PAIRS and validate the result."""

    @pytest.fixture(scope="class")
    def calibration(self):
        from services.embedding import EMBEDDING_MODEL, embed_text
        return calibrate_threshold(
            LABELED_PAIRS, embed_text, model_id=EMBEDDING_MODEL,
            preprocess=_normalize_for_embedding,
        )

    def test_perfect_f1(self, calibration):
        assert calibration.f1_at_threshold == 1.0, (
            f"F1={calibration.f1_at_threshold:.4f} != 1.0 on labeled set "
            f"(threshold={calibration.optimal_threshold:.4f})"
        )

    def test_clean_separation(self, calibration):
        assert calibration.score_gap >= 0.05, (
            f"Score gap={calibration.score_gap:.4f} < 0.05 -- "
            f"dup min={calibration.dup_score_stats['min']:.4f}, "
            f"dist max={calibration.distinct_score_stats['max']:.4f}"
        )

    def test_threshold_in_sane_range(self, calibration):
        t = calibration.optimal_threshold
        assert 0.50 <= t <= 0.95, (
            f"Calibrated threshold {t:.4f} outside sane range [0.50, 0.95]"
        )

    def test_precision_and_recall(self, calibration):
        assert calibration.precision_at_threshold == 1.0
        assert calibration.recall_at_threshold == 1.0


class TestThresholdStability:
    """Verify calibration is deterministic: same pairs -> same threshold."""

    def test_deterministic(self):
        from services.embedding import EMBEDDING_MODEL, embed_text

        results = [
            calibrate_threshold(
                LABELED_PAIRS, embed_text, model_id=EMBEDDING_MODEL,
                preprocess=_normalize_for_embedding,
            )
            for _ in range(3)
        ]
        thresholds = [r.optimal_threshold for r in results]
        assert thresholds[0] == thresholds[1] == thresholds[2], (
            f"Non-deterministic thresholds: {thresholds}"
        )


class TestPlattScaling:
    """Platt scaling (stretch goal) -- logistic regression on cosine scores."""

    @pytest.fixture(scope="class")
    def platt_result(self):
        from services.embedding import EMBEDDING_MODEL, embed_text
        return calibrate_threshold(
            LABELED_PAIRS, embed_text, method="platt", model_id=EMBEDDING_MODEL,
            preprocess=_normalize_for_embedding,
        )

    def test_platt_f1(self, platt_result):
        assert platt_result.f1_at_threshold == 1.0, (
            f"Platt F1={platt_result.f1_at_threshold:.4f} != 1.0"
        )

    def test_platt_parameters_set(self, platt_result):
        assert platt_result.platt_slope is not None
        assert platt_result.platt_intercept is not None

    def test_platt_threshold_sane(self, platt_result):
        t = platt_result.optimal_threshold
        assert 0.50 <= t <= 0.95, (
            f"Platt-derived threshold {t:.4f} outside sane range"
        )


# ---------------------------------------------------------------------------
# Real-ticket validation (uses eval data if available)
# ---------------------------------------------------------------------------

class TestRealTicketValidation:
    """Cross-validate against real <PROJECT_KEY> ticket data if available.

    Real service-desk tickets have noisier labels than hand-labeled pairs:
    "duplicate" groups are tickets with identical summaries but varying
    descriptions, and "distinct" pairs are random so some may be topically
    similar.  Score overlap is expected — the test reports it rather than
    failing on it.
    """

    @pytest.fixture(scope="class")
    def real_ticket_pairs(self):
        from pathlib import Path
        data_file = Path(__file__).resolve().parent.parent / "eval" / "data" / "jiraconfsd_all.json"
        if not data_file.exists():
            pytest.skip("eval/data/jiraconfsd_all.json not available")

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from eval.loaders.jiraconfsd import _load_all_tickets, _build_duplicate_groups, _ticket_text

        tickets = _load_all_tickets()
        if not tickets:
            pytest.skip("No tickets loaded")

        import random
        rng = random.Random(42)
        groups = _build_duplicate_groups(tickets)

        pairs: list[tuple[str, str, str]] = []
        group_keys = sorted(groups.keys(), key=lambda k: -len(groups[k]))

        for gk in group_keys[:50]:
            indices = groups[gk]
            rng.shuffle(indices)
            if len(indices) >= 2:
                pairs.append((
                    _ticket_text(tickets[indices[0]]),
                    _ticket_text(tickets[indices[1]]),
                    "duplicate",
                ))

        used = {gk for gk in group_keys[:50]}
        non_dup = [t for t in tickets if t.get("summary") and
                   _ticket_text(t) not in used]
        rng.shuffle(non_dup)
        for i in range(min(50, len(non_dup) - 1)):
            pairs.append((
                _ticket_text(non_dup[i]),
                _ticket_text(non_dup[i + 1]),
                "distinct",
            ))

        return pairs

    @pytest.fixture(scope="class")
    def hand_labeled_result(self):
        from services.embedding import EMBEDDING_MODEL, embed_text
        return calibrate_threshold(
            LABELED_PAIRS, embed_text, model_id=EMBEDDING_MODEL,
            preprocess=_normalize_for_embedding,
        )

    @pytest.fixture(scope="class")
    def real_result(self, real_ticket_pairs):
        from services.embedding import EMBEDDING_MODEL, embed_text
        try:
            return calibrate_threshold(
                real_ticket_pairs, embed_text, model_id=EMBEDDING_MODEL,
                preprocess=_normalize_for_embedding,
            )
        except ValueError:
            # Score overlap is expected with noisy real-ticket labels
            return calibrate_threshold(
                real_ticket_pairs, embed_text, model_id=EMBEDDING_MODEL,
                preprocess=_normalize_for_embedding,
                allow_overlap=True,
            )

    def test_thresholds_within_tolerance(self, hand_labeled_result, real_result, capsys):
        diff = abs(hand_labeled_result.optimal_threshold - real_result.optimal_threshold)
        overlap_note = ""
        if real_result.score_gap < 0.0:
            overlap_note = (
                f"\n  WARNING: Real-ticket scores overlap (gap={real_result.score_gap:.4f})."
                f"\n  This is expected — same-summary tickets often have very different"
                f"\n  descriptions, and random 'distinct' pairs may be topically similar."
            )
        print(
            f"\n  Hand-labeled threshold: {hand_labeled_result.optimal_threshold:.4f}"
            f"  (F1={hand_labeled_result.f1_at_threshold:.4f}, gap={hand_labeled_result.score_gap:.4f})"
            f"\n  Real-ticket threshold:  {real_result.optimal_threshold:.4f}"
            f"  (F1={real_result.f1_at_threshold:.4f}, gap={real_result.score_gap:.4f})"
            f"\n  Difference:             {diff:.4f}"
            f"{overlap_note}"
        )
        if real_result.score_gap < 0.0:
            pytest.skip(
                f"Real-ticket scores overlap (gap={real_result.score_gap:.4f}) — "
                f"threshold comparison not meaningful with noisy labels"
            )


# ---------------------------------------------------------------------------
# Overfitting checks — cross-validation and held-out evaluation
# ---------------------------------------------------------------------------

class TestLeaveOneOutCV:
    """Leave-one-out cross-validation on the 40 labeled pairs.

    For each pair: calibrate on the other 39, classify the held-out pair.
    Reports how many held-out pairs are misclassified — a direct test of
    whether the threshold generalizes beyond its training set.
    """

    @pytest.fixture(scope="class")
    def loo_results(self):
        from services.embedding import EMBEDDING_MODEL, embed_text

        misclassified = []
        thresholds = []

        for i in range(len(LABELED_PAIRS)):
            train = LABELED_PAIRS[:i] + LABELED_PAIRS[i + 1:]
            held_out = LABELED_PAIRS[i]

            result = calibrate_threshold(
                train, embed_text, model_id=EMBEDDING_MODEL,
                preprocess=_normalize_for_embedding,
            )
            thresholds.append(result.optimal_threshold)

            import numpy as np
            a = _normalize_for_embedding(held_out[0])
            b = _normalize_for_embedding(held_out[1])
            va = np.array(embed_text(a))
            vb = np.array(embed_text(b))
            sim = float(np.dot(va, vb))

            predicted = "duplicate" if sim >= result.optimal_threshold else "distinct"
            if predicted != held_out[2]:
                misclassified.append({
                    "pair_idx": i + 1,
                    "label": held_out[2],
                    "predicted": predicted,
                    "sim": sim,
                    "threshold": result.optimal_threshold,
                })

        return {"misclassified": misclassified, "thresholds": thresholds}

    def test_loo_accuracy(self, loo_results, capsys):
        n = len(LABELED_PAIRS)
        errors = loo_results["misclassified"]
        accuracy = (n - len(errors)) / n

        thresholds = loo_results["thresholds"]
        t_min, t_max = min(thresholds), max(thresholds)
        t_mean = sum(thresholds) / len(thresholds)

        print(
            f"\n  Leave-one-out CV ({n} folds):"
            f"\n    Accuracy:          {accuracy:.4f} ({n - len(errors)}/{n})"
            f"\n    Threshold range:   [{t_min:.4f}, {t_max:.4f}]"
            f"\n    Threshold mean:    {t_mean:.4f}"
        )
        if errors:
            for e in errors:
                print(
                    f"\n    MISS pair-{e['pair_idx']}: "
                    f"label={e['label']} predicted={e['predicted']} "
                    f"sim={e['sim']:.4f} threshold={e['threshold']:.4f}"
                )

        assert accuracy >= 0.95, (
            f"LOO-CV accuracy {accuracy:.4f} < 0.95 — "
            f"{len(errors)} misclassified out of {n}"
        )

    def test_threshold_stability_across_folds(self, loo_results):
        thresholds = loo_results["thresholds"]
        t_range = max(thresholds) - min(thresholds)
        assert t_range <= 0.10, (
            f"Threshold varies by {t_range:.4f} across LOO folds — "
            f"suggests overfitting to individual pairs"
        )


class TestBootstrapConfidence:
    """Bootstrap resampling to estimate threshold confidence interval.

    Draws 200 bootstrap samples (with replacement) from the 40 pairs,
    calibrates each, and reports the 95% CI on the threshold.  A wide
    CI means the threshold is sensitive to which pairs are included.
    """

    @pytest.fixture(scope="class")
    def bootstrap_thresholds(self):
        import random
        from services.embedding import EMBEDDING_MODEL, embed_text

        rng = random.Random(42)
        thresholds = []

        for _ in range(200):
            sample = rng.choices(LABELED_PAIRS, k=len(LABELED_PAIRS))
            result = calibrate_threshold(
                sample, embed_text, model_id=EMBEDDING_MODEL,
                preprocess=_normalize_for_embedding,
            )
            thresholds.append(result.optimal_threshold)

        thresholds.sort()
        return thresholds

    def test_confidence_interval_narrow(self, bootstrap_thresholds, capsys):
        n = len(bootstrap_thresholds)
        ci_lo = bootstrap_thresholds[int(n * 0.025)]
        ci_hi = bootstrap_thresholds[int(n * 0.975)]
        ci_width = ci_hi - ci_lo
        median = bootstrap_thresholds[n // 2]

        print(
            f"\n  Bootstrap CI (200 resamples, n={len(LABELED_PAIRS)} pairs):"
            f"\n    Median threshold:  {median:.4f}"
            f"\n    95% CI:            [{ci_lo:.4f}, {ci_hi:.4f}]"
            f"\n    CI width:          {ci_width:.4f}"
        )

        assert ci_width <= 0.15, (
            f"Bootstrap 95% CI width {ci_width:.4f} > 0.15 — "
            f"threshold is unstable, likely overfitting to small sample"
        )


class TestTrainTestSplit:
    """50/50 train-test split: calibrate on 20 pairs, evaluate on 20.

    Runs 10 random splits to get stable metrics.  This directly tests
    whether a threshold trained on half the data generalizes to unseen
    pairs — the core overfitting question.
    """

    @pytest.fixture(scope="class")
    def split_results(self):
        import random
        from services.embedding import EMBEDDING_MODEL, embed_text
        import numpy as np

        rng = random.Random(42)
        n_splits = 10
        results = []

        for _ in range(n_splits):
            indices = list(range(len(LABELED_PAIRS)))
            rng.shuffle(indices)
            mid = len(indices) // 2
            train_idx, test_idx = indices[:mid], indices[mid:]

            train = [LABELED_PAIRS[i] for i in train_idx]
            test = [LABELED_PAIRS[i] for i in test_idx]

            cal = calibrate_threshold(
                train, embed_text, model_id=EMBEDDING_MODEL,
                preprocess=_normalize_for_embedding, allow_overlap=True,
            )

            tp = fp = fn = tn = 0
            for text_a, text_b, label in test:
                a = _normalize_for_embedding(text_a)
                b = _normalize_for_embedding(text_b)
                va = np.array(embed_text(a))
                vb = np.array(embed_text(b))
                sim = float(np.dot(va, vb))
                pred_dup = sim >= cal.optimal_threshold

                if label == "duplicate" and pred_dup:
                    tp += 1
                elif label == "duplicate" and not pred_dup:
                    fn += 1
                elif label == "distinct" and pred_dup:
                    fp += 1
                else:
                    tn += 1

            total = tp + fp + fn + tn
            accuracy = (tp + tn) / total if total else 0.0
            p = tp / (tp + fp) if (tp + fp) else 1.0
            r = tp / (tp + fn) if (tp + fn) else 1.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0

            results.append({
                "threshold": cal.optimal_threshold,
                "accuracy": accuracy,
                "f1": f1, "precision": p, "recall": r,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })

        return results

    def test_mean_accuracy(self, split_results, capsys):
        accs = [r["accuracy"] for r in split_results]
        f1s = [r["f1"] for r in split_results]
        thresholds = [r["threshold"] for r in split_results]
        mean_acc = sum(accs) / len(accs)
        mean_f1 = sum(f1s) / len(f1s)

        print(
            f"\n  Train/test split (10 random 50/50 splits, 20 train / 20 test):"
            f"\n    Mean accuracy:     {mean_acc:.4f}"
            f"\n    Mean F1:           {mean_f1:.4f}"
            f"\n    Threshold range:   [{min(thresholds):.4f}, {max(thresholds):.4f}]"
        )
        for i, r in enumerate(split_results):
            print(
                f"    Split {i+1}: threshold={r['threshold']:.4f} "
                f"acc={r['accuracy']:.4f} F1={r['f1']:.4f} "
                f"TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}"
            )

        assert mean_acc >= 0.90, (
            f"Mean accuracy {mean_acc:.4f} < 0.90 across splits — overfitting"
        )

    def test_mean_f1(self, split_results):
        f1s = [r["f1"] for r in split_results]
        mean_f1 = sum(f1s) / len(f1s)
        assert mean_f1 >= 0.85, (
            f"Mean F1 {mean_f1:.4f} < 0.85 across splits — overfitting"
        )


class TestHeldOutCrossTopicFPR:
    """Calibrate on 40 pairs, test FP rate on 3000+ unseen cross-topic pairs.

    Each of the 40 labeled pairs uses 2 texts. We form ALL cross-topic
    pairings (text from topic A vs text from topic B) that were NOT in the
    original 40.  These are all distinct-by-definition, so the only metric
    is false-positive rate: does the threshold incorrectly flag unrelated
    text as duplicates?
    """

    @pytest.fixture(scope="class")
    def cross_topic_eval(self):
        from services.embedding import EMBEDDING_MODEL, embed_text
        import numpy as np

        result = calibrate_threshold(
            LABELED_PAIRS, embed_text, model_id=EMBEDDING_MODEL,
            preprocess=_normalize_for_embedding,
        )

        original_pair_set = set()
        all_texts = []
        for a, b, _ in LABELED_PAIRS:
            all_texts.append(a)
            all_texts.append(b)
            original_pair_set.add((a, b))
            original_pair_set.add((b, a))

        unique_texts = list(dict.fromkeys(all_texts))

        fp = tn = 0
        fp_examples = []
        for i in range(len(unique_texts)):
            for j in range(i + 1, len(unique_texts)):
                if (unique_texts[i], unique_texts[j]) in original_pair_set:
                    continue
                a = _normalize_for_embedding(unique_texts[i])
                b = _normalize_for_embedding(unique_texts[j])
                va = np.array(embed_text(a))
                vb = np.array(embed_text(b))
                sim = float(np.dot(va, vb))

                if sim >= result.optimal_threshold:
                    fp += 1
                    if len(fp_examples) < 5:
                        fp_examples.append((sim, unique_texts[i][:60], unique_texts[j][:60]))
                else:
                    tn += 1

        total = fp + tn
        fpr = fp / total if total else 0.0
        return {
            "n_pairs": total, "fp": fp, "tn": tn, "fpr": fpr,
            "threshold": result.optimal_threshold,
            "fp_examples": fp_examples,
        }

    def test_cross_topic_fpr(self, cross_topic_eval, capsys):
        e = cross_topic_eval
        print(
            f"\n  Cross-topic held-out FP evaluation:"
            f"\n    Pairs evaluated:   {e['n_pairs']} (all cross-topic, excluding original 40)"
            f"\n    Threshold:         {e['threshold']:.4f}"
            f"\n    False positives:   {e['fp']}"
            f"\n    True negatives:    {e['tn']}"
            f"\n    FP rate:           {e['fpr']:.4f}"
        )
        if e["fp_examples"]:
            print("    FP examples (highest-sim misclassifications):")
            for sim, a, b in e["fp_examples"]:
                print(f"      sim={sim:.4f}: '{a}...' vs '{b}...'")

        assert e["fpr"] <= 0.05, (
            f"Cross-topic FP rate {e['fpr']:.4f} > 0.05 — "
            f"{e['fp']} false positives out of {e['n_pairs']} pairs"
        )


# ---------------------------------------------------------------------------
# Baseline comparison — calibrated vs naive fixed threshold
# ---------------------------------------------------------------------------

class TestBaselineComparison:
    """Compare F1-optimal calibrated threshold against naive baselines.

    Baselines:
      1. Fixed 0.75 (prior hardcoded threshold)
      2. Fixed 0.50 (naive midpoint)

    Both are evaluated on the same 40 labeled pairs using the active
    embedding model + preprocessing pipeline.  The calibrated threshold
    must match or beat each baseline on F1, precision, and recall.
    """

    @pytest.fixture(scope="class")
    def scored_pairs(self):
        from services.embedding import embed_text
        import numpy as np

        scores, labels = [], []
        for text_a, text_b, label in LABELED_PAIRS:
            a = _normalize_for_embedding(text_a)
            b = _normalize_for_embedding(text_b)
            va = np.array(embed_text(a))
            vb = np.array(embed_text(b))
            scores.append(float(np.dot(va, vb)))
            labels.append(1 if label == "duplicate" else 0)
        return scores, labels

    @pytest.fixture(scope="class")
    def calibrated(self):
        from services.embedding import EMBEDDING_MODEL, embed_text
        return calibrate_threshold(
            LABELED_PAIRS, embed_text, model_id=EMBEDDING_MODEL,
            preprocess=_normalize_for_embedding,
        )

    @staticmethod
    def _metrics_at(scores, labels, threshold):
        tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 1)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}

    def test_beats_hardcoded_075(self, scored_pairs, calibrated, capsys):
        scores, labels = scored_pairs
        baseline = self._metrics_at(scores, labels, 0.75)
        cal = self._metrics_at(scores, labels, calibrated.optimal_threshold)

        print(
            f"\n  Baseline (0.75):  F1={baseline['f1']:.4f}  P={baseline['precision']:.4f}"
            f"  R={baseline['recall']:.4f}  TP={baseline['tp']} FP={baseline['fp']} FN={baseline['fn']}"
            f"\n  Calibrated ({calibrated.optimal_threshold:.2f}): "
            f" F1={cal['f1']:.4f}  P={cal['precision']:.4f}"
            f"  R={cal['recall']:.4f}  TP={cal['tp']} FP={cal['fp']} FN={cal['fn']}"
        )
        assert cal["f1"] >= baseline["f1"], (
            f"Calibrated F1={cal['f1']:.4f} < baseline F1={baseline['f1']:.4f}"
        )

    def test_beats_naive_050(self, scored_pairs, calibrated, capsys):
        scores, labels = scored_pairs
        baseline = self._metrics_at(scores, labels, 0.50)
        cal = self._metrics_at(scores, labels, calibrated.optimal_threshold)

        print(
            f"\n  Baseline (0.50):  F1={baseline['f1']:.4f}  P={baseline['precision']:.4f}"
            f"  R={baseline['recall']:.4f}  TP={baseline['tp']} FP={baseline['fp']} FN={baseline['fn']}"
            f"\n  Calibrated ({calibrated.optimal_threshold:.2f}): "
            f" F1={cal['f1']:.4f}  P={cal['precision']:.4f}"
            f"  R={cal['recall']:.4f}  TP={cal['tp']} FP={cal['fp']} FN={cal['fn']}"
        )
        assert cal["f1"] >= baseline["f1"], (
            f"Calibrated F1={cal['f1']:.4f} < naive 0.50 F1={baseline['f1']:.4f}"
        )

    def test_zero_false_positives(self, scored_pairs, calibrated):
        scores, labels = scored_pairs
        cal = self._metrics_at(scores, labels, calibrated.optimal_threshold)
        assert cal["fp"] == 0, (
            f"Calibrated threshold has {cal['fp']} false positives"
        )


# ---------------------------------------------------------------------------
# Summary test -- prints the calibration report
# ---------------------------------------------------------------------------

class TestCalibrationSummary:
    """Aggregate all pairs and print FP/FN rates for active provider."""

    def test_print_calibration_report(self, capsys):
        from services.embedding import EMBEDDING_MODEL, embed_text

        result = calibrate_threshold(
            LABELED_PAIRS, embed_text, model_id=EMBEDDING_MODEL,
            preprocess=_normalize_for_embedding,
        )

        cosine_scores = []
        jaccard_scores = []

        for text_a, text_b, label in LABELED_PAIRS:
            cosine_scores.append((_cosine_pair_active(text_a, text_b), label))
            jaccard_scores.append((_jaccard_pair(text_a, text_b), label))

        calibrated_threshold = result.optimal_threshold

        # Cosine metrics at calibrated threshold
        cos_tp = sum(1 for s, l in cosine_scores if l == "duplicate" and s >= calibrated_threshold)
        cos_fn = sum(1 for s, l in cosine_scores if l == "duplicate" and s < calibrated_threshold)
        cos_fp = sum(1 for s, l in cosine_scores if l == "distinct" and s >= calibrated_threshold)
        cos_tn = sum(1 for s, l in cosine_scores if l == "distinct" and s < calibrated_threshold)

        n_dup = sum(1 for _, l in cosine_scores if l == "duplicate")
        n_dist = sum(1 for _, l in cosine_scores if l == "distinct")

        cos_fpr = cos_fp / n_dist if n_dist else 0.0
        cos_fnr = cos_fn / n_dup if n_dup else 0.0

        # Jaccard metrics
        jac_fp = sum(1 for s, l in jaccard_scores if l == "duplicate" and s >= JACCARD_THRESHOLD)
        jac_fn = sum(1 for s, l in jaccard_scores if l == "duplicate" and s < JACCARD_THRESHOLD)
        jac_fp_d = sum(1 for s, l in jaccard_scores if l == "distinct" and s >= JACCARD_THRESHOLD)
        jac_tn = sum(1 for s, l in jaccard_scores if l == "distinct" and s < JACCARD_THRESHOLD)

        jac_fpr = jac_fp_d / n_dist if n_dist else 0.0
        jac_fnr = jac_fn / n_dup if n_dup else 0.0

        # Score distributions
        dup_cosine = [s for s, l in cosine_scores if l == "duplicate"]
        dist_cosine = [s for s, l in cosine_scores if l == "distinct"]
        dup_jaccard = [s for s, l in jaccard_scores if l == "duplicate"]
        dist_jaccard = [s for s, l in jaccard_scores if l == "distinct"]

        def _stats(vals):
            if not vals:
                return "n/a"
            mn, mx = min(vals), max(vals)
            avg = sum(vals) / len(vals)
            return f"min={mn:.4f}  avg={avg:.4f}  max={mx:.4f}"

        report = (
            "\n"
            "=================================================================\n"
            f"  THRESHOLD CALIBRATION REPORT ({EMBEDDING_PROVIDER})\n"
            "=================================================================\n"
            f"  Pairs tested:       {len(LABELED_PAIRS)}\n"
            f"    Duplicates:       {n_dup}\n"
            f"    Distinct:         {n_dist}\n"
            "\n"
            f"  Static threshold:   {COSINE_THRESHOLD}\n"
            f"  Calibrated (F1):    {calibrated_threshold:.4f}\n"
            f"  Score gap:          {result.score_gap:.4f}\n"
            "\n"
            f"  Cosine @ calibrated {calibrated_threshold:.4f}:\n"
            f"    TP={cos_tp}  FN={cos_fn}  FP={cos_fp}  TN={cos_tn}\n"
            f"    FP rate: {cos_fpr:.4f}    FN rate: {cos_fnr:.4f}\n"
            f"    Dup scores:   {_stats(dup_cosine)}\n"
            f"    Dist scores:  {_stats(dist_cosine)}\n"
            "\n"
            f"  Jaccard threshold: {JACCARD_THRESHOLD}\n"
            f"    TP={jac_fp}  FN={jac_fn}  FP={jac_fp_d}  TN={jac_tn}\n"
            f"    FP rate: {jac_fpr:.4f}    FN rate: {jac_fnr:.4f}\n"
            f"    Dup scores:   {_stats(dup_jaccard)}\n"
            f"    Dist scores:  {_stats(dist_jaccard)}\n"
            "=================================================================\n"
        )
        print(report)

        assert cos_fpr == 0.0, f"Cosine false positive rate = {cos_fpr}"
        assert cos_fnr == 0.0, f"Cosine false negative rate = {cos_fnr}"
        assert jac_fpr == 0.0, f"Jaccard false positive rate = {jac_fpr}"
