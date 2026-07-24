"""Tests for the embedding-based feedback scorer (ANTSE-295).

Validates that _score_similarity() uses cosine similarity via the shared
embedding service and produces reasonable scores across edge cases and
the 30-pair calibration set.
"""

import json
import pathlib
import difflib

import pytest
from unittest.mock import patch

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture()
def _mock_embed():
    """Mock embed_batch to return deterministic unit vectors."""
    import numpy as np

    def _fake_embed(texts):
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % 2**31)
            v = rng.randn(384).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v.tolist())
        return vecs

    with patch("services.embedding.embed_batch", side_effect=_fake_embed):
        yield


class TestScoreSimilarityEdgeCases:
    """Edge-case behaviour for the embedding-based scorer."""

    def test_empty_draft_returns_ignored(self, _mock_embed):
        from faq.auto_responder import _score_similarity

        score, cat = _score_similarity("", "some actual response")
        assert score == 0.0
        assert cat == "ignored"

    def test_empty_actual_returns_ignored(self, _mock_embed):
        from faq.auto_responder import _score_similarity

        score, cat = _score_similarity("draft text", "")
        assert score == 0.0
        assert cat == "ignored"

    def test_both_empty_returns_ignored(self, _mock_embed):
        from faq.auto_responder import _score_similarity

        score, cat = _score_similarity("", "")
        assert score == 0.0
        assert cat == "ignored"

    def test_whitespace_only_returns_ignored(self, _mock_embed):
        from faq.auto_responder import _score_similarity

        score, cat = _score_similarity("   ", "\n\t")
        assert score == 0.0
        assert cat == "ignored"

    def test_identical_strings_high_score(self):
        """Identical text should produce score ~1.0 (same embedding)."""
        with patch("services.embedding.embed_batch") as mock_eb:
            vec = [0.0] * 384
            vec[0] = 1.0
            mock_eb.return_value = [vec, vec]

            from faq.auto_responder import _score_similarity
            score, cat = _score_similarity("hello", "hello")
            assert score >= 0.99
            assert cat == "as_is"

    def test_orthogonal_vectors_low_score(self):
        """Orthogonal embeddings should produce score ~0.0."""
        with patch("services.embedding.embed_batch") as mock_eb:
            v1 = [0.0] * 384
            v1[0] = 1.0
            v2 = [0.0] * 384
            v2[1] = 1.0
            mock_eb.return_value = [v1, v2]

            from faq.auto_responder import _score_similarity
            score, cat = _score_similarity("hello", "goodbye")
            assert score < 0.01
            assert cat == "ignored"

    def test_score_clamped_to_0_1(self):
        """Score stays within [0, 1] even for adversarial embeddings."""
        with patch("services.embedding.embed_batch") as mock_eb:
            v = [1.0] * 384  # not unit-normalized
            mock_eb.return_value = [v, v]

            from faq.auto_responder import _score_similarity
            score, _ = _score_similarity("a", "b")
            assert 0.0 <= score <= 1.0

    def test_returns_tuple_of_float_and_str(self, _mock_embed):
        from faq.auto_responder import _score_similarity

        result = _score_similarity("test draft", "test actual")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)

    def test_all_categories_reachable(self):
        """Each category corresponds to a threshold band."""
        from faq.auto_responder import _score_similarity

        thresholds = [
            (0.95, "as_is"),
            (0.80, "lightly_edited"),
            (0.50, "heavily_rewritten"),
            (0.10, "ignored"),
        ]
        for target_score, expected_cat in thresholds:
            with patch("services.embedding.embed_batch") as mock_eb:
                import math
                v1 = [0.0] * 384
                v2 = [0.0] * 384
                v1[0] = 1.0
                v2[0] = target_score
                v2[1] = math.sqrt(1.0 - target_score**2)
                mock_eb.return_value = [v1, v2]

                score, cat = _score_similarity("a", "b")
                assert cat == expected_cat, f"score={score}, expected {expected_cat}, got {cat}"


class TestCalibrationSet:
    """Run the 30-pair calibration set and document difflib vs cosine scores."""

    @pytest.fixture()
    def pairs(self):
        with open(FIXTURES / "calibration_pairs.json") as f:
            return json.load(f)["pairs"]

    def test_calibration_set_has_30_pairs(self, pairs):
        assert len(pairs) == 30

    def test_difflib_vs_cosine_scores(self, pairs):
        """Compute both scores for each pair and print comparison table.

        This test always passes — its purpose is to document the score
        mapping between the old and new scorers.
        """
        from services.embedding import embed_batch

        results = []
        non_empty = [(p, p["draft"], p["actual"]) for p in pairs if p["actual"].strip()]

        if non_empty:
            drafts = [d for _, d, _ in non_empty]
            actuals = [a for _, _, a in non_empty]
            all_texts = drafts + actuals
            all_vecs = embed_batch(all_texts)

            for i, (pair, draft, actual) in enumerate(non_empty):
                difflib_score = difflib.SequenceMatcher(None, draft, actual).ratio()
                v_draft = all_vecs[i]
                v_actual = all_vecs[len(non_empty) + i]
                cosine_score = sum(a * b for a, b in zip(v_draft, v_actual))
                results.append({
                    "id": pair["id"],
                    "label": pair["label"],
                    "difflib": round(difflib_score, 4),
                    "cosine": round(cosine_score, 4),
                    "delta": round(cosine_score - difflib_score, 4),
                })

        for p in pairs:
            if not p["actual"].strip():
                results.append({
                    "id": p["id"],
                    "label": p["label"],
                    "difflib": 0.0,
                    "cosine": 0.0,
                    "delta": 0.0,
                })

        results.sort(key=lambda r: r["id"])

        print("\n=== Calibration: difflib vs cosine similarity ===")
        print(f"{'ID':>3} {'Label':<40} {'difflib':>8} {'cosine':>8} {'delta':>8}")
        print("-" * 72)
        for r in results:
            print(f"{r['id']:>3} {r['label']:<40} {r['difflib']:>8.4f} {r['cosine']:>8.4f} {r['delta']:>8.4f}")

        out_path = FIXTURES / "calibration_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {out_path}")

    def test_identical_pair_highest_cosine(self, pairs):
        """The identical pair (id=1) should have the highest cosine score."""
        from services.embedding import embed_batch

        identical = next(p for p in pairs if p["label"] == "identical")
        vecs = embed_batch([identical["draft"], identical["actual"]])
        score = sum(a * b for a, b in zip(vecs[0], vecs[1]))
        assert score > 0.99

    def test_unrelated_pairs_low_cosine(self, pairs):
        """Completely unrelated pairs should score below 0.5."""
        from services.embedding import embed_batch

        unrelated = [p for p in pairs if p["label"] in ("completely_different_topic", "completely_unrelated")]
        for pair in unrelated:
            vecs = embed_batch([pair["draft"], pair["actual"]])
            score = sum(a * b for a, b in zip(vecs[0], vecs[1]))
            assert score < 0.5, f"Pair {pair['id']} ({pair['label']}) scored {score:.4f}"


class TestDifflibRemoved:
    """Verify difflib is no longer used for feedback scoring."""

    def test_no_difflib_import(self):
        import inspect
        import faq.auto_responder as mod

        source = inspect.getsource(mod._score_similarity)
        assert "difflib" not in source
        assert "SequenceMatcher" not in source

    def test_module_does_not_import_difflib(self):
        import faq.auto_responder as mod
        source_file = pathlib.Path(mod.__file__)
        content = source_file.read_text()
        assert "import difflib" not in content
