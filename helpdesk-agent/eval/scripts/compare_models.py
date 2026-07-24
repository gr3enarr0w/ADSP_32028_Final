"""Final apples-to-apples model comparison on the held-out test set.

Compares 4 pre-trained and 2 fine-tuned bi-encoder models on retrieval
metrics (MRR, P@K, Recall@K) using the same test pairs and evaluation
protocol for every model.

Usage:
    USE_TF=0 python3 -m eval.scripts.compare_models
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np

# ── Model registry ──

# Import pre-trained model IDs and prefixes from compare_strategies
_PRETRAINED = {
    "minilm": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "q_prefix": "",
        "d_prefix": "",
    },
    "mpnet": {
        "id": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "q_prefix": "",
        "d_prefix": "",
    },
    "bge": {
        "id": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "q_prefix": "",
        "d_prefix": "",
    },
    "e5": {
        "id": "intfloat/e5-small-v2",
        "dim": 384,
        "q_prefix": "query: ",
        "d_prefix": "passage: ",
    },
}

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MODELS_DIR = _DATA_DIR / "models"

_FINETUNED = {
    "minilm-finetuned": {
        "path": str(_MODELS_DIR / "minilm-finetuned"),
        "base": "minilm",
        "q_prefix": "",
        "d_prefix": "",
    },
    "mpnet-finetuned": {
        "path": str(_MODELS_DIR / "mpnet-finetuned"),
        "base": "mpnet",
        "q_prefix": "",
        "d_prefix": "",
    },
}

ALL_MODELS = list(_PRETRAINED.keys()) + list(_FINETUNED.keys())


# ── Data loading ──

def load_test_pairs(path: Path | None = None) -> list[dict]:
    path = path or (_DATA_DIR / "training_pairs_test.json")
    data = json.loads(path.read_text())
    print(f"Loaded {len(data)} test pairs from {path.name}")
    return data


# ── Model loading & encoding ──

def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    if name in _PRETRAINED:
        model_id = _PRETRAINED[name]["id"]
        print(f"  Loading pre-trained: {model_id}")
        return SentenceTransformer(model_id)
    elif name in _FINETUNED:
        model_path = _FINETUNED[name]["path"]
        print(f"  Loading fine-tuned:  {model_path}")
        return SentenceTransformer(model_path)
    else:
        raise ValueError(f"Unknown model: {name}")


def _get_prefixes(name: str) -> tuple[str, str]:
    if name in _PRETRAINED:
        info = _PRETRAINED[name]
    else:
        info = _FINETUNED[name]
    return info["q_prefix"], info["d_prefix"]


def encode_texts(model, texts: list[str], prefix: str, batch_size: int = 128) -> np.ndarray:
    prefixed = [prefix + t for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False,
                        batch_size=batch_size)


# ── Metrics ──

def compute_retrieval_metrics(sim_matrix: np.ndarray) -> dict:
    """Compute retrieval metrics from a similarity matrix.

    sim_matrix[i][j] = similarity between anchor i and positive j.
    Ground truth: anchor i matches positive i (diagonal).
    """
    n = sim_matrix.shape[0]

    reciprocal_ranks = []
    precision_at = {1: [], 3: [], 5: []}
    recall_at = {1: [], 5: [], 10: []}

    for i in range(n):
        # Rank all positives by similarity to anchor i
        scores = sim_matrix[i]
        ranked_indices = np.argsort(-scores)  # descending

        # The correct positive is at index i
        correct_rank = int(np.where(ranked_indices == i)[0][0]) + 1  # 1-indexed

        reciprocal_ranks.append(1.0 / correct_rank)

        for k in precision_at:
            top_k = set(ranked_indices[:k].tolist())
            precision_at[k].append(1.0 if i in top_k else 0.0)

        for k in recall_at:
            top_k = set(ranked_indices[:k].tolist())
            # Single relevant doc per query, so recall@k = 1 if found, else 0
            recall_at[k].append(1.0 if i in top_k else 0.0)

    return {
        "mrr": statistics.mean(reciprocal_ranks),
        "p@1": statistics.mean(precision_at[1]),
        "p@3": statistics.mean(precision_at[3]),
        "p@5": statistics.mean(precision_at[5]),
        "r@1": statistics.mean(recall_at[1]),
        "r@5": statistics.mean(recall_at[5]),
        "r@10": statistics.mean(recall_at[10]),
    }


def mean_matched_cosine(sim_matrix: np.ndarray) -> float:
    """Mean cosine similarity along the diagonal (matched pairs)."""
    return float(np.mean(np.diag(sim_matrix)))


# ── Main ──

def run():
    pairs = load_test_pairs()
    n = len(pairs)

    anchors = [p["anchor"] for p in pairs]
    positives = [p["positive"] for p in pairs]

    results = {}

    for model_name in ALL_MODELS:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─' * 60}")

        t0 = time.perf_counter()
        model = _load_model(model_name)
        q_prefix, d_prefix = _get_prefixes(model_name)

        # Encode
        anchor_embs = encode_texts(model, anchors, q_prefix)
        positive_embs = encode_texts(model, positives, d_prefix)
        encode_time = time.perf_counter() - t0

        # Full similarity matrix: each anchor against ALL positives
        sim_matrix = anchor_embs @ positive_embs.T
        assert sim_matrix.shape == (n, n), f"Expected ({n},{n}), got {sim_matrix.shape}"

        # Metrics
        metrics = compute_retrieval_metrics(sim_matrix)
        metrics["mean_cos"] = mean_matched_cosine(sim_matrix)
        metrics["encode_s"] = encode_time

        results[model_name] = metrics

        print(f"    MRR={metrics['mrr']:.4f}  P@1={metrics['p@1']:.4f}  "
              f"R@10={metrics['r@10']:.4f}  mean_cos={metrics['mean_cos']:.4f}  "
              f"({encode_time:.1f}s)")

        # Free memory
        del model, anchor_embs, positive_embs, sim_matrix

    # ── Print comparison table ──
    print()
    print()
    w = 120
    print("=" * w)
    print("  FINAL MODEL COMPARISON — Held-Out Test Set ({} pairs)".format(n))
    print("=" * w)
    print()

    col = 22
    header = (f"  {'Model':<{col}} {'MRR':>7} {'P@1':>7} {'P@3':>7} {'P@5':>7} "
              f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'MeanCos':>8} {'Time(s)':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_name in ALL_MODELS:
        m = results[model_name]
        tag = ""
        if model_name in _FINETUNED:
            tag = " *"
        print(f"  {model_name + tag:<{col}} "
              f"{m['mrr']:>7.4f} {m['p@1']:>7.4f} {m['p@3']:>7.4f} {m['p@5']:>7.4f} "
              f"{m['r@1']:>7.4f} {m['r@5']:>7.4f} {m['r@10']:>7.4f} "
              f"{m['mean_cos']:>8.4f} {m['encode_s']:>8.1f}")

    # ── Delta table: fine-tuned vs pre-trained base ──
    print()
    print("=" * w)
    print("  FINE-TUNING IMPROVEMENT (delta vs pre-trained base)")
    print("=" * w)
    print()

    delta_header = (f"  {'Fine-tuned':<{col}} {'Base':<{col}} "
                    f"{'dMRR':>8} {'dP@1':>8} {'dP@3':>8} {'dR@5':>8} {'dR@10':>8} "
                    f"{'dMeanCos':>9}")
    print(delta_header)
    print("  " + "-" * (len(delta_header) - 2))

    for ft_name, ft_info in _FINETUNED.items():
        base_name = ft_info["base"]
        ft = results[ft_name]
        base = results[base_name]

        d_mrr = ft["mrr"] - base["mrr"]
        d_p1 = ft["p@1"] - base["p@1"]
        d_p3 = ft["p@3"] - base["p@3"]
        d_r5 = ft["r@5"] - base["r@5"]
        d_r10 = ft["r@10"] - base["r@10"]
        d_cos = ft["mean_cos"] - base["mean_cos"]

        print(f"  {ft_name:<{col}} {base_name:<{col}} "
              f"{d_mrr:>+8.4f} {d_p1:>+8.4f} {d_p3:>+8.4f} {d_r5:>+8.4f} {d_r10:>+8.4f} "
              f"{d_cos:>+9.4f}")

    # ── Winner ──
    print()
    print("=" * w)
    print("  WINNER")
    print("=" * w)
    print()

    ranked = sorted(results.items(), key=lambda t: t[1]["mrr"], reverse=True)
    best_name, best = ranked[0]
    runner_name, runner = ranked[1]

    print(f"  1st: {best_name:<22} MRR={best['mrr']:.4f}  P@1={best['p@1']:.4f}  R@10={best['r@10']:.4f}")
    print(f"  2nd: {runner_name:<22} MRR={runner['mrr']:.4f}  P@1={runner['p@1']:.4f}  R@10={runner['r@10']:.4f}")
    print()

    margin = best["mrr"] - runner["mrr"]
    print(f"  {best_name} wins by {margin:+.4f} MRR over {runner_name}")
    print()


if __name__ == "__main__":
    run()
