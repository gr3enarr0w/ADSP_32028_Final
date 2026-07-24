#!/usr/bin/env python3
"""
Standalone diagnostic CLI: visualize child-chunk dense embeddings from a Qdrant
collection to make embedding-collapse (all chunks landing on ~the same vector)
visually and numerically obvious, without live-debugging via raw vector pulls.

Pulls dense ("dense" named vector) vectors + payloads for CHILD chunks
(chunk_type=child) from the given collection, projects them to 2D via PCA
and/or UMAP, colors points by `source` payload field, saves PNG plot(s) to
disk, and prints a simple pairwise-cosine-similarity degeneracy check.

Usage examples:
    python scripts/visualize_embeddings.py --collection my_docs
    python scripts/visualize_embeddings.py --collection my_docs --method pca
    python scripts/visualize_embeddings.py --collection my_docs --method umap --limit 500
    python scripts/visualize_embeddings.py --collection my_docs --output viz/my_docs_embeddings --config config/config.yaml
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import EfficientRAG, load_config  # noqa: E402

# Matplotlib must use a non-interactive backend — this is a CLI tool that
# saves PNGs to disk, never displays inline (no notebook/GUI context to
# assume is present).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def fetch_child_vectors(rag: EfficientRAG, collection: str, limit: int = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Scroll all (or up to `limit`) CHILD-chunk points from the collection,
    pulling the "dense" named vector + payload for each.

    Filters on chunk_type=child using the same Filter/FieldCondition/MatchValue
    pattern EfficientRAG.retrieve() already uses to exclude zero-vector parent
    points from search — parent points carry a dummy all-zero "dense" vector
    (see EfficientRAG.ingest()) and would otherwise appear as a degenerate
    cluster-at-the-origin artifact in the projection, not a real signal about
    embedding collapse.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    full_coll = rag.get_collection_name(collection)
    child_filter = Filter(must=[FieldCondition(key="chunk_type", match=MatchValue(value="child"))])

    vectors: List[List[float]] = []
    payloads: List[Dict[str, Any]] = []
    next_offset = None
    batch_size = 256

    while True:
        remaining = None if limit is None else limit - len(vectors)
        if remaining is not None and remaining <= 0:
            break
        fetch_size = batch_size if remaining is None else min(batch_size, remaining)

        points, next_offset = rag.qdrant.scroll(
            collection_name=full_coll,
            scroll_filter=child_filter,
            with_vectors=True,
            with_payload=True,
            limit=fetch_size,
            offset=next_offset,
        )

        for point in points:
            vec = point.vector
            # Named-vector collections return a dict (e.g. {"dense": [...], "sparse": ...});
            # only "dense" is used for this diagnostic (matches EfficientRAG's storage layout).
            dense_vec = vec.get("dense") if isinstance(vec, dict) else vec
            if dense_vec is None:
                continue
            vectors.append(dense_vec)
            payloads.append(point.payload or {})

        if next_offset is None or not points:
            break

    return np.array(vectors, dtype=np.float64), payloads


def compute_degeneracy_check(vectors: np.ndarray, sample_size: int = 500, threshold: float = 0.98) -> Dict[str, Any]:
    """Compute mean pairwise cosine similarity across a random sample of
    points, plus the fraction of pairs above `threshold`. Independent of the
    2D visualization — a numeric cross-check that doesn't rely on eyeballing
    a plot, in case PCA/UMAP happen to visually spread out points that are
    nonetheless nearly identical in the full-dimensional space (or vice versa)."""
    n = vectors.shape[0]
    if n < 2:
        return {"n_points": n, "n_pairs_sampled": 0, "mean_cosine_similarity": None,
                "frac_above_threshold": None, "threshold": threshold}

    sample_n = min(sample_size, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=sample_n, replace=False)
    sample = vectors[idx]

    # Normalize (dense vectors are stored normalized at ingest time via
    # normalize_embeddings=True, but don't assume that holds for every
    # collection this tool might be pointed at).
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normed = sample / norms

    sim_matrix = normed @ normed.T
    iu = np.triu_indices(sample_n, k=1)
    pair_sims = sim_matrix[iu]

    mean_sim = float(np.mean(pair_sims))
    frac_above = float(np.mean(pair_sims > threshold))

    return {
        "n_points": n,
        "n_pairs_sampled": len(pair_sims),
        "mean_cosine_similarity": mean_sim,
        "frac_above_threshold": frac_above,
        "threshold": threshold,
    }


def print_degeneracy_report(report: Dict[str, Any]) -> None:
    print("\n=== Embedding Degeneracy Check ===")
    if report["n_pairs_sampled"] == 0:
        print(f"Only {report['n_points']} point(s) found — need at least 2 to compute pairwise similarity.")
        return

    print(f"Points available: {report['n_points']}")
    print(f"Pairs sampled: {report['n_pairs_sampled']}")
    print(f"Mean pairwise cosine similarity: {report['mean_cosine_similarity']:.4f}")
    pct_above = report["frac_above_threshold"] * 100
    print(f"Fraction of pairs with cosine similarity > {report['threshold']}: {pct_above:.2f}%")

    if report["frac_above_threshold"] > 0.5 or report["mean_cosine_similarity"] > 0.95:
        print(
            f"WARNING: {pct_above:.2f}% of sampled chunk pairs have cosine similarity > "
            f"{report['threshold']} — check for embedding collapse or a data pipeline bug "
            f"(e.g. every chunk embedding the same text, or all vectors zeroed/near-identical)."
        )
    else:
        print(
            "Healthy spread — similarities are not uniformly high, consistent with chunks "
            "encoding distinct content rather than a collapsed/degenerate embedding space."
        )


def project_pca(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    n_components = min(2, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(vectors)
    if n_components < 2:
        # Pad so downstream plotting code can always assume 2 columns.
        coords = np.pad(coords, ((0, 0), (0, 2 - n_components)))
    return coords, pca.explained_variance_ratio_


def project_umap(vectors: np.ndarray) -> np.ndarray:
    import umap

    n_points = vectors.shape[0]
    # UMAP's default n_neighbors=15 assumes large corpora; on small corpora
    # (fewer than ~15 points) this either errors or produces a misleading
    # layout, since there aren't enough neighbors to estimate local structure
    # from. Scale it down to fit the actual corpus size.
    n_neighbors = max(1, min(15, n_points - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
    return reducer.fit_transform(vectors)


def plot_projection(coords: np.ndarray, sources: List[str], title: str, output_path: str) -> None:
    unique_sources = sorted(set(sources))
    cmap = plt.get_cmap("tab20" if len(unique_sources) > 10 else "tab10")
    color_map = {src: cmap(i % cmap.N) for i, src in enumerate(unique_sources)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for src in unique_sources:
        mask = [s == src for s in sources]
        pts = coords[mask]
        label = src if src else "(unknown)"
        ax.scatter(pts[:, 0], pts[:, 1], label=label, alpha=0.7, s=30, color=color_map[src])

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize="small", markerscale=1.2, title="source")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize child-chunk dense embeddings from a Qdrant collection "
                    "(PCA and/or UMAP) and run a pairwise-cosine-similarity degeneracy check.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--collection", required=True, help="Collection name (unprefixed, same as ingest.py/retrieve.py)")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Cap on number of child points sampled (default: all)")
    parser.add_argument("--method", choices=["pca", "umap", "both"], default="both", help="Projection method(s) to run")
    parser.add_argument("--output", default="embeddings_viz", help="Output filename prefix for saved PNG(s)")
    parser.add_argument("--sim-sample-size", type=int, default=500, help="Number of points to sample for the pairwise cosine similarity degeneracy check")
    parser.add_argument("--sim-threshold", type=float, default=0.98, help="Cosine similarity threshold above which a pair is flagged as suspiciously similar")
    args = parser.parse_args()

    config = load_config(args.config)
    rag = EfficientRAG(config)

    print(f"Fetching child-chunk vectors from collection '{args.collection}'"
          f"{' (limit=' + str(args.limit) + ')' if args.limit else ''}...")
    vectors, payloads = fetch_child_vectors(rag, args.collection, limit=args.limit)

    if len(vectors) == 0:
        print(f"No child-chunk points found in collection '{args.collection}' (or its Qdrant-prefixed name "
              f"'{rag.get_collection_name(args.collection)}'). Nothing to visualize.")
        sys.exit(1)

    print(f"Fetched {len(vectors)} child-chunk vectors (dim={vectors.shape[1]}).")

    sources = [p.get("source") or "(unknown)" for p in payloads]

    # Degeneracy check — independent of the visualization, so it still gives
    # a signal even if the 2D projection is misleading or a plotting
    # dependency is missing.
    report = compute_degeneracy_check(vectors, sample_size=args.sim_sample_size, threshold=args.sim_threshold)
    print_degeneracy_report(report)

    random.seed(42)

    if args.method in ("pca", "both"):
        coords, explained_var = project_pca(vectors)
        print(f"\nPCA explained variance ratio (first 2 components): "
              f"{explained_var[0]:.4f}, {explained_var[1] if len(explained_var) > 1 else 0.0:.4f} "
              f"(total: {sum(explained_var[:2]):.4f})")
        pca_path = f"{args.output}_pca.png"
        plot_projection(coords, sources, f"PCA projection — {args.collection} ({len(vectors)} child chunks)", pca_path)

    if args.method in ("umap", "both"):
        if len(vectors) < 2:
            print("\nSkipping UMAP — need at least 2 points.")
        else:
            umap_coords = project_umap(vectors)
            n_neighbors_used = max(1, min(15, len(vectors) - 1))
            print(f"\nUMAP projection computed (n_neighbors={n_neighbors_used}, scaled down from default 15 "
                  f"since this corpus has {len(vectors)} points).")
            umap_path = f"{args.output}_umap.png"
            plot_projection(umap_coords, sources, f"UMAP projection — {args.collection} ({len(vectors)} child chunks)", umap_path)


if __name__ == "__main__":
    main()
