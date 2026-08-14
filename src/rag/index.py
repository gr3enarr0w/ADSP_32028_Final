"""
index.py — build & load the vector index over the Household-Cleaning slice.

Store backend is pluggable via `VECTOR_STORE` (see `vectorstore.py`):
  * "qdrant" (default) — embedded local (no server), or a server via QDRANT_URL.
  * "chroma"           — persistent, native metadata filtering, zero setup.

The embedding text is `title + features + top-3 review snippets + ingredients`
(`Product.embed_text`). Metadata carried on every vector supports the hybrid
retriever's filters and the rag.search citation payload.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config, get_config
from .embeddings import get_embedder
from .vectorstore import get_store

# Chroma metadata must be non-null scalars; use these sentinels for "missing".
_NUM_MISSING = -1.0
_STR_MISSING = ""


def _load_frames(cfg: Config):
    products = pd.read_parquet(os.path.join(cfg.processed_dir, "products.parquet"))
    reviews_path = os.path.join(cfg.processed_dir, "reviews.parquet")
    reviews = pd.read_parquet(reviews_path) if os.path.exists(reviews_path) else pd.DataFrame(
        columns=["product_id", "stars", "snippet"])
    return products, reviews


def _top_snippets(reviews: pd.DataFrame, pid: str, n: int = 3) -> list[str]:
    if reviews.empty:
        return []
    sub = reviews[reviews["product_id"] == pid]
    return sub["snippet"].head(n).tolist()


def _material(title: str, category: str) -> str:
    """Coarse material tag used as a cheap metadata filter (e.g. 'stainless steel')."""
    text = f"{title} {category}".lower()
    for m in ["stainless steel", "glass", "wood", "tile", "grout", "fabric",
              "porcelain", "ceramic", "chrome", "multi-surface"]:
        if m in text:
            return m
    return _STR_MISSING


def _meta_row(row: pd.Series) -> dict:
    def num(v):
        return float(v) if pd.notna(v) else _NUM_MISSING

    def s(v):
        return str(v) if pd.notna(v) and str(v) != "nan" else _STR_MISSING

    return {
        "doc_id": s(row["doc_id"]),
        "sku": s(row["sku"]),
        "asin": s(row["asin"]),
        "title": s(row["title"]),
        "brand": s(row["brand"]),
        "category": s(row["category"]),
        "material": _material(str(row["title"]), str(row["category"])),
        "price": num(row["price"]),
        "list_price": num(row["list_price"]),
        "rating": num(row["rating"]),
        "size_oz": num(row["size_oz"]),
        "price_per_oz": num(row["price_per_oz"]),
        "ingredients": s(row["ingredients"])[:1500],
        "url": s(row["url"]),
        "stock": s(row["stock"]),
    }


def build_index(cfg: Optional[Config] = None) -> dict:
    """Read processed parquet, embed, and (re)build the vector index."""
    cfg = cfg or get_config()
    products, reviews = _load_frames(cfg)
    if products.empty:
        raise RuntimeError("products.parquet is empty — run ingestion first.")

    embedder = get_embedder(cfg)
    sig = cfg.embedding_signature()

    ids, docs, metas = [], [], []
    seen = set()
    from .schema import Product

    for _, row in products.iterrows():
        snippets = _top_snippets(reviews, str(row["doc_id"]))
        p = Product(
            doc_id=str(row["doc_id"]), sku=str(row["sku"]), asin=str(row["asin"]),
            title=str(row["title"]), brand=str(row["brand"]), category=str(row["category"]),
            price=row["price"] if pd.notna(row["price"]) else None,
            list_price=row["list_price"] if pd.notna(row["list_price"]) else None,
            rating=row["rating"] if pd.notna(row["rating"]) else None,
            features=str(row["features"]).split(" | ") if row["features"] else [],
            ingredients=str(row["ingredients"]) if pd.notna(row["ingredients"]) else "",
            size_oz=row["size_oz"] if pd.notna(row["size_oz"]) else None,
            price_per_oz=row["price_per_oz"] if pd.notna(row["price_per_oz"]) else None,
            url=str(row["url"]), stock=str(row["stock"]),
        )
        did = p.doc_id or p.asin
        while did in seen:  # guarantee unique ids
            did = f"{did}_dup"
        seen.add(did)
        ids.append(did)
        docs.append(p.embed_text(snippets))
        metas.append(_meta_row(row))

    embeddings = embedder.embed(docs)

    store = get_store(cfg)
    res = store.rebuild(ids, embeddings, docs, metas, sig)

    manifest = {
        "collection": cfg.collection,
        "vector_store": cfg.vector_store,
        "embedding_signature": sig,
        "n_vectors": res["n_vectors"],
        "dim": res.get("dim", int(embeddings.shape[1]) if len(ids) else 0),
        "index_dir": cfg.index_dir,
    }
    with open(os.path.join(cfg.index_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_collection(cfg: Optional[Config] = None):
    """Deprecated shim — use `rag.vectorstore.get_store(cfg).load()`.

    Returns the loaded vector store (backend-agnostic), not a raw Chroma
    collection. Kept so older imports don't break.
    """
    return get_store(cfg).load()


if __name__ == "__main__":
    print(json.dumps(build_index(), indent=2))
