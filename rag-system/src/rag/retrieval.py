"""
retrieval.py — Hybrid retrieval for the Agentic RAG node.

Pipeline:
  1. Dense vector search (Chroma, cosine) with metadata pre-filter.
  2. Sparse BM25 keyword search over the same corpus, with the same filter.
  3. Reciprocal-Rank Fusion (RRF) of the two ranked lists (weighted by HYBRID_ALPHA).
  4. Optional cross-encoder reranking of the fused top-M.

Metadata filters (from the Planner) are applied to BOTH channels so budget /
brand / material / rating constraints are always honored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config, get_config
from .embeddings import get_embedder
from .schema import RagResult
from .vectorstore import get_store

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class RetrievedDoc:
    id: str
    document: str
    meta: dict
    vec_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    fused: float = 0.0
    rerank: Optional[float] = None

    @property
    def score(self) -> float:
        return self.rerank if self.rerank is not None else self.fused


# ---------------------------------------------------------------------------
# Filter translation
# ---------------------------------------------------------------------------
# NB: the dense-channel filter is translated to each backend's native filter
# language inside `vectorstore.py`. `match_meta` below is the store-agnostic
# Python predicate used to filter the *BM25* channel over the in-memory corpus.

def match_meta(meta: dict, filters: Optional[dict]) -> bool:
    """Store-agnostic predicate mirroring the dense-channel filter, applied to
    the BM25 candidates over the in-memory corpus."""
    if not filters:
        return True
    price = meta.get("price", -1.0)
    rating = meta.get("rating", -1.0)
    if filters.get("price_max") is not None and not (0 <= price <= float(filters["price_max"])):
        return False
    if filters.get("price_min") is not None and not (price >= float(filters["price_min"])):
        return False
    if filters.get("min_rating") is not None and not (rating >= float(filters["min_rating"])):
        return False
    if filters.get("brand") and str(meta.get("brand", "")).lower() != str(filters["brand"]).lower():
        return False
    if filters.get("material") and str(meta.get("material", "")).lower() != str(filters["material"]).lower():
        return False
    if filters.get("category_contains"):
        if str(filters["category_contains"]).lower() not in str(meta.get("category", "")).lower():
            return False
    return True


class HybridRetriever:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or get_config()
        self.store = get_store(self.cfg).load()
        # Pull the whole (small) corpus once to power BM25 + local filtering.
        self.ids, self.docs, self.metas = self.store.fetch_all()
        self._id_to_pos = {i: p for p, i in enumerate(self.ids)}

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tok(d) for d in self.docs]) if self.docs else None

        self._embedder = None
        self._reranker = None

    # -- lazy heavy resources ------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.cfg)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.cfg.reranker_model)
        return self._reranker

    # -- channels ------------------------------------------------------------
    def _vector_ranked(self, query: str, filters, n: int) -> list[str]:
        qvec = self.embedder.embed([query])[0]
        return self.store.query(qvec, min(n, max(1, len(self.ids))), filters)

    def _bm25_ranked(self, query: str, filters, n: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tok(query))
        order = np.argsort(scores)[::-1]
        out = []
        for pos in order:
            if scores[pos] <= 0:
                break
            if match_meta(self.metas[pos], filters):
                out.append(self.ids[pos])
            if len(out) >= n:
                break
        return out

    # -- public API ----------------------------------------------------------
    def search(self, query: str, k: Optional[int] = None,
               filters: Optional[dict] = None,
               use_reranker: Optional[bool] = None) -> list[RetrievedDoc]:
        cfg = self.cfg
        k = k or cfg.top_k
        n = cfg.candidate_k
        alpha = cfg.hybrid_alpha
        use_reranker = cfg.use_reranker if use_reranker is None else use_reranker

        vec_ids = self._vector_ranked(query, filters, n)
        bm_ids = self._bm25_ranked(query, filters, n)

        vec_rank = {i: r for r, i in enumerate(vec_ids, start=1)}
        bm_rank = {i: r for r, i in enumerate(bm_ids, start=1)}

        # Reciprocal Rank Fusion across the union of candidates.
        docs: dict[str, RetrievedDoc] = {}
        for i in set(vec_ids) | set(bm_ids):
            pos = self._id_to_pos.get(i)
            if pos is None:
                continue
            fused = 0.0
            if i in vec_rank:
                fused += alpha * (1.0 / (cfg.rrf_k + vec_rank[i]))
            if i in bm_rank:
                fused += (1 - alpha) * (1.0 / (cfg.rrf_k + bm_rank[i]))
            docs[i] = RetrievedDoc(
                id=i, document=self.docs[pos], meta=self.metas[pos],
                vec_rank=vec_rank.get(i), bm25_rank=bm_rank.get(i), fused=fused,
            )

        ranked = sorted(docs.values(), key=lambda d: d.fused, reverse=True)

        if use_reranker and ranked:
            top = ranked[: max(k * 3, 10)]
            try:
                pairs = [(query, d.document) for d in top]
                scores = self.reranker.predict(pairs)
                for d, s in zip(top, scores):
                    d.rerank = float(s)
                rest = ranked[len(top):]
                top.sort(key=lambda d: d.rerank, reverse=True)
                ranked = top + rest
            except Exception:  # noqa: BLE001 — reranker is optional; fall back to fused order
                pass

        return ranked[:k]

    def as_results(self, docs: list[RetrievedDoc]) -> list[RagResult]:
        out = []
        for d in docs:
            m = d.meta
            out.append(RagResult(
                sku=m.get("sku") or m.get("asin") or d.id,
                title=m.get("title", ""),
                price=None if m.get("price", -1) < 0 else round(float(m["price"]), 2),
                rating=None if m.get("rating", -1) < 0 else round(float(m["rating"]), 2),
                doc_id=m.get("doc_id") or d.id,
                brand=m.get("brand") or None,
                ingredients=m.get("ingredients") or None,
                url=m.get("url") or None,
                price_per_oz=None if m.get("price_per_oz", -1) < 0 else round(float(m["price_per_oz"]), 4),
                score=round(float(d.score), 6),
            ))
        return out


_RETRIEVER: Optional[HybridRetriever] = None


def get_retriever(cfg: Optional[Config] = None, refresh: bool = False) -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None or refresh:
        _RETRIEVER = HybridRetriever(cfg)
    return _RETRIEVER
