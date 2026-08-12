"""
embeddings.py — provider-swappable text embeddings (Model-Agnostic requirement).

Providers (set EMBEDDING_PROVIDER in .env):
  * "local"  -> sentence-transformers (default: all-MiniLM-L6-v2). No API key.
  * "openai" -> OpenAI embeddings (needs OPENAI_API_KEY).
  * "hash"   -> deterministic hashing embedder. No deps, no network — used by
                the test suite / CI so the whole pipeline runs offline. NOT for
                production quality, only for wiring/logic verification.

All embedders expose:
    .dim            -> int
    .embed(texts)   -> np.ndarray  shape (n, dim), L2-normalized (cosine-ready)
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from .config import Config, get_config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashEmbedder:
    """Feature-hashing embedder: token -> bucket, signed, then L2-normalized.

    Deterministic and dependency-free. Captures lexical overlap only (no
    semantics), which is enough to exercise retrieval/fusion logic in tests.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall((text or "").lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 7) & 1 else -1.0
            v[idx] += sign
        return v

    def embed(self, texts: List[str]) -> np.ndarray:
        mat = np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        return _l2_normalize(mat.astype(np.float32))


class LocalEmbedder:
    """sentence-transformers embedder (default, no API key)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        # method was renamed across ST versions — support both.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        vecs = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False, batch_size=64,
        )
        return vecs.astype(np.float32)


class OpenAIEmbedder:
    """OpenAI embeddings (needs OPENAI_API_KEY)."""

    def __init__(self, model_name: str):
        from openai import OpenAI  # lazy import

        self.client = OpenAI()
        self.model_name = model_name
        # dims for common models; overridden after first call if needed
        self.dim = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(model_name, 1536)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for i in range(0, len(texts), 256):  # respect batch limits
            batch = [t if t.strip() else " " for t in texts[i:i + 256]]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            out.extend([d.embedding for d in resp.data])
        mat = np.array(out, dtype=np.float32)
        self.dim = mat.shape[1]
        return _l2_normalize(mat)


def get_embedder(cfg: Config | None = None):
    cfg = cfg or get_config()
    provider = cfg.embedding_provider.lower()
    if provider == "local":
        return LocalEmbedder(cfg.embedding_model_local)
    if provider == "openai":
        return OpenAIEmbedder(cfg.embedding_model_openai)
    if provider == "hash":
        return HashEmbedder(cfg.hash_dim)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER={cfg.embedding_provider!r}")
