"""Shared embedding service — singleton model, normalized dense vectors.

Provider selection is controlled by the EMBEDDING_MODEL environment variable:

    sentence-transformers/all-MiniLM-L6-v2  (default — legacy, 384-dim)
        Self-hosted SentenceTransformer model.  No GCP credentials required.
        Low domain accuracy on JSM tickets (F1=0.54, threshold at floor 0.30).

    gemini-embedding-001  (recommended — 3072-dim)
        Vertex AI embedding model via the google-genai SDK.  Requires
        GOOGLE_SERVICE_ACCOUNT_JSON, GEMINI_PROJECT, and GEMINI_LOCATION.
        Uses task_type specifications (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT)
        for query vs document embeddings.  Batched in chunks of 250 per
        Vertex AI API limits.

Usage:
    from services.embedding import embed_text, embed_batch

    vec = embed_text("How do I migrate to Cloud?")
    vecs = embed_batch(["query one", "query two"])
"""

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "gemini-embedding-001"
)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))

# Vertex AI allows up to 250 texts per embed_content call.
_VERTEX_BATCH_SIZE = 250

# ── SentenceTransformer singleton (legacy MiniLM path) ──

_st_model = None


def _get_st_model():
    """Lazy-load the SentenceTransformer model (only when EMBEDDING_MODEL is a HF path)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Loaded SentenceTransformer model: %s", EMBEDDING_MODEL)
    return _st_model


# ── Task-type mapping: codebase uses "query"/"document"; Vertex AI uses enum strings ──

_TASK_TYPE_MAP = {
    "query": "RETRIEVAL_QUERY",
    "document": "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY": "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT": "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY": "SEMANTIC_SIMILARITY",
    "CLASSIFICATION": "CLASSIFICATION",
    "CLUSTERING": "CLUSTERING",
}


def _vertex_task_type(task_type: str) -> str:
    """Map codebase task_type strings to Vertex AI enum values."""
    return _TASK_TYPE_MAP.get(task_type, "RETRIEVAL_QUERY")


# ── Vertex AI embedding helpers ──

def _vertex_embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts using Vertex AI gemini-embedding-001.

    Splits into chunks of _VERTEX_BATCH_SIZE (≤250) as required by the API.
    Returns unit-normalized vectors (Vertex AI returns normalized embeddings).
    """
    from core.genai import get_genai_client

    client = get_genai_client()
    vt = _vertex_task_type(task_type)
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), _VERTEX_BATCH_SIZE):
        chunk = texts[start : start + _VERTEX_BATCH_SIZE]
        from google.genai import types as genai_types
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk,
            config=genai_types.EmbedContentConfig(
                task_type=vt,
                http_options=genai_types.HttpOptions(timeout=90_000),
            ),
        )
        for emb in response.embeddings:
            vec = list(emb.values)
            # Normalize to unit length (Vertex returns near-unit but let's be safe)
            arr = np.array(vec, dtype=np.float32)
            mag = float(np.linalg.norm(arr))
            if mag > 0:
                arr = arr / mag
            all_vectors.append(arr.tolist())

    return all_vectors


def _is_vertex_model() -> bool:
    """Return True when EMBEDDING_MODEL is a Vertex AI model identifier."""
    return not EMBEDDING_MODEL.startswith("sentence-transformers/")


# ── Public API ──

def embed_text(text: str, task_type: str = "query") -> list[float]:
    """Embed a single text string into a normalized dense vector.

    Args:
        text: Input text to embed.
        task_type: "query" or "document" — controls task-type hint for
            models that support it (e.g. Vertex AI gemini-embedding-001).

    Returns:
        Unit-normalized embedding as list[float].
        Dimension is model-dependent: 384 for MiniLM, 3072 for gemini-embedding-001.
    """
    if _is_vertex_model():
        results = _vertex_embed_texts([text], task_type)
        return results[0] if results else []

    model = _get_st_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(
    texts: list[str], task_type: str = "query"
) -> list[list[float]]:
    """Embed multiple texts, processing in chunks to respect API / memory limits.

    Args:
        texts: List of input texts.
        task_type: "query" or "document".

    Returns:
        List of unit-normalized embeddings, one per input text.
    """
    if not texts:
        return []

    if _is_vertex_model():
        return _vertex_embed_texts(texts, task_type)

    # SentenceTransformer path (MiniLM legacy)
    model = _get_st_model()
    all_vecs: list[np.ndarray] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        chunk = texts[start : start + EMBEDDING_BATCH_SIZE]
        vecs = model.encode(chunk, normalize_embeddings=True)
        all_vecs.append(vecs)

    result = np.vstack(all_vecs)
    return result.tolist()
