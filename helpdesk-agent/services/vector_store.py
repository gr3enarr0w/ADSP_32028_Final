"""Vector storage layer — store and query embedding vectors in SQLite BLOB columns.

Usage:
    from services.vector_store import store_embedding, query_similar

    store_embedding("ANTSE-100", embed_text("How to reset password"), "ticket")
    results = query_similar(embed_text("password reset"), k=5, entity_type="ticket")
"""

import logging
import struct

import numpy as np

from db import get_db, get_db_conn

log = logging.getLogger(__name__)

_TABLE_CONFIG = {
    "ticket": ("tickets", "ticket_key"),
    "article": ("generated_articles", "article_topic"),
    "kb_article": ("kb_articles", "page_id"),
    "few_shot": ("few_shot_examples", "example_id"),
    "corpus": ("responder_corpus_embeddings", "corpus_id"),
}


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def store_embedding(
    entity_id: str,
    vector: list[float],
    entity_type: str,
    conn=None,
) -> None:
    """Persist an embedding vector for a given entity.

    Args:
        entity_id: Primary key value (ticket_key, article_topic, page_id, or example_id).
        vector: Pre-normalized embedding from the embedding service.
        entity_type: One of "ticket", "article", "kb_article", or "few_shot".
    """
    if entity_type not in _TABLE_CONFIG:
        raise ValueError(f"Unknown entity_type: {entity_type!r}")

    table, id_col = _TABLE_CONFIG[entity_type]
    blob = _pack(vector)

    if conn is None:
        with get_db_conn() as conn:
            conn.execute(
                f"UPDATE {table} SET embedding = ? WHERE {id_col} = ?",
                (blob, entity_id),
            )
    else:
        conn.execute(
            f"UPDATE {table} SET embedding = ? WHERE {id_col} = ?",
            (blob, entity_id),
        )


def query_similar(
    vector: list[float],
    k: int = 5,
    entity_type: str = "ticket",
) -> list[tuple[str, float]]:
    """Find the k most similar entities by dot-product similarity.

    Embeddings are pre-normalized by the embedding service, so dot product
    equals cosine similarity.

    Args:
        vector: Query embedding (unit-normalized).
        k: Number of results to return.
        entity_type: One of "ticket", "article", "kb_article", or "few_shot".

    Returns:
        List of (entity_id, similarity_score) sorted by descending similarity.
    """
    if entity_type not in _TABLE_CONFIG:
        raise ValueError(f"Unknown entity_type: {entity_type!r}")

    table, id_col = _TABLE_CONFIG[entity_type]
    query_vec = np.array(vector, dtype=np.float32)

    conn = get_db()
    try:
        rows = conn.execute(
            f"SELECT {id_col}, embedding FROM {table} WHERE embedding IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    results: list[tuple[str, float]] = []
    for row in rows:
        stored_vec = _unpack(row["embedding"])
        sim = float(np.dot(query_vec, stored_vec))
        results.append((str(row[id_col]), sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
