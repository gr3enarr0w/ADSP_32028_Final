"""Dense embedding retrieval for responder lookups (cosine similarity)."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass

from db import get_db_conn
from services.embedding import embed_batch, embed_text
from services.vector_store import query_similar, store_embedding

log = logging.getLogger(__name__)

_ENTITY_TYPE = "corpus"
_SOURCE_TYPE_ALIASES = {
    "faq": "faq_sources",
    "faq_source": "faq_sources",
    "faq_sources": "faq_sources",
    "kb": "kb_articles",
    "kb_article": "kb_articles",
    "kb_articles": "kb_articles",
    "ticket": "tickets",
    "tickets": "tickets",
    "resolved_ticket": "tickets",
    "atlassian_doc": "atlassian_docs",
    "atlassian_docs": "atlassian_docs",
}


def _normalize_source_type(source_type: str) -> str:
    normalized = _SOURCE_TYPE_ALIASES.get(source_type.lower())
    if not normalized:
        raise ValueError(f"Unknown source_type: {source_type}")
    return normalized


def _corpus_id(source_type: str, doc_id: str) -> str:
    return f"{source_type}:{doc_id}"


def _join_text(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


@dataclass(slots=True)
class _DocEntry:
    doc_id: str
    source_type: str
    text: str
    corpus_id: str


class DenseRetriever:
    """In-memory dense retrieval index backed by responder_corpus_embeddings."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: list[_DocEntry] = []
        self._doc_lookup: dict[tuple[str, str], int] = {}
        self._built = False

    def _load_from_connection(self, conn) -> list[_DocEntry]:
        docs: list[_DocEntry] = []
        counts: dict[str, int] = {}

        source_queries = [
            (
                "faq_sources",
                """
                SELECT source_id, source_type, COALESCE(title, source_id, '') AS title
                FROM faq_sources
                ORDER BY source_type, title
                """,
                lambda row: _join_text(row["title"], row["source_type"], row["source_id"]),
                lambda row: row["source_id"],
            ),
            (
                "kb_articles",
                """
                SELECT page_id, title, body_text, labels, topics_covered
                FROM kb_articles
                ORDER BY title
                """,
                lambda row: _join_text(
                    row["title"],
                    row["body_text"],
                    row["labels"],
                    row["topics_covered"],
                ),
                lambda row: row["page_id"],
            ),
            (
                "tickets",
                """
                SELECT ticket_key, summary, resolution, status
                FROM tickets
                WHERE LOWER(COALESCE(status, '')) IN ('resolved', 'closed')
                  AND resolution IS NOT NULL
                  AND TRIM(resolution) != ''
                ORDER BY resolved_at DESC, updated_at DESC, ticket_key DESC
                LIMIT 500
                """,
                lambda row: _join_text(row["summary"], row["resolution"], row["status"]),
                lambda row: row["ticket_key"],
            ),
            (
                "atlassian_docs",
                """
                SELECT url, title, product
                FROM atlassian_docs
                ORDER BY product, title
                """,
                lambda row: _join_text(row["title"], row["product"], row["url"]),
                lambda row: row["url"],
            ),
        ]

        for source_type, sql, text_fn, doc_id_fn in source_queries:
            rows = conn.execute(sql).fetchall()
            counts[source_type] = len(rows)
            for row in rows:
                doc_id = str(doc_id_fn(row))
                docs.append(
                    _DocEntry(
                        doc_id=doc_id,
                        source_type=source_type,
                        text=text_fn(row),
                        corpus_id=_corpus_id(source_type, doc_id),
                    )
                )

        log.info(
            "[dense] source counts faq_sources=%d kb_articles=%d tickets=%d atlassian_docs=%d total=%d",
            counts.get("faq_sources", 0),
            counts.get("kb_articles", 0),
            counts.get("tickets", 0),
            counts.get("atlassian_docs", 0),
            len(docs),
        )
        return docs

    def _set_index(self, docs: list[_DocEntry]) -> None:
        self._docs = docs
        self._doc_lookup = {(doc.source_type, doc.doc_id): i for i, doc in enumerate(docs)}
        self._built = True

    def build(self, db=None) -> "DenseRetriever":
        """Load corpus documents, embed them, and persist vectors for retrieval."""
        if db is None:
            conn_cm = get_db_conn()
        elif hasattr(db, "execute"):
            conn_cm = nullcontext(db)
        elif hasattr(db, "get_db_conn"):
            conn_cm = db.get_db_conn()
        else:
            raise TypeError("db must be a connection or expose get_db_conn()")

        # ── Phase 1: READ ────────────────────────────────────────────────────────
        # Hold the DB connection only long enough to load source docs and the
        # existing embedding cache.  The connection is released before the
        # embed_batch() API call so we never hold a SQLite write lock while
        # waiting on a network round-trip that can take 20+ minutes.
        with conn_cm as conn:
            docs = self._load_from_connection(conn)
            if not docs:
                conn.execute("DELETE FROM responder_corpus_embeddings")
                with self._lock:
                    self._set_index([])
                return self

            # Load cached text keyed by corpus_id to detect unchanged docs.
            existing = {
                row["corpus_id"]: row["text"]
                for row in conn.execute(
                    "SELECT corpus_id, text FROM responder_corpus_embeddings"
                ).fetchall()
            }

        # Snapshot the in-memory live index AFTER closing the connection so the
        # stale-cleanup step later won't delete docs that add_document() may have
        # inserted between the SELECT above and the write phase below.
        with self._lock:
            live_corpus_ids = {doc.corpus_id for doc in self._docs}

        # Only re-embed docs that are new or whose text has changed.
        current_corpus_ids = {doc.corpus_id for doc in docs}
        needs_embed = [doc for doc in docs if existing.get(doc.corpus_id) != doc.text]

        log.info(
            "[dense] build: %d total, %d cached, %d need embedding",
            len(docs),
            len(docs) - len(needs_embed),
            len(needs_embed),
        )

        # ── Phase 2: EMBED (no DB connection held) ────────────────────────────
        new_vectors: list[list[float]] = []
        if needs_embed:
            t0 = time.monotonic()
            new_vectors = embed_batch(
                [doc.text for doc in needs_embed], task_type="document"
            )
            elapsed = time.monotonic() - t0

            # Guard against partial API returns — a short vector list would
            # leave some docs with metadata but no stored vector.
            if len(new_vectors) != len(needs_embed):
                raise ValueError(
                    f"embed_batch returned {len(new_vectors)} vectors "
                    f"but {len(needs_embed)} docs need embedding; "
                    "aborting build to prevent vector/metadata mismatch"
                )

            log.info(
                "[dense] build: embedded %d docs in %.1fs",
                len(needs_embed),
                elapsed,
            )

        # ── Phase 3: WRITE ────────────────────────────────────────────────────
        # Re-acquire a short-lived connection just for the DB writes.
        # Using nullcontext when a bare connection was passed so we don't
        # double-close a caller-managed connection.
        write_cm = nullcontext(conn) if hasattr(db, "execute") else get_db_conn()  # type: ignore[arg-type]
        with write_cm as wconn:
            if needs_embed:
                # Upsert only the changed/new rows so unchanged cached rows are
                # preserved.
                wconn.executemany(
                    """INSERT INTO responder_corpus_embeddings
                           (corpus_id, source_type, doc_id, text)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(corpus_id) DO UPDATE SET
                           source_type = excluded.source_type,
                           doc_id      = excluded.doc_id,
                           text        = excluded.text""",
                    [
                        (d.corpus_id, d.source_type, d.doc_id, d.text)
                        for d in needs_embed
                    ],
                )
                for doc, vector in zip(needs_embed, new_vectors):
                    store_embedding(doc.corpus_id, vector, _ENTITY_TYPE, conn=wconn)

            # Remove embeddings for corpus_ids that no longer exist in the source
            # tables AND are not currently tracked by the in-memory live index.
            # Docs added via add_document() live only in responder_corpus_embeddings
            # (not in source tables), so excluding live_corpus_ids prevents build()
            # from racing with add_document() and deleting those entries.
            stale = set(existing.keys()) - current_corpus_ids - live_corpus_ids
            if stale:
                wconn.executemany(
                    "DELETE FROM responder_corpus_embeddings WHERE corpus_id = ?",
                    [(cid,) for cid in stale],
                )
                log.info("[dense] build: removed %d stale embeddings", len(stale))

        with self._lock:
            self._set_index(docs)
        return self

    def add_document(self, doc_id: str, source_type: str, text: str) -> "DenseRetriever":
        """Add or replace a single document and refresh its embedding."""
        canonical_source = _normalize_source_type(source_type)
        entry = _DocEntry(
            doc_id=str(doc_id),
            source_type=canonical_source,
            text=text or "",
            corpus_id=_corpus_id(canonical_source, str(doc_id)),
        )
        # Embed and persist before acquiring the lock so concurrent searches
        # are not blocked by I/O.
        vector = embed_text(entry.text, task_type="document")

        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO responder_corpus_embeddings (corpus_id, source_type, doc_id, text)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(corpus_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    doc_id = excluded.doc_id,
                    text = excluded.text
                """,
                (entry.corpus_id, entry.source_type, entry.doc_id, entry.text),
            )
            store_embedding(entry.corpus_id, vector, _ENTITY_TYPE, conn=conn)

        with self._lock:
            if not self._built:
                log.warning(
                    "[dense] add_document called before build() — initialising empty index"
                )
                self._set_index([])

            key = (entry.source_type, entry.doc_id)
            existing_index = self._doc_lookup.get(key)
            if existing_index is None:
                self._doc_lookup[key] = len(self._docs)
                self._docs.append(entry)
            else:
                self._docs[existing_index] = entry

        return self

    def search(self, query: str, k: int = 10, min_score: float = 0.0) -> list[dict]:
        """Return top-k corpus matches by cosine similarity."""
        if not query or not query.strip():
            return []
        if k <= 0:
            return []

        # Embed outside the lock to avoid serializing concurrent searches.
        query_vec = embed_text(query, task_type="query")

        with self._lock:
            if not self._built or not self._docs:
                return []
            by_corpus_id = {doc.corpus_id: doc for doc in self._docs}

        candidate_k = max(k * 5, 25)
        raw_hits = [
            (cid, score)
            for cid, score in query_similar(
                query_vec, k=candidate_k, entity_type=_ENTITY_TYPE
            )
            if score >= min_score
        ]

        results: list[dict] = []
        for corpus_id, score in raw_hits:
            doc = by_corpus_id.get(corpus_id)
            if not doc:
                continue
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "source_type": doc.source_type,
                    "text": doc.text,
                    "score": float(score),
                }
            )
            if len(results) >= k:
                break
        return results


_RETRIEVER = DenseRetriever()


def get_retriever() -> DenseRetriever:
    return _RETRIEVER


def build(db=None) -> DenseRetriever:
    """Build the shared dense retrieval index."""
    return get_retriever().build(db)


def add_document(doc_id: str, source_type: str, text: str) -> DenseRetriever:
    """Add a document to the shared index."""
    return get_retriever().add_document(doc_id, source_type, text)


def search(query: str, k: int = 10, min_score: float = 0.0) -> list[dict]:
    """Search the shared index."""
    return get_retriever().search(query, k=k, min_score=min_score)


__all__ = [
    "DenseRetriever",
    "build",
    "add_document",
    "search",
    "get_retriever",
]
