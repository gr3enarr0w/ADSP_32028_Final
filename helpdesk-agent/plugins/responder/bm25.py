"""In-memory BM25 retrieval index for responder lookups."""

from __future__ import annotations

import logging
import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass

from db import get_db_conn
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
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


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace/punctuation while keeping acronyms intact."""
    return _TOKEN_RE.findall((text or "").lower())


def _join_text(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


@dataclass(slots=True)
class _DocEntry:
    doc_id: str
    source_type: str
    text: str
    tokens: list[str]


class BM25Index:
    """Simple in-memory BM25 corpus for responder retrieval."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: list[_DocEntry] = []
        self._doc_lookup: dict[tuple[str, str], int] = {}
        self._bm25: BM25Okapi | None = None
        self._built = False

    def _rebuild_model(self) -> None:
        if not self._docs:
            self._bm25 = None
            self._built = True
            return

        tokenized = [doc.tokens for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        self._built = True

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
                text = text_fn(row)
                docs.append(
                    _DocEntry(
                        doc_id=str(doc_id_fn(row)),
                        source_type=source_type,
                        text=text,
                        tokens=_tokenize(text),
                    )
                )

        log.info(
            "[bm25] source counts faq_sources=%d kb_articles=%d tickets=%d atlassian_docs=%d total=%d",
            counts.get("faq_sources", 0),
            counts.get("kb_articles", 0),
            counts.get("tickets", 0),
            counts.get("atlassian_docs", 0),
            len(docs),
        )
        return docs

    def build(self, db=None) -> "BM25Index":
        """Load all corpus documents from the DB and rebuild the index."""
        with self._lock:
            if db is None:
                conn_cm = get_db_conn()
            elif hasattr(db, "execute"):
                conn_cm = nullcontext(db)
            elif hasattr(db, "get_db_conn"):
                conn_cm = db.get_db_conn()
            else:
                raise TypeError("db must be a connection or expose get_db_conn()")

            with conn_cm as conn:
                self._docs = self._load_from_connection(conn)
                self._doc_lookup = {
                    (doc.source_type, doc.doc_id): i for i, doc in enumerate(self._docs)
                }
                self._rebuild_model()
            return self

    def add_document(self, doc_id: str, source_type: str, text: str) -> "BM25Index":
        """Add or replace a single document and rebuild the index."""
        canonical_source = _normalize_source_type(source_type)
        entry = _DocEntry(
            doc_id=str(doc_id),
            source_type=canonical_source,
            text=text or "",
            tokens=_tokenize(text),
        )

        with self._lock:
            if not self._built:
                self._rebuild_model()
            key = (entry.source_type, entry.doc_id)
            existing_index = self._doc_lookup.get(key)
            if existing_index is None:
                self._doc_lookup[key] = len(self._docs)
                self._docs.append(entry)
            else:
                self._docs[existing_index] = entry
            self._rebuild_model()
            return self

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return the top-k BM25 matches for a query."""
        if not query or not query.strip():
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        with self._lock:
            if not self._built:
                try:
                    self._rebuild_model()
                except Exception:
                    log.exception("BM25 lazy rebuild failed during search — returning empty results")
                    return []
            if not self._bm25 or not self._docs:
                return []

            scores = self._bm25.get_scores(tokens)
            ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
            return [
                {
                    "doc_id": self._docs[idx].doc_id,
                    "source_type": self._docs[idx].source_type,
                    "text": self._docs[idx].text,
                    "score": float(scores[idx]),
                }
                for idx in ranked
                if scores[idx] > 0
            ]


_INDEX = BM25Index()


def get_index() -> BM25Index:
    return _INDEX


def build(db=None) -> BM25Index:
    """Build the shared in-memory BM25 index."""
    return get_index().build(db)


def add_document(doc_id: str, source_type: str, text: str) -> BM25Index:
    """Add a document to the shared index and rebuild it."""
    return get_index().add_document(doc_id, source_type, text)


def search(query: str, k: int = 10) -> list[dict]:
    """Search the shared index, rebuilding lazily if not yet built."""
    return get_index().search(query, k=k)


__all__ = [
    "BM25Index",
    "build",
    "add_document",
    "search",
    "get_index",
    "_tokenize",
]
