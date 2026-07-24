#!/usr/bin/env python3
"""Backfill embedding vectors for all tickets, articles, and KB articles.

Iterates records missing embeddings, calls embed_text()/embed_batch(),
and stores vectors via the vector storage layer.

Usage:
    python -m scripts.backfill_vectors          # all entity types
    python -m scripts.backfill_vectors ticket    # tickets only
    python -m scripts.backfill_vectors article   # generated articles only
    python -m scripts.backfill_vectors kb_article  # KB articles only
"""

import logging
import sys

from db import get_db_conn, init_db
from services.embedding import embed_batch
from services.vector_store import store_embedding

log = logging.getLogger(__name__)

_ENTITY_CONFIGS = {
    "ticket": {
        "table": "tickets",
        "id_col": "ticket_key",
        "text_cols": ("summary", "description"),
    },
    "article": {
        "table": "generated_articles",
        "id_col": "article_topic",
        "text_cols": ("title", "body_html"),
    },
    "kb_article": {
        "table": "kb_articles",
        "id_col": "page_id",
        "text_cols": ("title", "body_text"),
    },
}

BATCH_SIZE = 64


def _build_text(row, text_cols: tuple[str, ...]) -> str:
    parts = [str(row[c] or "") for c in text_cols]
    return " ".join(p for p in parts if p).strip()


def backfill(entity_type: str) -> int:
    cfg = _ENTITY_CONFIGS[entity_type]
    table, id_col, text_cols = cfg["table"], cfg["id_col"], cfg["text_cols"]

    with get_db_conn() as conn:
        rows = conn.execute(
            f"SELECT {id_col}, {', '.join(text_cols)} "
            f"FROM {table} WHERE embedding IS NULL"
        ).fetchall()

    if not rows:
        log.info("[%s] No records need backfill", entity_type)
        return 0

    log.info("[%s] Backfilling %d records", entity_type, len(rows))
    stored = 0

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        texts = [_build_text(r, text_cols) for r in batch]
        ids = [str(r[id_col]) for r in batch]

        non_empty = [(i, t) for i, t in zip(ids, texts) if t]
        if not non_empty:
            continue

        batch_ids, batch_texts = zip(*non_empty)
        vectors = embed_batch(list(batch_texts), task_type="document")

        for eid, vec in zip(batch_ids, vectors):
            store_embedding(eid, vec, entity_type)
            stored += 1

        log.info("[%s] %d / %d", entity_type, min(start + BATCH_SIZE, len(rows)), len(rows))

    log.info("[%s] Backfilled %d records", entity_type, stored)
    return stored


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()

    types = sys.argv[1:] if len(sys.argv) > 1 else list(_ENTITY_CONFIGS)
    total = 0
    for entity_type in types:
        if entity_type not in _ENTITY_CONFIGS:
            log.error("Unknown entity type: %s (valid: %s)", entity_type, list(_ENTITY_CONFIGS))
            sys.exit(1)
        total += backfill(entity_type)
    log.info("Total backfilled: %d", total)


if __name__ == "__main__":
    main()
