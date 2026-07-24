"""Migrate all data from SQLite (jsm_data.db) to Postgres (DATABASE_URL).

Usage:
    DATABASE_URL=postgres://... python scripts/migrate_sqlite_to_postgres.py

Requires psycopg2-binary and the DATABASE_URL env var to be set.
Idempotent — uses ON CONFLICT DO NOTHING for all inserts, so safe to re-run.
"""

import os
import sys
import sqlite3
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        sys.exit("DATABASE_URL is not set — point it at the Neon (or any Postgres) connection string")

    data_dir = os.getenv("DATA_DIR", _ROOT)
    sqlite_path = os.path.join(data_dir, "jsm_data.db")
    if not os.path.exists(sqlite_path):
        sys.exit(f"SQLite database not found at {sqlite_path}")

    import psycopg2
    import psycopg2.extras

    from db import init_db

    log.info("Initializing Postgres schema at %s", database_url.split("@")[-1])
    init_db()

    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    dst = psycopg2.connect(database_url)
    dst.autocommit = False
    cur = dst.cursor()

    # Tables in dependency order (referenced tables first)
    tables = [
        "tickets",
        "ticket_comments",
        "ticket_classifications",
        "ticket_links",
        "linked_issues",
        "kb_coverage",
        "generated_articles",
        "slack_signals",
        "predictions",
        "doc_improvements",
        "kb_articles",
        "atlassian_docs",
        "faq_sources",
        "ai_draft_feedback",
        "doc_content_cache",
        "kb_cloud_scores",
        "response_examples",
        "ticket_csat",
        "job_state",
        "response_templates",
        "anomaly_baseline",
        "anomaly_scores",
        "ticket_clusters",
        "category_csat_correlations",
        "few_shot_examples",
        "responder_corpus_embeddings",
    ]

    total_rows = 0
    for table in tables:
        try:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
        except sqlite3.OperationalError as e:
            log.warning("Skipping %s: %s", table, e)
            continue

        if not rows:
            log.info("  %s: 0 rows (empty)", table)
            continue

        cols = rows[0].keys()
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(cols)
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT DO NOTHING"
        )

        batch = [tuple(r[c] for c in cols) for r in rows]
        psycopg2.extras.execute_batch(cur, sql, batch, page_size=500)
        dst.commit()
        log.info("  %s: %d rows migrated", table, len(rows))
        total_rows += len(rows)

    src.close()
    dst.close()
    log.info("Migration complete — %d total rows", total_rows)

    # Verify
    dst2 = psycopg2.connect(database_url)
    cur2 = dst2.cursor()
    log.info("Row count verification:")
    for table in tables:
        try:
            cur2.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur2.fetchone()[0]
            log.info("  %s: %d", table, count)
        except Exception:
            pass
    dst2.close()


if __name__ == "__main__":
    main()
