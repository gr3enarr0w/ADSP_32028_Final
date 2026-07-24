"""Database initialization and connection management.

Supports SQLite (local dev/testing) and PostgreSQL (production via Neon or any
Postgres-compatible service).

Backend selection:
  DATABASE_URL env var set → PostgreSQL (psycopg2)
  DATABASE_URL unset       → SQLite at DATA_DIR/jsm_data.db

The three public functions (init_db, get_db, get_db_conn) present the same
interface regardless of backend. SQL parameter style is unified:

  Positional:  ?  (SQLite-style — translated to %s for Postgres in the wrapper)
  Named:       :name  (SQLite-style — translated to %(name)s for Postgres)
  INSERT OR IGNORE → translated to INSERT ... ON CONFLICT DO NOTHING for Postgres

Callers use INSERT ... ON CONFLICT (...) DO UPDATE SET for upserts — this
syntax is supported in both SQLite 3.24+ and Postgres 9.5+.
"""

import os
import re
import logging
from contextlib import contextmanager

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
_data_dir = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_data_dir, "jsm_data.db")


def _is_postgres() -> bool:
    return bool(DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")))


# ── Postgres compatibility wrapper ───────────────────────────────────────────

class _PgConn:
    """Thin psycopg2 wrapper that mimics the sqlite3.Connection interface.

    Handles:
      - conn.execute(sql, params) / conn.executemany() / conn.executescript()
      - DictCursor so rows support both row["key"] and row[0] access
      - SQL dialect translation: ? → %s, :name → %(name)s, INSERT OR IGNORE
    """

    def __init__(self, conn):
        self._conn = conn

    @staticmethod
    def _adapt(sql: str) -> str:
        # Named params first (:name → %(name)s) to avoid double-replacement
        sql = re.sub(r":(\w+)", r"%(\1)s", sql)
        # Positional params (? → %s)
        sql = sql.replace("?", "%s")
        # INSERT OR IGNORE → INSERT + ON CONFLICT DO NOTHING
        if "INSERT OR IGNORE" in sql:
            sql = sql.replace("INSERT OR IGNORE", "INSERT")
            sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
        # SQLite datetime functions → Postgres equivalents
        sql = sql.replace("datetime('now')", "NOW()")
        sql = sql.replace("date('now')", "CURRENT_DATE")
        # SQLite DDL → Postgres equivalents (for inline CREATE TABLE in plugins)
        sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        # SQLite LIKE is case-insensitive for ASCII; Postgres LIKE is case-sensitive.
        # Replace LIKE with ILIKE so search behaviour matches across backends.
        sql = sql.replace(" LIKE ", " ILIKE ")
        # Translate PRAGMA table_info(<table>) → information_schema column query
        import re as _re
        pragma_match = _re.match(
            r"\s*PRAGMA\s+table_info\((\w+)\)\s*$", sql, _re.IGNORECASE
        )
        if pragma_match:
            table = pragma_match.group(1)
            # Emulate PRAGMA result shape: (cid, name, type, ...) so r[1] == name
            sql = (
                "SELECT ordinal_position - 1 AS cid, column_name AS name, "
                "data_type AS type, 0 AS notnull, NULL AS dflt_value, 0 AS pk "
                "FROM information_schema.columns "
                f"WHERE table_name = '{table}' AND table_schema = 'public' "
                "ORDER BY ordinal_position"
            )
        return sql

    def execute(self, sql: str, params=()):
        import psycopg2.extras
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        adapted = self._adapt(sql)
        if params:
            cur.execute(adapted, params)
        else:
            cur.execute(adapted)
        return cur

    def executemany(self, sql: str, params_seq):
        cur = self._conn.cursor()
        cur.executemany(self._adapt(sql), params_seq)
        return cur

    def executescript(self, script: str):
        cur = self._conn.cursor()
        for stmt in script.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        self._conn.commit()

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


# ── Schema templates ─────────────────────────────────────────────────────────
# __AUTO__ = SERIAL PRIMARY KEY (pg) | INTEGER PRIMARY KEY AUTOINCREMENT (sqlite)
# __NOW__  = NOW() (pg) | datetime('now') (sqlite)
# __BLOB__ = BYTEA (pg) | BLOB (sqlite)

_SCHEMA_TEMPLATE = """
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_key TEXT PRIMARY KEY,
        summary TEXT,
        description TEXT,
        status TEXT,
        resolution TEXT,
        request_type TEXT,
        affect_version TEXT,
        components TEXT,
        reporter_id TEXT,
        reporter_email TEXT,
        assignee_id TEXT DEFAULT '',
        created_at TEXT,
        resolved_at TEXT,
        updated_at TEXT,
        fetched_at TEXT,
        source TEXT,
        is_cloud INTEGER DEFAULT 0,
        is_uat_only INTEGER DEFAULT 0,
        resolution_summary TEXT,
        embedding __BLOB__
    );

    CREATE TABLE IF NOT EXISTS ticket_comments (
        comment_id TEXT,
        ticket_key TEXT,
        author_id TEXT,
        author_name TEXT,
        body TEXT,
        is_public INTEGER DEFAULT 0,
        created_at TEXT,
        UNIQUE(ticket_key, comment_id)
    );

    CREATE TABLE IF NOT EXISTS ticket_classifications (
        ticket_key TEXT PRIMARY KEY,
        category TEXT,
        issue_type TEXT,
        question_type TEXT,
        keywords TEXT,
        has_resolution INTEGER DEFAULT 0,
        resolution_summary TEXT,
        confidence REAL,
        affect_version TEXT,
        classified_at TEXT,
        model_version TEXT,
        sentiment_score REAL,
        sentiment_intensity TEXT
    );

    CREATE TABLE IF NOT EXISTS kb_coverage (
        theme TEXT PRIMARY KEY,
        category TEXT,
        ticket_count INTEGER,
        sample_issue_types TEXT,
        coverage_status TEXT,
        covered_by TEXT,
        doc_section TEXT,
        gap_detail TEXT,
        article_priority TEXT,
        assessed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS ticket_links (
        id __AUTO__,
        ticket_key TEXT NOT NULL,
        link_type TEXT NOT NULL,
        linked_key TEXT NOT NULL,
        direction TEXT NOT NULL,
        linked_summary TEXT,
        linked_status TEXT,
        linked_project TEXT,
        fetched_at TEXT,
        UNIQUE(ticket_key, linked_key, link_type)
    );

    CREATE TABLE IF NOT EXISTS generated_articles (
        id __AUTO__,
        article_topic TEXT NOT NULL,
        title TEXT NOT NULL,
        body_html TEXT NOT NULL,
        format TEXT DEFAULT 'how-to',
        source_ticket_keys TEXT,
        reference_urls TEXT,
        confluence_page_id TEXT,
        confluence_url TEXT,
        structural_fingerprint TEXT,
        semantic_embedding TEXT,
        embedding __BLOB__,
        status TEXT DEFAULT 'draft',
        generated_at TEXT DEFAULT (__NOW__),
        published_at TEXT,
        UNIQUE(article_topic)
    );

    CREATE TABLE IF NOT EXISTS slack_signals (
        id __AUTO__,
        channel TEXT,
        thread_ts TEXT,
        message_text TEXT,
        user_id TEXT,
        signal_type TEXT,
        sentiment TEXT,
        topic TEXT,
        related_tickets TEXT,
        ticket_key TEXT,
        is_resolved INTEGER DEFAULT 0,
        thread_replies TEXT,
        ingested_at TEXT DEFAULT (__NOW__),
        UNIQUE(channel, thread_ts)
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id __AUTO__,
        issue_type TEXT,
        category TEXT,
        risk_level TEXT,
        prediction TEXT,
        evidence TEXT,
        mitigation TEXT,
        uat_trend TEXT,
        open_blockers INTEGER DEFAULT 0,
        predicted_at TEXT DEFAULT (__NOW__)
    );

    CREATE TABLE IF NOT EXISTS linked_issues (
        issue_key TEXT PRIMARY KEY,
        project_key TEXT,
        summary TEXT,
        description TEXT,
        status TEXT,
        resolution TEXT,
        priority TEXT,
        issue_type TEXT,
        created_at TEXT,
        resolved_at TEXT,
        fetched_at TEXT
    );

    CREATE TABLE IF NOT EXISTS doc_improvements (
        id __AUTO__,
        theme TEXT NOT NULL,
        covered_by TEXT,
        doc_section TEXT,
        missing_topics TEXT,
        unclear_sections TEXT,
        edge_cases TEXT,
        suggested_additions TEXT,
        created_at TEXT DEFAULT (__NOW__),
        UNIQUE(theme)
    );

    CREATE TABLE IF NOT EXISTS kb_articles (
        page_id TEXT PRIMARY KEY,
        space_key TEXT,
        title TEXT,
        body_text TEXT,
        url TEXT,
        labels TEXT,
        topics_covered TEXT,
        fetched_at TEXT,
        embedding __BLOB__
    );

    CREATE TABLE IF NOT EXISTS atlassian_docs (
        url TEXT PRIMARY KEY,
        product TEXT NOT NULL,
        title TEXT,
        last_modified TEXT,
        fetched_at TEXT
    );

    CREATE TABLE IF NOT EXISTS faq_sources (
        id __AUTO__,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        title TEXT,
        content_hash TEXT,
        last_fetched TEXT,
        UNIQUE(source_type, source_id)
    );

    CREATE TABLE IF NOT EXISTS ai_draft_feedback (
        id __AUTO__,
        ticket_key TEXT NOT NULL,
        draft_comment_id TEXT NOT NULL,
        response_type TEXT NOT NULL,
        draft_customer_response TEXT NOT NULL,
        draft_admin_steps TEXT,
        actual_response TEXT,
        actual_comment_id TEXT,
        similarity_score REAL,
        feedback_category TEXT,
        agent_feedback TEXT,
        draft_mode TEXT,
        template_name TEXT,
        captured_at TEXT DEFAULT (__NOW__),
        UNIQUE(ticket_key, draft_comment_id)
    );

    CREATE TABLE IF NOT EXISTS doc_content_cache (
        url TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        fetched_at TEXT DEFAULT (__NOW__)
    );

    CREATE TABLE IF NOT EXISTS kb_cloud_scores (
        page_id TEXT PRIMARY KEY,
        title TEXT,
        url TEXT,
        cloud_applicable INTEGER DEFAULT 0,
        confidence INTEGER,
        scored_at TEXT DEFAULT (__NOW__)
    );

    CREATE TABLE IF NOT EXISTS response_examples (
        id __AUTO__,
        ticket_key TEXT NOT NULL UNIQUE,
        response_type TEXT NOT NULL,
        question_type TEXT,
        category TEXT,
        agent_response TEXT NOT NULL,
        agent_id TEXT,
        harvested_at TEXT DEFAULT (__NOW__)
    );

    CREATE TABLE IF NOT EXISTS few_shot_examples (
        example_id TEXT PRIMARY KEY,
        source_table TEXT NOT NULL,
        source_key TEXT NOT NULL,
        ticket_key TEXT,
        response_type TEXT NOT NULL,
        example_text TEXT NOT NULL,
        embedding __BLOB__,
        indexed_at TEXT DEFAULT (__NOW__),
        UNIQUE(source_table, source_key)
    );

    CREATE TABLE IF NOT EXISTS responder_corpus_embeddings (
        corpus_id TEXT PRIMARY KEY,
        source_type TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        text TEXT,
        embedding __BLOB__,
        UNIQUE(source_type, doc_id)
    );

    CREATE TABLE IF NOT EXISTS ticket_csat (
        ticket_key TEXT PRIMARY KEY,
        csat_score INTEGER,
        csat_comment TEXT,
        submitted_at TEXT,
        ingested_at TEXT DEFAULT (__NOW__)
    );


    CREATE TABLE IF NOT EXISTS job_state (
        job_name TEXT PRIMARY KEY,
        last_run_date TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS response_templates (
        template_name TEXT PRIMARY KEY,
        category TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        template_body TEXT NOT NULL,
        is_customized INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (__NOW__)
    );

    CREATE TABLE IF NOT EXISTS anomaly_baseline (
        segment TEXT PRIMARY KEY,
        rolling_mean REAL,
        rolling_std REAL,
        computed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS anomaly_scores (
        id __AUTO__,
        scored_at TEXT NOT NULL,
        segment TEXT,
        window_minutes INTEGER,
        ticket_count INTEGER,
        is_anomaly INTEGER DEFAULT 0,
        zscore REAL,
        category_breakdown TEXT
    );

    CREATE TABLE IF NOT EXISTS category_anomaly_baselines (
        category TEXT NOT NULL,
        segment TEXT NOT NULL,
        rolling_mean REAL NOT NULL,
        rolling_std REAL NOT NULL,
        computed_at TEXT NOT NULL,
        PRIMARY KEY (category, segment)
    );

    CREATE INDEX IF NOT EXISTS idx_category_anomaly_baselines_category
        ON category_anomaly_baselines(category);

    CREATE TABLE IF NOT EXISTS ticket_clusters (
        cluster_id INTEGER NOT NULL,
        run_date TEXT NOT NULL,
        run_type TEXT DEFAULT 'incident',
        window_days INTEGER DEFAULT 0,
        ticket_keys TEXT,
        label TEXT,
        size INTEGER,
        is_new INTEGER DEFAULT 0,
        growth_rate REAL,
        PRIMARY KEY (cluster_id, run_date, run_type)
    );

    CREATE TABLE IF NOT EXISTS category_csat_correlations (
        category TEXT NOT NULL,
        run_date TEXT NOT NULL,
        n_samples INTEGER,
        spearman_r __DOUBLE__,
        spearman_p __DOUBLE__,
        spearman_p_corrected __DOUBLE__,
        pearson_r __DOUBLE__,
        ci_lower __DOUBLE__,
        ci_upper __DOUBLE__,
        cv_variance __DOUBLE__,
        mean_csat __DOUBLE__,
        std_csat __DOUBLE__,
        acceptance_rate __DOUBLE__,
        mean_similarity __DOUBLE__,
        PRIMARY KEY (category, run_date)
    );

    CREATE TABLE IF NOT EXISTS retrieval_quality_log (
        id __AUTO__,
        run_date TEXT NOT NULL,
        mrr_bm25 __DOUBLE__,
        mrr_dense __DOUBLE__,
        mrr_rrf __DOUBLE__,
        mrr_weighted __DOUBLE__,
        mrr_learned __DOUBLE__,
        n_queries INTEGER,
        recorded_at TEXT DEFAULT (__NOW__),
        UNIQUE(run_date)
    );

    CREATE INDEX IF NOT EXISTS idx_ticket_comments_ticket_key ON ticket_comments(ticket_key);
    CREATE INDEX IF NOT EXISTS idx_response_templates_category ON response_templates(category, issue_type);
    CREATE INDEX IF NOT EXISTS idx_ai_draft_feedback_ticket_key ON ai_draft_feedback(ticket_key);
    CREATE INDEX IF NOT EXISTS idx_ticket_classifications_ticket_key ON ticket_classifications(ticket_key);
    CREATE INDEX IF NOT EXISTS idx_generated_articles_status ON generated_articles(status);
    CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at);
    CREATE INDEX IF NOT EXISTS idx_category_csat_correlations_run_date ON category_csat_correlations(run_date);
    CREATE INDEX IF NOT EXISTS idx_retrieval_quality_log_run_date ON retrieval_quality_log(run_date);
"""


def _schema_sql(dialect: str) -> str:
    if dialect == "postgres":
        return (
            _SCHEMA_TEMPLATE
            .replace("__AUTO__", "SERIAL PRIMARY KEY")
            .replace("__NOW__", "NOW()")
            .replace("__BLOB__", "BYTEA")
            .replace("__DOUBLE__", "DOUBLE PRECISION")
        )
    return (
        _SCHEMA_TEMPLATE
        .replace("__AUTO__", "INTEGER PRIMARY KEY AUTOINCREMENT")
        .replace("__NOW__", "datetime('now')")
        .replace("__BLOB__", "BLOB")
        .replace("__DOUBLE__", "REAL")
    )


# ── init_db ──────────────────────────────────────────────────────────────────

def _init_postgres():
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    for stmt in _schema_sql("postgres").split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)
    # Migration additions — idempotent via IF NOT EXISTS / IF EXISTS
    migrations = [
        "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS is_cloud INTEGER DEFAULT 0",
        "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS is_uat_only INTEGER DEFAULT 0",
        "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS assignee_id TEXT DEFAULT ''",
        "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS embedding BYTEA",
        "ALTER TABLE tickets DROP COLUMN IF EXISTS assignee",
        "ALTER TABLE ai_draft_feedback ADD COLUMN IF NOT EXISTS agent_feedback TEXT",
        "ALTER TABLE slack_signals ADD COLUMN IF NOT EXISTS is_resolved INTEGER DEFAULT 0",
        "ALTER TABLE slack_signals ADD COLUMN IF NOT EXISTS thread_replies TEXT",
        "ALTER TABLE atlassian_docs DROP COLUMN IF EXISTS body_text",
        "ALTER TABLE atlassian_docs DROP COLUMN IF EXISTS content_hash",
        "ALTER TABLE generated_articles ADD COLUMN IF NOT EXISTS structural_fingerprint TEXT",
        "ALTER TABLE generated_articles ADD COLUMN IF NOT EXISTS semantic_embedding TEXT",
        "ALTER TABLE generated_articles ADD COLUMN IF NOT EXISTS embedding BYTEA",
        "ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS embedding BYTEA",
        "ALTER TABLE ticket_classifications ADD COLUMN IF NOT EXISTS sentiment_score REAL",
        "ALTER TABLE ticket_classifications ADD COLUMN IF NOT EXISTS sentiment_intensity TEXT",
        "ALTER TABLE ai_draft_feedback ADD COLUMN IF NOT EXISTS draft_mode TEXT",
        "ALTER TABLE ai_draft_feedback ADD COLUMN IF NOT EXISTS template_name TEXT",
        "ALTER TABLE response_templates ADD COLUMN IF NOT EXISTS is_customized INTEGER DEFAULT 0",
        "ALTER TABLE ticket_clusters ADD COLUMN IF NOT EXISTS run_type TEXT DEFAULT 'incident'",
        "ALTER TABLE ticket_clusters ADD COLUMN IF NOT EXISTS window_days INTEGER DEFAULT 0",
        "ALTER TABLE ticket_clusters DROP CONSTRAINT IF EXISTS ticket_clusters_pkey",
        "ALTER TABLE ticket_clusters ADD PRIMARY KEY (cluster_id, run_date, run_type)",
        """CREATE TABLE IF NOT EXISTS category_anomaly_baselines (
            category TEXT NOT NULL,
            segment TEXT NOT NULL,
            rolling_mean REAL NOT NULL,
            rolling_std REAL NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (category, segment)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_category_anomaly_baselines_category ON category_anomaly_baselines(category)",
        """CREATE TABLE IF NOT EXISTS retrieval_quality_log (
            id SERIAL PRIMARY KEY,
            run_date TEXT NOT NULL,
            mrr_bm25 DOUBLE PRECISION,
            mrr_dense DOUBLE PRECISION,
            mrr_rrf DOUBLE PRECISION,
            mrr_weighted DOUBLE PRECISION,
            mrr_learned DOUBLE PRECISION,
            n_queries INTEGER,
            recorded_at TEXT DEFAULT NOW(),
            UNIQUE(run_date)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_retrieval_quality_log_run_date ON retrieval_quality_log(run_date)",
        "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS resolution_summary TEXT",
    ]
    for m in migrations:
        try:
            cur.execute(m)
        except Exception as exc:
            log.debug("Migration step skipped: %s — %s", m, exc)
            conn.rollback()
    conn.commit()
    conn.close()
    log.info("Postgres database initialized via %s", DATABASE_URL.split("@")[-1])


def _init_sqlite():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_schema_sql("sqlite"))

    # Idempotent column additions
    def _add_col_if_missing(table, col, defn):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")

    def _drop_col_if_present(table, col):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col in cols:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")

    _add_col_if_missing("tickets", "is_cloud", "INTEGER DEFAULT 0")
    _add_col_if_missing("tickets", "is_uat_only", "INTEGER DEFAULT 0")
    _add_col_if_missing("tickets", "assignee_id", "TEXT DEFAULT ''")
    _drop_col_if_present("tickets", "assignee")
    _add_col_if_missing("ai_draft_feedback", "agent_feedback", "TEXT")
    _add_col_if_missing("slack_signals", "is_resolved", "INTEGER DEFAULT 0")
    _add_col_if_missing("slack_signals", "thread_replies", "TEXT")
    _drop_col_if_present("atlassian_docs", "body_text")
    _drop_col_if_present("atlassian_docs", "content_hash")
    _add_col_if_missing("generated_articles", "structural_fingerprint", "TEXT")
    _add_col_if_missing("generated_articles", "semantic_embedding", "TEXT")
    for table in ("tickets", "kb_articles", "generated_articles"):
        _add_col_if_missing(table, "embedding", "BLOB")
    _add_col_if_missing("ticket_classifications", "sentiment_score", "REAL")
    _add_col_if_missing("ticket_classifications", "sentiment_intensity", "TEXT")
    _add_col_if_missing("ai_draft_feedback", "draft_mode", "TEXT")
    _add_col_if_missing("ai_draft_feedback", "template_name", "TEXT")
    _add_col_if_missing("response_templates", "is_customized", "INTEGER DEFAULT 0")
    _add_col_if_missing("tickets", "resolution_summary", "TEXT")
    _add_col_if_missing("ticket_clusters", "run_type", "TEXT DEFAULT 'incident'")
    _add_col_if_missing("ticket_clusters", "window_days", "INTEGER DEFAULT 0")
    cluster_pk = [
        row[1]
        for row in conn.execute("PRAGMA table_info(ticket_clusters)").fetchall()
        if row[5] > 0
    ]
    if cluster_pk != ["cluster_id", "run_date", "run_type"]:
        conn.execute("ALTER TABLE ticket_clusters RENAME TO ticket_clusters_old")
        conn.execute(
            """
            CREATE TABLE ticket_clusters (
                cluster_id INTEGER NOT NULL,
                run_date TEXT NOT NULL,
                run_type TEXT DEFAULT 'incident',
                window_days INTEGER DEFAULT 0,
                ticket_keys TEXT,
                label TEXT,
                size INTEGER,
                is_new INTEGER DEFAULT 0,
                growth_rate REAL,
                PRIMARY KEY (cluster_id, run_date, run_type)
            )
            """
        )
        # Determine which columns the old table has so we can preserve window_days
        # if it was already present (idempotent re-migrations).
        old_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(ticket_clusters_old)").fetchall()
        }
        window_days_expr = "window_days" if "window_days" in old_cols else "0"
        conn.execute(
            f"""
            INSERT INTO ticket_clusters (
                cluster_id,
                run_date,
                run_type,
                window_days,
                ticket_keys,
                label,
                size,
                is_new,
                growth_rate
            )
            SELECT
                cluster_id,
                run_date,
                COALESCE(run_type, 'incident'),
                {window_days_expr},
                ticket_keys,
                label,
                size,
                is_new,
                growth_rate
            FROM ticket_clusters_old
            """
        )
        conn.execute("DROP TABLE ticket_clusters_old")

    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    conn.close()
    log.info("SQLite database initialized: %s", DB_PATH)


def init_db():
    """Create tables and apply migrations for the configured backend."""
    if _is_postgres():
        _init_postgres()
    else:
        _init_sqlite()


# ── get_db / get_db_conn ─────────────────────────────────────────────────────

def get_db():
    """Return an open database connection.

    For Postgres: returns a _PgConn wrapper (psycopg2 connection).
    For SQLite:   returns a sqlite3.Connection with row_factory=sqlite3.Row.

    Callers are responsible for closing the connection.  Prefer get_db_conn()
    (the context manager) for automatic commit/rollback/close.
    """
    if _is_postgres():
        import psycopg2
        return _PgConn(psycopg2.connect(DATABASE_URL))
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def get_last_run_date(conn, job_name: str):
    """Return the last run date for a job_state entry, or None."""
    import datetime as _dt
    row = conn.execute(
        "SELECT last_run_date FROM job_state WHERE job_name = ?",
        (job_name,),
    ).fetchone()
    if row is None:
        return None
    return _dt.date.fromisoformat(row["last_run_date"])


def set_last_run_date(conn, job_name: str, run_date) -> None:
    """Upsert the last run date for a job_state entry."""
    conn.execute(
        """
        INSERT INTO job_state (job_name, last_run_date) VALUES (?, ?)
        ON CONFLICT(job_name) DO UPDATE SET last_run_date = excluded.last_run_date
        """,
        (job_name, run_date.isoformat()),
    )


@contextmanager
def get_db_conn():
    """Context manager for database connections.

    Auto-commits on clean exit, rolls back on exception, always closes.
    Callers can still call conn.commit() manually for mid-block commits.
    """
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
