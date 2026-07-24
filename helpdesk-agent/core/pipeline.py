"""Pipeline orchestrator — loads pipeline.yaml, coordinates plugin hooks.

Extracted from main.py to support a plugin architecture.  main.py keeps
its FastAPI routes and startup glue; this module owns the pipeline
lifecycle: config loading, scheduling, locking, and phase dispatch.

Usage from main.py
──────────────────
    from core.pipeline import (
        load_pipeline_config,
        trigger_pipeline_async,
        pipeline_scheduler,
        pipeline_status,
    )

    @app.on_event("startup")
    def startup():
        load_pipeline_config()           # reads pipeline.yaml (once)
        trigger_pipeline_async("startup") # kick off first cycle
        threading.Thread(target=pipeline_scheduler, daemon=True).start()
"""

# NOTE: _phase_export still imports from main.py at call time (deferred import).
# Importing main triggers FastAPI app creation, so those imports must stay deferred.

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import yaml  # PyYAML — already in requirements (transitive via several deps)

log = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────────────

_config: dict[str, Any] | None = None
_pipeline_lock = threading.Lock()
_last_pipeline_run: str | None = None
_last_pipeline_trigger: str | None = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Config loading ──────────────────────────────────────────────────


def _env_override(section: str, key: str, default: Any) -> Any:
    """Allow env vars to override any pipeline.yaml value.

    Convention:  PIPELINE_<SECTION>_<KEY>  (uppercased)
    e.g.  PIPELINE_SCHEDULE_REFRESH_MINUTES=10

    For lists, the env value is comma-separated.
    For bools, accepts 1/true/yes (case-insensitive).
    """
    env_key = f"PIPELINE_{section}_{key}".upper()
    env_val = os.getenv(env_key)
    if env_val is None:
        return default

    # Coerce to the same type as the default
    if isinstance(default, bool):
        return env_val.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(env_val)
    if isinstance(default, float):
        return float(env_val)
    if isinstance(default, list):
        return [v.strip() for v in env_val.split(",") if v.strip()]
    return env_val


def load_pipeline_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline.yaml (or env-specified override) and cache it.

    The config is loaded once per process.  Re-call with a path to force
    a reload (useful in tests).
    """
    global _config

    if _config is not None and path is None:
        return _config

    if path is None:
        path = os.getenv("PIPELINE_CONFIG", _PROJECT_ROOT / "pipeline.yaml")
    path = Path(path)

    if not path.exists():
        log.warning("pipeline.yaml not found at %s — using defaults", path)
        _config = {}
        return _config

    with open(path) as f:
        _config = yaml.safe_load(f) or {}

    log.info("Loaded pipeline config from %s", path)
    return _config


def get_config() -> dict[str, Any]:
    """Return the cached pipeline config (loads if needed)."""
    if _config is None:
        load_pipeline_config()
    return _config  # type: ignore[return-value]


def get_plugin_config(plugin_name: str) -> dict[str, Any]:
    """Return config dict for a specific plugin, with env overrides applied."""
    cfg = get_config()
    plugin_cfg: dict[str, Any] = (cfg.get("plugins") or {}).get(plugin_name) or {}

    # Apply env overrides for top-level keys
    result = {}
    for key, value in plugin_cfg.items():
        result[key] = _env_override(plugin_name, key, value)
    return result


def is_plugin_enabled(plugin_name: str) -> bool:
    """Check whether a plugin is enabled in the config."""
    cfg = get_plugin_config(plugin_name)
    return cfg.get("enabled", True)


def get_model(task: str) -> str:
    """Resolve which Gemini model to use for a given task.

    Falls back through:  models.<task>  →  models.base  →  GEMINI_MODEL env
    """
    cfg = get_config()
    models = cfg.get("models") or {}
    model = models.get(task, "")
    if model:
        return model
    base = models.get("base", "")
    if base:
        return base
    import config as _cfg
    return os.getenv("GEMINI_MODEL", _cfg.GEMINI_MODEL_GENERATION)


def get_schedule_config() -> dict[str, Any]:
    """Return schedule section with env overrides."""
    cfg = get_config()
    schedule = dict((cfg.get("schedule") or {}))
    schedule.setdefault("refresh_minutes", 5)
    schedule.setdefault("run_on_startup", True)
    schedule["refresh_minutes"] = int(
        _env_override("schedule", "refresh_minutes", schedule["refresh_minutes"])
    )
    schedule["run_on_startup"] = _env_override(
        "schedule", "run_on_startup", schedule["run_on_startup"]
    )
    return schedule


# ── Pipeline execution ──────────────────────────────────────────────

# Registry of plugin hooks.  Each entry is (name, callable).
# Plugins register themselves at import time via ``register_plugin``.
_plugin_hooks: list[tuple[str, Any]] = []

# Built-in phase order — maps plugin names to their on_schedule callables.
# Populated by ``_register_builtin_phases`` on first run.
_builtins_registered = False


def register_plugin(name: str, hook: Any) -> None:
    """Register a plugin's ``on_schedule()`` hook.

    Plugins should call this at module level::

        from core.pipeline import register_plugin
        register_plugin("my_plugin", my_on_schedule_func)
    """
    _plugin_hooks.append((name, hook))
    log.debug("Registered pipeline plugin: %s", name)


def _register_builtin_phases() -> None:
    """Lazily register the 10 built-in pipeline phases.

    Imports are deferred so this module can be imported early without
    pulling in the entire dependency tree.
    """
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    # We don't import the actual module functions here — instead we
    # define thin wrappers that respect the enabled flag and call the
    # existing code paths.  This keeps main.py's functions unchanged
    # while the orchestrator controls sequencing.

    register_plugin("ingest", _phase_ingest)
    register_plugin("analysis", _phase_analysis)
    register_plugin("resolution_summary", _phase_resolution_summary)
    # NOTE: feedback runs immediately after resolution_summary and BEFORE faq/responder
    # phases.  sentiment_intensity is therefore not available to the responder on the
    # same pipeline cycle as initial classification — it's available on the NEXT cycle
    # (or via on_ticket dispatch).  The on_ticket("classified") dispatch in
    # plugins/__init__.py provides same-cycle scoring for webhook-triggered tickets.
    # The phase ordering is intentional for batch sweeps.
    register_plugin("feedback", _phase_feedback)
    register_plugin("faq", _phase_faq)
    register_plugin("kb_index", _phase_kb_index)  # Confluence KB → kb_articles + embeddings (weekly, per-instance)
    register_plugin("responder", _phase_responder)
    register_plugin("export", _phase_export)
    register_plugin("alerting", _phase_alerting)
    register_plugin("mrr_monitor", _phase_mrr_monitor)  # Weekly MRR quality snapshot (ANTSE-450)


# ── Phase implementations (thin wrappers around existing code) ──────


def _phase_ingest() -> None:
    """Phase 1: Ingest tickets from Jira Cloud + scrub PII.

    Incremental strategy:
    - On first run (no ``job_state`` entry): performs a full JSM fetch via the
      servicedeskapi endpoint to capture all historical tickets.
    - On subsequent runs: uses JQL ``updated >= last_run_date`` so only tickets
      changed since the previous cycle are fetched, keeping each 5-minute cycle
      fast regardless of corpus size.
    - The ``last_run_date`` is recorded at the start of each cycle (not the end)
      so any ticket updated during the run is picked up on the next cycle.
    """
    import datetime as _dt

    cfg = get_plugin_config("ingest")

    from ingest.tickets import fetch_tickets_cloud, fetch_comments_cloud
    from db import get_db_conn, get_last_run_date, set_last_run_date

    project_keys = cfg.get("project_keys", ["<PROJECT_KEY>"])
    version = cfg.get("default_affect_version") or None

    # Capture today's date before fetching so any ticket updated mid-run is
    # caught on the next cycle rather than silently skipped.
    today = _dt.date.today()

    total = 0
    newly_ingested_keys: list[str] = []
    for project_key in project_keys:
        try:
            from ingest.oauth2lo import get_cloud_base_url as _get_base
            _cloud_id = _get_base("jsm").rstrip("/").split("/")[-1]
        except Exception as e:
            log.warning("Could not resolve cloud_id for ingest, using 'default': %s", e)
            _cloud_id = "default"
        job_name = f"ingest:{_cloud_id}:{project_key}"
        with get_db_conn() as conn:
            last_run = get_last_run_date(conn, job_name)
        if last_run is None:
            log.info(
                "No prior run recorded for %s — running full JSM sync.", project_key
            )
        else:
            log.info(
                "Last run for %s was %s — running incremental JQL sync.", project_key, last_run
            )
        fetched, keys = fetch_tickets_cloud(project_key, version, last_run_date=last_run)
        total += fetched
        newly_ingested_keys.extend(keys)
        # Persist the run date after a successful fetch.
        with get_db_conn() as conn:
            set_last_run_date(conn, job_name, today)

    log.info("Ingested %d tickets", total)

    # Fetch comments only for tickets touched in this cycle — avoids re-fetching
    # comments for all 8 000+ tickets on every pipeline run.
    if newly_ingested_keys:
        fetch_comments_cloud(newly_ingested_keys)

    # Scrub PII
    if cfg.get("scrub_pii", True):
        from ingest.scrubber import scrub_database
        scrub_result = scrub_database(dry_run=False)
        scrubbed = scrub_result["scrubbed_tickets"] + scrub_result["scrubbed_comments"]
        anon = scrub_result["anon_reporters"] + scrub_result["anon_authors"]
        if scrubbed or anon:
            log.info("Scrubbed %d text fields, anonymized %d identity fields", scrubbed, anon)


def _phase_analysis() -> None:
    """Phase 2: Classify unclassified tickets + re-classify newly resolved."""
    cfg = get_plugin_config("analysis")
    from analysis.classifier import classify_unclassified
    from db import get_db_conn

    ingest_cfg = get_plugin_config("ingest")
    version = ingest_cfg.get("default_affect_version") or None

    if cfg.get("reclassify_newly_resolved", True):
        with get_db_conn() as conn:
            stale = conn.execute("""
                SELECT COUNT(*) FROM tickets t
                JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
                WHERE t.resolution IS NOT NULL
                  AND (c.has_resolution = 0 OR c.has_resolution IS NULL)
            """).fetchone()[0]
            if stale > 0:
                conn.execute("""
                    DELETE FROM ticket_classifications WHERE ticket_key IN (
                        SELECT t.ticket_key FROM tickets t
                        JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
                        WHERE t.resolution IS NOT NULL
                          AND (c.has_resolution = 0 OR c.has_resolution IS NULL)
                    )
                """)
                log.info("Marked %d newly-resolved tickets for re-classification", stale)

    classified, errors = classify_unclassified(version)
    if classified:
        log.info("Classified %d tickets (%d errors)", classified, errors)


def _phase_resolution_summary() -> None:
    """Phase 2b: Backfill resolution_summary for resolved/closed tickets via Gemini.

    Cap is controlled by resolution_summary.max_tickets_per_cycle (default 50).
    Results are written to ticket_classifications.resolution_summary only — nothing
    is posted back to Jira.

    The connection is opened and closed manually so that per-row commits inside
    backfill_resolution_summaries work as intended.  Using ``get_db_conn()`` as a
    context manager would suppress the per-row commits and break the rollback
    contract on early exit.
    """
    from analysis.resolution_summary import backfill_resolution_summaries
    from db import get_db

    conn = get_db()
    try:
        rs_cfg = get_plugin_config("resolution_summary")
        max_tickets = rs_cfg.get("max_tickets_per_cycle", 50)
        generated = backfill_resolution_summaries(conn, max_tickets=max_tickets)
    finally:
        conn.close()
    if generated:
        log.info("Resolution summary: generated %d new summaries", generated)


def _phase_faq() -> None:
    """Phase 3: FAQ pipeline — analyze gaps, generate entries, publish to Confluence.

    Gap analysis + generation use a hybrid gate (industry best practice):
      - TTL gate: skip if last run < run_interval_hours ago (default 168h / 7 days)
      - Event gate: override TTL and run early if >= new_tickets_threshold new
        resolved/closed tickets have appeared since the last run
    Publishing runs every cycle (idempotent — only touches articles with no confluence_page_id).
    """
    import datetime as _dt

    from db import get_db_conn, get_last_run_date, set_last_run_date
    from generation.publisher import publish_article
    from ingest.oauth2lo import get_cloud_base_url as _get_base

    # ── Hybrid gate: weekend cron (Saturday UTC) + event-driven on new resolved tickets ──
    # TTL: run on Saturday UTC (= Friday midnight Eastern / Saturday 05:00 UTC).
    # Rationale: low-traffic weekend window, weekly cadence matches KB index refresh.
    now_utc = _dt.datetime.utcnow()
    today = now_utc.date()
    is_saturday_utc = now_utc.weekday() == 5  # Monday=0 … Saturday=5

    try:
        _cloud_id = _get_base("jsm").rstrip("/").split("/")[-1]
    except Exception as e:
        log.warning("Could not resolve cloud_id for faq_gap_analysis, using 'default': %s", e)
        _cloud_id = "default"
    _job_key = f"faq_gap_analysis:{_cloud_id}"

    faq_cfg = get_plugin_config("faq")
    gap_cfg = faq_cfg.get("gap_analysis", {})
    new_tickets_threshold = gap_cfg.get("new_tickets_threshold", 50)

    with get_db_conn() as conn:
        last_run = get_last_run_date(conn, _job_key)

        # Event gate: count resolved/closed tickets added since last run
        new_resolved = 0
        if last_run is not None:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM tickets
                WHERE LOWER(COALESCE(status, '')) IN ('resolved', 'closed')
                AND resolved_at >= %s
                """,
                (last_run.isoformat(),),
            ).fetchone()
            new_resolved = row[0] if row else 0

    # TTL gate: run if it's Saturday UTC AND hasn't run this week
    already_ran_this_week = last_run is not None and (today - last_run).days < 7
    ttl_elapsed = is_saturday_utc and not already_ran_this_week
    # First run (no last_run) fires immediately regardless of day
    if last_run is None:
        ttl_elapsed = True
    event_triggered = new_resolved >= new_tickets_threshold

    run_gap_analysis = ttl_elapsed or event_triggered
    if not run_gap_analysis:
        log.debug(
            "FAQ gap analysis: not Saturday UTC (weekday=%d) or ran %s, %d new resolved (threshold %d) — skipping",
            now_utc.weekday(),
            last_run or "never",
            new_resolved,
            new_tickets_threshold,
        )
    else:
        from faq.sources import gather_all_sources
        from faq.analyzer import analyze_faq_gaps
        from faq.generator import generate_all_faq_entries

        sources = gather_all_sources()
        gaps = analyze_faq_gaps(sources)
        missing = [g for g in gaps if g.get("coverage_status") in ("missing", "partial")]
        log.info("Gap analysis: %d themes, %d missing/partial", len(gaps), len(missing))

        if missing:
            try:
                generated, gen_errors = generate_all_faq_entries(sources)
                log.info("Generated %d FAQ entries (%d errors)", generated, gen_errors)
            except Exception as e:
                log.error("FAQ generation failed — not updating last_run_date, will retry next cycle: %s", e)
                return

        with get_db_conn() as conn:
            set_last_run_date(conn, _job_key, today)
        if event_triggered and ttl_elapsed:
            trigger = "event+scheduled"
        elif event_triggered:
            trigger = "event-triggered"
        else:
            trigger = "scheduled"
        log.info("FAQ gap analysis complete (%s) — next run in %d day(s) or after %d new resolved tickets",
                 trigger, interval_days, new_tickets_threshold)

    # ── Publish unpublished drafts (runs every cycle — idempotent) ───────────
    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT id FROM generated_articles
            WHERE format = 'faq' AND status = 'draft' AND confluence_page_id IS NULL
            ORDER BY generated_at
        """).fetchall()
    if rows:
        published = sum(1 for r in rows if publish_article(r["id"]))
        log.info("Published %d of %d FAQ articles to Confluence", published, len(rows))


def _phase_responder() -> None:
    """Phase 4: Auto-draft responses + capture feedback + harvest examples."""
    # Auto-draft sweep
    from plugins.responder.sweep import _auto_draft_sweep

    _auto_draft_sweep()

    # Feedback loop + few-shot ANN rebuild (plugins/responder on_schedule)
    from plugins.responder import plugin as responder_plugin

    responder_plugin.on_schedule()

    # Check Slack signals for resolved threads
    responder_cfg = get_plugin_config("responder")
    if responder_cfg.get("check_resolved_signals", True):
        from ingest.slack import check_resolved_signals
        check_resolved_signals()


def _phase_kb_index() -> None:
    """Phase 4b: Crawl Confluence KB spaces into kb_articles + build embeddings.

    Runs on a weekly cadence per instance (keyed by cloud_id + space so stage
    and prod maintain completely separate indexes).  On first run performs a
    full crawl; subsequent runs are incremental via fetched_at tracking.
    """
    from db import get_db_conn, get_last_run_date, set_last_run_date
    import datetime as _dt

    cfg = get_plugin_config("kb_index")
    spaces = cfg.get("confluence_spaces", ["HUB", "OMEGA"])
    if not spaces:
        return

    try:
        from ingest.oauth2lo import get_cloud_base_url as _get_base
        _cloud_id = _get_base("jsm").rstrip("/").split("/")[-1]
    except Exception as e:
        log.warning("Could not resolve cloud_id for kb_index, using 'default': %s", e)
        _cloud_id = "default"

    today = _dt.date.today()
    refresh_days = cfg.get("refresh_days", 7)

    from scripts.index_confluence_kb import _process_crawl, run_vector_backfill

    for space in spaces:
        job_name = f"kb_index:{_cloud_id}:{space}"
        with get_db_conn() as conn:
            last_run = get_last_run_date(conn, job_name)

        if last_run and (today - last_run).days < refresh_days:
            log.debug("kb_index: %s last indexed %s, skipping (< %d days)", space, last_run, refresh_days)
            continue

        log.info("kb_index: indexing space %s for instance %s", space, _cloud_id)
        try:
            per_space, upserted, skipped = _process_crawl((space,), dry_run=False)
            log.info("kb_index: %s — upserted=%d skipped=%d", space, upserted, skipped)
            with get_db_conn() as conn:
                set_last_run_date(conn, job_name, today)
        except Exception as e:
            log.error("kb_index: failed for space %s: %s", space, e)

    # Build/refresh embeddings for any kb_articles rows missing vectors
    try:
        stored, failures = run_vector_backfill()
        if stored:
            log.info("kb_index: vector backfill — stored=%d failures=%d", stored, failures)
    except Exception as e:
        log.error("kb_index: vector backfill failed: %s", e)


def _phase_export() -> None:
    """Phase 5: Refresh doc indexes + daily gap analysis."""
    cfg = get_plugin_config("export")

    if cfg.get("refresh_doc_indexes", True):
        try:
            from main import _refresh_doc_indexes
        except ImportError as e:
            raise ImportError(
                "core.pipeline requires main._refresh_doc_indexes — "
                "ensure main.py is on sys.path"
            ) from e
        _refresh_doc_indexes()

    # Gap analysis is handled exclusively by _phase_faq() with a 7-day job_state gate.
    # Do NOT call _auto_gap_analysis() here — it bypasses the gate and runs every cycle.


def _phase_feedback() -> None:
    """Phase 6: Daily CSAT ingestion for recently resolved tickets."""
    from plugins.feedback import plugin

    plugin.on_schedule()


def _phase_alerting() -> None:
    """Phase 7: Volume anomaly detection + daily topic clustering."""
    from plugins.alerting import plugin

    plugin.on_schedule()


def _phase_mrr_monitor() -> None:
    """Phase 8: Weekly MRR quality snapshot (ANTSE-450).

    Runs the retrieval quality snapshot at most once per 7 days.  Gated on
    ``job_state.last_run_date`` for ``mrr_snapshot:<cloud_id>`` — the underlying
    ``run_mrr_snapshot()`` function is idempotent per day, but this outer gate
    avoids touching the retrieval index more than once every 7 days.
    """
    import datetime as _dt

    from db import get_db_conn, get_last_run_date
    from plugins.responder.mrr_monitor import run_mrr_snapshot
    from ingest.oauth2lo import get_cloud_base_url as _get_base

    today = _dt.date.today()
    try:
        _cloud_id = _get_base("jsm").rstrip("/").split("/")[-1]
    except Exception as e:
        log.warning("Could not resolve cloud_id for mrr_monitor, using 'default': %s", e)
        _cloud_id = "default"
    _job_key = f"mrr_snapshot:{_cloud_id}"

    with get_db_conn() as conn:
        last_run = get_last_run_date(conn, _job_key)

    if last_run is not None and (today - last_run).days < 7:
        log.debug(
            "MRR snapshot ran %d day(s) ago — skipping until 7-day cadence",
            (today - last_run).days,
        )
        return

    log.info("Running weekly MRR snapshot (last run: %s)", last_run or "never")
    run_mrr_snapshot()

    from db import set_last_run_date
    with get_db_conn() as conn:
        set_last_run_date(conn, _job_key, today)
    log.debug("MRR snapshot complete — next run in 7 days")


# ── Orchestrator ────────────────────────────────────────────────────


def _run_pipeline_cycle(trigger: str = "internal") -> None:
    """Acquire the pipeline lock and run all enabled plugin hooks in order.

    Designed to run in a daemon thread (spawned by ``trigger_pipeline_async``).
    Acquires ``_pipeline_lock`` non-blockingly so that concurrent invocations
    (e.g. a scheduled tick arriving while a webhook-triggered cycle is still
    running) are silently dropped rather than queued.
    """
    global _last_pipeline_run, _last_pipeline_trigger

    if not _pipeline_lock.acquire(blocking=False):
        log.info("Pipeline already running, skipping %s trigger", trigger)
        return

    try:
        _register_builtin_phases()

        for name, hook in _plugin_hooks:
            if not is_plugin_enabled(name):
                log.debug("Skipping disabled plugin: %s", name)
                continue
            try:
                log.info("Pipeline phase: %s", name)
                hook()
            except Exception as e:
                log.error("Pipeline phase '%s' failed: %s", name, e)
                # Continue to next phase — one failure shouldn't block the rest

        log.info("Pipeline cycle complete")
        _last_pipeline_run = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _last_pipeline_trigger = trigger
    finally:
        _pipeline_lock.release()


def trigger_pipeline_async(trigger: str = "webhook") -> None:
    """Dispatch a pipeline cycle in a daemon thread (non-blocking).

    Safe to call from any thread — webhook handlers, scheduler, startup.
    If a cycle is already running the call is silently skipped (the lock
    check happens inside ``_run_pipeline_cycle`` so the thread exits
    immediately without blocking the caller).
    """
    t = threading.Thread(target=_run_pipeline_cycle, args=(trigger,), daemon=True)
    t.start()


def pipeline_scheduler() -> None:
    """Background thread loop — runs the pipeline on a recurring schedule.

    Intended to be started as a daemon thread from main.py::

        threading.Thread(target=pipeline_scheduler, daemon=True).start()
    """
    schedule = get_schedule_config()
    interval = schedule["refresh_minutes"] * 60

    while True:
        time.sleep(interval)
        log.info("Starting scheduled pipeline refresh...")
        trigger_pipeline_async(trigger="schedule")


def pipeline_status() -> dict[str, Any]:
    """Return current pipeline status for the /api/health endpoint."""
    schedule = get_schedule_config()
    return {
        "pipeline_running": _pipeline_lock.locked(),
        "last_pipeline_run": _last_pipeline_run,
        "last_trigger": _last_pipeline_trigger,
        "refresh_interval_minutes": schedule["refresh_minutes"],
    }
