"""FastAPI application for AI Helpdesk Agent.

Runs the full pipeline (ingest → scrub → classify → FAQ analyze/generate/export)
on a background schedule so data stays fresh for lookups.
"""

import hmac
import json
import logging
import sys
import os
import threading
import time

# Add project root to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Depends, Header, HTTPException, Query, Request

from db import init_db, get_db, get_db_conn
from analysis.gaps import get_gap_summary
from faq.router import faq_router, _verify_token
from faq.slack_handler import slack_router

log = logging.getLogger(__name__)

# Pipeline refresh interval (default: 15 minutes)
REFRESH_INTERVAL = int(os.getenv("PIPELINE_REFRESH_MINUTES", "5")) * 60

# Jira webhook secret (optional — logs warning if unset)
JIRA_WEBHOOK_SECRET = os.getenv("JIRA_WEBHOOK_SECRET", "")

# Disable OpenAPI docs in production
DISABLE_DOCS = os.getenv("DISABLE_DOCS", "").lower() in ("1", "true")

_docs_kwargs = {}
if DISABLE_DOCS:
    _docs_kwargs = {"docs_url": None, "redoc_url": None, "openapi_url": None}

app = FastAPI(
    title="AI Helpdesk Agent",
    description="AI-Powered JSM Tier 1 Technician — ticket classification, automated response drafting, FAQ generation, and knowledge base management",
    version="0.2.0",
    **_docs_kwargs,
)

app.include_router(faq_router, prefix="/api/faq", tags=["FAQ"])
app.include_router(slack_router, prefix="/api/slack", tags=["Slack"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


_pipeline_thread = None


_last_doc_refresh = None


def _refresh_doc_indexes():
    """Refresh Atlassian docs and Confluence KB indexes (daily)."""
    global _last_doc_refresh
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)
    if _last_doc_refresh and (now - _last_doc_refresh).total_seconds() < 86400:
        return  # already refreshed today

    # Atlassian support + developer doc sitemaps (lightweight URL+title index)
    from faq.atlassian_docs import fetch_all_doc_sites
    total = fetch_all_doc_sites()
    log.info("Atlassian docs index refreshed: %d URLs", total)

    # Confluence KB spaces are searched on-demand via CQL API in faq/lookup.py
    # — no bulk crawl needed
    _last_doc_refresh = now


_last_gap_analysis = None


def _auto_gap_analysis():
    """Daily automated FAQ gap detection + generation + Slack enrichment.

    Part A: Finds resolved ticket themes not covered by existing FAQs,
    generates new FAQ articles for gaps with 3+ tickets.
    Part B: Enriches existing FAQs with verified ✅ Slack Q&A threads.
    Exports all to Google Doc.
    """
    global _last_gap_analysis
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)
    if _last_gap_analysis and (now - _last_gap_analysis).total_seconds() < 86400:
        return
    _last_gap_analysis = now

    log.info("Running automated gap analysis...")

    # Part A — Gap detection + FAQ generation
    client = None  # guard for Part B if Gemini init fails
    try:
        from faq.auto_responder import _get_genai_client
        from config import GEMINI_MODEL_GENERATION
        from utils.gemini import parse_json_response

        client = _get_genai_client()

        with get_db_conn() as conn:
            faqs = {r[0].lower() for r in conn.execute(
                "SELECT title FROM generated_articles WHERE format = 'faq'"
            ).fetchall()}

            # GROUP_CONCAT is SQLite-only; collect ticket_keys in Python instead.
            # HAVING uses COUNT(*) directly — column aliases are not valid in Postgres HAVING.
            themes = conn.execute("""
                SELECT tc.category, tc.issue_type,
                       COUNT(*) as ticket_count
                FROM tickets t
                JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
                WHERE t.resolution IS NOT NULL
                  AND tc.resolution_summary IS NOT NULL
                  AND tc.resolution_summary != ''
                GROUP BY tc.category, tc.issue_type
                HAVING COUNT(*) >= 3
                ORDER BY ticket_count DESC
            """).fetchall()

        gaps = []
        for t in themes:
            cat_l = (t["category"] or "").lower()
            issue_l = (t["issue_type"] or "").lower()
            covered = any(
                cat_l in f or issue_l in f
                or any(w in f for w in issue_l.split() if len(w) > 4)
                for f in faqs
            )
            if not covered:
                gaps.append(t)

        generated = 0
        for gap in gaps[:10]:
            # Fetch sample ticket keys in Python to avoid GROUP_CONCAT (SQLite-only).
            with get_db_conn() as conn:
                key_rows = conn.execute("""
                    SELECT t.ticket_key
                    FROM tickets t
                    JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
                    WHERE t.resolution IS NOT NULL
                      AND tc.resolution_summary IS NOT NULL
                      AND tc.resolution_summary != ''
                      AND tc.category = ?
                      AND tc.issue_type = ?
                    LIMIT 3
                """, (gap["category"], gap["issue_type"])).fetchall()
            keys = [r[0] for r in key_rows]
            if not keys:
                continue
            with get_db_conn() as conn:
                samples = conn.execute(
                    """
                    SELECT t.summary, tc.resolution_summary
                    FROM tickets t
                    JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
                    WHERE t.ticket_key IN ({})
                    """.format(",".join(["?"] * len(keys))),
                    keys,
                ).fetchall()

            context = "\n".join(
                f"- {s['summary']}: {(s['resolution_summary'] or '')[:200]}"
                for s in samples
            )

            prompt = f"""Generate a concise FAQ article for our Atlassian Cloud migration knowledge base.

Topic: {gap['category']} > {gap['issue_type']}
Ticket count: {gap['ticket_count']}
Sample resolutions:
{context}

Write in Q&A format with:
- A clear question as the title (starting with "How do I..." or "Why can't I...")
- A concise answer (200-400 words)
- Step-by-step instructions where applicable
- Links to official Atlassian Cloud documentation (use real support.atlassian.com URLs)

Return valid JSON only (no markdown fencing):
{{"title": "...", "body_html": "<p>...</p>"}}"""

            try:
                response = client.models.generate_content(model=GEMINI_MODEL_GENERATION, contents=prompt)
                result = parse_json_response(response.text)
                if isinstance(result, list):
                    result = result[0]
                if result and result.get("body_html"):
                    sample_keys_str = ",".join(keys)[:500]
                    with get_db_conn() as conn:
                        conn.execute("""
                            INSERT INTO generated_articles
                                (article_topic, title, body_html, format, source_ticket_keys, status)
                            VALUES (?, ?, ?, 'faq', ?, 'draft')
                            ON CONFLICT (article_topic) DO UPDATE SET
                                title=EXCLUDED.title, body_html=EXCLUDED.body_html,
                                format=EXCLUDED.format,
                                source_ticket_keys=EXCLUDED.source_ticket_keys,
                                status=EXCLUDED.status
                        """, (f"{gap['category']} > {gap['issue_type']}",
                              result["title"], result["body_html"], sample_keys_str))
                    generated += 1
            except Exception as e:
                log.debug("FAQ generation failed for %s: %s", gap["issue_type"], e)

        if generated:
            log.info("Gap analysis: generated %d new FAQ articles", generated)

    except Exception as e:
        log.error("Gap analysis Part A failed: %s", e)

    # Part B — FAQ enrichment from ✅ Slack threads
    if client is None:
        log.warning("Gap analysis Part B skipped: Gemini client unavailable")
        return
    try:
        import json as _json
        from ingest.slack import get_resolved_threads

        resolved = get_resolved_threads(limit=100)
        if not resolved:
            log.info("Gap analysis: no resolved Slack threads to process")
        else:
            with get_db_conn() as conn:
                faqs = conn.execute(
                    "SELECT id, title, body_html FROM generated_articles WHERE format = 'faq'"
                ).fetchall()

            enriched = 0
            for faq in faqs:
                title_words = {w.lower() for w in faq["title"].split() if len(w) > 4}
                matching = [
                    r for r in resolved
                    if sum(1 for w in title_words if w in r["question"].lower()) >= 2
                ]
                if len(matching) < 2:
                    continue

                slack_context = "\n---\n".join(
                    f"Q: {qa['question'][:150]}\nA: {' '.join(qa['answers'][:3])[:300]}"
                    for qa in matching[:5]
                )

                prompt = f"""Update this FAQ article by incorporating verified answers from Slack (✅ resolved).

CURRENT FAQ:
Title: {faq['title']}
Content: {faq['body_html'][:2000]}

VERIFIED SLACK Q&A:
{slack_context[:3000]}

Rules: Keep ALL existing content. Only ADD new insights, tips, or common follow-ups from Slack.
Return valid JSON: {{"title": "{faq['title']}", "body_html": "<p>updated...</p>"}}"""

                try:
                    response = client.models.generate_content(model=GEMINI_MODEL_GENERATION, contents=prompt)
                    result = parse_json_response(response.text)
                    if isinstance(result, list):
                        result = result[0]
                    if result and result.get("body_html"):
                        with get_db_conn() as conn:
                            conn.execute("UPDATE generated_articles SET body_html = ? WHERE id = ?",
                                (result["body_html"], faq["id"]))
                        enriched += 1
                except Exception as e:
                    log.debug("FAQ enrichment failed for %s: %s", faq["title"][:30], e)

            if enriched:
                log.info("Gap analysis: enriched %d FAQs from Slack", enriched)

    except Exception as e:
        log.error("Gap analysis Part B failed: %s", e)

    # Export to Google Doc
    try:
        from faq.google_docs import write_faq_entries
        from utils.html_parser import parse_faq_html
        with get_db_conn() as conn:
            rows = conn.execute("""
                SELECT title, body_html FROM generated_articles
                WHERE format = 'faq' ORDER BY generated_at
            """).fetchall()
        if rows:
            entries = [{"topic": r["title"], **parse_faq_html(r["body_html"])} for r in rows]
            write_faq_entries(entries)
            log.info("Gap analysis: exported %d FAQ entries to Google Doc", len(entries))
    except Exception as e:
        log.error("FAQ export failed: %s", e)


def _run_pipeline_cycle():
    """Thin shim — delegates all pipeline execution to core.pipeline._run_pipeline_cycle.

    All phases run in order via core.pipeline:
        ingest → analysis → resolution_summary → feedback → faq →
        kb_index → responder → export → alerting → mrr_monitor

    This function is retained for import compatibility only; do not add
    phase logic here.
    """
    from core.pipeline import _run_pipeline_cycle as _core_run
    _core_run()


def _pipeline_scheduler():
    """Background thread that runs the pipeline on a schedule (fallback for webhooks)."""
    while True:
        time.sleep(REFRESH_INTERVAL)
        log.info("Starting scheduled pipeline refresh...")
        _trigger_pipeline_async(trigger="schedule")


_plugins_ready = False


def _load_plugins_background(application):
    global _pipeline_thread, _plugins_ready
    from plugins import register_plugins
    register_plugins(application)
    _plugins_ready = True
    from config import FAQ_API_TOKEN
    if not FAQ_API_TOKEN:
        log.warning("FAQ_API_TOKEN not set — API endpoints are UNPROTECTED")
    if not JIRA_WEBHOOK_SECRET:
        log.warning("JIRA_WEBHOOK_SECRET not set — webhook endpoint is UNPROTECTED")
    # Delay the startup pipeline trigger by 30s to let dense_retrieval.build()
    # (called synchronously in responder.register()) fully release its transient
    # numpy allocations before the 4-worker classifier ThreadPoolExecutor spins up.
    # Both compete for the same memory budget; sequential execution keeps peak < 4Gi.
    def _deferred_startup():
        time.sleep(30)
        _trigger_pipeline_async("startup")

    threading.Thread(target=_deferred_startup, daemon=True).start()
    _pipeline_thread = threading.Thread(target=_pipeline_scheduler, daemon=True)
    _pipeline_thread.start()
    log.info("Pipeline scheduler started (refresh every %d minutes)", REFRESH_INTERVAL // 60)


@app.on_event("startup")
def startup():
    init_db()
    threading.Thread(target=_load_plugins_background, args=(app,), daemon=True).start()


@app.get("/api/health")
def health():
    from ingest.oauth2lo import oauth_configured
    from core.pipeline import pipeline_status
    status = pipeline_status()
    return {
        "status": "ok",
        "warming_up": not _plugins_ready,
        "pipeline_running": status["pipeline_running"],
        "last_pipeline_run": status["last_pipeline_run"],
        "last_trigger": status["last_trigger"],
        "refresh_interval_minutes": status["refresh_interval_minutes"],
        "oauth_configured": {
            "jira": oauth_configured("jira"),
            "confluence": oauth_configured("confluence"),
        },
    }


@app.get("/api/ready")
def ready():
    """Readiness probe — returns 503 during plugin warmup so traffic is held until loaded."""
    if not _plugins_ready:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "warming_up"})
    return {"status": "ready"}


# ── Jira Webhook — triggers pipeline on ticket changes ──


def _trigger_pipeline_async(trigger="webhook"):
    """Dispatch a pipeline cycle in a daemon thread (non-blocking).

    Always threads the call so neither the scheduler thread nor the
    webhook handler blocks for the full cycle duration.
    """
    from core.pipeline import trigger_pipeline_async as _core_trigger
    _core_trigger(trigger=trigger)


def _dispatch_plugin_ticket_event(ticket_key: str, event: str, payload: dict) -> None:
    """Dispatch ticket events to plugin on_ticket hooks."""
    try:
        from plugins import dispatch_on_ticket
        dispatch_on_ticket(ticket_key, event, payload)
    except Exception as exc:
        log.warning("Plugin on_ticket dispatch failed for %s (%s): %s", ticket_key, event, exc)


def _map_webhook_to_plugin_event(webhook_event: str) -> str | None:
    """Map Jira webhook event names to plugin on_ticket event names."""
    if webhook_event in ("jira:issue_created", "issue_created"):
        return "created"
    if webhook_event in ("jira:issue_updated", "issue_updated", "jira:issue_resolved", "issue_resolved"):
        return "updated"
    if webhook_event == "comment_created":
        return "commented"
    # comment_updated and comment_deleted are intentionally not mapped — no plugin
    # currently handles these events. Add mappings here when needed.
    # Other unmapped event types (e.g., sprint events) also return None and are silently ignored.
    return None


@app.post("/api/webhook/jira")
def jira_webhook(
    payload: dict,
    secret: str | None = None,
    x_webhook_secret: str | None = Header(None, alias="X-Webhook-Secret"),
):
    """Receive Jira webhook for ticket create/update/resolve events.

    Accepts the shared secret via either:
      - X-Webhook-Secret request header (preferred)
      - ?secret= query parameter (Jira automation rule URL param)

    Configure in Jira: POST to https://<service>/api/webhook/jira?secret=<JIRA_WEBHOOK_SECRET>
    Events: issue_created, issue_updated, issue_resolved — Project: <PROJECT_KEY>
    """
    if JIRA_WEBHOOK_SECRET:
        provided = x_webhook_secret or secret or ""
        if not provided or not hmac.compare_digest(provided, JIRA_WEBHOOK_SECRET):
            raise HTTPException(status_code=401, detail="Invalid webhook secret")
    # If no secret configured, startup warning already logged — allow through

    event = payload.get("webhookEvent", payload.get("issue_event_type_name", "unknown"))
    key = payload.get("issue", {}).get("key", "unknown")
    log.info("Webhook received: %s for %s", event, key)

    # Route webhook events through plugin on_ticket hooks (single dispatch path).
    plugin_event = _map_webhook_to_plugin_event(event)
    if key != "unknown" and plugin_event:
        plugin_payload = dict(payload)
        if plugin_event == "commented":
            comment_data = payload.get("comment", {})
            plugin_payload["comment_id"] = str(comment_data.get("id", ""))
            body = comment_data.get("body", {})
            if isinstance(body, dict):
                from ingest.tickets import _extract_adf_text
                plugin_payload["comment_body"] = _extract_adf_text(body)
            else:
                plugin_payload["comment_body"] = str(body) if body else ""
        threading.Thread(
            target=_dispatch_plugin_ticket_event,
            args=(key, plugin_event, plugin_payload),
            daemon=True,
        ).start()

    threading.Thread(target=_trigger_pipeline_async, daemon=True).start()
    return {"status": "accepted", "event": event, "issue": key}


@app.post("/api/auto-respond/{ticket_key}")
def trigger_auto_respond(ticket_key: str, _token: str = Depends(_verify_token)):
    """Manually trigger an AI-drafted response for a ticket."""
    from plugins.responder import handle_new_ticket
    success = handle_new_ticket(ticket_key)
    if success:
        return {"status": "posted", "ticket": ticket_key}
    raise HTTPException(
        status_code=404,
        detail=f"No matching FAQ/KB content found for {ticket_key}, or draft failed",
    )


@app.post("/api/review-tickets")
def review_tickets(ticket_keys: list[str], _token: str = Depends(_verify_token)):
    """Batch review tickets and post disposition comments."""
    from plugins.responder import batch_review_tickets
    results = batch_review_tickets(ticket_keys)
    return results


@app.get("/api/feedback/stats")
def feedback_stats(_token: str = Depends(_verify_token)):
    """Auto-responder feedback loop statistics."""
    from plugins.responder import get_feedback_stats
    return get_feedback_stats()


@app.get("/api/reports/summary")
def dashboard_summary(_token: str = Depends(_verify_token)):
    """Dashboard overview data."""
    from reporting.reports import build_dashboard_data
    return build_dashboard_data()


@app.get("/api/tickets")
def list_tickets(
    status: str | None = None,
    version: str | None = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    _token: str = Depends(_verify_token),
):
    """List tickets with optional filters."""
    with get_db_conn() as conn:
        conditions = []
        params = []

        if status:
            conditions.append("t.status = ?")
            params.append(status)
        if version:
            conditions.append("t.affect_version = ?")
            params.append(version)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        rows = conn.execute(f"""
            SELECT
                t.*,
                c.category,
                c.issue_type,
                c.confidence,
                cs.csat_score,
                cs.csat_comment,
                cs.submitted_at AS csat_submitted_at,
                cs.ingested_at AS csat_ingested_at
            FROM tickets t
            LEFT JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
            LEFT JOIN ticket_csat cs ON t.ticket_key = cs.ticket_key
            {where}
            ORDER BY t.created_at DESC
            LIMIT ? OFFSET ?
        """, params).fetchall()

        total = conn.execute(f"SELECT COUNT(*) FROM tickets t {where}", params[:-2]).fetchone()[0]

    return {"total": total, "tickets": [dict(r) for r in rows]}


@app.get("/api/tickets/{key}")
def get_ticket(key: str, _token: str = Depends(_verify_token)):
    """Single ticket with classification and comments."""
    with get_db_conn() as conn:
        ticket = conn.execute("SELECT * FROM tickets WHERE ticket_key = ?", (key,)).fetchone()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")

        classification = conn.execute("""
            SELECT
                c.*,
                cs.csat_score,
                cs.csat_comment,
                cs.submitted_at AS csat_submitted_at,
                cs.ingested_at AS csat_ingested_at
            FROM ticket_classifications c
            LEFT JOIN ticket_csat cs ON c.ticket_key = cs.ticket_key
            WHERE c.ticket_key = ?
        """, (key,)).fetchone()
        comments = conn.execute(
            "SELECT * FROM ticket_comments WHERE ticket_key = ? ORDER BY created_at", (key,)
        ).fetchall()
        links = conn.execute(
            "SELECT * FROM ticket_links WHERE ticket_key = ?", (key,)
        ).fetchall()

    return {
        "ticket": dict(ticket),
        "classification": dict(classification) if classification else None,
        "comments": [dict(c) for c in comments],
        "links": [dict(l) for l in links],
    }


@app.get("/api/classifications")
def classification_summary(version: str | None = None, _token: str = Depends(_verify_token)):
    """Classification breakdown by category and issue type."""
    from reporting.reports import build_ticket_summary
    return build_ticket_summary(version)


@app.get("/api/gaps")
def gap_analysis(_token: str = Depends(_verify_token)):
    """KB coverage gap analysis."""
    summary = get_gap_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="No gap analysis data. Run 'gaps' command first.")
    return summary


@app.get("/api/articles")
def list_articles(status: str | None = None, _token: str = Depends(_verify_token)):
    """List generated articles."""
    with get_db_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT id, article_topic, title, format, status, generated_at, published_at FROM generated_articles WHERE status = ? ORDER BY generated_at DESC",
                (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, article_topic, title, format, status, generated_at, published_at FROM generated_articles ORDER BY generated_at DESC"
            ).fetchall()
    return {"articles": [dict(r) for r in rows]}


@app.get("/api/articles/{article_id}")
def get_article(article_id: int, _token: str = Depends(_verify_token)):
    """Single article with full content."""
    with get_db_conn() as conn:
        article = conn.execute("SELECT * FROM generated_articles WHERE id = ?", (article_id,)).fetchone()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    result = dict(article)
    # Parse JSON fields
    for field in ["source_ticket_keys", "reference_urls"]:
        if result.get(field):
            try:
                result[field] = json.loads(result[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return result


@app.post("/api/articles/{article_id}/publish")
def publish_article_endpoint(article_id: int, space_key: str | None = None, _token: str = Depends(_verify_token)):
    """Publish an article to Confluence."""
    from generation.publisher import publish_article
    success = publish_article(article_id, space_key)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to publish article")
    return {"status": "published", "article_id": article_id}


@app.get("/api/stats")
def stats(_token: str = Depends(_verify_token)):
    """Database statistics."""
    from analysis.ml.eda import ticket_summary_stats
    return ticket_summary_stats()


@app.get("/api/predictions")
def list_predictions(_token: str = Depends(_verify_token)):
    """List risk predictions."""
    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            ORDER BY CASE risk_level WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                                      WHEN 'medium' THEN 2 ELSE 3 END,
                     predicted_at DESC
        """).fetchall()
    return {"predictions": [dict(r) for r in rows]}


@app.get("/api/linked-issues")
def list_linked_issues(project: str | None = None, _token: str = Depends(_verify_token)):
    """List linked issue details."""
    with get_db_conn() as conn:
        if project:
            rows = conn.execute(
                "SELECT * FROM linked_issues WHERE project_key = ? ORDER BY issue_key",
                (project,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM linked_issues ORDER BY issue_key").fetchall()
    return {"linked_issues": [dict(r) for r in rows]}


@app.get("/api/doc-improvements")
def list_doc_improvements(_token: str = Depends(_verify_token)):
    """List documentation improvement suggestions."""
    with get_db_conn() as conn:
        rows = conn.execute("SELECT * FROM doc_improvements ORDER BY theme").fetchall()
    results = []
    for r in rows:
        d = dict(r)
        for field in ["missing_topics", "unclear_sections", "edge_cases"]:
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(d)
    return {"improvements": results}


@app.get("/api/kb-articles")
def list_kb_articles(space: str | None = None, _token: str = Depends(_verify_token)):
    """List crawled KB articles."""
    with get_db_conn() as conn:
        if space:
            rows = conn.execute(
                "SELECT page_id, space_key, title, url, labels, topics_covered, fetched_at FROM kb_articles WHERE space_key = ? ORDER BY title",
                (space,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT page_id, space_key, title, url, labels, topics_covered, fetched_at FROM kb_articles ORDER BY title"
            ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        for field in ["labels", "topics_covered"]:
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(d)
    return {"kb_articles": results}
