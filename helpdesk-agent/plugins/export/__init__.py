"""Export plugin — Google Docs/Sheets/Slides output and Confluence publishing.

Re-exports from faq.google_*, generation.*, and reporting.reports.
"""

import logging

from plugins._protocol import BasePlugin

# -- Re-exports from faq.google_docs --
from faq.google_docs import (
    read_doc,
    read_all_source_docs,
    write_faq_entries,
)

# -- Re-exports from faq.google_sheets --
from faq.google_sheets import read_sheet, read_source_sheets

# -- Re-exports from faq.google_slides --
from faq.google_slides import read_slides

# -- Re-exports from generation.articles --
from generation.articles import (
    determine_article_format,
    gather_article_sources,
    generate_articles,
)

# -- Re-exports from generation.publisher --
from generation.publisher import publish_article

# -- Re-exports from reporting.reports --
from reporting.reports import build_ticket_summary, build_dashboard_data

log = logging.getLogger(__name__)

__all__ = [
    "read_doc",
    "read_all_source_docs",
    "write_faq_entries",
    "read_sheet",
    "read_source_sheets",
    "read_slides",
    "determine_article_format",
    "gather_article_sources",
    "generate_articles",
    "publish_article",
    "build_ticket_summary",
    "build_dashboard_data",
    "plugin",
]


class ExportPlugin(BasePlugin):
    """Export FAQ entries to Google Docs."""

    name = "export"

    def on_schedule(self) -> None:
        from db import get_db
        log.info("[export] scheduled run — exporting FAQs to Google Docs")
        conn = get_db()
        rows = conn.execute("""
            SELECT topic, question, answer, steps, known_limitations
            FROM faq_entries WHERE exported_at IS NULL
            ORDER BY created_at
        """).fetchall()
        conn.close()

        if not rows:
            log.info("[export] no unexported FAQ entries")
        else:
            entries = [dict(r) for r in rows]
            write_faq_entries(entries)
            log.info("[export] exported %d FAQ entries", len(entries))

        from core.pipeline import get_plugin_config
        from plugins.responder.templates import sync_templates_from_confluence

        cfg = get_plugin_config("responder")
        page_id = cfg.get("confluence_template_page_id", "").strip()
        if page_id:
            result = sync_templates_from_confluence(page_id)
            log.info("[export] template sync: %s", result)


plugin = ExportPlugin()
