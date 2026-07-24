"""Feedback plugin — sentiment scoring and CSAT ingestion for JSM tickets."""

import datetime as dt
import logging

from db import get_db_conn, get_last_run_date, set_last_run_date
from plugins._protocol import BasePlugin
from plugins.feedback.csat import ingest_csat
from plugins.feedback.pipeline import score_classification, score_unscored_classifications

log = logging.getLogger(__name__)

JOB_NAME = "csat_ingest"

__all__ = ["plugin", "score_classification", "score_unscored_classifications"]


class FeedbackPlugin(BasePlugin):
    """Sentiment scoring and daily CSAT ingestion."""

    name = "feedback"

    def on_ticket(self, ticket_key: str, event: str, payload: dict) -> None:
        """Score sentiment when a ticket has been classified."""
        if event not in ("classified", "created", "updated", "commented"):
            return
        try:
            score_classification(ticket_key)
        except Exception as exc:
            log.warning("[feedback] on_ticket sentiment failed for %s: %s", ticket_key, exc)

    def on_schedule(self) -> None:
        today = dt.date.today()

        # Resolve cloud_id so the job_state key is instance-scoped.
        try:
            from ingest.oauth2lo import get_cloud_base_url as _get_base
            _cloud_id = _get_base("jsm").rstrip("/").split("/")[-1]
        except Exception:
            _cloud_id = "default"
        _job_key = f"{JOB_NAME}:{_cloud_id}"

        # CSAT ingestion (once per day)
        with get_db_conn() as conn:
            last_run = get_last_run_date(conn, _job_key)
            if last_run != today:
                log.info("[feedback] scheduled run — ingesting CSAT for recently resolved tickets")
                stats = ingest_csat(conn)
                set_last_run_date(conn, _job_key, today)
                log.info(
                    "[feedback] CSAT ingest complete — checked=%d found=%d errors=%d",
                    stats["checked"], stats["found"], stats["errors"],
                )
                if stats["errors"] > 0 and stats["found"] == 0 and stats["checked"] > 0:
                    log.warning(
                        "[feedback] CSAT ingest: all %d tickets errored, none ingested — "
                        "job_state written, no automatic retry today. Check JSM API connectivity.",
                        stats["errors"],
                    )

                from plugins.feedback.correlator import compute_category_correlations

                results = compute_category_correlations(conn)
                log.info(
                    "[feedback] CSAT correlator: %d categories processed",
                    len(results),
                )

        # Sentiment backfill for any unscored classifications
        scored, errors = score_unscored_classifications()
        if scored:
            log.info("[feedback] sentiment backfill — scored %d (%d errors)", scored, errors)


plugin = FeedbackPlugin()
