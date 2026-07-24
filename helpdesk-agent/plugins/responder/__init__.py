"""Responder plugin — AI-powered draft responses and disposition reviews.

Dispatches ticket events to the appropriate handler (new ticket, /ai-lookup,
/ai-review, emoji feedback, AI assist) and runs the auto-draft sweep on
schedule.
"""

import logging

from core.pipeline import get_plugin_config
from plugins._protocol import BasePlugin

from .drafting import (
    _get_ticket_context,
    _get_latest_customer_reply,
    _lookup_and_draft,
    _draft_response,
    _build_few_shot_block,
    _fetch_doc_content,
)
from .ann_fewshot import ANNFewShotIndex
from .gates import (
    _check_status_gate,
    _check_age_gate,
    _check_content_gate,
    _get_confidence_threshold,
    check_classification_gates,
    _has_pending_draft,
    _SKIP_STATUSES,
)
from .jira_comments import (
    _jira_request,
    _delete_comment,
    _post_draft_comment,
    _post_review_comment,
    _post_self_check_and_decide,
    _text_to_adf_paragraphs,
    _has_agent_response,
    _check_assignee_allowed,
)
from .feedback import (
    handle_feedback_emoji,
    _store_draft_record,
    _score_similarity,
    capture_feedback,
    harvest_response_examples,
    get_few_shot_examples,
    get_organic_examples,
    get_feedback_stats,
    FEEDBACK_EMOJI_MAP,
    QUESTION_TYPE_MAP,
)
from .review import (
    review_ticket,
    handle_review_trigger,
    batch_review_tickets,
    handle_ai_assist,
)
from .lookup import lookup

log = logging.getLogger(__name__)

AI_LOOKUP_TRIGGER = "/ai-lookup"
AI_REVIEW_TRIGGER = "/ai-review"
AI_ASSIST_EMOJI = "🤖"


# ── Entry-point functions (preserve original signatures) ──────


def handle_new_ticket(ticket_key: str) -> bool:
    """Main entry point — look up matches, draft response, post to ticket.

    Only drafts responses for tickets assigned to allowed agents
    (AUTO_RESPOND_ASSIGNEES). Returns True if a draft was posted, False otherwise.
    """
    log.info("Auto-responder processing %s", ticket_key)

    allowed, assignee_id = _check_assignee_allowed(ticket_key)
    if not allowed:
        return False

    if not _check_status_gate(ticket_key):
        return False

    if not _check_age_gate(ticket_key):
        return False

    if not _check_content_gate(ticket_key):
        return False

    responder_cfg = get_plugin_config("responder")
    confidence_threshold = _get_confidence_threshold(responder_cfg)
    if not check_classification_gates(ticket_key, responder_cfg):
        return False

    # M5 Router: confidence + sentiment + novelty gate
    from analysis.router import route_ticket
    routing = route_ticket(ticket_key, confidence_threshold=confidence_threshold)
    if routing.route == "human_review":
        log.info(
            "Ticket %s → human_review (M5 router: %s) — skipping auto-draft",
            ticket_key, routing.reason,
        )
        return False
    if routing.route == "flag_only":
        log.info(
            "Ticket %s → flag_only (M5 router: %s) — skipping auto-draft",
            ticket_key, routing.reason,
        )
        return False

    if _has_agent_response(ticket_key, assignee_account_id=assignee_id):
        return False

    context = _get_ticket_context(ticket_key)
    if not context:
        return False
    summary, desc = context

    return _lookup_and_draft(ticket_key, summary, desc)


def handle_ai_lookup_trigger(ticket_key: str, trigger_comment_id: str) -> bool:
    """Handle an on-demand /ai-lookup trigger from an internal comment.

    Skips the _has_agent_response check (technician explicitly requested).
    Includes the latest customer reply as additional context.
    Deletes the trigger comment after processing.
    """
    log.info("AI lookup triggered for %s (comment %s)", ticket_key, trigger_comment_id)

    allowed, _ = _check_assignee_allowed(ticket_key)
    if not allowed:
        _delete_comment(ticket_key, trigger_comment_id)
        return False

    context = _get_ticket_context(ticket_key)
    if not context:
        _delete_comment(ticket_key, trigger_comment_id)
        return False
    summary, desc = context

    # Append latest customer reply as follow-up context
    customer_reply = _get_latest_customer_reply(ticket_key)
    if customer_reply:
        desc = desc + "\n\nCUSTOMER FOLLOW-UP:\n" + customer_reply

    result = _lookup_and_draft(ticket_key, summary, desc, force=True)
    _delete_comment(ticket_key, trigger_comment_id)
    return result


# ── Plugin instance ───────────────────────────────────────────


class ResponderPlugin(BasePlugin):
    name = "responder"

    def register(self, app, config) -> None:
        """Build the shared BM25 and dense retrieval indexes during startup."""
        from .bm25 import build as build_bm25
        from .dense_retrieval import build as build_dense

        log.info("Building responder BM25 index")
        try:
            build_bm25()
        except Exception:
            log.exception("Failed to build responder BM25 index at startup")

        log.info("Building responder dense retrieval index")
        try:
            build_dense()
        except Exception:
            log.exception("Failed to build responder dense retrieval index at startup")

    def on_ticket(self, ticket_key: str, event: str, payload: dict) -> None:
        """Dispatch ticket events to the appropriate handler."""
        if event == "created":
            handle_new_ticket(ticket_key)
        elif event == "commented":
            comment_id = payload.get("comment_id", "")
            comment_body = payload.get("comment_body", "")

            if comment_body.strip() == AI_LOOKUP_TRIGGER:
                handle_ai_lookup_trigger(ticket_key, comment_id)
            elif comment_body.strip() == AI_REVIEW_TRIGGER:
                handle_review_trigger(ticket_key, comment_id)
            elif comment_body.strip() == AI_ASSIST_EMOJI:
                handle_ai_assist(ticket_key, comment_id)
            elif comment_body.strip() in FEEDBACK_EMOJI_MAP:
                handle_feedback_emoji(ticket_key, comment_id, comment_body.strip())

    def on_schedule(self) -> None:
        """Capture feedback, harvest examples, and rebuild the few-shot ANN index."""
        cfg = get_plugin_config("responder")
        if cfg.get("capture_feedback", True):
            try:
                capture_feedback()
            except Exception:
                log.exception("capture_feedback failed")
        if cfg.get("harvest_response_examples", True):
            try:
                harvest_response_examples()
            except Exception:
                log.exception("harvest_response_examples failed")
        try:
            ANNFewShotIndex.rebuild_daily()
        except Exception:
            log.exception("Responder few-shot ANN rebuild failed")


plugin = ResponderPlugin()
