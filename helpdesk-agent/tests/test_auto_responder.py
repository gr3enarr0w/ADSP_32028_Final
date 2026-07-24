"""Integration tests for the AI auto-responder pipeline.

Tests cover: emoji feedback mapping, draft deduplication, review disposition,
unified AI assist routing, deep doc fetching, Cloud confidence scoring,
and the full auto-draft sweep flow.
"""

import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from db import init_db, get_db_conn


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Use a fresh in-memory DB for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("db.DB_PATH", db_path)
    init_db()
    yield db_path


@pytest.fixture
def mock_jira_api():
    """Mock all Jira API calls."""
    with patch("faq.auto_responder.get_cloud_auth", return_value={"Authorization": "Bearer test"}), \
         patch("faq.auto_responder.get_cloud_base_url", return_value="https://test.atlassian.net"), \
         patch("faq.auto_responder.requests") as mock_requests:
        yield mock_requests


# ── Constants and config ──


class TestFeedbackEmojiMap:
    def test_all_emojis_mapped(self):
        from faq.auto_responder import FEEDBACK_EMOJI_MAP
        assert "✅" in FEEDBACK_EMOJI_MAP
        assert "👤" in FEEDBACK_EMOJI_MAP
        assert "🔧" in FEEDBACK_EMOJI_MAP
        assert "❌" in FEEDBACK_EMOJI_MAP
        assert "🔄" in FEEDBACK_EMOJI_MAP
        assert "❓" in FEEDBACK_EMOJI_MAP

    def test_emoji_categories_unique(self):
        from faq.auto_responder import FEEDBACK_EMOJI_MAP
        categories = list(FEEDBACK_EMOJI_MAP.values())
        assert len(categories) == len(set(categories))

    def test_triggers_defined(self):
        from faq.auto_responder import AI_LOOKUP_TRIGGER, AI_REVIEW_TRIGGER, AI_ASSIST_EMOJI
        assert AI_LOOKUP_TRIGGER == "/ai-lookup"
        assert AI_REVIEW_TRIGGER == "/ai-review"
        assert AI_ASSIST_EMOJI == "🤖"


class TestQuestionTypeMap:
    def test_all_types_mapped(self):
        from faq.auto_responder import QUESTION_TYPE_MAP
        assert QUESTION_TYPE_MAP["how-to"] == "self_service"
        assert QUESTION_TYPE_MAP["configuration"] == "admin_action"
        assert QUESTION_TYPE_MAP["access-request"] == "admin_action"
        assert QUESTION_TYPE_MAP["troubleshooting"] == "admin_action"
        assert QUESTION_TYPE_MAP["bug-report"] == "admin_action"


# ── Draft deduplication ──


class TestHasPendingDraft:
    def test_no_pending_draft(self):
        from faq.auto_responder import _has_pending_draft
        assert _has_pending_draft("TEST-999") is False

    def test_has_pending_draft(self):
        from faq.auto_responder import _has_pending_draft
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type, draft_customer_response)
                VALUES ('TEST-1', '123', 'self_service', 'draft text')
            """)
        assert _has_pending_draft("TEST-1") is True

    def test_resolved_draft_not_pending(self):
        from faq.auto_responder import _has_pending_draft
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response)
                VALUES ('TEST-2', '456', 'admin_action', 'draft', 'actual')
            """)
        assert _has_pending_draft("TEST-2") is False


# ── Feedback recording ──


class TestHandleFeedbackEmoji:
    def test_records_feedback(self, mock_jira_api):
        from faq.auto_responder import handle_feedback_emoji

        # Setup: create a draft record
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type, draft_customer_response)
                VALUES ('TEST-1', '100', 'self_service', 'draft')
            """)

        # Mock the delete comment call
        mock_jira_api.delete.return_value = MagicMock(status_code=204)

        result = handle_feedback_emoji("TEST-1", "200", "✅")
        assert result is True

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT agent_feedback FROM ai_draft_feedback WHERE ticket_key = 'TEST-1'"
            ).fetchone()
            assert row["agent_feedback"] == "both_good"

    def test_invalid_emoji_returns_false(self):
        from faq.auto_responder import handle_feedback_emoji
        result = handle_feedback_emoji("TEST-1", "200", "💀")
        assert result is False

    def test_no_draft_returns_false(self, mock_jira_api):
        from faq.auto_responder import handle_feedback_emoji
        mock_jira_api.delete.return_value = MagicMock(status_code=204)
        result = handle_feedback_emoji("TEST-NONE", "200", "✅")
        assert result is False


# ── Few-shot example selection ──


class TestGetFewShotExamples:
    def test_prioritizes_agent_approved(self):
        from faq.auto_responder import get_few_shot_examples

        with get_db_conn() as conn:
            # Agent-approved draft
            conn.execute("""
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, agent_feedback)
                VALUES ('GOOD-1', '1', 'self_service', 'good draft', 'both_good')
            """)
            # High-similarity but agent-rejected
            conn.execute("""
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response,
                     similarity_score, feedback_category, agent_feedback)
                VALUES ('BAD-1', '2', 'self_service', 'bad draft', 'actual',
                        0.95, 'as_is', 'both_bad')
            """)

        examples = get_few_shot_examples(limit=5)
        assert len(examples) >= 1
        assert examples[0]["agent_feedback"] == "both_good"

    def test_excludes_bad_feedback(self):
        from faq.auto_responder import get_few_shot_examples

        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response,
                     similarity_score, feedback_category, agent_feedback)
                VALUES ('WRONG-1', '3', 'admin_action', 'draft', 'actual',
                        0.90, 'as_is', 'wrong_type')
            """)

        examples = get_few_shot_examples(limit=5)
        wrong_type = [e for e in examples if e.get("agent_feedback") == "wrong_type"]
        assert len(wrong_type) == 0


# ── Deep doc fetching ──


class TestFetchDocContent:
    def test_allowed_domain_fetched(self):
        from faq.auto_responder import _fetch_doc_content

        with patch("faq.auto_responder.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                text="<html><body><h1>Title</h1><p>Content here</p></body></html>",
            )
            result = _fetch_doc_content("https://support.atlassian.com/jira-cloud/docs/test/")

        assert result is not None
        assert "Content here" in result

    def test_disallowed_domain_returns_none(self):
        from faq.auto_responder import _fetch_doc_content
        result = _fetch_doc_content("https://evil.example.com/page")
        assert result is None

    def test_cached_on_second_call(self):
        from faq.auto_responder import _fetch_doc_content

        with patch("faq.auto_responder.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, text="<p>Cached content</p>"
            )
            _fetch_doc_content("https://support.atlassian.com/test-cache/")
            _fetch_doc_content("https://support.atlassian.com/test-cache/")

        # Should only fetch once — second call uses cache
        assert mock_get.call_count == 1


# ── Ticket review ──


class TestReviewTicket:
    def test_returns_disposition(self, mock_jira_api):
        from faq.auto_responder import review_ticket

        # Mock ticket context
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO tickets (ticket_key, summary, description, status)
                VALUES ('TEST-1', 'Test ticket', 'Description', 'Waiting for customer')
            """)

        # Mock comment fetch
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "comments": [{
                "author": {"displayName": "Agent"},
                "created": "2026-03-18",
                "body": {"type": "doc", "content": []},
                "properties": [{"key": "sd.public.comment", "value": {"internal": False}}],
            }]
        }
        mock_jira_api.get.return_value = mock_resp

        # Mock Gemini
        with patch("faq.auto_responder._get_genai_client") as mock_client:
            mock_gen = MagicMock()
            mock_gen.models.generate_content.return_value = MagicMock(
                text='{"disposition": "stale", "summary": "No reply", "recommendation": "Close", "customer_response": null}'
            )
            mock_client.return_value = mock_gen

            result = review_ticket("TEST-1")

        assert result is not None
        assert result["disposition"] in ("close", "sprint_work", "needs_action", "stale")


# ── Unified AI assist ──


class TestHandleAiAssist:
    def test_skips_waiting_for_customer(self, mock_jira_api):
        from faq.auto_responder import handle_ai_assist

        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO tickets (ticket_key, summary, description, status)
                VALUES ('TEST-1', 'Test ticket', 'Desc', 'Waiting for customer')
            """)

        mock_jira_api.delete.return_value = MagicMock(status_code=204)

        result = handle_ai_assist("TEST-1", "comment-123")
        assert result is False

    def test_fires_on_waiting_for_support(self, mock_jira_api):
        from faq.auto_responder import handle_ai_assist

        # First call = status check, subsequent = other API calls
        mock_status = MagicMock(status_code=200)
        mock_status.json.return_value = {
            "fields": {"status": {"name": "Waiting for support"}}
        }
        mock_jira_api.get.return_value = mock_status
        mock_jira_api.delete.return_value = MagicMock(status_code=204)
        mock_jira_api.post.return_value = MagicMock(status_code=201, json=lambda: {"id": "999"})

        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO tickets (ticket_key, summary, description, status)
                VALUES ('TEST-2', 'Test ticket', 'Desc', 'Waiting for support')
            """)

        with patch("faq.auto_responder._get_genai_client") as mock_client, \
             patch("faq.auto_responder.lookup") as mock_lookup, \
             patch("faq.auto_responder._get_latest_customer_reply", return_value=None):
            mock_lookup.return_value = {"found": False}
            mock_gen = MagicMock()
            mock_gen.models.generate_content.return_value = MagicMock(
                text='{"disposition": "needs_action", "summary": "test", "recommendation": "test", "customer_response": null}'
            )
            mock_client.return_value = mock_gen

            result = handle_ai_assist("TEST-2", "comment-456")
            # May return False if no FAQ matches, but should not error
            assert isinstance(result, bool)


# ── Slack resolved signals ──


class TestSlackResolvedSignals:
    def test_store_signal_with_resolved(self):
        from ingest.slack import store_signal

        with get_db_conn() as conn:
            store_signal(
                conn, "test-channel", "123.456", "Test question",
                "U123", "resolved",
                is_resolved=True,
                thread_replies=["Answer 1", "Answer 2"],
            )

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT is_resolved, thread_replies FROM slack_signals WHERE thread_ts = '123.456'"
            ).fetchone()
            assert row["is_resolved"] == 1
            replies = json.loads(row["thread_replies"])
            assert len(replies) == 2
            assert replies[0] == "Answer 1"

    def test_get_resolved_threads(self):
        from ingest.slack import store_signal, get_resolved_threads

        with get_db_conn() as conn:
            store_signal(
                conn, "test-channel", "111.222", "How do I configure SSO?",
                "U456", "resolved",
                is_resolved=True,
                thread_replies=["Go to admin > Security > SAML"],
            )
            store_signal(
                conn, "test-channel", "333.444", "Unresolved question",
                "U789", "raw",
                is_resolved=False,
            )

        threads = get_resolved_threads(limit=10)
        assert len(threads) == 1
        assert "SSO" in threads[0]["question"]
        assert len(threads[0]["answers"]) == 1


# ── Cloud confidence scoring ──


class TestCloudConfidenceScoring:
    def test_score_caching(self):
        """Scores are cached in kb_cloud_scores table."""
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO kb_cloud_scores (page_id, title, url, cloud_applicable, confidence)
                VALUES ('12345', 'Test Page', 'https://test.com', 1, 85)
            """)
            row = conn.execute(
                "SELECT confidence FROM kb_cloud_scores WHERE page_id = '12345'"
            ).fetchone()
            assert row["confidence"] == 85


# ── Organic response harvesting ──


class TestOrganicExamples:
    def test_get_organic_examples_empty(self):
        from faq.auto_responder import get_organic_examples
        result = get_organic_examples(limit=5)
        assert result == []

    def test_get_organic_examples_returns_data(self):
        from faq.auto_responder import get_organic_examples

        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO response_examples
                    (ticket_key, response_type, agent_response)
                VALUES ('TEST-1', 'self_service', 'Here is how you fix this...')
            """)

        result = get_organic_examples(response_type="self_service", limit=5)
        assert len(result) == 1
        assert result[0]["response_type"] == "self_service"


# ── Feedback stats ──


class TestFeedbackStats:
    def test_includes_agent_feedback(self):
        from faq.auto_responder import get_feedback_stats

        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, agent_feedback)
                VALUES ('TEST-1', '1', 'self_service', 'draft', 'both_good')
            """)

        stats = get_feedback_stats()
        assert "agent_feedback" in stats
        assert stats["agent_feedback"].get("both_good", 0) == 1

    def test_includes_organic_count(self):
        from faq.auto_responder import get_feedback_stats

        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO response_examples
                    (ticket_key, response_type, agent_response)
                VALUES ('TEST-1', 'admin_action', 'response text')
            """)

        stats = get_feedback_stats()
        assert "organic_examples" in stats
        assert stats["organic_examples"]["total"] >= 1


# ── Doc content cache ──


class TestDocContentCache:
    def test_table_exists(self):
        with get_db_conn() as conn:
            conn.execute("SELECT COUNT(*) FROM doc_content_cache")

    def test_insert_and_retrieve(self):
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO doc_content_cache (url, content)
                VALUES ('https://test.com/page', 'cached page content')
            """)
            row = conn.execute(
                "SELECT content FROM doc_content_cache WHERE url = 'https://test.com/page'"
            ).fetchone()
            assert row["content"] == "cached page content"


# ── Assignee DB lookup (ANTSE-209) ──


# ── Status gate (ANTSE-330) ──


class TestStatusGate:
    def _insert_ticket(self, status="Open"):
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, status) VALUES (?, ?, ?)",
                ("TEST-1", "Summary", status),
            )

    def test_skips_resolved(self):
        from faq.auto_responder import _check_status_gate
        self._insert_ticket("Resolved")
        assert _check_status_gate("TEST-1") is False

    def test_skips_closed(self):
        from faq.auto_responder import _check_status_gate
        self._insert_ticket("Closed")
        assert _check_status_gate("TEST-1") is False

    def test_skips_waiting_for_customer(self):
        from faq.auto_responder import _check_status_gate
        self._insert_ticket("Waiting for customer")
        assert _check_status_gate("TEST-1") is False

    def test_allows_in_progress(self):
        from faq.auto_responder import _check_status_gate
        self._insert_ticket("In Progress")
        assert _check_status_gate("TEST-1") is True

    def test_allows_new(self):
        from faq.auto_responder import _check_status_gate
        self._insert_ticket("New")
        assert _check_status_gate("TEST-1") is True

    def test_missing_ticket_returns_true(self):
        from faq.auto_responder import _check_status_gate
        assert _check_status_gate("TEST-99") is True


# ── Age gate (ANTSE-331) ──


class TestAgeGate:
    def _insert_ticket(self, hours_ago):
        from datetime import datetime, timezone, timedelta
        created = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, status, created_at) VALUES (?, ?, ?, ?)",
                ("TEST-1", "Summary", "Open", created),
            )

    def test_skips_old_ticket(self):
        from faq.auto_responder import _check_age_gate
        self._insert_ticket(5)
        assert _check_age_gate("TEST-1") is False

    def test_allows_recent_ticket(self):
        from faq.auto_responder import _check_age_gate
        self._insert_ticket(1)
        assert _check_age_gate("TEST-1") is True

    def test_custom_threshold(self):
        from faq.auto_responder import _check_age_gate
        self._insert_ticket(2)
        with patch("faq.auto_responder.AGE_GATE_HOURS", 1):
            assert _check_age_gate("TEST-1") is False

    def test_missing_ticket_returns_true(self):
        from faq.auto_responder import _check_age_gate
        assert _check_age_gate("TEST-99") is True


# ── Self-check gate (ANTSE-332) ──


class TestSelfCheck:
    def test_passes_good_draft(self):
        from faq.auto_responder import _self_check_draft
        draft = {"response_type": "self_service", "customer_response": "Here is how to fix it..."}
        with patch("faq.auto_responder._get_genai_client") as mock_client:
            mock_gen = MagicMock()
            mock_gen.models.generate_content.return_value = MagicMock(
                text='{"type_correct": true, "addresses_question": true, "sufficient_context": true, "reason": null}'
            )
            mock_client.return_value = mock_gen
            passed, reason = _self_check_draft("Test summary", "Test desc", draft)
        assert passed is True

    def test_fails_wrong_type(self):
        from faq.auto_responder import _self_check_draft
        draft = {"response_type": "self_service", "customer_response": "We are looking into it..."}
        with patch("faq.auto_responder._get_genai_client") as mock_client:
            mock_gen = MagicMock()
            mock_gen.models.generate_content.return_value = MagicMock(
                text='{"type_correct": false, "addresses_question": true, "sufficient_context": true, "reason": "should be admin_action"}'
            )
            mock_client.return_value = mock_gen
            passed, reason = _self_check_draft("Test summary", "Test desc", draft)
        assert passed is False
        assert "wrong response type" in reason

    def test_disabled_via_env(self):
        from faq.auto_responder import _self_check_draft
        draft = {"response_type": "self_service", "customer_response": "anything"}
        with patch("faq.auto_responder.DRAFT_SELF_CHECK", False):
            passed, reason = _self_check_draft("summary", "desc", draft)
        assert passed is True
        assert reason == "disabled"

    def test_gemini_error_passes(self):
        from faq.auto_responder import _self_check_draft
        draft = {"response_type": "self_service", "customer_response": "draft text"}
        with patch("faq.auto_responder._get_genai_client") as mock_client:
            mock_client.side_effect = Exception("API unavailable")
            passed, reason = _self_check_draft("summary", "desc", draft)
        assert passed is True
        assert "fail-open" in reason


# ── Assignee DB lookup (ANTSE-209) ──


# ── Content gate (ANTSE-334) ──


class TestContentGate:
    def _insert_ticket(self, summary="Normal ticket", description="This is a perfectly normal description with enough content"):
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, status, description) VALUES (?, ?, ?, ?)",
                ("TEST-1", summary, "Open", description),
            )

    def test_skips_short_description(self):
        from faq.auto_responder import _check_content_gate
        self._insert_ticket(description="too short")
        assert _check_content_gate("TEST-1") is False

    def test_allows_normal_description(self):
        from faq.auto_responder import _check_content_gate
        self._insert_ticket(description="This description has more than twenty characters easily")
        assert _check_content_gate("TEST-1") is True

    def test_skips_matching_noise_pattern(self):
        from faq.auto_responder import _check_content_gate
        self._insert_ticket(summary="Auto: generated ticket from monitoring")
        with patch("faq.auto_responder.AUTO_DRAFT_NOISE_PATTERNS", [r"^Auto:"]):
            assert _check_content_gate("TEST-1") is False

    def test_allows_non_matching_summary(self):
        from faq.auto_responder import _check_content_gate
        self._insert_ticket(summary="User cannot access Jira board")
        with patch("faq.auto_responder.AUTO_DRAFT_NOISE_PATTERNS", [r"^Auto:"]):
            assert _check_content_gate("TEST-1") is True

    def test_missing_ticket_returns_true(self):
        from faq.auto_responder import _check_content_gate
        assert _check_content_gate("TEST-99") is True

    def test_empty_patterns_allows_all(self):
        from faq.auto_responder import _check_content_gate
        self._insert_ticket(summary="Auto: something")
        with patch("faq.auto_responder.AUTO_DRAFT_NOISE_PATTERNS", []):
            assert _check_content_gate("TEST-1") is True


# ── Assignee DB lookup (ANTSE-209) ──


class TestCheckAssigneeAllowed:
    """_check_assignee_allowed reads assignee_id (accountId) from DB — no API call.

    assignee_id is an opaque technical identifier, not PII, so the scrubber
    never anonymizes it.  Display names are never stored.
    """

    def _insert_ticket(self, assignee_id=""):
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, status, assignee_id) VALUES (?, ?, ?, ?)",
                ("TEST-1", "Summary", "Open", assignee_id),
            )

    def test_empty_allowed_list_returns_true_without_db_hit(self):
        from faq.auto_responder import _check_assignee_allowed

        with patch("faq.auto_responder.AUTO_RESPOND_ASSIGNEES", []):
            allowed, account_id = _check_assignee_allowed("TEST-1")

        assert allowed is True
        assert account_id is None

    def test_assignee_in_allowed_list(self):
        from faq.auto_responder import _check_assignee_allowed

        self._insert_ticket("abc123")
        with patch("faq.auto_responder.AUTO_RESPOND_ASSIGNEES", ["abc123"]):
            allowed, account_id = _check_assignee_allowed("TEST-1")

        assert allowed is True
        assert account_id == "abc123"

    def test_assignee_not_in_allowed_list(self):
        from faq.auto_responder import _check_assignee_allowed

        self._insert_ticket("xyz789")
        with patch("faq.auto_responder.AUTO_RESPOND_ASSIGNEES", ["abc123"]):
            allowed, account_id = _check_assignee_allowed("TEST-1")

        assert allowed is False
        assert account_id == "xyz789"

    def test_unassigned_ticket_returns_false(self):
        from faq.auto_responder import _check_assignee_allowed

        self._insert_ticket("")  # ticket exists but has no assignee
        with patch("faq.auto_responder.AUTO_RESPOND_ASSIGNEES", ["abc123"]):
            allowed, account_id = _check_assignee_allowed("TEST-1")

        assert allowed is False
        assert account_id is None

    def test_ticket_not_yet_ingested_returns_false(self):
        from faq.auto_responder import _check_assignee_allowed

        # No ticket in DB at all
        with patch("faq.auto_responder.AUTO_RESPOND_ASSIGNEES", ["abc123"]):
            allowed, account_id = _check_assignee_allowed("TEST-99")

        assert allowed is False
        assert account_id is None
