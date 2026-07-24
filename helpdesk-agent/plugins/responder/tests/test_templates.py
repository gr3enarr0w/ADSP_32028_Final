"""Tests for response template system (ANTSE-311)."""

from unittest.mock import patch

import pytest

from db import get_db_conn, init_db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr("db.DB_PATH", str(tmp_path / "test.db"))
    init_db()
    from plugins.responder.templates import seed_templates

    seed_templates()
    yield


def _insert_classification(
    ticket_key: str,
    category: str,
    issue_type: str = "Access request",
    confidence: float = 0.9,
    question_type: str = "access-request",
) -> None:
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO ticket_classifications
                (ticket_key, category, issue_type, question_type, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticket_key, category, issue_type, question_type, confidence),
        )


class TestSeedTemplates:
    def test_seeds_at_least_three_templates(self):
        with get_db_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM response_templates").fetchone()[0]
        assert count >= 3

    def test_seed_is_idempotent(self):
        from plugins.responder.templates import seed_templates

        assert seed_templates() == 0

    def test_seed_rewrites_stale_configuration_template(self):
        from plugins.responder.templates import seed_templates

        with get_db_conn() as conn:
            conn.execute(
                """
                UPDATE response_templates
                SET template_body = ?, is_customized = 0
                WHERE template_name = ? AND category = ? AND issue_type = ?
                """,
                (
                    "---CUSTOMER---\nold body\n---ADMIN---\nold steps",
                    "configuration_howto",
                    "Configuration",
                    "*",
                ),
            )

        assert seed_templates() == 0

        with get_db_conn() as conn:
            body = conn.execute(
                """
                SELECT template_body
                FROM response_templates
                WHERE template_name = ?
                """,
                ("configuration_howto",),
            ).fetchone()[0]

        assert body.count("---CUSTOMER---") == 1
        assert "Configuration request checklist:" in body
        assert "old body" not in body

    def test_seed_does_not_rewrite_customized_templates(self):
        from plugins.responder.templates import seed_templates

        with get_db_conn() as conn:
            conn.execute(
                """
                UPDATE response_templates
                SET template_body = ?, is_customized = 1
                WHERE template_name = ?
                """,
                (
                    "---CUSTOMER---\ncustom body\n---ADMIN---\ncustom steps",
                    "configuration_howto",
                ),
            )

        seed_templates()

        with get_db_conn() as conn:
            body = conn.execute(
                """
                SELECT template_body
                FROM response_templates
                WHERE template_name = ?
                """,
                ("configuration_howto",),
            ).fetchone()[0]

        assert "custom body" in body
        assert "Configuration request checklist:" not in body


class TestFindTemplate:
    def test_wildcard_match_by_category(self):
        from plugins.responder.templates import find_template

        tpl = find_template("Access", "SSO redirect loop after migration")
        assert tpl is not None
        assert tpl["template_name"] == "access_request"

    def test_exact_match_preferred(self):
        from plugins.responder.templates import find_template, seed_templates

        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO response_templates
                    (template_name, category, issue_type, template_body)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "sso_loop",
                    "Access",
                    "SSO redirect loop after migration",
                    "---CUSTOMER---\nSSO help\n---ADMIN---\n[STEPS]",
                ),
            )

        tpl = find_template("Access", "SSO redirect loop after migration")
        assert tpl["template_name"] == "sso_loop"

    def test_no_match_for_unknown_category(self):
        from plugins.responder.templates import find_template

        assert find_template("Notifications", "Email alert missing") is None


class TestExtractStepsFallback:
    _EMPTY_MATCHES = {
        "ticket_matches": [],
        "kb_matches": [],
        "faq_matches": [],
        "atlassian_matches": [],
    }

    def test_access_category_fallback(self):
        from plugins.responder.templates import _extract_steps

        steps = _extract_steps(self._EMPTY_MATCHES, category="Access")
        assert "Rover profile" in steps
        assert "Provision access" in steps

    def test_permissions_category_fallback(self):
        from plugins.responder.templates import _extract_steps

        steps = _extract_steps(self._EMPTY_MATCHES, category="Permissions")
        assert "permission change needed" in steps
        assert "admin.atlassian.com > Directory > Groups" in steps

    def test_configuration_category_fallback(self):
        from plugins.responder.templates import _extract_steps

        steps = _extract_steps(self._EMPTY_MATCHES, category="Configuration")
        assert "configuration setting being requested" in steps
        assert "Escalate to the platform team" in steps


class TestFillSlots:
    def test_replaces_all_named_slots(self):
        from plugins.responder.templates import fill_slots

        body = "System: [SYSTEM_NAME]\n[STEPS]\nDoc: [KB_URL]"
        filled = fill_slots(
            body,
            {
                "SYSTEM_NAME": "Jira",
                "STEPS": "1. Do the thing",
                "KB_URL": "[Guide](https://example.com)",
            },
        )
        assert "Jira" in filled
        assert "1. Do the thing" in filled
        assert "[Guide](https://example.com)" in filled
        assert "[" not in filled or "SYSTEM_NAME" not in filled

    def test_drops_empty_label_lines(self):
        from plugins.responder.templates import fill_slots

        body = "Reference: [KB_URL]\nAccess verification steps:\n[STEPS]\nDocumentation: [KB_URL]"
        filled = fill_slots(
            body,
            {
                "SYSTEM_NAME": "Jira",
                "STEPS": "1. Step",
                "KB_URL": "",
            },
        )

        assert "Reference:" not in filled
        assert "Documentation:" not in filled
        assert "Access verification steps:" in filled
        assert "1. Step" in filled


class TestExtractSystemName:
    def test_extracts_from_summary_first(self):
        from plugins.responder.templates import _extract_system_name

        # Bug: regex was matching across summary+description boundary
        # causing "RHEL Engineering We have a Confluence space..." capture
        summary = "Update Confluence space permissions for RHEL Engineering"
        description = 'We have a Confluence space named "RHEL Product Management" where we document future RHEL releases.'

        result = _extract_system_name(summary, description)
        assert result == "RHEL Engineering"
        assert "RHEL Product Management" not in result
        assert "Confluence space" not in result

    def test_access_to_pattern(self):
        from plugins.responder.templates import _extract_system_name

        result = _extract_system_name("Need access to Jira admin", "")
        assert result == "Jira admin"

    def test_configure_pattern(self):
        from plugins.responder.templates import _extract_system_name

        result = _extract_system_name("Configure webhook for Confluence", "")
        assert result == "webhook for Confluence"

    def test_falls_back_to_description(self):
        from plugins.responder.templates import _extract_system_name

        summary = "Access request"
        description = "Please grant access to Bitbucket Cloud."

        result = _extract_system_name(summary, description)
        assert result == "Bitbucket Cloud"

    def test_quoted_text_fallback(self):
        from plugins.responder.templates import _extract_system_name

        # Quoted text is used as fallback when no pattern matches
        result = _extract_system_name("Access request", 'Please grant access to "JIRA Admin".')
        # Pattern "access to X" matches and captures quoted content with period boundary
        assert '"JIRA Admin"' in result or "JIRA Admin" in result

    def test_returns_summary_when_no_match(self):
        from plugins.responder.templates import _extract_system_name

        result = _extract_system_name("Random access thing", "Some description")
        assert result == "Random access thing"

    def test_returns_default_when_empty(self):
        from plugins.responder.templates import _extract_system_name

        result = _extract_system_name("", "")
        assert result == "the requested resource"


class TestExtractKbLink:
    def test_returns_empty_when_no_source(self):
        from plugins.responder.templates import _extract_kb_link

        assert _extract_kb_link({}) == ("", "")
        assert _extract_kb_link({"kb_matches": [], "faq_matches": [], "atlassian_matches": []}) == ("", "")


class TestTryTemplateDraft:
    _MATCHES = {
        "found": True,
        "kb_matches": [
            {"title": "Manage permissions", "url": "https://support.atlassian.com/permissions"},
        ],
        "ticket_matches": [],
        "faq_matches": [],
        "atlassian_matches": [],
    }

    @patch("plugins.responder.templates.get_plugin_config")
    def test_returns_draft_when_confidence_and_template_match(self, mock_cfg):
        from plugins.responder.templates import try_template_draft

        mock_cfg.return_value = {
            "templates_enabled": True,
            "template_confidence_threshold": 0.75,
        }
        _insert_classification("TPL-1", "Access", confidence=0.92)

        draft = try_template_draft(
            "TPL-1",
            "Need access to Jira admin",
            "Please grant access to Jira admin console",
            self._MATCHES,
        )

        assert draft is not None
        assert draft["draft_mode"] == "template"
        assert draft["template_name"] == "access_request"
        assert draft["customer_response"]
        assert draft["admin_steps"]
        # Verify the KB match title was used in admin steps
        assert "Manage permissions" in draft["admin_steps"]
        # Confirm the old hardcoded fallback URL is no longer injected
        assert "https://support.atlassian.com" not in (draft["admin_steps"] or "")

    @patch("plugins.responder.templates.get_plugin_config")
    def test_returns_none_when_confidence_below_threshold(self, mock_cfg):
        from plugins.responder.templates import try_template_draft

        mock_cfg.return_value = {
            "templates_enabled": True,
            "template_confidence_threshold": 0.75,
        }
        _insert_classification("TPL-2", "Access", confidence=0.55)

        draft = try_template_draft(
            "TPL-2",
            "Need access",
            "Please grant access",
            self._MATCHES,
        )
        assert draft is None

    @patch("plugins.responder.templates.get_plugin_config")
    def test_returns_none_when_templates_disabled(self, mock_cfg):
        from plugins.responder.templates import try_template_draft

        mock_cfg.return_value = {"templates_enabled": False}
        _insert_classification("TPL-3", "Access", confidence=0.95)

        draft = try_template_draft("TPL-3", "Need access", "desc", self._MATCHES)
        assert draft is None

    @patch("plugins.responder.templates.get_plugin_config")
    def test_self_service_for_how_to_configuration(self, mock_cfg):
        from plugins.responder.templates import try_template_draft

        mock_cfg.return_value = {
            "templates_enabled": True,
            "template_confidence_threshold": 0.75,
        }
        _insert_classification(
            "TPL-4",
            "Configuration",
            issue_type="Workflow scheme setup",
            confidence=0.88,
            question_type="how-to",
        )

        draft = try_template_draft(
            "TPL-4",
            "Configure workflow scheme",
            "How do I configure the workflow scheme for my space?",
            self._MATCHES,
        )

        assert draft is not None
        assert draft["response_type"] == "self_service"

    @patch("plugins.responder.templates.get_plugin_config")
    def test_can_build_template_without_lookup_matches(self, mock_cfg):
        from plugins.responder.templates import try_template_draft

        mock_cfg.return_value = {
            "templates_enabled": True,
            "template_confidence_threshold": 0.75,
        }
        _insert_classification("TPL-7", "Access", confidence=0.9)

        draft = try_template_draft(
            "TPL-7",
            "Need access to Bitbucket",
            "Please grant access to repository tools",
            {"found": False, "kb_matches": [], "ticket_matches": [], "faq_matches": [], "atlassian_matches": []},
        )
        assert draft is not None
        assert draft["draft_mode"] == "template"


class TestLookupAndDraftIntegration:
    @patch("plugins.responder.jira_comments._post_self_check_and_decide", return_value=(True, "ok"))
    @patch("plugins.responder.jira_comments._post_draft_comment", return_value="comment-1")
    @patch("plugins.responder.feedback._store_draft_record")
    @patch("plugins.responder.drafting._draft_response")
    @patch("plugins.responder.lookup.lookup")
    @patch("plugins.responder.gates._has_pending_draft", return_value=False)
    @patch("plugins.responder.templates.get_plugin_config")
    def test_uses_template_without_gemini(
        self,
        mock_cfg,
        mock_pending,
        mock_lookup,
        mock_gemini,
        mock_store,
        mock_post,
        mock_self_check,
    ):
        from plugins.responder.drafting import _lookup_and_draft

        mock_cfg.return_value = {
            "templates_enabled": True,
            "template_confidence_threshold": 0.75,
        }
        mock_lookup.return_value = {
            "found": True,
            "kb_matches": [{"title": "KB", "url": "https://support.atlassian.com/kb"}],
            "ticket_matches": [],
            "faq_matches": [],
            "atlassian_matches": [],
        }
        _insert_classification("TPL-5", "Permissions", confidence=0.91)

        assert _lookup_and_draft("TPL-5", "Permissions issue", "Need edit access to Confluence") is True
        mock_gemini.assert_not_called()
        mock_post.assert_called_once()
        draft_arg = mock_post.call_args[0][1]
        assert draft_arg.get("draft_mode") == "template"

    @patch("plugins.responder.jira_comments._post_self_check_and_decide", return_value=(True, "ok"))
    @patch("plugins.responder.jira_comments._post_draft_comment", return_value="comment-2")
    @patch("plugins.responder.feedback._store_draft_record")
    @patch("plugins.responder.drafting._draft_response")
    @patch("plugins.responder.lookup.lookup")
    @patch("plugins.responder.gates._has_pending_draft", return_value=False)
    @patch("plugins.responder.templates.get_plugin_config")
    def test_falls_back_to_gemini_when_no_template(
        self,
        mock_cfg,
        mock_pending,
        mock_lookup,
        mock_gemini,
        mock_store,
        mock_post,
        mock_self_check,
    ):
        from plugins.responder.drafting import _lookup_and_draft

        mock_cfg.return_value = {
            "templates_enabled": True,
            "template_confidence_threshold": 0.75,
        }
        mock_lookup.return_value = {
            "found": True,
            "kb_matches": [{"title": "KB", "url": "https://support.atlassian.com/kb"}],
            "ticket_matches": [],
            "faq_matches": [],
            "atlassian_matches": [],
        }
        _insert_classification("TPL-6", "Notifications", confidence=0.95)
        mock_gemini.return_value = {
            "response_type": "admin_action",
            "customer_response": "We are looking into this.",
            "admin_steps": "1. Check settings",
            "missing_info": None,
        }

        assert _lookup_and_draft("TPL-6", "Alert missing", "I did not get the email alert") is True
        mock_gemini.assert_called_once()

    @patch("plugins.responder.jira_comments._post_self_check_and_decide", return_value=(True, "ok"))
    @patch("plugins.responder.jira_comments._post_draft_comment", return_value="comment-3")
    @patch("plugins.responder.feedback._store_draft_record")
    @patch("plugins.responder.drafting._draft_response")
    @patch("plugins.responder.lookup.lookup")
    @patch("plugins.responder.gates._has_pending_draft", return_value=False)
    @patch("plugins.responder.templates.get_plugin_config")
    def test_uses_template_when_lookup_misses(
        self,
        mock_cfg,
        _mock_pending,
        mock_lookup,
        mock_gemini,
        _mock_store,
        mock_post,
        _mock_self_check,
    ):
        from plugins.responder.drafting import _lookup_and_draft

        mock_cfg.return_value = {
            "templates_enabled": True,
            "template_confidence_threshold": 0.75,
        }
        mock_lookup.return_value = {
            "found": False,
            "kb_matches": [],
            "ticket_matches": [],
            "faq_matches": [],
            "atlassian_matches": [],
        }
        _insert_classification("TPL-8", "Access", confidence=0.95)

        assert _lookup_and_draft("TPL-8", "Need access", "Please grant Jira admin access") is True
        mock_gemini.assert_not_called()
        mock_post.assert_called_once()


SAMPLE_CONFLUENCE_STORAGE_HTML = """
<h2>access_request</h2>
<p>Category: Access / *</p>
<p>---CUSTOMER---</p>
<p>Thank you for your access request regarding [SYSTEM_NAME].</p>
<p>---ADMIN---</p>
<p>Access verification steps:</p>
<p>[STEPS]</p>
<p>Reference: [KB_URL]</p>
<h2>permissions_change</h2>
<p>Category: Permissions / *</p>
<p>---CUSTOMER---</p>
<p>We received your permissions request for [SYSTEM_NAME].</p>
<p>---ADMIN---</p>
<p>Permission change checklist:</p>
<p>[STEPS]</p>
"""


def _mock_confluence_response(html: str, status_code: int = 200):
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.status_code = status_code
    if status_code == 200:
        resp.json.return_value = {"body": {"storage": {"value": html}}}
    else:
        resp.text = "Unauthorized"
        resp.json.side_effect = ValueError("no json")
    return resp


class TestSyncTemplatesFromConfluence:
    @patch("plugins.responder.templates.requests.get")
    @patch("plugins.responder.templates.get_cloud_base_url", return_value="https://test.atlassian.net")
    @patch("plugins.responder.templates.get_cloud_auth", return_value={"Authorization": "Bearer test"})
    def test_sync_templates_from_confluence_parses_page(self, _mock_auth, _mock_base, mock_get):
        from plugins.responder.templates import sync_templates_from_confluence

        mock_get.return_value = _mock_confluence_response(SAMPLE_CONFLUENCE_STORAGE_HTML)

        with get_db_conn() as conn:
            conn.execute("DELETE FROM response_templates")
            conn.execute("DELETE FROM job_state")

        result = sync_templates_from_confluence("12345")

        assert result == {"synced": 2, "skipped": 0, "error": None}
        with get_db_conn() as conn:
            rows = conn.execute(
                """
                SELECT template_name, category, issue_type, template_body
                FROM response_templates
                ORDER BY template_name
                """
            ).fetchall()

        assert len(rows) == 2
        assert rows[0]["template_name"] == "access_request"
        assert rows[0]["category"] == "Access"
        assert rows[0]["issue_type"] == "*"
        assert "---CUSTOMER---" in rows[0]["template_body"]
        assert "[SYSTEM_NAME]" in rows[0]["template_body"]
        assert rows[1]["template_name"] == "permissions_change"
        assert rows[1]["category"] == "Permissions"

    @patch("plugins.responder.templates.requests.get")
    @patch("plugins.responder.templates.get_cloud_base_url", return_value="https://test.atlassian.net")
    @patch("plugins.responder.templates.get_cloud_auth", return_value={"Authorization": "Bearer test"})
    def test_sync_templates_skips_when_hash_unchanged(self, _mock_auth, _mock_base, mock_get):
        from plugins.responder.templates import sync_templates_from_confluence

        mock_get.return_value = _mock_confluence_response(SAMPLE_CONFLUENCE_STORAGE_HTML)

        with get_db_conn() as conn:
            conn.execute("DELETE FROM response_templates")
            conn.execute("DELETE FROM job_state")

        first = sync_templates_from_confluence("12345")
        second = sync_templates_from_confluence("12345")

        assert first == {"synced": 2, "skipped": 0, "error": None}
        assert second == {"synced": 0, "skipped": 1, "error": None}
        assert mock_get.call_count == 2

    @patch("plugins.responder.templates.requests.get")
    @patch("plugins.responder.templates.get_cloud_base_url", return_value="https://test.atlassian.net")
    @patch("plugins.responder.templates.get_cloud_auth", return_value={"Authorization": "Bearer test"})
    def test_sync_templates_returns_error_on_401(self, _mock_auth, _mock_base, mock_get):
        from plugins.responder.templates import sync_templates_from_confluence

        mock_get.return_value = _mock_confluence_response("", status_code=401)

        result = sync_templates_from_confluence("12345")

        assert result["synced"] == 0
        assert result["skipped"] == 0
        assert result["error"] is not None
        assert "401" in result["error"]

    @patch("plugins.responder.templates.requests.get")
    @patch("plugins.responder.templates.get_cloud_base_url", return_value="https://test.atlassian.net")
    @patch("plugins.responder.templates.get_cloud_auth", return_value={"Authorization": "Bearer test"})
    def test_sync_templates_replaces_missing_rows(self, _mock_auth, _mock_base, mock_get):
        from plugins.responder.templates import sync_templates_from_confluence

        html = """
        <h2>access_request</h2>
        <p>Category: Access / *</p>
        <p>---CUSTOMER---</p>
        <p>Updated access template.</p>
        """
        mock_get.return_value = _mock_confluence_response(html)

        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO response_templates
                    (template_name, category, issue_type, template_body)
                VALUES (?, ?, ?, ?)
                """,
                ("legacy_template", "Legacy", "*", "old body"),
            )

        result = sync_templates_from_confluence("12345")

        assert result == {"synced": 1, "skipped": 0, "error": None}
        with get_db_conn() as conn:
            names = [
                row["template_name"]
                for row in conn.execute(
                    "SELECT template_name FROM response_templates ORDER BY template_name"
                ).fetchall()
            ]

        assert names == ["access_request"]

    @patch("plugins.responder.templates.requests.get")
    @patch("plugins.responder.templates.get_cloud_base_url", return_value="https://test.atlassian.net")
    @patch("plugins.responder.templates.get_cloud_auth", return_value={"Authorization": "Bearer test"})
    def test_sync_templates_returns_error_on_malformed_page(self, _mock_auth, _mock_base, mock_get):
        from plugins.responder.templates import sync_templates_from_confluence

        mock_get.return_value = _mock_confluence_response("<h2>broken</h2><p>no category</p>")

        with get_db_conn() as conn:
            conn.execute("DELETE FROM job_state")

        result = sync_templates_from_confluence("12345")

        assert result["synced"] == 0
        assert result["skipped"] == 0
        assert result["error"] is not None
        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT last_run_date FROM job_state WHERE job_name = ?",
                ("template_sync_12345",),
            ).fetchone()

        assert row is None
