"""Tests for output-layer cosine dedup gate in faq/google_docs.py."""

import pytest

from faq.dedup import COSINE_THRESHOLD, is_duplicate_of_sections


# ── is_duplicate_of_sections (pure function, no DB / Google API) ──


class TestIsDuplicateOfSections:
    """Unit tests for the section-level dedup helper."""

    def test_identical_content_detected(self):
        article = "How do I migrate my Jira project data to Cloud?"
        sections = [
            "How do I migrate my Jira project data to Cloud?",
            "Billing and payment setup for organization",
        ]
        is_dup, idx, sim = is_duplicate_of_sections(article, sections)
        assert is_dup is True
        assert idx == 0
        assert sim >= 0.99

    def test_different_content_passes(self):
        article = "Configure SAML single sign-on authentication for your organization"
        sections = [
            "How to update your billing and credit card payment method",
            "Steps for requesting a new Jira Cloud workspace",
        ]
        is_dup, idx, sim = is_duplicate_of_sections(article, sections)
        assert is_dup is False
        assert idx is None

    def test_empty_sections_allows_all(self):
        article = "Any article content should pass when there are no existing sections"
        is_dup, idx, sim = is_duplicate_of_sections(article, [])
        assert is_dup is False
        assert idx is None
        assert sim == 0.0

    def test_empty_article_not_duplicate(self):
        sections = ["Some existing section content about migration"]
        is_dup, idx, sim = is_duplicate_of_sections("", sections)
        assert is_dup is False
        assert sim == 0.0

    def test_none_article_not_duplicate(self):
        sections = ["Existing content"]
        is_dup, idx, sim = is_duplicate_of_sections(None, sections)
        assert is_dup is False
        assert sim == 0.0

    def test_paraphrased_content_detected(self):
        """Same words reordered should still be caught."""
        article = (
            "To migrate your data from Jira Server to Jira Cloud you need to "
            "install the migration assistant, run the pre-migration assessment, "
            "review the results, and start the migration process."
        )
        sections = [
            (
                "You need to install the migration assistant to migrate your data "
                "from Jira Server to Jira Cloud. Run the pre-migration assessment, "
                "review the results, and start the migration process."
            ),
        ]
        is_dup, idx, sim = is_duplicate_of_sections(article, sections)
        assert is_dup is True
        assert idx == 0

    def test_custom_threshold(self):
        """A very low threshold should flag even loosely related content."""
        article = "Information about migrating Jira server data"
        sections = ["Steps to migrate your Jira data to the cloud platform"]
        is_dup, _, _ = is_duplicate_of_sections(article, sections, threshold=0.1)
        assert is_dup is True

    def test_high_threshold_allows_similar(self):
        """Threshold of 1.0 should only catch exact matches."""
        article = "How to migrate Jira data to Cloud safely"
        sections = ["Steps for migrating Jira data to Cloud platform"]
        is_dup, _, sim = is_duplicate_of_sections(article, sections, threshold=1.0)
        assert is_dup is False
        assert sim < 1.0

    def test_matches_correct_section_index(self):
        """Should return the index of the best-matching section."""
        article = "Configure billing and payment methods"
        sections = [
            "How to set up SAML SSO for Jira Cloud",
            "Configure billing and payment methods for your account",
            "Migration steps for moving to Cloud",
        ]
        is_dup, idx, sim = is_duplicate_of_sections(article, sections, threshold=0.5)
        assert is_dup is True
        assert idx == 1

    def test_multiple_sections_all_empty_allows(self):
        sections = ["", "   ", ""]
        is_dup, idx, sim = is_duplicate_of_sections("real content here", sections)
        assert is_dup is False


# ── Section splitting (via _extract_text) ──


class TestExtractTextForDedup:
    """Test _extract_text section splitting used by the dedup gate."""

    def test_splits_by_heading(self):
        from faq.google_docs import _extract_text

        doc = {
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "paragraphStyle": {"namedStyleType": "HEADING_2"},
                            "elements": [{"textRun": {"content": "Migration FAQ\n"}}],
                        }
                    },
                    {
                        "paragraph": {
                            "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                            "elements": [{"textRun": {"content": "How to migrate.\n"}}],
                        }
                    },
                    {
                        "paragraph": {
                            "paragraphStyle": {"namedStyleType": "HEADING_2"},
                            "elements": [{"textRun": {"content": "Billing FAQ\n"}}],
                        }
                    },
                    {
                        "paragraph": {
                            "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                            "elements": [{"textRun": {"content": "Payment info.\n"}}],
                        }
                    },
                ],
            }
        }
        sections = _extract_text(doc)
        assert len(sections) == 2
        assert sections[0]["heading"] == "Migration FAQ"
        assert sections[0]["content"] == "How to migrate."
        assert sections[1]["heading"] == "Billing FAQ"
        assert sections[1]["content"] == "Payment info."

    def test_empty_doc_returns_no_sections(self):
        from faq.google_docs import _extract_text

        doc = {"body": {"content": []}}
        assert _extract_text(doc) == []

    def test_content_without_headings(self):
        from faq.google_docs import _extract_text

        doc = {
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                            "elements": [{"textRun": {"content": "Just plain text.\n"}}],
                        }
                    },
                ],
            }
        }
        sections = _extract_text(doc)
        assert len(sections) == 1
        assert sections[0]["heading"] == ""
        assert sections[0]["content"] == "Just plain text."


# ── Integration: write_faq_entries dedup filtering ──


class TestWriteFaqEntriesDedup:
    """Test that write_faq_entries filters duplicate entries.

    These tests mock the Google Docs API but exercise the real dedup logic.
    """

    def _make_doc_json(self, sections: list[tuple[str, str]]) -> dict:
        """Build a minimal Google Docs JSON with heading/content pairs."""
        content = []
        for heading, body_text in sections:
            if heading:
                content.append({
                    "paragraph": {
                        "paragraphStyle": {"namedStyleType": "HEADING_2"},
                        "elements": [{"textRun": {"content": f"{heading}\n"}}],
                    }
                })
            content.append({
                "paragraph": {
                    "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                    "elements": [{"textRun": {"content": f"{body_text}\n"}}],
                }
            })
        # Ensure endIndex exists on last element
        if content:
            content[-1]["endIndex"] = 100
        else:
            content.append({"endIndex": 1})
        return {"body": {"content": content}}

    def test_duplicate_entry_skipped(self, monkeypatch):
        """An entry matching an existing doc section should be filtered out."""
        from faq import google_docs

        existing_doc = self._make_doc_json([
            ("Migration FAQ", "How to migrate your Jira data to Cloud safely"),
        ])

        mock_service = _MockDocsService(existing_doc)
        monkeypatch.setattr(google_docs, "_get_docs_service", lambda: mock_service)
        monkeypatch.setattr(google_docs, "FAQ_OUTPUT_DOC_ID", "fake-doc-id")

        entries = [
            {
                "topic": "Migration FAQ",
                "question": "How to migrate your Jira data to Cloud safely",
                "answer": "How to migrate your Jira data to Cloud safely",
                "steps": [],
                "known_limitations": "",
            },
        ]

        result = google_docs.write_faq_entries(entries)
        assert result is True
        # batchUpdate should NOT have been called because the only entry was a dup
        assert mock_service.batch_update_called is False

    def test_new_entry_written(self, monkeypatch):
        """An entry NOT matching any existing section should be written."""
        from faq import google_docs

        existing_doc = self._make_doc_json([
            ("Billing FAQ", "How to update your credit card and payment methods"),
        ])

        mock_service = _MockDocsService(existing_doc)
        monkeypatch.setattr(google_docs, "_get_docs_service", lambda: mock_service)
        monkeypatch.setattr(google_docs, "FAQ_OUTPUT_DOC_ID", "fake-doc-id")

        entries = [
            {
                "topic": "SSO Setup",
                "question": "How to configure SAML single sign-on?",
                "answer": "Navigate to the admin console authentication section.",
                "steps": ["Open admin console", "Select authentication"],
                "known_limitations": "",
            },
        ]

        result = google_docs.write_faq_entries(entries)
        assert result is True
        assert mock_service.batch_update_called is True

    def test_empty_doc_allows_all_entries(self, monkeypatch):
        """When the doc is empty, all entries should pass through."""
        from faq import google_docs

        existing_doc = {"body": {"content": [{"endIndex": 1}]}}

        mock_service = _MockDocsService(existing_doc)
        monkeypatch.setattr(google_docs, "_get_docs_service", lambda: mock_service)
        monkeypatch.setattr(google_docs, "FAQ_OUTPUT_DOC_ID", "fake-doc-id")

        entries = [
            {
                "topic": "New Topic",
                "question": "A brand new question?",
                "answer": "A detailed answer about something entirely new.",
                "steps": [],
                "known_limitations": "",
            },
        ]

        result = google_docs.write_faq_entries(entries)
        assert result is True
        assert mock_service.batch_update_called is True

    def test_mixed_entries_partial_filter(self, monkeypatch):
        """One dup and one new — only the new one should be written."""
        from faq import google_docs

        existing_doc = self._make_doc_json([
            ("Migration FAQ", "How to migrate your Jira data to Cloud safely"),
        ])

        mock_service = _MockDocsService(existing_doc)
        monkeypatch.setattr(google_docs, "_get_docs_service", lambda: mock_service)
        monkeypatch.setattr(google_docs, "FAQ_OUTPUT_DOC_ID", "fake-doc-id")

        entries = [
            {
                "topic": "Migration FAQ",
                "question": "How to migrate your Jira data to Cloud safely",
                "answer": "How to migrate your Jira data to Cloud safely",
                "steps": [],
                "known_limitations": "",
            },
            {
                "topic": "Brand New Topic",
                "question": "Something completely different?",
                "answer": "An answer about configuring SSO authentication.",
                "steps": ["Step one", "Step two"],
                "known_limitations": "",
            },
        ]

        result = google_docs.write_faq_entries(entries)
        assert result is True
        assert mock_service.batch_update_called is True


# ── Mock helpers ──


class _MockDocsService:
    """Minimal mock for Google Docs API service."""

    def __init__(self, doc_json: dict):
        self._doc = doc_json
        self.batch_update_called = False

    def documents(self):
        return self

    def get(self, documentId=None):
        return self

    def batchUpdate(self, documentId=None, body=None):
        self.batch_update_called = True
        return self

    def execute(self):
        if self.batch_update_called:
            return {}
        return self._doc
