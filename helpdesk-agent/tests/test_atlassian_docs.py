"""Tests for Atlassian docs sitemap-only indexing pipeline.

Covers: title derivation, sitemap parsing, DB storage, search, and
the end-to-end lookup fallback path (no internal matches → Atlassian docs).
"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from db import init_db, get_db_conn
from faq.atlassian_docs import (
    _title_from_url,
    _is_cloud_url,
    _product_from_url,
    fetch_doc_site,
    search_atlassian_docs,
    read_atlassian_docs,
)


SAMPLE_SITEMAP_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://support.atlassian.com/jira-service-management-cloud/docs/configure-sla-policies/</loc>
    <lastmod>2026-01-15</lastmod>
  </url>
  <url>
    <loc>https://support.atlassian.com/jira-service-management-cloud/docs/set-up-queues/</loc>
    <lastmod>2026-02-01</lastmod>
  </url>
  <url>
    <loc>https://support.atlassian.com/jira-service-management-cloud/docs/manage-request-types/</loc>
  </url>
  <url>
    <loc>https://support.atlassian.com/jira-service-management-cloud/server/old-page/</loc>
  </url>
  <url>
    <loc>https://confluence.atlassian.com/doc/some-dc-page/</loc>
  </url>
</urlset>
"""


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("db.DB_PATH", db_path), \
         patch("faq.atlassian_docs.get_db_conn", get_db_conn):
        init_db()
        yield db_path


# --- Unit tests ---

class TestTitleFromUrl:
    def test_basic_slug(self):
        url = "https://support.atlassian.com/jira-service-management-cloud/docs/configure-sla-policies/"
        assert _title_from_url(url) == "Configure Sla Policies"

    def test_underscores(self):
        url = "https://example.com/docs/some_feature_guide/"
        assert _title_from_url(url) == "Some Feature Guide"

    def test_trailing_slash(self):
        url = "https://support.atlassian.com/jira-cloud/docs/set-up-queues/"
        assert _title_from_url(url) == "Set Up Queues"

    def test_no_path(self):
        assert _title_from_url("https://example.com/") == ""
        assert _title_from_url("https://example.com") == ""


class TestIsCloudUrl:
    def test_cloud_url(self):
        assert _is_cloud_url("https://support.atlassian.com/jira-service-management-cloud/docs/foo/")

    def test_rejects_server(self):
        assert not _is_cloud_url("https://support.atlassian.com/jira-service-management-cloud/server/foo/")

    def test_rejects_data_center(self):
        assert not _is_cloud_url("https://support.atlassian.com/jira/data-center/foo/")

    def test_rejects_wrong_host(self):
        assert not _is_cloud_url("https://confluence.atlassian.com/doc/foo/")


class TestProductFromUrl:
    def test_jsm(self):
        assert _product_from_url("https://support.atlassian.com/jira-service-management-cloud/docs/foo/") == "jira-service-management-cloud"

    def test_unknown(self):
        assert _product_from_url("https://example.com/foo/") == "unknown"


# --- Integration tests ---

class TestFetchDocSite:
    def test_indexes_sitemap_urls(self):
        """fetch_doc_site parses sitemap XML and stores URL+title in DB."""
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_SITEMAP_XML.encode()
        mock_resp.raise_for_status = MagicMock()

        with patch("faq.atlassian_docs.requests.get", return_value=mock_resp):
            stored = fetch_doc_site(
                "https://support.atlassian.com/jira-service-management-cloud/resources/"
            )

        # 3 cloud URLs (server + wrong-host filtered out)
        assert stored == 3

        docs = read_atlassian_docs()
        assert len(docs) == 3

        titles = {d["title"] for d in docs}
        assert "Configure Sla Policies" in titles
        assert "Set Up Queues" in titles
        assert "Manage Request Types" in titles

        # Verify no body_text column
        assert "body_text" not in docs[0]

    def test_empty_sitemap(self):
        mock_resp = MagicMock()
        mock_resp.content = b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
        mock_resp.raise_for_status = MagicMock()

        with patch("faq.atlassian_docs.requests.get", return_value=mock_resp):
            stored = fetch_doc_site(
                "https://support.atlassian.com/jira-service-management-cloud/resources/"
            )
        assert stored == 0

    def test_sitemap_fetch_failure(self):
        import requests as req
        with patch("faq.atlassian_docs.requests.get", side_effect=req.RequestException("timeout")):
            stored = fetch_doc_site(
                "https://support.atlassian.com/jira-service-management-cloud/resources/"
            )
        assert stored == 0


class TestSearchAtlassianDocs:
    def _seed_docs(self):
        """Insert test docs directly."""
        with get_db_conn() as conn:
            for url, product, title in [
                ("https://support.atlassian.com/jsm-cloud/docs/configure-sla-policies/",
                 "jira-service-management-cloud", "Configure Sla Policies"),
                ("https://support.atlassian.com/jsm-cloud/docs/set-up-queues/",
                 "jira-service-management-cloud", "Set Up Queues"),
                ("https://support.atlassian.com/jira-cloud/docs/manage-workflows/",
                 "jira-cloud", "Manage Workflows"),
            ]:
                conn.execute(
                    "INSERT INTO atlassian_docs (url, product, title, fetched_at) VALUES (?, ?, ?, datetime('now'))",
                    (url, product, title),
                )

    def test_search_by_title_keyword(self):
        self._seed_docs()
        with get_db_conn() as conn:
            results = search_atlassian_docs(conn, ["configure"])
        assert len(results) == 1
        assert results[0]["title"] == "Configure Sla Policies"

    def test_search_multiple_words(self):
        self._seed_docs()
        with get_db_conn() as conn:
            results = search_atlassian_docs(conn, ["queues", "workflows"])
        assert len(results) == 2

    def test_search_no_match(self):
        self._seed_docs()
        with get_db_conn() as conn:
            results = search_atlassian_docs(conn, ["nonexistent"])
        assert results == []

    def test_search_empty_words(self):
        with get_db_conn() as conn:
            results = search_atlassian_docs(conn, [])
        assert results == []

    def test_results_have_no_body_text(self):
        self._seed_docs()
        with get_db_conn() as conn:
            results = search_atlassian_docs(conn, ["configure"])
        assert "body_text" not in results[0]
        assert "url" in results[0]
        assert "product" in results[0]
        assert "title" in results[0]


class TestLookupFallback:
    """End-to-end: lookup() falls back to Atlassian docs when no internal matches exist."""

    def _seed_atlassian_only(self):
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO atlassian_docs (url, product, title, fetched_at) VALUES (?, ?, ?, datetime('now'))",
                ("https://support.atlassian.com/jsm-cloud/docs/configure-sla-policies/",
                 "jira-service-management-cloud", "Configure Sla Policies"),
            )

    def test_lookup_returns_atlassian_match_when_no_internal(self):
        self._seed_atlassian_only()

        # Patch CLOUD_URL and mock the live Confluence CQL search
        with patch("faq.lookup.CLOUD_URL", "https://test.atlassian.net"), \
             patch("faq.lookup.requests.get", return_value=MagicMock(status_code=200, json=lambda: {"results": []})):
            from faq.lookup import lookup
            result = lookup("configure policies")

        assert result["found"] is True
        assert len(result["atlassian_matches"]) == 1
        assert result["atlassian_matches"][0]["title"] == "Configure Sla Policies"
        assert "body_text" not in result["atlassian_matches"][0]
        assert result["faq_matches"] == []
        assert result["kb_matches"] == []
        assert result["ticket_matches"] == []

    def test_lookup_includes_atlassian_alongside_internal_matches(self):
        self._seed_atlassian_only()

        # Add an internal FAQ match
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO generated_articles (article_topic, title, body_html, format)
                VALUES (?, ?, ?, 'faq')
            """, ("configure policies", "How to Configure Policies", "<p>Steps here</p>"))

        with patch("faq.lookup.CLOUD_URL", "https://test.atlassian.net"), \
             patch("faq.lookup.requests.get", return_value=MagicMock(status_code=200, json=lambda: {"results": []})):
            from faq.lookup import lookup
            result = lookup("configure policies")

        assert result["found"] is True
        assert len(result["faq_matches"]) >= 1
        # Atlassian docs now always supplement internal matches
        assert len(result["atlassian_matches"]) >= 1
