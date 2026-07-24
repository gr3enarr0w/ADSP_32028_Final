import pytest
from unittest.mock import patch, MagicMock

from scripts.index_confluence_kb import (
    _build_cql,
    _strip_storage_html,
    _match_topics,
    _resolve_next_url,
    parse_page,
    crawl_confluence_pages,
)


def test_build_cql():
    assert _build_cql(("HUB", "OMEGA")) == "space IN (HUB,OMEGA) AND type=page"
    assert _build_cql(("ENG",)) == "space IN (ENG) AND type=page"


def test_strip_storage_html():
    html = "<p>Hello <strong>world</strong></p>"
    assert _strip_storage_html(html) == "Hello world"
    assert _strip_storage_html(None) == ""


def test_match_topics():
    assert _match_topics("Access Request", "Need access to Jira") == "Access"
    assert "Workflow" in _match_topics("Workflow status", "Change transition")
    # Case-insensitive
    assert "Integration" in _match_topics("integration testing", "API setup")
    # Multiple topics
    topics = _match_topics("UI/UX layout", "Performance is slow")
    assert "UI/UX" in topics
    assert "Performance" in topics


def test_resolve_next_url():
    base = "https://api.atlassian.com/ex/confluence/123"
    # Absolute URL
    assert _resolve_next_url(base, "https://example.com/next") == "https://example.com/next"
    # Gateway root path
    assert _resolve_next_url(base, "/ex/confluence/123/rest/api/content/search?next=true") == "https://api.atlassian.com/ex/confluence/123/rest/api/content/search?next=true"
    # Relative path without /ex/
    assert _resolve_next_url(base, "/rest/api/content/search?next=true") == f"{base}/rest/api/content/search?next=true"
    assert _resolve_next_url(base, "rest/api") == f"{base}/rest/api"


def test_parse_page_valid():
    page = {
        "id": "1001",
        "title": "Setup Guide",
        "space": {"key": "HUB"},
        "body": {
            "storage": {
                "value": "<p>This is a setup guide.</p>" + (" filler text" * 20)
            }
        },
        "metadata": {
            "labels": {
                "results": [{"name": "guide"}, {"name": "setup"}]
            }
        }
    }

    parsed = parse_page(page, "https://example.com", "2026-06-04T00:00:00Z")
    assert parsed is not None
    assert parsed["page_id"] == "1001"
    assert parsed["space_key"] == "HUB"
    assert parsed["title"] == "Setup Guide"
    assert "This is a setup guide" in parsed["body_text"]
    assert parsed["url"] == "https://example.com/wiki/spaces/HUB/pages/1001"
    assert parsed["labels"] == "guide,setup"
    assert parsed["fetched_at"] == "2026-06-04T00:00:00Z"


def test_parse_page_too_short():
    page = {
        "id": "1002",
        "title": "Short",
        "body": {
            "storage": {
                "value": "<p>Too short</p>"
            }
        }
    }
    assert parse_page(page, "https://example.com", "2026-06-04T00:00:00Z") is None


def test_parse_page_no_id():
    page = {
        "title": "No ID",
        "body": {
            "storage": {
                "value": "<p>Content</p>" + (" filler text" * 20)
            }
        }
    }
    # Should be skipped because page_id is blank
    assert parse_page(page, "https://example.com", "2026-06-04T00:00:00Z") is None


@patch("scripts.index_confluence_kb._get_search")
@patch("scripts.index_confluence_kb.get_cloud_auth")
@patch("scripts.index_confluence_kb.get_cloud_base_url")
def test_crawl_confluence_pages_pagination(mock_base, mock_auth, mock_search):
    mock_base.return_value = "https://base"
    mock_auth.return_value = {"Authorization": "Bearer token"}

    # Page 1
    resp1 = MagicMock()
    resp1.status_code = 200
    resp1.json.return_value = {
        "results": [{"id": "1"}],
        "_links": {"next": "/next-page"}
    }

    # Page 2
    resp2 = MagicMock()
    resp2.status_code = 200
    resp2.json.return_value = {
        "results": [{"id": "2"}],
        "_links": {}  # No next link
    }

    # _get_search returns (response, headers)
    mock_search.side_effect = [(resp1, mock_auth.return_value), (resp2, mock_auth.return_value)]

    pages = crawl_confluence_pages(("HUB",))

    assert len(pages) == 2
    assert pages[0]["id"] == "1"
    assert pages[1]["id"] == "2"
    assert mock_search.call_count == 2

    # Second call must use the resolved next URL, not the original search endpoint
    second_call_url = mock_search.call_args_list[1][0][0]
    assert second_call_url == "https://base/next-page"
    # params must be None on subsequent pages (cursor embedded in URL)
    second_call_params = mock_search.call_args_list[1][1]["params"]
    assert second_call_params is None
