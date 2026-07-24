from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from db import get_db_conn
from plugins.responder import bm25, dense_retrieval
from plugins.responder.eval_set import get_eval_queries
from plugins.responder.lookup import lookup, rank_in_matches


class _FakeCursor:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self):
        self.queries: list[str] = []

    def execute(self, sql: str, params):
        self.queries.append(sql)
        if "FROM generated_articles" in sql:
            return _FakeCursor(
                [
                    {
                        "id": 101,
                        "article_topic": "oauth",
                        "title": "OAuth FAQ",
                        "body_html": "<p>Use OAuth 2LO</p>",
                        "confluence_url": "https://example.com/faq-101",
                    }
                ]
            )
        if "FROM kb_articles" in sql:
            return _FakeCursor(
                [
                    {
                        "page_id": "KB-200",
                        "title": "KB OAuth",
                        "url": "https://example.com/kb-200",
                        "body_text": "OAuth details",
                        "labels": "oauth",
                        "space_key": "HUB",
                    }
                ]
            )
        if "FROM tickets t" in sql:
            return _FakeCursor(
                [
                    {
                        "ticket_key": "ANTSE-300",
                        "summary": "OAuth fails",
                        "status": "resolved",
                        "resolution": "Updated token scopes",
                        "resolution_summary": "Regenerated token",
                        "category": "Authentication",
                        "issue_type": "OAuth",
                    }
                ]
            )
        if "FROM atlassian_docs" in sql:
            return _FakeCursor(
                [
                    {
                        "url": "https://support.atlassian.com/oauth",
                        "title": "OAuth docs",
                        "product": "jira-cloud",
                    }
                ]
            )
        return _FakeCursor([])


@contextmanager
def _fake_db_conn():
    yield _FakeConn()


def test_lookup_enriches_all_source_types():
    with (
        patch(
            "plugins.responder.lookup.search",
            return_value=[
                {"source_type": "faq_sources", "doc_id": "101", "text": "faq", "score": 0.9},
                {"source_type": "kb_articles", "doc_id": "KB-200", "text": "kb", "score": 0.8},
                {"source_type": "tickets", "doc_id": "ANTSE-300", "text": "ticket", "score": 0.7},
                {
                    "source_type": "atlassian_docs",
                    "doc_id": "https://support.atlassian.com/oauth",
                    "text": "doc",
                    "score": 0.6,
                },
            ],
        ),
        patch("plugins.responder.lookup.get_db_conn", _fake_db_conn),
    ):
        result = lookup("oauth issue")

    assert result["found"] is True
    assert result["faq_matches"][0]["id"] == 101
    assert result["kb_matches"][0]["page_id"] == "KB-200"
    assert result["ticket_matches"][0]["ticket_key"] == "ANTSE-300"
    assert result["atlassian_matches"][0]["url"] == "https://support.atlassian.com/oauth"


def test_lookup_returns_legacy_shape_without_response_draft():
    with patch("plugins.responder.lookup.search", return_value=[]), patch(
        "plugins.responder.lookup._legacy_lookup",
        return_value={
            "found": True,
            "query": "legacy",
            "faq_matches": [],
            "kb_matches": [],
            "ticket_matches": [],
            "atlassian_matches": [],
            "response_draft": "legacy text",
        },
    ):
        result = lookup("legacy")

    assert sorted(result.keys()) == sorted(
        ["found", "query", "faq_matches", "kb_matches", "ticket_matches", "atlassian_matches"]
    )
    assert "response_draft" not in result


def _db_populated() -> bool:
    try:
        with get_db_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM tickets").fetchone()
            return bool(row and row["c"] >= 100)
    except Exception:
        return False


@pytest.mark.skipif(not _db_populated(), reason="DB not populated")
def test_lookup_mrr_on_real_db():
    queries = get_eval_queries(min_queries=200, max_queries=300)
    if len(queries) < 200:
        pytest.skip("Need at least 200 real eval queries")

    # Explicitly build indexes for this DB-backed integration assertion.
    bm25.build()
    dense_retrieval.build()

    total_mrr = 0.0
    for query, source_type, doc_id in queries:
        rank = rank_in_matches(lookup(query), source_type, doc_id)
        if rank and rank <= 10:
            total_mrr += 1.0 / rank

    mrr = total_mrr / len(queries)
    assert mrr >= 0.5, f"MRR@10 was {mrr:.3f}, expected >= 0.5"
