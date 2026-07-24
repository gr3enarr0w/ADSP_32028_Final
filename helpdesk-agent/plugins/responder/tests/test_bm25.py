"""Tests for the responder BM25 retrieval index."""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

import pytest

import db as db_mod
from db import get_db_conn, init_db
from plugins.responder import bm25


def _seed_corpus() -> None:
    with get_db_conn() as conn:
        conn.execute(
            "INSERT INTO faq_sources (source_type, source_id, title, content_hash, last_fetched) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            ("google_doc", "FAQ-1", "OAuth 2LO setup guide", "hash-1"),
        )
        conn.execute(
            "INSERT INTO faq_sources (source_type, source_id, title, content_hash, last_fetched) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            ("google_doc", "FAQ-2", "<PROJECT_KEY> servicedeskapi reference", "hash-2"),
        )
        conn.execute(
            "INSERT INTO faq_sources (source_type, source_id, title, content_hash, last_fetched) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            ("google_doc", "FAQ-3", "Password reset FAQ", "hash-3"),
        )

        conn.execute(
            "INSERT INTO kb_articles (page_id, title, body_text, labels, topics_covered, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("KB-1", "Configure SLA policies", "Queues and request types.", "sla", "queues",),
        )
        conn.execute(
            "INSERT INTO kb_articles (page_id, title, body_text, labels, topics_covered, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("KB-2", "Manage request types", "Workflow portal details.", "portal", "workflow",),
        )
        conn.execute(
            "INSERT INTO kb_articles (page_id, title, body_text, labels, topics_covered, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("KB-3", "Permission scheme troubleshooting", "Project roles and access.", "permissions", "roles",),
        )

        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description, status, resolution, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("ANTSE-1", "SSO login loop", "User is stuck on SSO redirect.", "resolved", "Clear browser cache to fix the SSO login loop."),
        )
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description, status, resolution, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("ANTSE-2", "OAuth client secret rotation", "Integration broke after a secret change.", "closed", "Rotate the OAuth client secret after compromise."),
        )
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description, status, resolution, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("ANTSE-3", "Workflow transition condition", "Request type transition is failing.", "resolved", "Fix the workflow transition condition on the request type."),
        )

        conn.execute(
            "INSERT INTO atlassian_docs (url, product, title, fetched_at) VALUES (?, ?, ?, datetime('now'))",
            ("https://support.atlassian.com/jira-service-management-cloud/docs/set-up-queues/", "jira-service-management-cloud", "Set up queues"),
        )
        conn.execute(
            "INSERT INTO atlassian_docs (url, product, title, fetched_at) VALUES (?, ?, ?, datetime('now'))",
            ("https://developer.atlassian.com/cloud/jira/platform/rest/v3/authentication/", "jira-platform-api", "REST API authentication"),
        )
        conn.execute(
            "INSERT INTO atlassian_docs (url, product, title, fetched_at) VALUES (?, ?, ?, datetime('now'))",
            ("https://support.atlassian.com/jira-cloud/docs/manage-workflows/", "jira-cloud", "Manage workflows"),
        )


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(bm25, "_INDEX", bm25.BM25Index())
    init_db()
    _seed_corpus()
    bm25.build()
    yield


def test_tokenize_preserves_acronyms():
    assert bm25._tokenize("<PROJECT_KEY> OAuth servicedeskapi 2LO") == [
        "jiraconfsd",
        "oauth",
        "servicedeskapi",
        "2lo",
    ]


def test_build_logs_source_counts(caplog):
    caplog.set_level(logging.INFO)
    bm25.build()
    assert any("faq_sources=3" in record.message for record in caplog.records)
    assert any("kb_articles=3" in record.message for record in caplog.records)
    assert any("tickets=3" in record.message for record in caplog.records)
    assert any("atlassian_docs=3" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("query", "expected_doc_id", "expected_source_type"),
    [
        ("oauth 2lo", "FAQ-1", "faq_sources"),
        ("jiraconfsd servicedeskapi", "FAQ-2", "faq_sources"),
        ("password reset", "FAQ-3", "faq_sources"),
        ("sla policies", "KB-1", "kb_articles"),
        ("request types", "KB-2", "kb_articles"),
        ("project roles", "KB-3", "kb_articles"),
        ("browser cache sso", "ANTSE-1", "tickets"),
        ("oauth client secret", "ANTSE-2", "tickets"),
        ("workflow transition condition", "ANTSE-3", "tickets"),
        ("rest api authentication", "https://developer.atlassian.com/cloud/jira/platform/rest/v3/authentication/", "atlassian_docs"),
    ],
)
def test_manual_keyword_queries_return_relevant_results(query, expected_doc_id, expected_source_type):
    bm25.build()
    results = bm25.search(query, k=3)
    assert results
    assert results[0]["doc_id"] == expected_doc_id
    assert results[0]["source_type"] == expected_source_type
    assert "score" in results[0]
    assert "text" in results[0]


def test_responder_register_builds_index(monkeypatch):
    from plugins.responder import ResponderPlugin

    fresh = bm25.BM25Index()
    monkeypatch.setattr(bm25, "_INDEX", fresh)
    assert not bm25.get_index()._built
    ResponderPlugin().register(None, {})
    assert bm25.get_index()._built
    assert len(bm25.get_index().search("oauth 2lo", k=1)) == 1


def test_add_document_updates_search_results():
    bm25.build()
    bm25.add_document(
        "KB-NEW",
        "kb_articles",
        "Escalate Jira Service Management SLA breach notifications for service desk queues.",
    )

    results = bm25.search("sla breach notifications", k=3)
    assert results[0]["doc_id"] == "KB-NEW"
    assert results[0]["source_type"] == "kb_articles"


def test_add_document_upsert_no_duplicate():
    """Test A — calling add_document twice with same (source_type, doc_id) replaces, not duplicates."""
    bm25.build()
    index = bm25.get_index()

    bm25.add_document("KB-NEW", "kb_articles", "First version of the document about oauth.")
    count_after_first = len(index._docs)

    bm25.add_document("KB-NEW", "kb_articles", "Second version mentioning token refresh instead.")
    count_after_second = len(index._docs)

    assert count_after_second == count_after_first, (
        f"Expected {count_after_first} docs after upsert, got {count_after_second}"
    )

    results = bm25.search("token refresh", k=3)
    assert results, "Expected at least one result after upsert"
    assert results[0]["doc_id"] == "KB-NEW"
    assert "token refresh" in results[0]["text"]


def test_search_concurrent_thread_safety():
    """Test B — 4 concurrent threads searching a built index all return results without raising."""
    bm25.build()

    errors: list[Exception] = []
    results_per_thread: list[list[dict]] = [[] for _ in range(4)]

    def search_worker(idx: int) -> None:
        try:
            results_per_thread[idx] = bm25.search("oauth sso", k=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=search_worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread(s) raised exceptions: {errors}"
    for i, results in enumerate(results_per_thread):
        assert isinstance(results, list), f"Thread {i} returned non-list: {results!r}"
    for i, result in enumerate(results_per_thread):
        assert result, f"Thread {i} returned empty results — index may not have been built"


def main() -> None:
    """Simple manual smoke script for local keyword testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_mod.DB_PATH = str(Path(tmpdir) / "bm25.db")
        bm25._INDEX = bm25.BM25Index()
        init_db()
        _seed_corpus()
        bm25.build()

        queries = [
            "oauth 2lo",
            "jiraconfsd servicedeskapi",
            "password reset",
            "sla policies",
            "request types",
            "project roles",
            "browser cache sso",
            "oauth client secret",
            "workflow transition condition",
            "rest api authentication",
        ]
        for query in queries:
            top = bm25.search(query, k=1)
            print(f"{query!r} -> {top[0]['source_type']}:{top[0]['doc_id']} score={top[0]['score']:.3f}")


if __name__ == "__main__":
    main()
