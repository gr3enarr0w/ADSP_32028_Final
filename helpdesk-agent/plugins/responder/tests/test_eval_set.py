"""Tests for scalable retrieval eval set construction."""

from __future__ import annotations

import json

import pytest

import db as db_mod
from core.pipeline import load_pipeline_config
from db import init_db
from plugins.responder.eval_set import (
    PRODUCTION_MIN_QUERIES,
    TEST_MIN_QUERIES,
    build_eval_manifest,
    build_eval_queries_from_db,
    get_eval_queries,
    load_eval_manifest,
    save_eval_manifest,
)
from plugins.responder.tests.test_retrieval import _seed_extended_corpus


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    init_db()
    _seed_extended_corpus()
    load_pipeline_config()
    yield


def test_build_eval_queries_scales_beyond_seed_minimum():
    queries = build_eval_queries_from_db(max_queries=100)
    assert len(queries) >= TEST_MIN_QUERIES


def test_build_eval_queries_includes_keyword_and_semantic():
    queries = build_eval_queries_from_db(max_queries=100)
    types = {q.query_type for q in queries}
    assert "keyword" in types
    assert "semantic" in types


def test_build_eval_queries_covers_all_source_types():
    queries = build_eval_queries_from_db(max_queries=100)
    sources = {q.source_type for q in queries}
    assert "tickets" in sources
    assert "kb_articles" in sources
    assert "faq_sources" in sources
    assert "atlassian_docs" in sources


def test_manifest_round_trip(tmp_path):
    manifest = build_eval_manifest(max_queries=50)
    path = save_eval_manifest(manifest, tmp_path / "eval.json")
    loaded = load_eval_manifest(path)
    assert loaded is not None
    assert loaded.count == manifest.count
    assert loaded.queries[0].query == manifest.queries[0].query


def test_get_eval_queries_builds_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_RESPONDER_RETRIEVAL_EVAL_MIN_QUERIES", str(TEST_MIN_QUERIES))
    load_pipeline_config()
    queries = get_eval_queries(
        min_queries=TEST_MIN_QUERIES,
        max_queries=100,
        path=tmp_path / "missing.json",
        prefer_file=True,
    )
    assert len(queries) >= TEST_MIN_QUERIES


def test_get_eval_queries_prefers_file_when_large_enough(tmp_path):
    manifest = build_eval_manifest(max_queries=100)
    path = save_eval_manifest(manifest, tmp_path / "eval.json")
    queries = get_eval_queries(
        min_queries=min(40, manifest.count),
        max_queries=100,
        path=path,
        prefer_file=True,
    )
    assert len(queries) == manifest.count


def test_production_minimum_is_two_hundred():
    assert PRODUCTION_MIN_QUERIES == 200


def test_manifest_json_is_valid(tmp_path):
    manifest = build_eval_manifest(max_queries=40)
    path = save_eval_manifest(manifest, tmp_path / "eval.json")
    data = json.loads(path.read_text())
    assert data["count"] == 40
    assert isinstance(data["queries"], list)
