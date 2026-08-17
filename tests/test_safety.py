"""Tests for rag.safety — the allowlist and hazard checks the prompts promise.

These exist because a safety rule stated only inside a prompt is not a safety
rule: it holds exactly as long as the model behaves. Everything here is
deterministic and needs no model.
"""
import pytest

from rag import safety


# ---------------------------------------------------------------------------
# host parsing / allowlist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://www.amazon.com/dp/B0X", "amazon.com"),
    ("https://Amazon.COM/dp/B0X", "amazon.com"),
    ("http://smile.amazon.com:8080/dp/B0X", "smile.amazon.com"),
    ("https://user:pw@target.com/p/1", "target.com"),
    ("", ""),
    (None, ""),
])
def test_host_of(url, expected):
    assert safety.host_of(url) == expected


def test_allowlist_accepts_listed_host_and_subdomains():
    assert safety.is_allowed("https://www.amazon.com/dp/B0X")
    assert safety.is_allowed("https://smile.amazon.com/dp/B0X")


def test_allowlist_rejects_unknown_and_lookalike_hosts():
    assert not safety.is_allowed("https://random-blog.example/post")
    # The classic suffix-matching bug: a lookalike host that merely *contains*
    # an allowed domain must not pass.
    assert not safety.is_allowed("https://amazon.com.phishing.example/dp/B0X")
    assert not safety.is_allowed("https://notamazon.com/dp/B0X")


def test_allowlist_is_env_overridable(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ALLOWLIST", "example.org, example.net")
    assert safety.get_allowlist() == ("example.org", "example.net")
    assert safety.is_allowed("https://example.org/x")
    assert not safety.is_allowed("https://amazon.com/x")


def test_wildcard_disables_filtering(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ALLOWLIST", "*")
    assert safety.is_allowed("https://anything.example/x")


# ---------------------------------------------------------------------------
# result filtering
# ---------------------------------------------------------------------------

def test_filter_web_results_partitions_and_explains():
    results = [
        {"title": "A", "url": "https://www.amazon.com/dp/1"},
        {"title": "B", "url": "https://sketchy.example/deal"},
        {"title": "C"},  # no url at all
    ]
    kept, blocked = safety.filter_web_results(results)
    assert [r["title"] for r in kept] == ["A"]
    assert [r["title"] for r in blocked] == ["B", "C"]
    assert all("blocked_reason" in b for b in blocked)
    assert "not on the allowlist" in blocked[0]["blocked_reason"]
    assert "no url" in blocked[1]["blocked_reason"]


def test_filter_web_results_handles_empty_and_none():
    assert safety.filter_web_results([]) == ([], [])
    assert safety.filter_web_results(None) == ([], [])


def test_filter_does_not_mutate_input():
    results = [{"title": "B", "url": "https://sketchy.example/deal"}]
    safety.filter_web_results(results)
    assert "blocked_reason" not in results[0]


# ---------------------------------------------------------------------------
# hazard flags
# ---------------------------------------------------------------------------

def test_bleach_and_ammonia_flags():
    assert "bleach_ammonia" in safety.hazard_flags(
        "Can I mix bleach and ammonia to clean the tile?")


def test_bleach_and_vinegar_flags():
    assert "bleach_acid" in safety.hazard_flags(
        "Is it fine to combine bleach with vinegar?")


def test_flags_span_multiple_texts():
    """The transcript and the retrieved ingredients together can be unsafe
    even when neither is on its own."""
    flags = safety.hazard_flags("Use this with my other spray",
                                "Ingredients: sodium hypochlorite",
                                "Ingredients: ammonium hydroxide")
    assert "bleach_ammonia" in flags


def test_ordinary_query_flags_nothing():
    assert safety.hazard_flags(
        "Recommend an eco-friendly stainless-steel cleaner under fifteen dollars."
    ) == []
    assert safety.hazard_flags("") == []
    assert safety.hazard_flags(None) == []


def test_caution_only_when_flagged():
    assert safety.caution_for(["bleach_ammonia"]) == safety.HAZARD_CAUTION
    assert safety.caution_for([]) == ""
