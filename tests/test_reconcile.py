"""Offline smoke tests for rag.reconcile — no index/network required.

Follows the convention in tests/test_pipeline.py (plain pytest, no fixtures
needed here since reconcile() takes plain dicts, not the retriever/index).
"""
from rag.reconcile import reconcile

RAG_ITEM = {
    "sku": "SKU-GREE-000",
    "title": "Steel-Safe Eco Stainless Steel Cleaner & Polish, 16 oz",
    "price": 12.49,
    "rating": 4.6,
    "brand": "GreenGleam",
    "ingredients": "Water, Caprylyl/Capryl Glucoside ...",
    "doc_id": "74d9f6149b125affab1f3b8d14798b0b",
    "url": "https://www.amazon.com/dp/B0SAMPLE000",
    "price_per_oz": 0.7806,
    "score": 2.97,
}


def test_reconcile_returns_expected_shape():
    out = reconcile([RAG_ITEM], [])
    assert set(out.keys()) == {"items", "unmatched_web"}
    assert len(out["items"]) == 1
    item = out["items"][0]
    assert item["live_match"] is None
    assert item["discrepancy"] is None
    # all original rag fields must survive untouched
    for k, v in RAG_ITEM.items():
        assert item[k] == v


def test_brand_match_flags_price_discrepancy_over_threshold():
    web = [{
        "title": "GreenGleam Steel-Safe Eco Stainless Steel Cleaner, 16oz",
        "url": "https://example.com/greengleam-steel-safe",
        "snippet": "GreenGleam's popular stainless steel cleaner, now $14.99.",
        "price": 14.99,  # ~20% over 12.49
        "availability": None,
    }]
    out = reconcile([RAG_ITEM], web)
    item = out["items"][0]
    assert item["live_match"]["url"] == web[0]["url"]
    assert item["discrepancy"] is not None
    assert item["discrepancy"]["type"] == "price"
    # private fact must remain the grounded baseline, never overwritten
    assert item["price"] == 12.49


def test_small_price_difference_under_threshold_is_not_flagged():
    web = [{
        "title": "GreenGleam Steel-Safe Eco Stainless Steel Cleaner, 16oz",
        "url": "https://example.com/greengleam-steel-safe",
        "snippet": "GreenGleam stainless steel cleaner.",
        "price": 12.99,  # ~4% over 12.49, under 10% default threshold
        "availability": None,
    }]
    out = reconcile([RAG_ITEM], web)
    assert out["items"][0]["discrepancy"] is None


def test_none_price_and_availability_never_crash_or_flag():
    """Current web.search always returns price/availability=None; must be a no-op."""
    web = [{
        "title": "Steel-Safe Eco Stainless Steel Cleaner & Polish",
        "url": "https://example.com/steel-safe",
        "snippet": "Buy the eco stainless steel cleaner online.",
        "price": None,
        "availability": None,
    }]
    out = reconcile([RAG_ITEM], web)
    item = out["items"][0]
    assert item["live_match"] is not None  # fuzzy title still matches
    assert item["discrepancy"] is None


def test_no_match_leaves_web_result_unmatched():
    web = [{
        "title": "Completely unrelated grocery item",
        "url": "https://example.com/x",
        "snippet": "n/a",
        "price": None,
        "availability": None,
    }]
    out = reconcile([RAG_ITEM], web)
    assert out["items"][0]["live_match"] is None
    assert out["unmatched_web"] == web


def test_availability_conflict_is_flagged():
    web = [{
        "title": "GreenGleam Steel-Safe Eco Stainless Steel Cleaner",
        "url": "https://example.com/greengleam-oos",
        "snippet": "GreenGleam cleaner.",
        "price": None,
        "availability": "Out of Stock",
    }]
    out = reconcile([RAG_ITEM], web)
    assert out["items"][0]["discrepancy"]["type"] == "availability"


def test_empty_inputs_are_safe():
    assert reconcile([], []) == {"items": [], "unmatched_web": []}
    out = reconcile([], [{"title": "x", "url": "u", "snippet": "s",
                           "price": None, "availability": None}])
    assert out["items"] == []
    assert len(out["unmatched_web"]) == 1
