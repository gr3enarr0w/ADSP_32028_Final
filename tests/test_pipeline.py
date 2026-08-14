"""Smoke tests for the RAG pipeline (offline hash embedder)."""
from rag.schema import parse_price, parse_rating, parse_size_oz, price_per_oz, split_features
from rag.rag_search import rag_search

REQUIRED_KEYS = {"sku", "title", "price", "rating", "brand", "ingredients", "doc_id"}


# ---- normalization --------------------------------------------------------

def test_parse_price():
    assert parse_price("$ 12.49") == 12.49
    assert parse_price("1,234.00") == 1234.0
    assert parse_price("") is None
    assert parse_price("nan") is None


def test_parse_rating():
    assert parse_rating("4.6") == 4.6
    assert parse_rating("4.6 out of 5 stars") == 4.6
    assert parse_rating("9.9") is None  # out of range


def test_parse_size_and_ppo():
    assert parse_size_oz("16 Fl Oz") == 16.0
    assert round(parse_size_oz("500 ml"), 1) == 16.9
    assert price_per_oz(12.49, 16.0) == round(12.49 / 16.0, 4)
    assert price_per_oz(10.0, None) is None


def test_split_features_drops_boilerplate():
    feats = split_features("Make sure this fits by entering your model number. | Plant-based | Streak-free")
    assert feats == ["Plant-based", "Streak-free"]


# ---- retrieval / rag.search ----------------------------------------------

def test_rag_search_returns_hits():
    out = rag_search("stainless steel cleaner", k=3)
    assert out["count"] > 0
    assert len(out["results"]) == out["count"]


def test_rag_search_schema():
    out = rag_search("plant-based dish soap", k=2)
    for r in out["results"]:
        assert REQUIRED_KEYS.issubset(r.keys())
        assert isinstance(r["doc_id"], str) and r["doc_id"]


def test_price_filter_is_honored():
    out = rag_search("cleaner", k=10, filters={"price_max": 10})
    assert out["count"] > 0
    assert all(r["price"] is not None and r["price"] <= 10 + 1e-6 for r in out["results"])


def test_material_filter_is_honored():
    out = rag_search("cleaner", k=10, filters={"material": "glass"})
    for r in out["results"]:
        # material filter should keep glass/cooktop products; title reflects it
        assert any(w in r["title"].lower() for w in ["glass", "window", "cooktop", "stove"])


# ---- groundedness metric --------------------------------------------------

def test_groundedness_score():
    import run_eval
    assert run_eval.groundedness_score(["a", "b"], ["a", "b", "c"]) == 1.0
    assert run_eval.groundedness_score(["a", "x"], ["a", "b"]) == 0.5
    assert run_eval.groundedness_score([], ["a"]) == 0.0
