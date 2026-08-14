"""
reconcile.py — private/live reconciliation logic for the Retriever node.

Source spec: prompts/retriever_tool_instructions.md, "## Reconciliation
(when both are used)" section (see also the "Do / Don't" section referenced
below). This module is a standalone, importable helper that the shared
LangGraph Retriever node (assembled elsewhere by the team) will call after it
has invoked both `rag.search` (Shane, `rag_search.py` /
`mcp/README_mcp_rag.md`) and `web.search` (Clark, `web-search-mcp/web_search.py`).
It does not itself call either tool.

Per the spec:
  - Match private (`rag.search`) <-> live (`web.search`) items by `sku`, then
    `brand`, then fuzzy `title`.
  - If prices differ by > 10% or availability conflicts, keep the private
    fact as the grounded baseline and flag the discrepancy with BOTH
    citations (`doc_id` for the private item, `url` for the live item) so the
    Answerer can say "listed at $X; currently ~$Y online".
  - Never let a live result silently overwrite a private fact.

Schema notes (re-checked against the two READMEs before implementing):
  - rag.search results: {sku, title, price, rating, brand, ingredients,
    doc_id, url, price_per_oz, score}  (mcp/README_mcp_rag.md). No
    `availability` field exists on the private side today.
  - web.search results: {title, url, snippet, price, availability}
    (web-search-mcp/web_search.py). `sku` does not exist on the web side, and
    `price`/`availability` are *currently always None* in the shipped
    implementation (a documented known limitation) — so sku-matching across
    sources is moot today and price/availability discrepancies will rarely
    fire until a teammate fills those fields in. The code below still
    implements the full matching/discrepancy chain so it "just works" the
    day that limitation is lifted, and treats `None` price/availability as
    "no conflict, nothing to flag" rather than crashing.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

# Fuzzy title-match acceptance threshold (SequenceMatcher ratio, 0..1).
# 0.6 was tested against real rag.search output and produced false positives
# (e.g. two unrelated stainless-steel-cleaner titles scored 0.61) — raised to
# 0.72, which still passes near-identical titles (typically >0.85) while
# rejecting merely-same-category ones.
TITLE_MATCH_THRESHOLD = 0.72


def _normalize(text: Optional[str]) -> str:
    """Lowercase + collapse whitespace for comparison. Never raises on None."""
    if not text:
        return ""
    return " ".join(text.lower().split())


def _title_similarity(a: Optional[str], b: Optional[str]) -> float:
    """difflib.SequenceMatcher ratio over normalized titles. 0.0 if either is empty."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _brand_in_text(brand: Optional[str], *texts: Optional[str]) -> bool:
    """True if the rag item's brand name appears verbatim in any of the given texts
    (web title/snippet). Used as the exact-match step of the fallback chain."""
    nb = _normalize(brand)
    if not nb:
        return False
    return any(nb in _normalize(t) for t in texts)


# Tiers for candidate ranking — higher tier always outranks a lower one,
# regardless of the score within each tier (sku beats brand beats title).
_TIER_SKU = 3
_TIER_BRAND = 2
_TIER_TITLE = 1


def _best_candidate(rag_item: dict, web_item: dict) -> Optional[tuple[int, float]]:
    """Score one (rag_item, web_item) pair. Returns (tier, score) or None if
    the pair doesn't clear any matching rule at all.

    Matching rules (per retriever_tool_instructions.md), adapted to the
    actual schemas confirmed in the module docstring:
      1. sku — web.search results carry no `sku` field today, so this tier is
         unreachable in practice; implemented so it activates automatically
         if a teammate ever adds a comparable identifier to web results.
      2. brand — exact match: the rag item's `brand` appears verbatim in the
         web result's title or snippet.
      3. fuzzy title — SequenceMatcher ratio >= TITLE_MATCH_THRESHOLD.
    """
    rag_sku = rag_item.get("sku")
    if rag_sku and web_item.get("sku") and web_item["sku"] == rag_sku:
        return (_TIER_SKU, 1.0)

    if _brand_in_text(rag_item.get("brand"), web_item.get("title"), web_item.get("snippet")):
        return (_TIER_BRAND, 1.0)

    ratio = _title_similarity(rag_item.get("title"), web_item.get("title"))
    if ratio >= TITLE_MATCH_THRESHOLD:
        return (_TIER_TITLE, ratio)

    return None


def _match_all(rag_results: list[dict], web_results: list[dict]) -> dict[int, int]:
    """Global one-to-one matching between rag_results and web_results indices.

    A naive per-rag-item "take the first candidate that clears the bar" (the
    original implementation) is order-dependent: if rag item A is processed
    before the true best match B, and A's title happens to also clear the
    fuzzy threshold against the same web result, A steals it and B is left
    unmatched or paired with something worse. Verified against real
    rag.search output: two different stainless-steel-cleaner products both
    cleared the old 0.6 threshold against the same web result.

    Fix: score every (rag, web) pair, then greedily assign highest-tier /
    highest-score pairs first, skipping either side once it's claimed. This
    guarantees the strongest match wins regardless of list order.

    Returns: {rag_index: web_index} for matched pairs only.
    """
    candidates = []  # (tier, score, rag_idx, web_idx)
    for ri, rag_item in enumerate(rag_results):
        for wi, web_item in enumerate(web_results):
            scored = _best_candidate(rag_item, web_item)
            if scored is not None:
                tier, score = scored
                candidates.append((tier, score, ri, wi))

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    claimed_rag: set[int] = set()
    claimed_web: set[int] = set()
    assignment: dict[int, int] = {}
    for _tier, _score, ri, wi in candidates:
        if ri in claimed_rag or wi in claimed_web:
            continue
        assignment[ri] = wi
        claimed_rag.add(ri)
        claimed_web.add(wi)

    return assignment


def _detect_discrepancy(
    rag_item: dict, web_match: dict, price_conflict_threshold: float
) -> Optional[dict]:
    """Compare a matched pair and return a discrepancy dict, or None if none applies.

    Never mutates either input — the private fact stays the grounded baseline;
    this only produces a side-channel flag for the Answerer to surface.
    """
    rag_price = rag_item.get("price")
    web_price = web_match.get("price")
    if rag_price is not None and web_price is not None:
        try:
            rag_price_f, web_price_f = float(rag_price), float(web_price)
        except (TypeError, ValueError):
            rag_price_f = web_price_f = None
        if rag_price_f is not None and rag_price_f > 0:
            pct_diff = abs(web_price_f - rag_price_f) / rag_price_f
            if pct_diff > price_conflict_threshold:
                return {
                    "type": "price",
                    "detail": (
                        f"listed at ${rag_price_f:g}; currently ~${web_price_f:g} online "
                        f"({pct_diff:.0%} difference)"
                    ),
                }

    # Availability: rag_search's schema (README_mcp_rag.md) has no
    # availability field at all, so there is no private-side value to
    # "conflict" with in the strict sense. Per spec intent ("if availability
    # conflicts, flag it"), we treat a live-only "unavailable" signal as
    # noteworthy: the private catalog doesn't track stock, but the web says
    # it's currently unavailable, which the Answerer should mention.
    web_availability = web_match.get("availability")
    if web_availability is not None:
        normalized = str(web_availability).strip().lower()
        if normalized in {"out of stock", "unavailable", "false", "no"}:
            return {
                "type": "availability",
                "detail": (
                    f"private catalog does not track live availability; "
                    f"web source reports '{web_availability}'"
                ),
            }

    return None


def reconcile(
    rag_results: list[dict],
    web_results: list[dict],
    price_conflict_threshold: float = 0.10,
) -> dict:
    """Reconcile `rag.search` results (grounded baseline) with `web.search` results.

    Implements prompts/retriever_tool_instructions.md's "Reconciliation (when
    both are used)" section: match private <-> live items by sku, then brand,
    then fuzzy title; flag price (>10% diff) or availability conflicts
    without ever letting the live result silently overwrite the private fact.

    Args:
        rag_results: the `results` list from a `rag.search` response
            (schema: mcp/README_mcp_rag.md).
        web_results: the `results` list from a `web.search` response
            (schema: web-search-mcp/web_search.py). `price`/`availability`
            may legitimately be None for every item (current known
            limitation) — handled gracefully as "nothing to flag".
        price_conflict_threshold: fractional price-difference threshold
            above which a discrepancy is flagged (default 0.10 = 10%,
            per spec).

    Returns:
        {
          "items": [
              {**rag_item, "live_match": {"url", "price", "availability"} | None,
               "discrepancy": {"type": "price"|"availability", "detail": str} | None},
              ...
          ],
          "unmatched_web": [<web result dicts with no matched private item>],
        }
    """
    assignment = _match_all(rag_results, web_results)  # {rag_idx: web_idx}, one-to-one
    items = []

    for ri, rag_item in enumerate(rag_results):
        # Build the output item without mutating rag_item's own price/rating/etc.
        out_item = dict(rag_item)

        wi = assignment.get(ri)
        if wi is None:
            out_item["live_match"] = None
            out_item["discrepancy"] = None
        else:
            web_match = web_results[wi]
            out_item["live_match"] = {
                "url": web_match.get("url"),
                "price": web_match.get("price"),
                "availability": web_match.get("availability"),
            }
            out_item["discrepancy"] = _detect_discrepancy(
                rag_item, web_match, price_conflict_threshold
            )

        items.append(out_item)

    matched_web_idx = set(assignment.values())
    unmatched_web = [w for wi, w in enumerate(web_results) if wi not in matched_web_idx]

    return {"items": items, "unmatched_web": unmatched_web}


if __name__ == "__main__":
    import json

    # --- Example 1: brand + price conflict (>10%) -------------------------
    rag_hits = [
        {
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
    ]
    web_hits = [
        {
            "title": "GreenGleam Steel-Safe Eco Stainless Steel Cleaner, 16oz",
            "url": "https://example.com/greengleam-steel-safe",
            "snippet": "GreenGleam's popular stainless steel cleaner, now $14.99.",
            "price": 14.99,  # +20% vs 12.49 -> should flag
            "availability": None,
        },
        {
            "title": "Unrelated Product Entirely",
            "url": "https://example.com/unrelated",
            "snippet": "Nothing to do with cleaners.",
            "price": None,
            "availability": None,
        },
    ]
    result = reconcile(rag_hits, web_hits)
    print("=== Example 1: brand match + price discrepancy ===")
    print(json.dumps(result, indent=2))
    assert result["items"][0]["discrepancy"]["type"] == "price"
    assert result["items"][0]["price"] == 12.49  # private fact untouched
    assert len(result["unmatched_web"]) == 1

    # --- Example 2: current web.search reality — price/availability None --
    web_hits_noop = [
        {
            "title": "Steel-Safe Eco Stainless Steel Cleaner & Polish",
            "url": "https://example.com/steel-safe",
            "snippet": "Buy the eco stainless steel cleaner online.",
            "price": None,
            "availability": None,
        }
    ]
    result2 = reconcile(rag_hits, web_hits_noop)
    print("\n=== Example 2: fuzzy title match, no price data (no crash, no flag) ===")
    print(json.dumps(result2, indent=2))
    assert result2["items"][0]["live_match"] is not None
    assert result2["items"][0]["discrepancy"] is None

    # --- Example 3: no match at all ----------------------------------------
    result3 = reconcile(rag_hits, [
        {"title": "Completely different item", "url": "https://example.com/x",
         "snippet": "n/a", "price": None, "availability": None}
    ])
    print("\n=== Example 3: no match ===")
    print(json.dumps(result3, indent=2))
    assert result3["items"][0]["live_match"] is None
    assert len(result3["unmatched_web"]) == 1

    # --- Example 4: availability-only discrepancy ---------------------------
    web_hits_unavailable = [
        {
            "title": "GreenGleam Steel-Safe Eco Stainless Steel Cleaner",
            "url": "https://example.com/greengleam-oos",
            "snippet": "GreenGleam cleaner.",
            "price": None,
            "availability": "Out of Stock",
        }
    ]
    result4 = reconcile(rag_hits, web_hits_unavailable)
    print("\n=== Example 4: availability discrepancy ===")
    print(json.dumps(result4, indent=2))
    assert result4["items"][0]["discrepancy"]["type"] == "availability"

    print("\nAll __main__ self-tests passed.")
