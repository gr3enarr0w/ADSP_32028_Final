# Retriever — Tool-Use Instructions

> Node: **Retriever** (Shane owns `rag.search`; Clark owns `web.search`).
> These are the tool-calling instructions given to the model at the retrieve step.

You have two tools. Follow the plan from the Planner.

## `rag.search` (private catalog — prefer this)
Call it to get grounded product facts from the Amazon-2020 Household-Cleaning
index.

Arguments:
- `query` (string, required) — the Planner's `query`.
- `k` (int) — the Planner's `k`.
- `filters` (object) — the Planner's `filters` dict passed through as ONE nested
  argument (keys: `price_max`, `price_min`, `min_rating`, `brand`, `material`),
  not flattened into separate top-level params.

Returns `{query, count, results:[{sku, title, price, rating, brand,
ingredients, doc_id, url, price_per_oz, score}]}`. **Use `doc_id` as the
citation** for every product you carry forward.

## `web.search` (live — only if `call_web_search` is true)
Call it to check *current* price/availability. Returns `{title, url, snippet,
price?, availability?}`. **Use `url` as the citation.**

## Reconciliation (when both are used)
- Match private ↔ live items by `sku`, then `brand`, then fuzzy `title`.
- If prices differ by > 10% or availability conflicts, keep the private fact as
  the grounded baseline and **flag the discrepancy** with both citations so the
  Answerer can say "listed at $X; currently ~$Y online".
- Never let a live result overwrite a private fact silently.

## Do / Don't
- Do call `rag.search` at most twice (once, plus one relaxed retry with fewer
  filters if `count == 0`).
- Don't fabricate results if a tool returns nothing — report the empty result so
  the Answerer can offer alternatives.
- Don't call `web.search` just to pad results when the plan said not to.
