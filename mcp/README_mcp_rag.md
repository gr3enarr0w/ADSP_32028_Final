# `rag.search` — MCP tool (Shane)

Half of the two-tool MCP server. Clark's server merges this with `web.search`.
This directory ships a **standalone** server so `rag.search` runs and is
testable on its own.

## Run

```bash
# from repo root
export PYTHONPATH=src
python mcp/rag_mcp_server.py          # serves over stdio

# inspect interactively
npx @modelcontextprotocol/inspector python mcp/rag_mcp_server.py
```

Requires a built index (`bash scripts/build_index.sh`).

## Tool: `rag.search`

Queries the private Amazon Product Dataset 2020 (Household-Cleaning slice) via
hybrid retrieval (dense vector + BM25 + optional cross-encoder rerank).

### Input schema

| field | type | required | description |
|---|---|---|---|
| `query` | string | yes | Natural-language product need |
| `k` | integer (1–20) | no (default 5) | Number of products to return |
| `filters` | object | no | Structured constraints from the Planner, as a **single nested dict** (see below). |

`filters` is passed as one nested argument (matching `planner.md`'s output
schema and `rag.rag_search()`), **not** as flat top-level params. Recognized keys
(all optional):

| filter key | type | description |
|---|---|---|
| `price_max` | number | Max price in USD |
| `price_min` | number | Min price in USD |
| `min_rating` | number | Minimum average star rating |
| `brand` | string | Exact brand filter |
| `material` | string | Surface tag: `stainless steel`, `glass`, `wood`, … |

Unknown keys are ignored; `{}` or omitting `filters` means no filtering.

### Example call

```json
{ "query": "eco-friendly stainless steel cleaner under $15",
  "k": 3,
  "filters": { "price_max": 15, "material": "stainless steel" } }
```

### Output schema

```json
{
  "query": "eco-friendly stainless steel cleaner under $15",
  "count": 3,
  "results": [
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
      "score": 2.97
    }
  ]
}
```

`doc_id` is the **private-catalog citation id** the UI shows next to the spoken
answer. Required keys per the syllabus: `sku, title, price, rating, brand?,
ingredients?, doc_id`; `url`, `price_per_oz`, `score` are extras.

## Planner contract

- Prefer `rag.search` for grounded product facts.
- If the user asks for **current** price/availability ("now", "latest", "in
  stock today"), the planner should ALSO call `web.search` and reconcile by
  SKU/brand/title similarity.
- Every returned `doc_id` must be surfaced as an on-screen citation.

## Logging

Each call logs `query`, `filters`, hit count, and latency to **stderr** (stdout
is reserved for the stdio JSON-RPC channel). Clark's combined server adds
timestamped request/response + source-URL logging for `web.search`.
