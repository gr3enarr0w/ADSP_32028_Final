# `web.search` — MCP tool (Clark)

Half of the two-tool MCP server. `combined_mcp_server.py` merges this with
Shane's `rag.search` (imported unchanged from `src/rag`) into one
`FastMCP` server. This directory also ships the **standalone** research-tool
server (`server.py`) so the underlying `web_search()` wrapper can run and be
tested on its own.

## Run

```bash
# from repo root
cd web-search-mcp
pip install -r requirements.txt
cp .env.example .env   # add provider API keys (Exa/Brave/Tavily/Gemini/Linkup/Newsdata)

python combined_mcp_server.py   # serves BOTH rag.search + web.search over stdio

# inspect interactively
npx @modelcontextprotocol/inspector python combined_mcp_server.py
```

Requires the RAG index to already be built (`bash ../scripts/build_index.sh`)
since `rag.search` is served from the same process.

`web_search.py` also remains runnable directly for manual testing of just
the web-search half:

```bash
python web_search.py
```

## Tool: `web.search`

Body defined in `web_search.py`'s `web_search()` — an async, thin,
JSON-serializable wrapper over `orchestrator.search()` that keeps the tool
layer dumb and the multi-provider routing/escalation logic
(Exa/Brave/Tavily/Gemini/Linkup/Newsdata) in `orchestrator.py`.

Use it for CURRENT, live-web facts — not grounded in the private
Amazon-2020 catalog. Prefer `rag.search` unless the user asks for *current*
price/availability or something outside the private catalog.

### Input schema

| field | type | required | description |
|---|---|---|---|
| `query` | string | yes | Natural-language search query, e.g. "current price of OXO Good Grips stainless steel cleaner" |
| `k` | integer | no (default 5) | Number of results to return |

### Example call

```json
{ "query": "current price of OXO Good Grips stainless steel cleaner", "k": 3 }
```

### Output schema

```json
{
  "query": "current price of OXO Good Grips stainless steel cleaner",
  "count": 3,
  "results": [
    {
      "title": "OXO Good Grips Stainless Steel Cleaner & Polish, 14 oz",
      "url": "https://www.example-retailer.com/oxo-good-grips-stainless-steel-cleaner",
      "snippet": "OXO Good Grips Stainless Steel Cleaner keeps appliances streak-free. Currently $11.99 at ExampleRetailer, in stock.",
      "price": null,
      "availability": null
    }
  ]
}
```

`url` is the **citation** for every `web.search` result carried forward.

**Known limitation (not fabricated):** per `web_search()`'s own docstring,
the underlying providers are general-purpose web search, not
product-specific scrapers, so `price` and `availability` are always `null`
here rather than guessed — even when a price is visible in `snippet` text
(as in the example above).

**Future work:** parse `price`/`availability` out of `snippet` text, or run
an Apify extract (see `providers/apify_provider.py`) on the returned
product `url`s to pull structured price/availability data.

## Planner contract

- Prefer `rag.search` for grounded product facts; only call `web.search`
  when the Planner sets `call_web_search: true` (user asked for *current*
  price/availability, or something outside the private catalog).
- When both tools are used, reconcile results by matching private ↔ live
  items on **`sku`, then `brand`, then fuzzy `title`**.
- If prices differ by **more than 10%** or availability conflicts, the
  **private fact wins** as the grounded baseline — flag the discrepancy
  with both citations so the Answerer can say "listed at $X; currently ~$Y
  online." Never let a live result silently overwrite a private fact.
- Every `web.search` result surfaced to the user must cite its `url`.
- Don't call `web.search` just to pad results when the plan said not to;
  don't fabricate results if it returns nothing — report the empty result
  so the Answerer can offer alternatives.

## Caching & rate-limiting

`web_search()` keeps a simple in-process, in-memory TTL cache and a
minimum-interval rate limit in front of `orchestrator.search()` — this is
separate from `usage_tracker.py`'s monthly-quota bookkeeping (calls vs.
limit per provider), which is a different concern.

- **Cache**: keyed on the normalized `(query, k)` pair (query lowercased and
  stripped). Default TTL is **120s**, configurable via the `WEB_SEARCH_CACHE_TTL`
  env var (spec range: 60–300s). A cache hit returns the stored result
  immediately, without calling `orchestrator.search()` again, and logs a
  line to stderr. Pass `use_cache=False` to `web_search()` to bypass the
  cache for a single call.
- **Rate limit**: a minimum-interval throttle between provider calls.
  Default interval is **1.0s**, configurable via the `WEB_SEARCH_MIN_INTERVAL`
  env var. If a call comes in sooner than that after the previous provider
  call, `web_search()` awaits (`asyncio.sleep`) the remaining time before
  proceeding — no request is dropped, just delayed.

Both live in module-level state in `web_search.py` (no Redis, no disk
persistence) — in-process memory only, matching how `usage_tracker.py` keeps
its own persistence simple.

## Logging

Each call logs timestamped request/response plus source URLs to **stderr**
(stdout is reserved for the stdio JSON-RPC channel) — this is the
"Clark's combined server adds timestamped request/response + source-URL
logging for `web.search`" behavior referenced in `mcp/README_mcp_rag.md`'s
Logging section.
