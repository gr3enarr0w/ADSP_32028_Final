# Clark's deliverables — full handoff

Everything Clark owns for the Voice-Commerce Agentic RAG project, how it works,
how it was verified, and what to watch out for. Companion to
[`README_mcp_web.md`](README_mcp_web.md) (tool schema reference) — this doc is
the narrative version: what each piece does, why it's built the way it is, and
what broke during testing.

## Scope

| Deliverable | File(s) |
|---|---|
| `web.search` MCP tool | `web_search.py` |
| Combined two-tool MCP server | `combined_mcp_server.py` |
| `web.search` schema doc | `README_mcp_web.md` |
| Timestamped + source-URL logging | inside `combined_mcp_server.py` |
| Private/live reconciliation | `../src/rag/reconcile.py` |
| End-to-end Colab demo | `../notebooks/colab_end_to_end_demo.ipynb` |

Not Clark's: the Router/Planner/Answerer-Critic LangGraph nodes, ASR/TTS, and
the Streamlit UI (rest of the team).

## How each piece works

### 1. `web.search` (`web_search.py`)

A thin async wrapper around the multi-provider research orchestrator already
vendored into this directory (`orchestrator.py` + `providers/`). Schema,
known `price`/`availability` limitation, and the caching/rate-limiting
details are documented once, in [`README_mcp_web.md`](README_mcp_web.md) —
not repeated here. What's specific to *how it's wired*, not *what it
returns*:

- Calls `orchestrator.search(query, mode="auto", max_results=k)`, which
  auto-classifies the query and routes it to **one** best-fit provider
  (Exa/Brave/Tavily/Gemini/Linkup/Newsdata) — not a fan-out across all of
  them. See `orchestrator.py`'s `MODE_PRIMARY`/`MODE_FALLBACK` tables.
- Loads `web-search-mcp/.env` itself (`load_dotenv()` at import time) since
  neither `orchestrator.py` nor the providers do — without this, provider API
  keys set via `.env` are silently never read.
- If no provider has a configured key, `orchestrator.py`'s `_init_provider`
  catches the resulting exception, logs a warning, and returns `[]` —
  `web_search()` then returns `{count: 0, results: []}` rather than raising.
  This graceful-degradation path is intentional and tested, and the same
  pattern the TTL cache/rate-limit and `rag.tts.speak()` both follow.

### 2. Combined MCP server (`combined_mcp_server.py`)

Registers **both** tools on one `FastMCP("combined-tools")` instance:

- `rag.search` — imports `rag_search` **unchanged** from
  `src/rag/rag_search.py` (Shane's code, not touched).
- `web.search` — imports `web_search` from this directory's `web_search.py`.

Two path-resolution problems were found and fixed here (both are Clark-side
fixes; neither touches `src/rag/config.py`, which is correct in isolation).
**Historical context:** at the time these bugs were found, the repo layout
had `rag-system/` nested as a sibling of `web-search-mcp/` (it has since been
flattened, so `src/`, `prompts/`, etc. now live directly at the repo root
alongside `web-search-mcp/`), which is what made the path resolution tricky
in the first place:

1. **`.env` discovery across sibling directories.** `rag.config`'s
   `load_dotenv()` walks *upward* from CWD. Since `web-search-mcp/` and
   `rag-system/` were siblings back then, running the combined server from
   `web-search-mcp/` (its documented run command) never found
   `rag-system/.env`. Fixed by loading it explicitly via a path computed from
   `Path(__file__).resolve()`. That fix still holds today with the flattened
   layout — it locates the repo-root `.env` regardless of CWD.
2. **Relative data paths.** The root `.env`'s `INDEX_DIR=data/index` etc.
   are relative, meant to be resolved with CWD=repo root (formerly
   `rag-system/`). When the combined server runs from `web-search-mcp/`,
   those resolve to the wrong directory (`web-search-mcp/data/index`, which
   doesn't exist) → `Collection household_cleaning not found`. Fixed by
   rewriting the three data-path env vars to absolute paths (anchored on the
   repo root, not CWD) before `rag.config` reads them.

Logging: both tools log a timestamped (`_ts()` helper, UTC ISO-8601) line per
call to **stderr** (stdout is reserved for the stdio JSON-RPC channel).
`web.search` additionally logs a second line listing every returned source
URL — the "MCP logging" grading criterion calls this out specifically for the
live-search tool.

### 3. Reconciliation (`../src/rag/reconcile.py`)

Implements `../prompts/retriever_tool_instructions.md`'s
reconciliation contract: match private (`rag.search`) ↔ live (`web.search`)
items by `sku`, then `brand`, then fuzzy `title`; private fact always wins as
the grounded baseline; discrepancies (>10% price difference, or a live-only
availability signal) are flagged, never silently applied.

**A real bug was found and fixed during end-to-end testing.** The first
implementation matched each rag item to a web item independently, in list
order, with no exclusion once a web item was claimed. Tested against real
`rag.search` output: two *different* stainless-steel-cleaner products both
scored above the 0.6 fuzzy-title threshold against the *same* web result, and
whichever rag item was processed first "stole" the match — producing a false
price-discrepancy claim on the wrong product. Fixed by switching to a global
one-to-one assignment (score every `(rag, web)` pair, greedily assign
highest-confidence pairs first, each side claimed at most once) and raising
the threshold to 0.72. Re-verified against the same real data: now matches
correctly.

### 4. Colab demo (`../notebooks/colab_end_to_end_demo.ipynb`)

Generated deterministically from `../scripts/build_colab_demo.py` (same
convention as Shane's `scripts/build_notebook.py`). Clones the
repo, installs both subprojects' dependencies, configures the RAG pipeline
via `os.environ` (hash embedder — offline, deterministic, no model download),
builds the index, runs the 16-test offline suite, then exercises `rag.search`,
the combined server's tool discovery, both tools via the real MCP
`call_tool()` path, and reconciliation.

Verified twice: once via `jupyter nbconvert --execute` in a fresh clone
(confirms it works headless, no Colab-specific dependency), and once live in
Colab itself with real Brave/Tavily keys (confirms real live search results
reconcile correctly against the private catalog).

## Provider API keys in Colab

Section "2b" of the notebook checks `google.colab.userdata` for
`BRAVE_SEARCH_API_KEY` / `TAVILY_API_KEY` first (silent), falling back to a
masked prompt only if neither is set. Add keys via Colab's 🔑 **Secrets**
panel (left sidebar) rather than pasting them into a cell — they're then
scoped to your account, never touch the notebook file or git history.

**Gotcha hit during setup:** a key pasted into the Secrets panel came out
47 characters instead of the real 31 — the tail of the key
(`-L3top2iBt0avhQb`) had been duplicated onto the end, almost certainly a
double-paste/drag-select artifact. Brave's API correctly rejected it as
`SUBSCRIPTION_TOKEN_INVALID` (a real validation error, not a network/IP
issue — confirmed by testing the *same* literal key from a different
machine, where it worked). If `web.search` fails with a 4xx from a provider
whose key you're sure is right, check the secret's actual length/content
before assuming the key was revoked.

## Known gaps not fixed (flagged, not blocking)

- `web-search-mcp/requirements.txt` originally didn't list `newsdataapi` or
  `google-auth`, both needed transitively by `providers/newsdata_provider.py`
  and `providers/gemini_provider.py` — **fixed**, both now listed.
- `orchestrator.search()` doesn't enforce `max_results` as a hard cap on the
  `sources` list it returns — **fixed** in `web_search.py` with a defensive
  slice.
- `orchestrator.py` is single-provider-per-call by design (routes to one best
  provider, escalates only via a separate `research_escalate` tool, not
  automatically). So a single `web.search` call trying two different provider
  keys (e.g. Brave *and* Tavily) in sequence is not how it works today — if
  the routed provider fails, that call returns 0 results rather than falling
  through to the next key. Worth knowing if a demo run shows 0 hits despite
  multiple keys being configured.

## Running everything locally (no Colab)

```bash
# 1. Build the index (from repo root)
cp .env.example .env   # optionally edit EMBEDDING_PROVIDER=hash for a fast/offline run
bash scripts/build_index.sh

# 2. Run the offline test suite
PYTHONPATH=src python -m pytest tests/ -q

# 3. Serve both tools
cd web-search-mcp
cp .env.example .env   # add provider API keys for web.search
pip install -r requirements.txt
python combined_mcp_server.py   # stdio MCP server, both tools

# or inspect interactively:
npx @modelcontextprotocol/inspector python combined_mcp_server.py
```
