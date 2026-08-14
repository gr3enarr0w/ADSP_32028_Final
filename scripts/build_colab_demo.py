"""Generate notebooks/colab_end_to_end_demo.ipynb.

Run from anywhere:
    python build_colab_demo.py /path/to/repo/notebooks/colab_end_to_end_demo.ipynb

Mirrors the convention already used by scripts/build_notebook.py
(Shane's CP1 ingestion notebook): the notebook is a build artifact generated
deterministically from this script, not hand-edited.
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("colab_end_to_end_demo.ipynb")

nb = new_notebook()
cells = []

cells.append(new_markdown_cell(
    "# Voice-Commerce Agentic RAG — End-to-End Colab Demo\n"
    "\n"
    "Runs the currently-implemented half of the project **entirely inside this "
    "notebook**, no local setup required:\n"
    "\n"
    "1. Clone the repo\n"
    "2. Install dependencies\n"
    "3. Build the private-catalog vector index (`rag.search`'s data)\n"
    "4. Run the offline test suite as a sanity check\n"
    "5. Call `rag.search` directly\n"
    "6. Stand up the **combined two-tool MCP server** (`rag.search` + `web.search`) "
    "and call both tools through the real MCP tool-discovery/tool-call path\n"
    "7. Reconcile private + live results (price/availability discrepancy flagging)\n"
    "\n"
    "**Scope note:** this covers the RAG pipeline `src/` (Shane) and `web-search-mcp/` "
    "(Clark) — the Router/Planner/Answerer-Critic LangGraph nodes, ASR/TTS, and "
    "the Streamlit UI are the rest of the team's parts and aren't reproduced here.\n"
    "\n"
    "**`web.search` note:** without live provider API keys (Exa/Brave/Tavily/"
    "Gemini/Linkup/Newsdata), this notebook still runs end-to-end — the tool "
    "degrades gracefully to zero results (by design, see step 6b) rather than "
    "failing, and step 7 falls back to a clearly-labeled simulated live result so "
    "the reconciliation logic is still demonstrated meaningfully. Paste any one "
    "key in step 2b for a real live call."
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell("## 1. Clone the repository"))
cells.append(new_code_cell(
    "import os\n"
    "\n"
    "REPO_URL = \"https://github.com/gr3enarr0w/ADSP_32028_Final.git\"\n"
    "REPO_DIR = \"ADSP_32028_Final\"\n"
    "\n"
    "if not os.path.isdir(REPO_DIR):\n"
    "    !git clone -q {REPO_URL}\n"
    "else:\n"
    "    print(f\"{REPO_DIR}/ already present — pulling latest instead of re-cloning\")\n"
    "    !cd {REPO_DIR} && git pull -q\n"
    "\n"
    "%cd {REPO_DIR}\n"
    "REPO_ROOT = os.getcwd()\n"
    "print(\"repo root:\", REPO_ROOT)"
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 2. Install dependencies\n"
    "\n"
    "Installs both subprojects' pinned requirements. `sentence-transformers`/"
    "`torch` (pulled in by `requirements-rag.txt`) aren't strictly needed for "
    "this run — we use the offline, deterministic **hash embedder** (§3) so the "
    "notebook doesn't depend on downloading an embedding model — but they're "
    "installed anyway for fidelity with the documented setup in "
    "`README_shane.md`."
))
cells.append(new_code_cell(
    "!pip install -q -r requirements-rag.txt\n"
    "!pip install -q -r web-search-mcp/requirements.txt"
))

cells.append(new_markdown_cell(
    "### 2b. (Optional) live `web.search` provider API key\n"
    "\n"
    "Paste **one** key to see a real live web result reconciled against the "
    "private catalog in §7. Leave blank to skip — everything else still runs.\n"
    "\n"
    "Uses Colab's `google.colab.userdata` secrets store when available (Colab's "
    "🔑 icon in the left sidebar), falling back to a plain masked prompt "
    "otherwise."
))
cells.append(new_code_cell(
    "from getpass import getpass\n"
    "\n"
    "def _get_secret(name: str) -> str:\n"
    "    try:\n"
    "        from google.colab import userdata\n"
    "        val = userdata.get(name)\n"
    "        if val:\n"
    "            return val\n"
    "    except Exception:\n"
    "        pass\n"
    "    return \"\"\n"
    "\n"
    "# Try Colab secrets first (silent); only prompt interactively if none are set.\n"
    "PROVIDER_KEYS = {\n"
    "    \"BRAVE_SEARCH_API_KEY\": _get_secret(\"BRAVE_SEARCH_API_KEY\"),\n"
    "    \"TAVILY_API_KEY\": _get_secret(\"TAVILY_API_KEY\"),\n"
    "}\n"
    "if not any(PROVIDER_KEYS.values()):\n"
    "    try:\n"
    "        entered = getpass(\"Brave Search API key (or Enter to skip): \")\n"
    "    except Exception:\n"
    "        # No interactive stdin available (e.g. a non-interactive \"run all\"\n"
    "        # outside Colab's own frontend) — skip rather than block.\n"
    "        entered = \"\"\n"
    "    if entered:\n"
    "        PROVIDER_KEYS[\"BRAVE_SEARCH_API_KEY\"] = entered\n"
    "\n"
    "for k, v in PROVIDER_KEYS.items():\n"
    "    if v:\n"
    "        os.environ[k] = v\n"
    "\n"
    "HAVE_LIVE_KEY = any(PROVIDER_KEYS.values())\n"
    "print(\"live web.search key provided:\", HAVE_LIVE_KEY)"
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 3. Configure the RAG pipeline\n"
    "\n"
    "`src/rag/config.py` reads everything from environment variables "
    "(model-agnostic by design — see `.env.example`). We set these "
    "directly with `os.environ` rather than writing a `.env` file: `config.py`'s "
    "data-path defaults already resolve to absolute paths anchored on its own "
    "file location (not on the notebook's CWD), which sidesteps a relative-path "
    "footgun we hit during local testing (documented in "
    "`web-search-mcp/combined_mcp_server.py`'s comments) — so the fewer paths we "
    "set explicitly, the more robust this cell is to *where* later cells `cd`.\n"
    "\n"
    "**`EMBEDDING_PROVIDER=hash`** is an offline, deterministic embedder built "
    "for exactly this (tests/CI) — swap to `local` (sentence-transformers, no "
    "API key, downloads a small model) for closer-to-production embedding "
    "quality; both are config-only swaps, no code changes."
))
cells.append(new_code_cell(
    "os.environ[\"EMBEDDING_PROVIDER\"] = \"hash\"   # swap to \"local\" for real embeddings\n"
    "os.environ[\"HASH_DIM\"] = \"384\"\n"
    "os.environ[\"USE_RERANKER\"] = \"false\"          # reranker needs sentence-transformers' cross-encoder\n"
    "print(\"EMBEDDING_PROVIDER =\", os.environ[\"EMBEDDING_PROVIDER\"])"
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 4. Build the vector index\n"
    "Ingests the shipped 24-row sample slice and builds the Qdrant (embedded, "
    "local-mode) index — the same script documented in `README_shane.md`."
))
cells.append(new_code_cell(
    "!bash scripts/build_index.sh"
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 5. Sanity check: run the offline test suite\n"
    "16 tests covering ingestion, retrieval/filters, the MCP tool body, and the "
    "reconciliation logic (`tests/`) — all offline, no network calls."
))
cells.append(new_code_cell(
    "!PYTHONPATH=src python -m pytest tests/ -q"
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 6. `rag.search` — direct call\n"
    "Private-catalog hybrid retrieval (vector + BM25 + metadata filters), "
    "called directly as a plain Python function first, before going through "
    "the MCP layer in the next section."
))
cells.append(new_code_cell(
    "import sys\n"
    "sys.path.insert(0, os.path.join(REPO_ROOT, \"src\"))\n"
    "\n"
    "from rag.rag_search import rag_search\n"
    "import pandas as pd\n"
    "\n"
    "rag_out = rag_search(\n"
    "    query=\"eco-friendly stainless steel cleaner under $15\",\n"
    "    k=3,\n"
    "    filters={\"price_max\": 15, \"material\": \"stainless steel\"},\n"
    ")\n"
    "print(f\"{rag_out['count']} hits for {rag_out['query']!r}\")\n"
    "pd.DataFrame(rag_out[\"results\"])[[\"sku\", \"title\", \"brand\", \"price\", \"rating\", \"doc_id\"]]"
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 7. The combined two-tool MCP server\n"
    "\n"
    "`web-search-mcp/combined_mcp_server.py` registers **both** `rag.search` "
    "(Shane — imported unchanged from `src/rag/rag_search.py`) and "
    "`web.search` (Clark — thin wrapper over a multi-provider research "
    "orchestrator) on one `FastMCP` server instance. Below we import the "
    "module in-process and drive it through FastMCP's own tool-manager, "
    "exercising the real tool-discovery + tool-call path (as a real MCP client "
    "over stdio would) rather than calling the underlying functions directly."
))
cells.append(new_code_cell(
    "sys.path.insert(0, os.path.join(REPO_ROOT, \"web-search-mcp\"))\n"
    "import combined_mcp_server as mcp_server\n"
    "\n"
    "tools = await mcp_server.mcp.list_tools()\n"
    "for t in tools:\n"
    "    print(f\"- {t.name}: {t.description.splitlines()[0]}\")"
))

cells.append(new_markdown_cell(
    "### 7a. `rag.search` via the MCP tool-call path\n"
    "Watch the log line printed to stderr below the output — timestamped "
    "request/response logging, per the MCP-logging grading criterion."
))
cells.append(new_code_cell(
    "result = await mcp_server.mcp.call_tool(\n"
    "    \"rag.search\", {\"query\": \"plant-based dish soap gentle on hands\", \"k\": 3}\n"
    ")\n"
    "import json\n"
    "\n"
    "def _tool_json(call_result):\n"
    "    \"\"\"FastMCP's call_tool() returns a list of content blocks; unwrap the\n"
    "    first text block's JSON payload.\"\"\"\n"
    "    return json.loads(call_result[0].text)\n"
    "\n"
    "rag_via_mcp = _tool_json(result)\n"
    "pd.DataFrame(rag_via_mcp[\"results\"])[[\"sku\", \"title\", \"brand\", \"price\", \"rating\"]]"
))

cells.append(new_markdown_cell(
    "### 7b. `web.search` via the MCP tool-call path\n"
    "With no key provided (§2b), this degrades gracefully to `count: 0` — by "
    "design, not a crash (the underlying multi-provider orchestrator warns and "
    "moves on when a provider has no API key configured). With a key, it "
    "returns real live results, each carrying a `url` citation."
))
cells.append(new_code_cell(
    "result = await mcp_server.mcp.call_tool(\n"
    "    \"web.search\",\n"
    "    {\"query\": \"current price of OXO Good Grips stainless steel cleaner\", \"k\": 3},\n"
    ")\n"
    "web_via_mcp = _tool_json(result)\n"
    "print(f\"{web_via_mcp['count']} live hits\" + (\"\" if HAVE_LIVE_KEY else \" (expected: 0, no provider key was set in §2b)\"))\n"
    "pd.DataFrame(web_via_mcp[\"results\"]) if web_via_mcp[\"results\"] else web_via_mcp"
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 8. Reconciliation — merging private + live results\n"
    "\n"
    "Per `prompts/retriever_tool_instructions.md`: match private "
    "(`rag.search`) ↔ live (`web.search`) items by `sku`, then `brand`, then "
    "fuzzy `title`; the private fact always stays the grounded baseline, and "
    "price differences over 10% (or a live-only availability signal) are "
    "flagged with both citations rather than silently overwritten.\n"
    "\n"
    "If no live results came back in §7b (no API key), we fall back to one "
    "**clearly-labeled simulated** live result — matched against the top hit "
    "from §6 — purely so the reconciliation/discrepancy-flagging logic is "
    "still visibly exercised end-to-end."
))
cells.append(new_code_cell(
    "from rag.reconcile import reconcile\n"
    "\n"
    "web_results_for_reconcile = web_via_mcp[\"results\"]\n"
    "simulated = False\n"
    "if not web_results_for_reconcile:\n"
    "    simulated = True\n"
    "    top = rag_out[\"results\"][0]\n"
    "    web_results_for_reconcile = [{\n"
    "        \"title\": top[\"title\"],\n"
    "        \"url\": \"https://retailer.example.com/simulated-listing\",\n"
    "        \"snippet\": \"Simulated live listing for demo purposes (no provider API key set).\",\n"
    "        \"price\": round(top[\"price\"] * 1.2, 2),  # +20% -> should trip the >10% discrepancy flag\n"
    "        \"availability\": \"in_stock\",\n"
    "    }]\n"
    "\n"
    "print(\"live results are\", \"SIMULATED (no API key)\" if simulated else \"REAL\")\n"
    "reconciled = reconcile(rag_out[\"results\"], web_results_for_reconcile)\n"
    "\n"
    "rows = []\n"
    "for item in reconciled[\"items\"]:\n"
    "    rows.append({\n"
    "        \"title\": item[\"title\"],\n"
    "        \"private_price\": item[\"price\"],\n"
    "        \"live_price\": (item[\"live_match\"] or {}).get(\"price\"),\n"
    "        \"discrepancy\": (item[\"discrepancy\"] or {}).get(\"detail\"),\n"
    "    })\n"
    "pd.DataFrame(rows)"
))

# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## Summary\n"
    "\n"
    "- **`rag.search`** (Shane): hybrid vector+BM25 retrieval over the private "
    "Amazon-2020 Household-Cleaning slice, with metadata filters and `doc_id` "
    "citations. Schema: `mcp/README_mcp_rag.md`.\n"
    "- **`web.search`** (Clark): live multi-provider web search, `url` "
    "citations, degrades gracefully with no API key. Schema: "
    "`web-search-mcp/README_mcp_web.md`.\n"
    "- **Combined MCP server** (Clark): both tools discoverable and callable "
    "over one `FastMCP` instance, timestamped request/response + source-URL "
    "logging. `web-search-mcp/combined_mcp_server.py`.\n"
    "- **Reconciliation** (Clark): private fact always wins as the grounded "
    "baseline; conflicts are flagged, never silently overwritten. "
    "`src/rag/reconcile.py`.\n"
    "\n"
    "Not covered by this notebook (rest of the team's parts): Router/Planner/"
    "Answerer-Critic LangGraph orchestration, ASR/TTS, and the Streamlit UI."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": [], "name": "colab_end_to_end_demo.ipynb"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print("wrote", OUT)
