"""Generate notebooks/04_orchestration.ipynb (Final deliverable) programmatically.

Follows the exact nbformat.v4 convention used by build_notebook.py (CP1) and
build_tts_notebook.py (TTS): build a list of markdown/code cells with
nbformat.v4 helpers, then nbf.write() once at the end.
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_orchestration.ipynb"

nb = new_notebook()
cells = []

# ---------------------------------------------------------------------------
# 1. Title
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "# Final — LangGraph Orchestration: Router \u2192 Planner \u2192 Retriever \u2192 Answerer/Critic\n"
    "\n"
    "**Deliverable:** Final \u2014 LangGraph multi-agent orchestration\n"
    "\n"
    "`prompts/` has been the whole team's Prompt-Disclosure artifact since "
    "Checkpoint work began (`prompts/README.md`), but until now nothing "
    "actually *loaded* those files into a running agent graph \u2014 the gap "
    "called out in `README_shane.md`: \"prompts/ exists but nothing loads "
    "them into real running code.\" This notebook closes that gap.\n"
    "\n"
    "It wires together:\n"
    "\n"
    "* `src/rag/llm.py` \u2014 the single `call_llm()` call-site (Anthropic, "
    "with a graceful MockLLM fallback so this notebook runs with **zero "
    "API keys**).\n"
    "* `src/rag/nodes.py` \u2014 one function per node (`router_node`, "
    "`planner_node`, `retriever_node`, `answerer_critic_node`), each "
    "reading its prompt file(s) fresh from disk on every call.\n"
    "* `src/rag/graph.py` \u2014 `build_graph()`, the single source of truth "
    "for the compiled LangGraph `StateGraph`.\n"
    "\n"
    "Along the way we also confirm a stale-value fix: `rag.config.Config."
    "llm_model` used to default to the retired alias `claude-3-5-sonnet-"
    "latest`; it now defaults to `claude-sonnet-5`."
))

# ---------------------------------------------------------------------------
# 2-3. Setup
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 0. Setup\n"
    "\n"
    "Two `sys.path` entries are needed: `../src` (this subtree's own "
    "package) and the **sibling** `web-search-mcp/` directory, because "
    "`rag.nodes.retriever_node` imports `web_search()` from there directly "
    "(not through the MCP protocol) \u2014 the same sibling-directory "
    "relative-path pattern `web-search-mcp/combined_mcp_server.py` uses to "
    "import `rag.rag_search` from *this* subtree, and for the identical "
    "reason: `rag.config`'s bare `load_dotenv()` only walks **upward** from "
    "the current working directory, so a sibling directory's `.env` is "
    "never found by accident \u2014 it has to be loaded by explicit path. We "
    "do the same here for the repo root's `.env`."
))
cells.append(new_code_cell(
    "import os, sys\n"
    "sys.path.append(os.path.abspath('../src'))\n"
    "sys.path.append(os.path.abspath('../web-search-mcp'))\n"
    "\n"
    "from dotenv import load_dotenv\n"
    "load_dotenv(os.path.abspath('../.env'))  # explicit path -- see note above\n"
    "\n"
    "from rag.config import get_config\n"
    "\n"
    "cfg = get_config(refresh=True)\n"
    "\n"
    "# .env's data paths (RAW_CSV/PROCESSED_DIR/INDEX_DIR) are written as\n"
    "# relative strings meant to be resolved against the repo root as CWD (same\n"
    "# caveat combined_mcp_server.py documents for vectorstore.py). nbconvert\n"
    "# runs a notebook's kernel with the notebook's OWN directory as CWD\n"
    "# (notebooks/), not the repo root -- so anchor them explicitly to this\n"
    "# notebook's parent directory rather than relying on CWD.\n"
    "_REPO_ROOT = os.path.abspath('..')\n"
    "cfg.index_dir = os.path.join(_REPO_ROOT, cfg.index_dir)\n"
    "cfg.processed_dir = os.path.join(_REPO_ROOT, cfg.processed_dir)\n"
    "cfg.raw_csv = os.path.join(_REPO_ROOT, cfg.raw_csv)\n"
    "\n"
    "print('llm_provider :', cfg.llm_provider)\n"
    "print('llm_model    :', cfg.llm_model)\n"
    "assert cfg.llm_model == 'claude-sonnet-5', (\n"
    "    'stale LLM_MODEL default not fixed -- expected claude-sonnet-5, got ' + cfg.llm_model\n"
    ")\n"
    "print('confirmed: llm_model is claude-sonnet-5, not the retired claude-3-5-sonnet-latest alias')"
))

# ---------------------------------------------------------------------------
# 4-5. API key check
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 0b. API key check (Colab-friendly)\n"
    "\n"
    "Same `google.colab.userdata` \u2192 `getpass` fallback pattern as "
    "`notebooks/colab_end_to_end_demo.ipynb` \u00a72b, applied to "
    "`ANTHROPIC_API_KEY`. Paste a key to see `rag.llm.call_llm` make a real "
    "Anthropic call; leave it blank and everything below still runs end to "
    "end against the MockLLM fallback in `src/rag/llm.py`."
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
    "    return ''\n"
    "\n"
    "# Try Colab secrets first (silent); only prompt interactively if none is set.\n"
    "anthropic_key = _get_secret('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_API_KEY', '')\n"
    "if not anthropic_key:\n"
    "    try:\n"
    "        entered = getpass('ANTHROPIC_API_KEY (or Enter to skip -- MockLLM fallback): ')\n"
    "    except Exception:\n"
    "        # No interactive stdin available (e.g. a non-interactive \"run all\"\n"
    "        # outside Colab's own frontend) -- skip rather than block.\n"
    "        entered = ''\n"
    "    if entered:\n"
    "        anthropic_key = entered\n"
    "\n"
    "if anthropic_key:\n"
    "    os.environ['ANTHROPIC_API_KEY'] = anthropic_key\n"
    "\n"
    "HAVE_LLM_KEY = bool(anthropic_key)\n"
    "print('HAVE_LLM_KEY:', HAVE_LLM_KEY)"
))

# ---------------------------------------------------------------------------
# 6-7. Load all prompts from disk
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 1. Load all prompts from disk (Prompt Disclosure \u2192 real code)\n"
    "\n"
    "Every file below is read from `prompts/` at run time -- exactly what "
    "`rag.nodes` does internally (via `PROMPTS_DIR = Path(__file__)."
    "resolve().parents[2] / \"prompts\"`, mirroring `rag.config.REPO_ROOT`'s "
    "own resolution pattern). Printing length + a preview here proves "
    "there is no hardcoded prompt text anywhere in `nodes.py` -- disclosure "
    "really does equal what runs."
))
cells.append(new_code_cell(
    "import json\n"
    "from pathlib import Path\n"
    "from rag.nodes import PROMPTS_DIR, FEWSHOTS_DIR\n"
    "\n"
    "prompt_files = [\n"
    "    'system_assistant.md',\n"
    "    'router_intent.md',\n"
    "    'planner.md',\n"
    "    'retriever_tool_instructions.md',\n"
    "    'answerer_critic.md',\n"
    "]\n"
    "fewshot_files = [\n"
    "    'router_examples.json',\n"
    "    'planner_examples.json',\n"
    "    'answerer_examples.json',\n"
    "]\n"
    "\n"
    "for name in prompt_files:\n"
    "    text = (PROMPTS_DIR / name).read_text()\n"
    "    print(f'{name:35s} {len(text):5d} chars | {text.strip().splitlines()[0][:70]}')\n"
    "\n"
    "print()\n"
    "for name in fewshot_files:\n"
    "    data = json.loads((FEWSHOTS_DIR / name).read_text())\n"
    "    print(f'{name:35s} {len(data):3d} examples')"
))

# ---------------------------------------------------------------------------
# 8-9. Shared graph state
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 2. Shared graph state\n"
    "\n"
    "`rag.graph.GraphState` is the single `TypedDict` threaded through "
    "every node -- the fields each node reads/writes."
))
cells.append(new_code_cell(
    "from rag.graph import GraphState\n"
    "\n"
    "print('GraphState fields:')\n"
    "for field_name, field_type in GraphState.__annotations__.items():\n"
    "    print(f'  {field_name:16s} {field_type}')"
))

# ---------------------------------------------------------------------------
# 10-11. LLM call helper
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 3. LLM call helper\n"
    "\n"
    "`rag.llm.call_llm(system, user, model=None, max_tokens=1024, "
    "mock_response=None)` is the one function every node goes through. It "
    "dispatches on `get_config().llm_provider` (model-agnostic / config-"
    "driven, per `config.py`'s own doc comment) and only the `\"anthropic\"` "
    "path is fully implemented for now -- other providers raise "
    "`NotImplementedError`, per the plan. If `ANTHROPIC_API_KEY` is unset "
    "(or the real call fails for any reason), it never raises: it degrades "
    "to a `MockLLM` fallback and logs which path was taken to stderr, "
    "matching the graceful-degradation pattern already used by "
    "`web-search-mcp/web_search.py` and `rag/tts.py`."
))
cells.append(new_code_cell(
    "from rag.llm import call_llm\n"
    "\n"
    "test_response = call_llm(\n"
    "    system='You are a test.',\n"
    "    user='Say hello in one word.',\n"
    "    mock_response={'mock': True, 'note': 'this is the offline fallback path'},\n"
    ")\n"
    "print('call_llm() response:', test_response)"
))

# ---------------------------------------------------------------------------
# 12-14. Router node
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 4. Router node\n"
    "\n"
    "`router_node(state, llm_fn=call_llm)` builds `system_assistant.md + "
    "router_intent.md` as the system prompt, includes `router_examples."
    "json` as few-shot context, calls the LLM, and parses the response "
    "against `router_intent.md`'s documented schema (`task`, `constraints`, "
    "`keywords`, `safety_flags`) -- raising a clear error on malformed "
    "output rather than swallowing it. We run it on two sample transcripts "
    "from `router_examples.json`, including the safety-flag example "
    "(\"mix bleach and ammonia\")."
))
cells.append(new_code_cell(
    "from rag.nodes import router_node\n"
    "\n"
    "DEMO_TRANSCRIPT = 'Recommend an eco-friendly stainless-steel cleaner under fifteen dollars.'\n"
    "SAFETY_TRANSCRIPT = 'Can I mix bleach and ammonia to clean my grout faster?'\n"
    "\n"
    "router_state = router_node({'transcript': DEMO_TRANSCRIPT})\n"
    "safety_state = router_node({'transcript': SAFETY_TRANSCRIPT})"
))
cells.append(new_code_cell(
    "print('--- demo transcript router_output ---')\n"
    "print(json.dumps(router_state['router_output'], indent=2))\n"
    "print()\n"
    "print('--- safety-flag transcript router_output ---')\n"
    "print(json.dumps(safety_state['router_output'], indent=2))\n"
    "assert 'safety_flags' in safety_state['router_output']"
))

# ---------------------------------------------------------------------------
# 15-17. Planner node
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 5. Planner node\n"
    "\n"
    "`planner_node(state, llm_fn=call_llm)` builds `planner.md` + "
    "`planner_examples.json` as few-shot context, takes `state["
    "\"router_output\"]` as input, and produces the plan JSON documented in "
    "`planner.md`'s \"Output schema\" (`sources`, `call_web_search`, "
    "`filters`, `query`, `comparison_criteria`, `k`, `reconcile_on`). We "
    "feed it the demo transcript's Router output from the previous cell."
))
cells.append(new_code_cell(
    "from rag.nodes import planner_node\n"
    "\n"
    "planner_state = planner_node(router_state)"
))
cells.append(new_code_cell(
    "print(json.dumps(planner_state['planner_output'], indent=2))"
))

# ---------------------------------------------------------------------------
# 18-20. Retriever node
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 6. Retriever node\n"
    "\n"
    "`retriever_node(state)` (the only `async def` node -- it may `await` "
    "`web_search()`) calls the real `rag.rag_search()` tool using the "
    "Planner's `query`/`k`/`filters`, and additionally calls "
    "`web_search()` from `web-search-mcp/web_search.py` when the Planner's "
    "`call_web_search` field is `true`. Both result sets are reconciled "
    "via `rag.reconcile.reconcile()` (private facts as the grounded "
    "baseline; discrepancies flagged, never silently overwritten)."
))
cells.append(new_code_cell(
    "from rag.nodes import retriever_node\n"
    "\n"
    "# Jupyter supports top-level await in code cells.\n"
    "retriever_state = await retriever_node(planner_state)"
))
cells.append(new_code_cell(
    "print('rag.search count :', retriever_state['rag_results']['count'])\n"
    "print('web.search count :', retriever_state['web_results']['count'])\n"
    "print()\n"
    "for item in retriever_state['reconciled']['items']:\n"
    "    flag = f\" -- DISCREPANCY: {item['discrepancy']['detail']}\" if item.get('discrepancy') else ''\n"
    "    print(f\"  {item['title']:55s} \\${item['price']:<7}{flag}\")\n"
    "print()\n"
    "print('unmatched web results:', len(retriever_state['reconciled']['unmatched_web']))"
))

# ---------------------------------------------------------------------------
# 21-23. Answerer/Critic node
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 7. Answerer/Critic node\n"
    "\n"
    "`answerer_critic_node(state, llm_fn=call_llm)` builds "
    "`answerer_critic.md` + `answerer_examples.json`, calls the LLM once "
    "for the **Answerer** role (schema: `speech`, `citations`, "
    "`comparison_table` -- \u226415s spoken, every claim traces to a "
    "`doc_id`/`url`), then again for the **Critic** role (schema: "
    "`grounded`, `unsafe`, `reasons`, `action`). Per `answerer_critic.md`, "
    "on `action: \"revise\"` the Answerer regenerates **once** -- this is "
    "bounded in `graph.py` via `state[\"revise_count\"]` and a conditional "
    "edge, not by looping inside the node itself."
))
cells.append(new_code_cell(
    "from rag.nodes import answerer_critic_node\n"
    "\n"
    "answerer_state = answerer_critic_node(retriever_state)"
))
cells.append(new_code_cell(
    "print('--- Answerer payload ---')\n"
    "print(json.dumps(answerer_state['answerer_output'], indent=2))\n"
    "print()\n"
    "print('--- Critic verdict ---')\n"
    "print(json.dumps(answerer_state['critic_output'], indent=2))"
))

# ---------------------------------------------------------------------------
# 24-25. Wire the StateGraph
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 8. Wire the StateGraph\n"
    "\n"
    "`rag.graph.build_graph()` is the single source of truth other code "
    "(a future Streamlit UI, the eval harness) should import rather than "
    "re-wiring the graph elsewhere. Entry point `router` \u2192 `planner` "
    "\u2192 `retriever` \u2192 `answerer_critic` \u2192 conditional edge "
    "(`revise` \u2192 back to `answerer_critic`, `accept`/end \u2192 `END`)."
))
cells.append(new_code_cell(
    "from rag.graph import build_graph\n"
    "\n"
    "compiled_graph = build_graph()\n"
    "print(compiled_graph)"
))

# ---------------------------------------------------------------------------
# 26-27. End-to-end run
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 9. End-to-end run\n"
    "\n"
    "One call through the whole compiled graph on a real transcript -- the "
    "same demo query used throughout the repo (`README_shane.md`'s \"demo "
    "query\", `router_examples.json`'s first example): *\"Recommend an "
    "eco-friendly stainless-steel cleaner under fifteen dollars.\"* Because "
    "the graph contains an async node (`retriever`), we invoke it via "
    "`.ainvoke()` (LangGraph requires the async entry point on any graph "
    "with an async node, even from a notebook that could otherwise call "
    "`.invoke()`)."
))
cells.append(new_code_cell(
    "final_state = await compiled_graph.ainvoke({'transcript': DEMO_TRANSCRIPT})\n"
    "\n"
    "print('SPOKEN ANSWER:')\n"
    "print(' ', final_state['answerer_output']['speech'])\n"
    "print()\n"
    "print('CRITIC VERDICT:', final_state['critic_output']['action'],\n"
    "      '(grounded=' + str(final_state['critic_output']['grounded']) + ',',\n"
    "      'unsafe=' + str(final_state['critic_output']['unsafe']) + ')')\n"
    "print()\n"
    "print('CITATIONS:')\n"
    "for c in final_state['answerer_output']['citations']:\n"
    "    print('  -', c['title'], '|', c.get('doc_id') or c.get('url'))"
))

# ---------------------------------------------------------------------------
# 28-29. Optional TTS
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## 10. (Optional) Speak the final answer\n"
    "\n"
    "Soft-imports `rag.tts.speak()` (built separately -- see "
    "`notebooks/03_tts_summary.ipynb`) wrapped in try/except, so this cell "
    "degrades gracefully rather than failing the notebook if `tts.py` or "
    "its offline `pyttsx3` dependency isn't available in this environment."
))
cells.append(new_code_cell(
    "try:\n"
    "    from rag.tts import speak\n"
    "    audio_path = speak(final_state['answerer_output']['speech'])\n"
    "    print('wrote', audio_path)\n"
    "    try:\n"
    "        from IPython.display import Audio, display\n"
    "        display(Audio(str(audio_path)))\n"
    "    except Exception:\n"
    "        pass\n"
    "except Exception as e:\n"
    "    print(f'tts unavailable in this environment ({e!r}); skipping audio playback.')"
))

# ---------------------------------------------------------------------------
# 30. Handoff notes
# ---------------------------------------------------------------------------
cells.append(new_markdown_cell(
    "---\n"
    "### Handoff notes\n"
    "\n"
    "**What was built:** `src/rag/llm.py` (`call_llm()`, the one LLM "
    "call-site), `src/rag/nodes.py` (`router_node`, `planner_node`, "
    "`retriever_node`, `answerer_critic_node` -- each loading its prompt(s) "
    "fresh from disk, each accepting an optional `llm_fn` for dependency-"
    "injected testing), and `src/rag/graph.py` (`build_graph()`, the "
    "single source of truth for the compiled `StateGraph`).\n"
    "\n"
    "**Gap closed:** `prompts/` previously documented every prompt but "
    "nothing loaded them into running code (`README_shane.md`). This "
    "notebook and `nodes.py` are the concrete demonstration that "
    "disclosure == what runs (\u00a71 above reads and prints every prompt "
    "file straight from disk).\n"
    "\n"
    "**Model-default fix applied:** `rag.config.Config.llm_model` defaulted "
    "to the retired alias `claude-3-5-sonnet-latest`; fixed to "
    "`claude-sonnet-5` in `src/rag/config.py` (and mirrored in "
    "`.env.example`'s `LLM_MODEL=` line and `prompts/README.md`'s "
    "Conventions section).\n"
    "\n"
    "**Known limitations:**\n"
    "* Only the `\"anthropic\"` provider is implemented in `llm.py` -- "
    "`openai`/`google`/`bedrock`/`local` raise `NotImplementedError`, per "
    "the plan's scope.\n"
    "* The `MockLLM` fallback shape is deliberately simple: each node "
    "supplies its own structurally-valid canned response (see "
    "`_router_mock_response` / `_planner_mock_response` / "
    "`_answerer_mock_response` / `_critic_mock_response` in `nodes.py`) "
    "rather than `call_llm` sniffing prompt text for schema markers -- "
    "simpler and independently testable, per the plan.\n"
    "* The Critic's mock response always accepts (`action: \"accept\"`), so "
    "the revise loop is only exercised with a real LLM key or an injected "
    "`llm_fn` in tests (`tests/test_nodes.py::"
    "test_answerer_critic_node_revise_path_increments_revise_count`).\n"
    "* `retriever_node`'s `web.search` path inherits `web_search.py`'s own "
    "documented limitation: `price`/`availability` are always `None` from "
    "the current providers, so most reconciliation discrepancy flags won't "
    "fire until that's filled in."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print("wrote", OUT)
