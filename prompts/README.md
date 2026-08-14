# `prompts/` — Prompt Disclosure (Final deliverable, 5 pts)

**Owner:** Shane (assembly) · contributions from the whole team's node authors.

This folder is the single, auditable source of every prompt the assistant uses.
It satisfies the grading requirement:

> *Prompt Disclosure — Include all key prompts: system prompts, router/planner
> tool prompts, few-shot examples; show how prompts map to nodes/tools.*

## Prompt → node/tool map

| File | LangGraph node | Owner | Purpose |
|---|---|---|---|
| `system_assistant.md` | global (all LLM calls) | shared | Persona, grounding & safety rules, citation contract |
| `router_intent.md` | **Router** | Alison | Extract task + constraints (budget/material/brand) + safety flags → JSON |
| `planner.md` | **Planner** | Victoria | Choose sources (private vs live), filters, comparison criteria → JSON plan |
| `retriever_tool_instructions.md` | **Retriever** | Shane/Clark | How the agent calls `rag.search` & `web.search`; reconciliation rules |
| `answerer_critic.md` | **Answerer** + **Critic** | Victoria | Compose ≤15s cited spoken answer; verify grounding & safety, spoken via `tts.speak()` |
| `fewshots/router_examples.json` | Router | Alison | Few-shot intent-parsing examples |
| `fewshots/planner_examples.json` | Planner | Victoria | Few-shot planning examples |
| `fewshots/answerer_examples.json` | Answerer | Victoria | Few-shot cited-answer examples |

## Tool schemas referenced by the prompts

* `rag.search` — private Amazon-2020 retrieval. Schema: `mcp/README_mcp_rag.md`.
* `web.search` — live price/availability. Schema: Clark's `../web-search-mcp/README_mcp_web.md`.

## Conventions

* **Model-agnostic.** Prompts assume a tool-calling chat model. Default is
  Claude (`LLM_MODEL=claude-sonnet-5`); swap via `.env` with no prompt
  changes. Placeholders use `{{double_braces}}`.
* **Grounding is mandatory.** Every product claim must trace to a `doc_id`
  (private) or a `url` (live). The Critic rejects ungrounded answers.
* **Loading.** Nodes load these files at startup (e.g.
  `Path("prompts/router_intent.md").read_text()`), so disclosure == what runs.
  This is now demonstrated concretely, not just stated as intent: every node
  in `src/rag/nodes.py` reads its prompt file(s) fresh from disk on each call
  (no hardcoded prompt text), and `notebooks/04_orchestration.ipynb` proves it
  by loading and printing every prompt + few-shot file before wiring the
  LangGraph graph together.
