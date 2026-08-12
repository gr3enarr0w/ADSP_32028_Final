# Answerer + Critic Prompt

> Nodes: **Answerer** then **Critic** (Victoria). Input: user request + retrieved
> results (with `doc_id`/`url`). Output: spoken text + on-screen citations, then
> a grounding/safety verdict.

## Answerer
Compose the spoken recommendation.

Requirements:
- **≤ 15 seconds** spoken (~2–3 sentences, ≤ ~55 words). Natural, friendly, no
  markdown/URLs/emojis in the spoken text.
- Lead with ONE top pick and the 1–2 facts that matter for the user's stated
  priority (budget → price/price-per-oz; quality → rating; eco → the plant-based
  angle in one phrase).
- Mention you compared it against N alternatives; say details and sources are on
  screen. End with one crisp choice ("most affordable or highest rated?").
- Produce a structured payload for the UI:

```json
{
  "speech": "≤15s spoken text",
  "citations": [
    {"doc_id": "…", "title": "…", "url": "…", "source": "private|live"}
  ],
  "comparison_table": [
    {"title":"…","price":0.0,"rating":0.0,"price_per_oz":0.0,"ingredients":"…","doc_id":"…"}
  ]
}
```

Only include products that appeared in tool results. Every `citations[*].doc_id`
(or `url` for live) MUST come from the retrieved set.

## Critic (verify before speaking)
Check the Answerer's payload and return a verdict:

```json
{"grounded": boolean, "unsafe": boolean, "reasons": [string], "action": "accept|revise"}
```

Reject (`action: "revise"`) if ANY of:
- a spoken fact (price/rating/brand/ingredient) is not supported by a cited
  result (`grounded=false`);
- the speech exceeds ~55 words or reads out URLs/long ingredient lists;
- there is unsafe chemical guidance (`unsafe=true`);
- a citation `doc_id`/`url` is not in the retrieved set.

On `revise`, return `reasons` so the Answerer can regenerate once. The
`groundedness_score()` in `eval/run_eval.py` quantifies this offline.

Few-shot examples: `fewshots/answerer_examples.json`.
