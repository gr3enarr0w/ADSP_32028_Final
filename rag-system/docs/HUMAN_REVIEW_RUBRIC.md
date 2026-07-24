# Human Review Rubric — Judge-Independent Grounding Labels

This is the rubric for the **judge-independent** manual review tier in
`scripts/evaluate_ragas.py` (`--export-human-review` / `--summarize-human-review`).

## Why this exists

Every other quality signal in this project comes from an LLM judge — RAGAS's
`context_precision` / `context_recall` / `faithfulness` / `answer_relevancy` /
`answer_correctness`, the closed-book comparison arm, the Claude web_search
comparison arm, and even the multi-judge Ollama cross-check
(`--multi-judge-crosscheck`). That is a real gap, not a hypothetical one: this
project has already hit repeated judge-reliability problems in practice —

- a Gemini-based second judge was tried as a faithfulness cross-check and
  explicitly rejected because the readily available model version was too
  stale to be a trustworthy comparison point, and
- a full statistical rigor pass found that several RAGAS metric deltas were
  **not distinguishable from judge noise** at reasonable sample sizes.

Adding a fourth or fifth LLM judge does not fix that gap — it's still an LLM
judging LLM output. This rubric adds the one signal that isn't: a human
reading the actual (question, retrieved context, generated answer) triple and
making a call.

## The rubric

Read the `question`, the `retrieved_contexts`, and the `generated_answer`.
Assign **exactly one** of these three labels to `human_label`:

| Label | Definition |
|---|---|
| **Grounded** | Every factual claim in the answer is traceable to the retrieved context. |
| **Partially grounded** | Some claims are supported by context, others are unsupported / inferred / from general knowledge. |
| **Not grounded** | The answer contradicts the retrieved context, or ignores it entirely — e.g. it answers from general knowledge when relevant context WAS available and should have been used, or it hallucinates specifics (numbers, names, dates) not present anywhere in the retrieved context. |

Use these **exact three strings**, case-sensitive, no trailing punctuation,
so `summarize_human_review()` can parse them without guessing at synonyms:

```
Grounded
Partially grounded
Not grounded
```

### A note on "Partially grounded" and the `[C]`/`[G]` tags

`generate_answer()` in `scripts/evaluate_ragas.py` already asks the model to
tag each claim inline as `[C]` (derived from retrieved context) or `[G]`
(supplemented from the model's own general knowledge) — see that function's
docstring for the full attribution-tagged blending policy. Because of this, a
**"Partially grounded" label should naturally correlate with answers that
contain both `[C]` and `[G]` tags.** If you label an answer "Partially
grounded" but it's tagged entirely `[C]` (or entirely `[G]`), or you label an
answer "Grounded" and it's full of `[G]` tags, that's worth a note in
`human_comment` — it may mean the model's own self-tagging doesn't match
reality, which is itself a useful finding.

## The `human_comment` column

Use it freely for anything qualitative:

- What worked (retrieval found the right passage, answer used it well).
- What should be improved (retrieval missed the obviously relevant chunk,
  answer ignored good context, answer's `[C]`/`[G]` self-tagging looked wrong).
- Anything that made the label a close call.

## How the output gets used

`summarize_human_review(csv_path, ragas_csv_path=None)` reads a filled-in
copy of the exported CSV and prints:

1. A count/percentage breakdown per label (plus a warning if any
   `human_label` value doesn't exactly match one of the three strings above).
2. If a saved RAGAS per-question results CSV is also given
   (`--compare-ragas`), a cross-tabulation of mean RAGAS scores
   (`faithfulness`, `answer_relevancy`, `answer_correctness`,
   `context_precision`, `context_recall` — whichever columns are present) by
   human label, plus an explicit sanity check: **mean `faithfulness` for
   "Partially grounded" rows should be lower than for "Grounded" rows.** This
   is a real, falsifiable check the rubric can be validated against, not just
   a summary table — if it fails, that's flagged as evidence the human rubric
   and the LLM judge may be measuring different things, worth investigating
   rather than ignoring.

## Usage

```bash
# 1. Export a sample for review (runs real retrieval + generation)
python scripts/evaluate_ragas.py \
    --testset testset.json \
    --collection my_collection \
    --export-human-review human_review.csv

# 2. Open human_review.csv, fill in human_label (+ optional human_comment) for each row.

# 3. Summarize the filled-in labels
python scripts/evaluate_ragas.py --summarize-human-review human_review.csv

# 3b. ...or cross-tabulate against a saved RAGAS results CSV
python scripts/evaluate_ragas.py \
    --summarize-human-review human_review.csv \
    --compare-ragas ragas_results.csv
```

The exported CSV's first line is a `#`-prefixed comment restating the three
valid `human_label` values (skipped automatically by
`summarize_human_review()`'s `pandas.read_csv(..., comment="#")`), so the
rubric summary is visible even to a reviewer who opens the raw CSV without
this doc handy. The full rubric always lives here and in this module's
docstring — if the two ever disagree, this file and the module docstring
should be reconciled, not a third version invented.
