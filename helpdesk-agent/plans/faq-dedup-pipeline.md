# FAQ Dedup Pipeline Spec

## Problem
Generated FAQ articles accumulate duplicates across pipeline runs. The ~400-page output Google Doc is bloated with near-identical content. Current dedup is limited to `UNIQUE(article_topic)` which misses semantically identical articles with different titles/wording.

## Scope
5 sequential stories: ANTSE-286 → ANTSE-289 → ANTSE-287 → ANTSE-288 → ANTSE-290

Each phase builds on the prior. Each gets its own commit/PR but they must execute in order.

---

## Phase 1: MinHash Structural Fingerprinting (ANTSE-286)

Fast O(n) first pass — catches structurally identical articles with minor wording edits.

### Steps
1. Add `datasketch` to `requirements.lock`
2. Schema migration in `db.py`: add `fingerprint TEXT` column to `generated_articles` (existing ALTER TABLE pattern at lines 229-237)
3. Create `faq/dedup.py` — shared dedup module:
   - `compute_minhash(text, num_perm=128) → MinHash` (word 3-gram shingles)
   - `serialize_fingerprint(mh) → str` / `deserialize_fingerprint(stored) → MinHash`
   - `check_structural_duplicate(new_mh, existing_fps, threshold=0.9) → list[tuple[topic, similarity]]`
4. Modify `faq/generator.py` before INSERT (~line 271): compute fingerprint, check against existing, skip if >0.9 Jaccard
5. Create `scripts/backfill_fingerprints.py` — backfill existing articles
6. Create `tests/test_dedup.py` — 10 duplicate pairs + 10 distinct pairs

### Files
- Modify: `requirements.lock`, `db.py`, `faq/generator.py`
- Create: `faq/dedup.py`, `scripts/backfill_fingerprints.py`, `tests/test_dedup.py`

### Done when
- Fingerprint computed and stored on every article write
- Structural near-duplicates (>90% shingle overlap) flagged on 10 known-duplicate test pairs
- Existing articles backfilled

---

## Phase 2: Similarity Threshold Calibration (ANTSE-289)

Set the cosine similarity threshold from evidence, not intuition.

### Steps
1. Create `scripts/calibrate_threshold.py`:
   - Read all `generated_articles` from DB
   - Select 30-40 article pairs (15-20 known dupes, 15-20 distinct)
   - Compute cosine similarity for each pair
   - Sweep threshold 0.80-0.99, compute FP/FN rates at each step
   - Output optimal threshold
2. Create `data/calibration_pairs.json` — persistent labeled pairs for re-calibration
3. Update `config.py`: add `EMBEDDING_SIMILARITY_THRESHOLD` with calibrated value
4. Document results in `docs/dedup-calibration.md`

### Files
- Create: `scripts/calibrate_threshold.py`, `data/calibration_pairs.json`, `docs/dedup-calibration.md`
- Modify: `config.py`

### Done when
- 30-40 pairs labeled and scores recorded
- Threshold set with FP rate and FN rate documented
- Threshold value committed to config, not hardcoded

---

## Phase 3: Embedding Cosine Dedup Gate — generator.py (ANTSE-287)

Catches semantic duplicates that MinHash misses (different wording, identical meaning). Stubs against Vertex AI until Epic 2.

### Steps
1. Add to `config.py`: `EMBEDDING_MODEL`, `EMBEDDING_SIMILARITY_THRESHOLD` (from Phase 2), `EMBEDDING_ENABLED` (default `false`), `VERTEX_EMBEDDING_MODEL`
2. Create `faq/embedding.py`:
   - `get_embedding(text) → list[float]` — Vertex AI when enabled, deterministic hash-based stub when not
   - `cosine_similarity(a, b) → float` — pure Python
   - `check_embedding_duplicate(new_text, existing_embeddings, threshold) → (is_dup, max_score, matching_topic)`
3. Schema migration in `db.py`: add `embedding TEXT` column to `generated_articles`
4. Modify `faq/generator.py` before INSERT (~line 271): compute embedding, compare against existing, skip if above threshold
5. Extend `scripts/backfill_fingerprints.py` to also backfill embeddings

### Files
- Modify: `config.py`, `db.py`, `faq/generator.py`, `scripts/backfill_fingerprints.py`
- Create: `faq/embedding.py`

### Done when
- INSERT skipped for any article whose similarity exceeds threshold
- DB article count grows at expected rate (no new duplicates)
- Swap to Vertex AI is config-only (`EMBEDDING_ENABLED=true` + model name)

---

## Phase 4: Embedding Cosine Dedup Gate — google_docs.py (ANTSE-288)

DB gate (Phase 3) prevents storage duplicates. This gate prevents the output doc from accumulating duplicates.

### Steps
1. Modify `faq/google_docs.py` `write_faq_entries()` (~line 110):
   - Before writing, compute embedding for each entry in the list
   - Compare each entry against all earlier entries in the batch
   - Remove entries above cosine threshold (keep first occurrence)
   - Log: "Wrote N FAQ entries to Google Doc (M near-duplicates removed)"
2. Import `get_embedding()` and `cosine_similarity()` from `faq/embedding.py`

### Files
- Modify: `faq/google_docs.py`

### Done when
- No semantically equivalent article appended to output doc
- Tested against 5 known near-duplicate sections — all 5 skipped

---

## Phase 5: Initial Google Doc Cleanup (ANTSE-290)

One-time cleanup of the existing ~400-page doc using all the dedup tools built in Phases 1-4.

### Steps
1. Create `scripts/cleanup_google_doc.py`:
   - Read output doc via `FAQ_OUTPUT_DOC_ID` from config
   - Parse sections using `_extract_text()` logic from `faq/google_docs.py`
   - Record "before" section count
   - For each section pair: compute MinHash Jaccard + embedding cosine similarity
   - Flag near-duplicates, produce report
   - `--dry-run` (default): report only
   - `--execute`: back up doc, rewrite with duplicates removed
   - Record "after" section count
   - Sample 20 remaining sections for manual review

### Files
- Create: `scripts/cleanup_google_doc.py`

### Done when
- Output doc section count before and after recorded
- All remaining sections confirmed unique (sampled review of 20)
- Script retained in repo for future use

---

## Shared Files Across Phases

| File | Phases | Role |
|------|--------|------|
| `faq/dedup.py` | 1, 5 | MinHash fingerprinting |
| `faq/embedding.py` | 2, 3, 4, 5 | Embedding + cosine similarity |
| `faq/generator.py` | 1, 3 | INSERT gate (both MinHash + embedding) |
| `faq/google_docs.py` | 4 | Output doc dedup gate |
| `db.py` | 1, 3 | Schema migrations (fingerprint, embedding columns) |
| `config.py` | 2, 3 | Threshold + embedding config |
| `scripts/backfill_fingerprints.py` | 1, 3 | Backfill existing articles |

## Prerequisite
ANTSE-291 (pod memory bump) should land before Phase 3 goes live — embedding models need the extra RAM.

## Verification (end-to-end after all phases)
1. `pytest tests/test_dedup.py` passes
2. Run backfill script — all articles have fingerprints + embeddings
3. Run pipeline — no new duplicates written to DB or doc
4. Run `cleanup_google_doc.py --dry-run` — report shows reduction
5. Run `cleanup_google_doc.py --execute` — doc section count drops, remaining sections verified unique
