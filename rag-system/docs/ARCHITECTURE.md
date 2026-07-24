# Hermes RAG (Qdrant) — Architecture

**Status of this document:** current-state baseline as of 2026-07-19, rewritten from scratch
against the actual code (`scripts/*.py`, `config/config.yaml`) rather than carried forward from
the previous revision. It exists to be diffed against for an upcoming round of changes
(a faithfulness-metric fix, HyDE query rewriting, query-adaptive parent-chunk expansion, MMR
diversity reranking, and a statistical-power decision on the tuning program itself) — every claim
below was verified against the code or a live-run artifact under `/tmp/nfcorpus_eval_v2/`, not
carried over from the prior doc. `docs/CHANGELOG.md` is the full dated historical record (bugs
found/fixed, design changes, live-verification notes) and is NOT modified by this rewrite — read
it for "how did we get here"; read this document for "what does the system do right now."

Code map:
- `scripts/utils.py` — `EfficientRAG` class + `load_config()`: Qdrant client, hierarchical
  parent/child chunking, dense+sparse hybrid storage/retrieval, quantization guard, cross-encoder
  reranking, CRAG-style context review, Adaptive-RAG-lite retrieval gate.
- `scripts/ingest.py` — CLI wrapper around `EfficientRAG.ingest()`.
- `scripts/retrieve.py` — CLI wrapper + `retrieve_context()`/`format_context()`, the high-level
  function Hermes calls via `execute_code`.
- `scripts/evaluate_ragas.py` — RAGAS evaluation, generation (blended `[C]`/`[G]` prompt +
  verify-and-repair), closed-book/web-search comparison arms, Ollama and NeuralWatt multi-judge
  faithfulness cross-checks, judge-independent human review tier.
- `scripts/visualize_embeddings.py` — standalone diagnostic (PCA/UMAP + cosine-similarity
  degeneracy check) for embedding-collapse debugging.
- `config/config.yaml` — default runtime config, layered under by each script's own
  `load_config()`-driven defaults.
- `docs/HUMAN_REVIEW_RUBRIC.md` — the canonical Grounded/Partially grounded/Not grounded rubric.
- `docs/CHANGELOG.md` — full dated history (bugs, live-verification results, design decisions).

---

## 1. Full pipeline walkthrough

### 1.1 Ingestion (`EfficientRAG.ingest()`, `scripts/utils.py:499-591`)

1. **Load.** A source is loaded to raw text by one of three loaders, dispatched by `ingest.py`'s
   `--path`/`--url`/`--api` flags:
   - `_load_local_file()` (`utils.py:284-302`): `.pdf` via `pypdf.PdfReader`, `.docx` via
     `python-docx`, `.txt`/`.md`/`.markdown` read as plain UTF-8, anything else falls back to a
     best-effort plain-text read.
   - `_fetch_url()` (`utils.py:304-344`): tries `trafilatura.extract()` first (readability-style
     boilerplate removal — added 2026-07-16 after boilerplate/nav-chrome was found leaking into
     embedded chunks), falls back to BeautifulSoup tag-stripping (`script`/`style`/`nav`/`footer`/
     `header`/`aside` removed) if trafilatura returns nothing. Output capped at 50,000 characters
     either way.
   - `_fetch_api()` (`utils.py:346-358`): generic `requests.request()`, JSON-pretty-printed if the
     response is `application/json`, capped at 30,000 characters.
   - A doc ID is `hashlib.md5(source).hexdigest()[:12]` for all three source types.
2. **Hierarchical parent-child chunking** (`_hierarchical_chunk()`, `utils.py:372-473`, called with
   recursive separators `["\n\n", "\n", ". ", " ", ""]`):
   - Sizes are **token counts**, measured via the embedding model's own HuggingFace tokenizer
     (`_token_len()`, `utils.py:360-370` — `self.embed_model.tokenizer.encode(text,
     add_special_tokens=False)`), **not character counts**. This changed from character-based
     sizing on 2026-07-16 (see CHANGELOG); the current defaults were re-derived, not relabeled.
   - **Current defaults** (`config/config.yaml` / `load_config()`, both in sync):
     `child_chunk_size: 100`, `child_chunk_overlap: 20`, `parent_chunk_size: 400` (tokens).
     `bge-small-en-v1.5`'s max sequence length is 512 tokens, so both are comfortably inside it.
   - Text is first recursively split into **parent chunks** (`parent_size=400`,
     `overlap=child_chunk_overlap // 2`), then each parent is recursively split again into
     **child chunks** (`child_size=100`, `overlap=20`). Every child carries its parent's
     `parent_id` (`"{doc_id}_parent_{p_idx}"`) in metadata.
   - The recursive splitter tokenizes each candidate part once up front (O(num_parts), not
     O(num_parts²)) and falls back to raw token-window slicing (encode once, slice, decode) only
     if no separator is found at all in a chunk that's still too large.
3. **Per-chunk embedding and payload construction** (`ingest()`'s loop):
   - **Parent chunks are never embedded.** They're stored as a separate Qdrant point with a
     **zero dummy `"dense"` vector** (never searched, only looked up by deterministic point ID —
     `_parent_point_id()`, `utils.py:195-200`, a UUID5 derived from the `parent_id` string) and
     `chunk_type: "parent"` in the payload. This was previously a design gap — a prior revision of
     this project had `fetch_parents: true` in config with **no code that ever read it** and no
     parent record stored at all (fixed 2026-07-16; see CHANGELOG "Bug 4").
   - **Child chunks** are embedded twice:
     - Dense: `self.embed_model.encode(text, normalize_embeddings=True)` (default
       `BAAI/bge-small-en-v1.5`, 384 dims).
     - Sparse (if `indexing.hybrid_search: true`, the default): `_sparse_vector_for()`
       (`utils.py:488-497`) via `fastembed.SparseTextEmbedding("Qdrant/bm25")` — BM25-style,
       zero-training, deterministic term/IDF weighting.
   - Both vectors are written to one Qdrant point via **named vectors** `"dense"` and `"sparse"`
     (`vectors_config`/`sparse_vectors_config` in `create_collection()`).
   - **Contextual summary** (if `indexing.use_contextual_summaries: true`, the default): computed
     per child chunk via `self.summary_fn` if injected (`config["_summary_fn"]`, not a
     YAML-serializable key), else `_generate_summary()` (`utils.py:475-486`) — **a naive
     first-2-3-sentence extractive placeholder, NOT an LLM call**, despite "contextual summaries"
     being advertised as an LLM-driven feature in `SKILL.md`/`README.md`. This is a known,
     long-standing gap, not new.
   - Child payload text is capped at 2000 characters (`payload["text"] = chunk.text[:2000]`).
4. **Quantization guard** (`_resolve_quantization_config()`/`_effective_quantization_label()`,
   `utils.py:202-247`, run once inside `create_collection()`): the *effective* quantization mode is
   derived from the **actual dense embedding dimension**, not blindly from `indexing.quantization`:
   - `binary` iff (`dim >= binary_dim_threshold` (default 1024) AND `mode == "light"`) OR
     `force_binary: true`.
   - else `scalar` (INT8, `quantile=0.99`, `always_ram=True`) if `mode` is `"light"` or `"balanced"`.
   - else `none` (mode `"full"`).
   - With the current default config (`mode: light`, `dim: 384`), this **always routes to scalar
     INT8**, never binary — binary quantization is documented by Qdrant to degrade badly below
     1024 dimensions. The effective (not requested) mode is reported in every `ingest()` call's
     returned stats dict (`stats["quantization"]`), so a mismatch between requested and effective
     mode is always visible after the fact.
5. **Collection reuse guard** (`create_collection()`, `utils.py:249-282`): checks
   `qdrant.collection_exists()` before creating. If it exists and `force_recreate` is false, logs
   "reusing" and returns without touching Qdrant. `ingest.py`'s directory-ingest loop only passes
   `force_recreate=True` for the **first** file in a `--recursive` run (fixing a real data-loss bug
   where every subsequent file's ingest wiped the whole collection — see CHANGELOG "Bug 1"/"Bug 2").
6. Returned stats: `points_ingested`, `child_points_ingested`, `parent_points_ingested`, `mode`,
   effective `quantization`.

### 1.2 Retrieval (`EfficientRAG.retrieve()`, `scripts/utils.py:749-915`)

Order of operations, in the code's actual sequence:

1. **Adaptive-RAG-lite gate** (`_classify_retrieval_need()`, `utils.py:593-661`; gated on
   `retrieval.adaptive_retrieval_enabled`, **default `false`**). When enabled, a single cheap LLM
   call classifies the query as `"none"` (general-knowledge, skip retrieval — `retrieve()` returns
   `[]` immediately) or `"light"` (proceed with retrieval, overriding `top_k`/`oversampling` for
   this call only via `adaptive_light_top_k`/`adaptive_light_oversampling`, a shallow-copied config
   so `self.config` is never mutated). Biased toward `"light"` when ambiguous. Provider is one of
   `"ollama"` (default, model `glm-5.2:cloud`) / `"neuralwatt"` / `"claude"` (Haiku-class); an
   unknown provider raises `ValueError` immediately, but any call/parse/network error on a valid
   provider fails open to `adaptive_retrieval_fallback_strategy` (default `"light"`) — a classifier
   hiccup degrades to "gate did nothing," never to silently skipping retrieval. The outcome is
   recorded on `self._last_retrieval_strategy` (`"none"`/`"light"`/`None` if the gate is off),
   read by `retrieve_context()` in `retrieve.py:87-88` after the call, not returned by `retrieve()`
   itself (non-breaking for existing callers). v1 scope is deliberately `"none"`/`"light"` only —
   no third "deep" tier (Self-RAG/FLARE were both considered and ruled out as needing
   infrastructure — fine-tuning, streaming-with-confidence-signals — this project doesn't have).
2. **Query embedding.** Dense (`bge-small-en-v1.5`) always; sparse (`Qdrant/bm25`) also if
   `indexing.hybrid_search: true` (default).
3. **Hybrid search + client-side RRF fusion** (when hybrid search is on): two **independent**
   `qdrant.query_points()` calls (`using="dense"`, `using="sparse"`), each fetching up to
   `retrieval.rerank_candidate_pool` (default 40) hits, both filtered to `chunk_type: "child"`
   (parent points, carrying a dummy zero vector, are explicitly excluded from search — a fix
   alongside the parent-chunk feature landing; before that fix, zero-vector parent points could
   surface as search hits themselves). The two ranked lists are fused **client-side**: each point
   accumulates `1 / (rrf_k + rank)` per leg (rank 1-indexed), summed across legs when a point
   appears in both, then sorted by summed score. **This is a deliberate design choice, not an
   oversight or missing feature**: Qdrant's server-side `FusionQuery(fusion=Fusion.RRF)` does not
   expose a tunable `k` ([qdrant/qdrant#5116](https://github.com/qdrant/qdrant/issues/5116)), so
   this project fuses manually to make `rrf_k` (config key `retrieval.rrf_k`, default 60) tunable.
   If hybrid search is off, this step is a single plain dense `query_points()` call instead.
4. **Cross-encoder rerank** (`retrieval.rerank`, default `true`): the fused/dense candidate hits
   are scored as `(query, chunk_text)` pairs by a `sentence-transformers.CrossEncoder`
   (default `cross-encoder/ms-marco-MiniLM-L-6-v2`, lazily instantiated in `__init__` only when
   `rerank` is enabled), sorted descending, truncated to `retrieval.rerank_top_n` (default 8,
   falls back to `top_k` if unset). If reranking is off, the candidate pool is simply truncated to
   `top_k` by its original (fused or dense) score.
5. **Parent-chunk expansion** (`retrieval.fetch_parents`, default `true`): for each surviving hit,
   the full parent-chunk text is looked up by deterministic point ID
   (`qdrant.retrieve(ids=[...])`) and swapped into the hit's `"text"` field, while the original
   matched child text is preserved under a new `"child_text"` key. Lookup failures (e.g. points
   ingested before this feature existed) fall back gracefully to child-only text — never raises.
6. **CRAG-style context review/distillation** (`review_and_distill_context()`, `utils.py:1011-1123`;
   gated on `retrieval.context_review_enabled`, **default `false`**, OFF by default and
   experimental). Runs **last**, after reranking and parent expansion, on the hits that would
   otherwise go straight to generation. Per hit: splits `text` into sentences
   (`_split_into_sentences()`, a regex heuristic with a common-abbreviation guard — no new NLP
   dependency), scores every sentence's relevance to the query in **one batched LLM call per hit**
   (`_score_sentence_relevance()`, Haiku-class model), discards sentences below
   `retrieval.context_review_threshold` (default 0.3, deliberately lenient — losing a genuinely
   relevant sentence costs `context_recall` more than keeping a mildly noisy one costs
   `context_precision`), and recomposes survivors into a denser `text`. Explicitly targets
   `context_precision`, **not** faithfulness — see Known Issue #3 below.
7. Formatted results (`utils.py:856-869`) carry `score`, `text` (possibly parent-expanded),
   `child_text`, `source`, `parent_id`, `summary` (naive placeholder unless `_summary_fn` was
   injected), `tags`, `chunk_type`.

`retrieve.py`'s `retrieve_context()` (`retrieve.py:72-100`) wraps `EfficientRAG.retrieve()` for
`execute_code`, and `format_context()` (`retrieve.py:17-69`) renders the hit list into a
citation-ready text block, distinguishing "gate deliberately skipped retrieval" from "retrieval ran
and found nothing" via the `retrieval_skipped` flag.

### 1.3 Generation (`scripts/evaluate_ragas.py`)

1. **Blended `[C]`/`[G]` attribution prompt** (`generate_answer()`, `evaluate_ragas.py:1706-1780`).
   Replaced a strict "answer ONLY from context, refuse otherwise" prompt on 2026-07-16 after that
   policy was measured to cause a **75% refusal rate** on thin/tangential context, driving
   `answer_relevancy` below a no-retrieval closed-book baseline. The current prompt: use CONTEXT
   wherever it applies, tag context-derived claims `[C]`; may supplement with general knowledge
   when context is thin/missing, tag those `[G]`; on conflict, say so and prefer CONTEXT; never
   fabricate `[C]`-tagged specifics; end with a `Grounding: <fully|partially context-based |
   general-knowledge-based>` line. Uses `_build_anthropic_client()` (direct Anthropic if
   `ANTHROPIC_API_KEY` set, else `AnthropicVertex` if `ANTHROPIC_VERTEX_PROJECT_ID` set).
   **Faithfulness is expected to read lower under this policy by design** — see Known Issue #3.
2. **Verify-and-repair** (`verify_and_repair_answer()`, `evaluate_ragas.py:1541-1703`; opt-in via
   `generate_answer(..., verify_and_repair=True)` / CLI `--verify-repair`, **off by default**).
   Extracts every `[C]`-tagged claim (`_extract_c_tagged_claims()`), and — if any exist — batches
   **all** of them into a single verification LLM call asking whether each is actually traceable
   to the retrieved context. Unsupported claims are either **retagged** (`[C]`→`[G]`, when the
   claim is accurate general knowledge that was mistagged) or **rewritten** (tightened to a more
   literal extraction the judge supplies, still tagged `[C]`, when it drew on context but
   embellished). Motivated by a real A/B finding: the blended prompt fixed the refusal-rate
   problem but caused a genuine faithfulness regression (-0.30 to -0.35 absolute, p<0.001 vs. the
   strict prompt) traced by manual inspection mostly to **prose style** around `[C]`-tagged claims
   (expository framing tripping RAGAS's sentence-level judge), not ungrounded content — informed by
   the Self-Correcting RAG ablation (arXiv:2604.10734), which found context-side cleaning alone
   left faithfulness flat but output-side review/correction measurably fixed it (AP 0.58→0.85 in
   that paper). **Implemented, off by default, and — per the live eval harness under
   `/tmp/nfcorpus_eval_v2/` — never actually included in any of the tuned configs that were
   statistically compared against baseline** (see Known Issue #2).

### 1.4 Evaluation (`scripts/evaluate_ragas.py`)

| Arm / function | What it scores | Contexts used |
|---|---|---|
| `run_ragas_evaluation()` (`:2107-2214`) | `context_precision`, `context_recall` always; + `faithfulness`, `answer_relevancy`, `answer_correctness` if answers were generated | `EfficientRAG.retrieve()` via `retrieve_context()` |
| `run_closed_book_evaluation()` (`:1824-1925`) | `answer_relevancy`, `answer_similarity` (embedding cosine vs. reference — chosen over `answer_correctness`'s atomic-statement TP/FP/FN decomposition, which is a poor fit for NFCorpus's full-document reference-proxy) | none (zero retrieval, parametric knowledge only) |
| `run_claude_websearch_evaluation()` (`:2010-2104`) | `faithfulness`, `answer_relevancy`, `answer_correctness` | Claude's own native `web_search` tool results (title+URL), never NFCorpus gold passages |
| `run_multi_judge_faithfulness_crosscheck()` (`:641-736`) | re-scores `faithfulness` only, from a saved RAGAS results CSV, via 3 local Ollama models (`glm-5.2:cloud`, `kimi-k2.7-code:cloud`, `minimax-m3:cloud`) | reuses saved `retrieved_contexts`/`response` |
| `run_neuralwatt_multi_judge_consensus()` (`:852-1284`) | re-scores `faithfulness` only via 3 NeuralWatt-hosted models in parallel, aggregated by **geometric median** (Weiszfeld's algorithm), flagged for `high_disagreement` (`delta_m > 0.3`) / `suspiciously_unanimous` (`delta_m < 0.05`), with a **zero-NaN guarantee**: per-row escalation chain NeuralWatt → single-row Ollama-cloud fallback → Claude tiebreaker → loud `"unrecoverable"` flag if all three fail | same as above |
| `export_for_human_review()` / `summarize_human_review()` (`:1287-1517`) | judge-**independent** human labels (`Grounded` / `Partially grounded` / `Not grounded`, `docs/HUMAN_REVIEW_RUBRIC.md`), cross-tabulated against a saved RAGAS CSV with a falsifiable check ("Partially grounded" should mean-score lower on `faithfulness` than "Grounded") | reuses `generate_answer()` output |

The RAGAS judge LLM/embeddings (`_default_judge_llm()`/`_default_judge_embeddings()`,
`:394-474`) default to direct Anthropic/OpenAI, falling back to Claude-on-Vertex /
`gemini-embedding-001` (this project's GCP org blocks the `text-embedding-*` family) when only
Google Application Default Credentials are configured. `max_tokens=4096` is set explicitly on both
judge-LLM constructors — a prior bug (fixed 2026-07-16) left both at the wrapper default of 1024,
causing `LLMDidNotFinishException`/scattered `NaN`s under RAGAS's 3-generation self-consistency
sampling.

---

## 2. Complete config reference

Every key in `config/config.yaml` and `load_config()`'s `default_config` dict
(`scripts/utils.py:58-116`), by section. Both are checked to be in sync except where noted.

### `qdrant`
| Key | Default | Description |
|---|---|---|
| `host` | `localhost` | Qdrant server host. |
| `port` | `6333` | Qdrant server port. |
| `collection_prefix` | `rag_` | Prepended to every collection name via `get_collection_name()` — callers always use the bare name. |

### `embedding`
| Key | Default | Description |
|---|---|---|
| `model_name` | `BAAI/bge-small-en-v1.5` | Dense sentence-transformer model. |
| `dimension` | `384` | Dense embedding dimension — drives the quantization guard. |
| `sparse_model_name` | `Qdrant/bm25` | fastembed sparse (BM25-style) model for hybrid search. |

### `indexing`
| Key | Default | Description |
|---|---|---|
| `mode` | `light` | `light` \| `balanced` \| `full` — storage/quality tradeoff preset feeding the quantization guard. |
| `child_chunk_size` | `100` | Child (search-optimized) chunk size, in **tokens** (embedding model's own tokenizer). |
| `child_chunk_overlap` | `20` | Token overlap between adjacent child chunks. |
| `parent_chunk_size` | `400` | Parent (full-context) chunk size, in tokens. |
| `use_hierarchical` | `true` | Enables parent-child chunking (present in config but not separately branched on in current code — `_hierarchical_chunk()` is always the chunker used). |
| `use_contextual_summaries` | `true` | Generates a per-child summary at ingest time (naive placeholder unless `_summary_fn` injected). |
| `quantization` | `binary` | *Requested* mode — `binary` \| `scalar_int8` \| `none`; the *effective* mode is guard-derived, see §1.1 step 4. |
| `hybrid_search` | `true` | Enables sparse vector storage + hybrid dense+sparse retrieval. |
| `force_binary` | `false` | Escape hatch overriding the dimension guard (accuracy risk below `binary_dim_threshold`). |
| `binary_dim_threshold` | `1024` | Minimum dense dimension required for binary quantization (absent `force_binary`). |

### `retrieval`
| Key | Default | Description |
|---|---|---|
| `top_k_children` | `8` | Base top-k for child chunk retrieval. |
| `oversampling` | `3.0` | Oversampling factor used when reranking is off (`fetch_limit = top_k * oversampling`), applied client-side (candidate pool sizing) — not passed to Qdrant as a `SearchParams(quantization=QuantizationSearchParams(...))` argument, since no code in this file constructs one (`SearchParams`/`QuantizationSearchParams` aren't even imported from `qdrant_client.models`). |
| `fetch_parents` | `true` | Enables parent-chunk context expansion at retrieval time. |
| `rerank` | `true` | Enables the cross-encoder rerank stage. |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking. |
| `rerank_candidate_pool` | `40` | Hits fetched per leg before reranking. |
| `rerank_top_n` | `8` | Hits kept after reranking (falls back to `top_k` if unset). |
| `rrf_k` | `60` | Client-side RRF rank-fusion smoothing constant. **Present in `config/config.yaml` but MISSING from `load_config()`'s `default_config["retrieval"]` dict in `utils.py`.** Not a live bug — `EfficientRAG.__init__` (`utils.py:168`) has its own independent runtime default of 60 for this exact key — but it is a real, confirmed pre-existing parity gap between the two "sources of default truth." Confirmed via direct side-by-side key comparison of both dicts: **this is still the only such mismatch** across all four config sections. |
| `context_review_enabled` | `false` | Enables the CRAG-style context review/distillation pass. Off by default, experimental. |
| `context_review_threshold` | `0.3` | Per-sentence relevance score below which a sentence is discarded during context review. |
| `adaptive_retrieval_enabled` | `false` | Enables the Adaptive-RAG-lite retrieval gate. |
| `adaptive_retrieval_model` | `glm-5.2:cloud` | Model used for the gate's classification call. |
| `adaptive_retrieval_provider` | `ollama` | `ollama` \| `neuralwatt` \| `claude` — unknown values raise `ValueError` immediately. |
| `adaptive_light_top_k` | `8` | `top_k` override applied for this call when the gate picks `"light"`. |
| `adaptive_light_oversampling` | `3.0` | `oversampling` override applied for this call when the gate picks `"light"`. |
| `adaptive_retrieval_fallback_strategy` | `light` | Strategy to fail open to on any classifier call/parse/provider error. |

`load_config()` performs a **shallow, top-level merge** (`default_config.update(user_config)`) —
a user config file's nested `retrieval`/`indexing` dicts fully *replace* the corresponding default
sub-dict rather than deep-merging key-by-key. A partial override YAML (e.g. only
`retrieval: {rerank: false}`) will silently drop every other `retrieval.*` default. This is worth
knowing before adding new config-driven behavior for the upcoming changes.

---

## 3. Models and providers currently wired in

| Provider / model | Where used | Role |
|---|---|---|
| **Claude, direct API** (`ANTHROPIC_API_KEY`) — `claude-opus-4-8` | `_build_anthropic_client()` (`evaluate_ragas.py:356-391`), used by `generate_answer()`, `generate_closed_book_answer()` | Primary generation model for the blended `[C]`/`[G]` answer and the closed-book comparison arm. |
| **Claude, direct API — Haiku** — `claude-haiku-4-5` | `review_and_distill_context()`, `_classify_retrieval_need()` (provider=`"claude"`), `verify_and_repair_answer()` | Cheap/fast model for per-sentence context-review scoring, adaptive-gate classification, and `[C]`-claim verification. |
| **Claude on Vertex AI** — default `claude-sonnet-4-5@20250929` (`ANTHROPIC_VERTEX_MODEL`) | `_build_anthropic_client()` Vertex branch; also `verify_and_repair_answer()`'s judge model on the Vertex path (no guaranteed Haiku deployment there, so it reuses the generation model) | Fallback generation model when only `ANTHROPIC_VERTEX_PROJECT_ID` (GCP ADC) is configured, no API key needed. |
| **Claude on Vertex AI — Haiku** — default `claude-haiku-4-5@20251001` (`ANTHROPIC_VERTEX_HAIKU_MODEL`) | Same Vertex-path cheap-model call sites as the direct-API Haiku row above | Vertex equivalent of the direct-API Haiku role. |
| **Claude on Vertex AI, `web_search` tool** — `claude-sonnet-4-5@20250929` | `generate_claude_websearch_answer()` (`evaluate_ragas.py:1928-2007`) | Comparison arm: Claude's own native web retrieval + generation, zero custom RAG. Always uses `AnthropicVertex` directly (never falls back to the direct API). |
| **RAGAS judge LLM** — `ChatAnthropic(model="claude-sonnet-4-5")` direct, or `ChatAnthropicVertex(model="claude-sonnet-4-5@20250929")` fallback | `_default_judge_llm()` (`evaluate_ragas.py:394-440`) | Default evaluator for all RAGAS metrics across every arm except the multi-judge/human tiers. `max_tokens=4096` explicit on both paths. |
| **OpenAI embeddings** (`OPENAI_API_KEY`) — `OpenAIEmbeddings()` | `_default_judge_embeddings()` (`:443-474`) | Preferred embeddings backend for `answer_relevancy`/`answer_similarity`. |
| **Vertex AI embeddings** — `gemini-embedding-001` (`VERTEX_EMBEDDING_MODEL`) | `_default_judge_embeddings()` Vertex fallback | Fallback embeddings backend (this GCP org's allowlist blocks `text-embedding-*`). |
| **Ollama (local), `glm-5.2:cloud`** | Default `retrieval.adaptive_retrieval_model`; one of 3 `DEFAULT_OLLAMA_JUDGE_MODELS`; fallback target for NeuralWatt's `glm-5.2-short` | Adaptive-gate classifier (default provider); multi-judge faithfulness cross-check member. |
| **Ollama (local), `kimi-k2.7-code:cloud`** | One of 3 `DEFAULT_OLLAMA_JUDGE_MODELS`; fallback target for NeuralWatt's `kimi-k2.7-code` | Multi-judge faithfulness cross-check member. |
| **Ollama (local), `minimax-m3:cloud`** | One of 3 `DEFAULT_OLLAMA_JUDGE_MODELS` | Multi-judge faithfulness cross-check member only — **not** used as a NeuralWatt fallback target (NeuralWatt has no `minimax` model; `qwen3.5:397b-cloud` is used instead for that specific mapping). |
| **Ollama (local), `qwen3.5:397b-cloud`** | Fallback target for NeuralWatt's `qwen3.5-397b` | Single-row/whole-model retry target inside the NeuralWatt escalation chain, never a `DEFAULT_OLLAMA_JUDGE_MODELS` member itself. |
| **NeuralWatt, `glm-5.2-short`** | `NEURALWATT_MODELS` (`evaluate_ragas.py:203-220`); one of 3 `run_neuralwatt_multi_judge_consensus()` panel members; `retrieval.adaptive_retrieval_provider: "neuralwatt"` option | Generation+judging candidate; NeuralWatt consensus panel member. |
| **NeuralWatt, `kimi-k2.7-code`** | Same as above | Judging (native JSON mode) — panel member. |
| **NeuralWatt, `qwen3.5-397b`** | Same as above | Backup/third consensus panel member (replaces `minimax-m3`, which NeuralWatt doesn't offer). |

**Note on NeuralWatt's "unconfirmed" status:** `_neuralwatt_llm()`'s own docstring and the
`NEURALWATT_MODELS` catalog comment call NeuralWatt's OpenAI-API-compatibility and base URL
"UNCONFIRMED" / "not yet live-verified," and multiple `docs/CHANGELOG.md` entries repeat that
framing. This is **stale relative to reality**: `/tmp/nfcorpus_eval_v2/neuralwatt_battery_v2_summary.txt`
and 6 corresponding `neuralwatt_multijudge_v2_results2_*.csv`-derived JSON files record a real,
completed 3-model × 6-run × 40-row (720 judge calls) live NeuralWatt battery with a working
zero-NaN escalation chain (see Known Issues, and §5 below). The code comments should be treated as
out of date on this specific point — flagged here rather than silently corrected, since the code
itself wasn't in scope for this doc-only rewrite.

---

## 4. Known Issues / Open Gaps

This section is the load-bearing part of this document for planning the next round of changes —
each item below cites the actual finding, not a restatement of the task list.

### 4.1 Repeated null result: tuned (hybrid+rerank only) is not statistically distinguishable from baseline

Across every judge and every experiment run against this project's live 40-question ×3-repeat
NFCorpus harness (`/tmp/nfcorpus_eval_v2/`), the "tuned" retrieval configuration has never beaten
baseline by a statistically meaningful margin on `faithfulness`:

| Comparison | N | mean delta | 95% CI | p-value | Verdict |
|---|---|---|---|---|---|
| Original non-adaptive tuned − baseline (Claude judge, paired Wilcoxon, per-question mean of 3 repeats) | 40 | +0.0218 | [-0.0229, +0.0713] | 0.4358 | not distinguishable from noise |
| Same comparison, NeuralWatt panel (geometric-median-aggregated, 120 pooled paired deltas across 3 repeats) | 120 pairs | +0.0203 | [-0.0124, +0.0542] | 0.7483 (Wilcoxon) | not distinguishable from noise |
| Same comparison, original Claude judge column re-pooled the same way | 118 pairs | +0.0168 | [-0.0184, +0.0538] | 0.7044 (Wilcoxon) | not distinguishable from noise |
| Adaptive-RAG-lite: adaptive_tuned − adaptive_baseline (gate enabled both arms) | 40 | +0.0004 | [-0.0518, +0.0535] | 0.9100 | not distinguishable from noise |
| Adaptive-RAG-lite diagnostic: adaptive_baseline − non_adaptive_baseline (gate-only effect) | 40 | -0.0419 | [-0.0902, +0.0020] | 0.1387 | not distinguishable from noise |
| Adaptive-RAG-lite, gate-affected-subset only (8 questions where the gate ever picked "none") | 8 | -0.1131 | [-0.2587, +0.0314] | 0.2500 | not distinguishable from noise |

(Sources: `/tmp/nfcorpus_eval_v2/adaptive_validation_summary.txt`,
`/tmp/nfcorpus_eval_v2/neuralwatt_battery_v2_summary.txt`.) The adaptive gate's own explicit
conclusion, verbatim from its validation run: *"Gating retrieval per-query DID NOT help close the
tuned-vs-baseline significance gap... The dilution hypothesis is NOT supported by this experiment
as run."* Two independent judges (a single Claude judge and a 3-model NeuralWatt geometric-median
panel) reached the **same direction of effect** (tuned slightly higher) and the **same
significance verdict** (not significant) — this is not a single-judge artifact.

### 4.2 CRAG context-review and verify-and-repair have never been in a tested "tuned" config — active execution gap

`config_rigor2_tuned.yaml` — the config that actually generated the `results2_tuned_r1/r2/r3.csv`
files underlying essentially every "tuned" comparison in this project, confirmed by timestamp
correlation and by grepping `driver_rigor2_tuned_r1.log`/`eval_rigor2_tuned_r1.log` for any
context-review/CRAG or verify/repair trace (there is none) — does **not** set
`context_review_enabled`, and the eval driver never passed `--verify-repair`. The 2026-07-19
Adaptive-RAG-lite config (`config_adaptive_tuned.yaml`) intentionally preserved this same gap for
comparability, with an explicit inline note:

> `config_rigor2_tuned.yaml`... does NOT set `context_review_enabled` and the eval driver did not
> pass `--verify-repair`. So the actual "tuned" arm this project has been comparing against
> baseline is hybrid_search+rerank ONLY, not hybrid+rerank+CRAG+verify-repair...

**Both features exist, are implemented, are individually live-verified in isolation (see
`docs/CHANGELOG.md`'s 2026-07-16 entries), and are OFF in every "tuned" config that has ever been
statistically compared against baseline.** This is a real, confirmed gap in the evaluation program
itself, not a code defect — tracked as an active task, not yet closed.

### 4.3 RAGAS faithfulness is scored over the whole answer, penalizing `[G]` claims identically to hallucinations

`run_ragas_evaluation()` (and every re-judging path — Ollama cross-check, NeuralWatt consensus,
Claude tiebreaker) constructs each `SingleTurnSample.response` from `generate_answer()`'s raw
output string, `[C]`/`[G]` tags and all, and hands the whole thing to RAGAS's `Faithfulness()`
metric unmodified (`evaluate_ragas.py:2160-2171`, `:995-1030`, `:805-849`). There is no
tag-aware pre-processing anywhere in this pipeline that isolates `[C]`-tagged spans from
`[G]`-tagged spans before scoring. This means an intentionally-tagged `[G]` (general-knowledge)
claim — which `generate_answer()`'s own prompt explicitly *permits and asks the model to tag
honestly* — is scored by RAGAS's sentence-level faithfulness judge exactly like an unattributed
hallucination, because the judge only sees prose, not the tags' semantics. `evaluate_ragas.py`'s
own module docstring and `docs/CHANGELOG.md` both already frame the resulting lower faithfulness
number as "expected, not a regression" and instruct readers to track `answer_correctness`/
`answer_relevancy` alongside it — but that framing is a workaround for the metric/policy mismatch,
not a fix to it. This is a real, unfixed metric/policy mismatch for this project's blended
generation prompt, tracked as an active task.

### 4.4 N=40 is likely underpowered by roughly one to two orders of magnitude for the effect sizes actually being measured

The mean deltas found across every comparison in §4.1 cluster in the **~0.0004 to ~0.02** absolute
range on a 0-1 faithfulness scale — an order of magnitude or more below both the observed 95% CI
half-widths (~0.05-0.15) and the same-config judge-noise floor independently measured by re-running
the identical frozen tuned/baseline configs 3× each with a non-deterministic NeuralWatt judge panel
(std-dev across repeats: baseline 0.0146, tuned 0.0075, geometric-median-aggregated; 0.0038/0.0052
using the original Claude judge column — `neuralwatt_battery_v2_summary.txt` section (b)). Detecting
an effect that small with any reasonable statistical power at N=40 questions would require an effect
size well outside what's actually being observed; the current sample size is not built to resolve
differences this small. Formal power analysis puts the N required to reliably detect an effect this
small, given the observed per-question variance, at roughly **N≈1,200** — 30x the current testset.
**Independently**, published NFCorpus/BEIR benchmark-family literature reports near-zero real
hybrid+rerank gains on this exact benchmark family — meaning the repeated null results in §4.1 may
be **the correct expected answer for this technique on this benchmark**, not evidence that anything
in the pipeline is broken or mistuned.

**DECISION (2026-07-19), superseding the "not yet resolved" framing above:** three options were
evaluated — (a) expand the testset toward the ~1,200 that full power for the *observed* ~0.02
effect would require (or a smaller stopgap in the ~200-400 range), (b) accept the low ceiling as
the correct answer for hybrid+rerank on this benchmark and redirect tuning effort to a technique
with a much larger expected effect size, (c) switch to a different/additional benchmark corpus.
**Adopted: (b), primary — redirect effort to HyDE (§4.5) — combined with a bounded, deprioritized
version of (a) as a secondary, not-yet-scheduled option. (c) was evaluated and rejected.**

- **(a) Full expansion to ~1,200 — rejected as the near-term move, but mechanically feasible.**
  `/tmp/nfcorpus_eval_v2/expand_testset.py` (used for the 20→40 expansion) generalizes cleanly:
  it draws new queries from `datasets.load_dataset("BeIR/nfcorpus-qrels")["test"]` filtered to
  queries with at least one score>0 judged document present in the **full** BeIR NFCorpus corpus
  (not just this project's local doc sample) — the same run that produced the 40-query set logged
  **323 valid candidate query IDs** in that pool, of which only 40 have been consumed so far. So
  the corpus/qrels source itself is not the bottleneck: up to roughly the ~283 remaining valid
  query IDs (≈300-320 total) are reachable by rerunning the identical method (new `random.Random`
  seed, skip already-used question texts, take the next N) without changing the sampling
  methodology at all. What makes full (~1,200) expansion impractical is **cost, not corpus
  availability**: it would require ~3-4x more distinct queries than NFCorpus's entire valid-qrels
  pool actually contains (1,200 > ~323), meaning genuine expansion past ~300-320 questions would
  require abandoning the "one gold doc from qrels per query" methodology entirely (e.g. admitting
  lower-relevance-score judged docs as gold, or synthesizing new queries) — a materially different,
  unvalidated construction method, not a bigger run of the existing one. Even at the mechanically
  reachable ~300-320 ceiling, per-config-repeat wall-clock time already observed at N=40 (NeuralWatt
  3-model panel: ~60-100 minutes per config-repeat, bounded by the slowest judge model) would scale
  roughly linearly to many hours per config-repeat, and this project's methodology requires 3
  repeats × multiple arms (baseline/tuned, and now also an adaptive-gate variant) × multiple judges
  — a multi-day compute/cost commitment to fully resolve statistical power for one technique
  (hybrid+rerank) whose real-world effect size independent literature already suggests is close to
  zero on this exact benchmark family. Buying full power to precisely measure "not much" is a poor
  trade.
- **(b) Accept the ceiling, redirect to HyDE — adopted, primary.** The same power-analysis logic
  that makes ~0.02 need ~1,200 questions runs in reverse for a technique with a much larger expected
  effect: required N scales with `(noise / effect)²`, so an effect an order of magnitude larger than
  hybrid+rerank's ~0.02 needs roughly two orders of magnitude fewer questions for the same power.
  HyDE's cited +40-58% gains on comparable corpora (§4.5) are plausibly that much larger — if so,
  the existing N=40 testset, with its already-measured judge-noise floor
  (`neuralwatt_battery_v2_summary.txt` section (b): std-dev 0.0038-0.0146 depending on judge/config),
  is *already* adequately powered to detect a HyDE effect if one is real, at zero additional testset
  cost. This also matches this project's own existing priority ordering — §4.5 already flags HyDE as
  the highest-expected-value untested technique — so this decision closes the "should we chase power
  or redirect" question in favor of the path this project was already leaning toward, rather than
  reopening it.
- **(c) Switch benchmark corpus — evaluated and rejected.** The literature finding motivating this
  whole decision is explicitly that near-zero hybrid+rerank gains are a **benchmark-family**
  (BEIR-wide) pattern, not an NFCorpus-specific quirk — so there is no found evidence that a
  different BEIR-family corpus would surface a larger hybrid+rerank effect; switching corpora would
  likely reproduce the same null result on a different dataset while paying real costs: losing
  comparability with every existing result in §4.1 (which docs/ARCHITECTURE.md §5 and
  `MANIFEST.md` both establish as the reason to reuse the existing corpus/testset), and needing a
  new gold-document/reference-proxy construction methodology built and validated from scratch. No
  research or existing project artifact identifies a specific alternative corpus with a documented
  larger expected hybrid+rerank effect, so this option has cost without an identified benefit and is
  not pursued.

**Concrete next steps implied by this decision (not executed as part of this decision task):**
1. **Immediate priority:** implement and validate HyDE (§4.5) against the *existing* 40-question
   `testset_v2_40q.json`, using the exact same paired-Wilcoxon + bootstrap-CI + 3-repeat methodology
   as §4.1/§5. Do not build a new or larger testset for this first pass — N=40 is plausible to be
   adequately powered for HyDE's expected effect size, and reusing the existing harness keeps this
   result directly comparable to every existing hybrid+rerank number in §4.1.
2. **Stop treating null hybrid+rerank/faithfulness results as diagnostic of pipeline quality.** Per
   this decision, a null result on that specific comparison at N=40 is the expected, literature-
   consistent answer, not a signal that something in ingestion/retrieval/generation is broken —
   don't re-litigate §4.1 as an open bug.
3. **Deprioritized, not currently scheduled:** if a future need arises to re-test hybrid+rerank (or
   validate a HyDE result) with a tighter CI than N=40 provides, extend `expand_testset.py` with a
   new seed (e.g. `SEED=123`, distinct from 7/42/99 already used) to grow `testset_v2_40q.json`
   toward the ~120-150 range (a ~3-4x increase, mechanically identical to the 20→40 expansion:
   shuffle the valid-qids pool, skip question texts already in the existing set, take the next N,
   write gold docs not already in `docs/`, write the combined set to a NEW file such as
   `testset_v2_120q.json` without overwriting `testset_v2_40q.json`, and re-run the MANIFEST.md
   sanity checks — doc count, query count, zero duplicate question texts). This narrows the 95% CI
   by roughly `sqrt(40/120) ≈ 0.58x`, enough to additionally rule out moderate (~0.05+) effects with
   reasonable confidence, at a cost that scales roughly linearly (not the 30x blowup of full (a)) —
   but this is explicitly a lower-priority, not-yet-scheduled option, subordinate to step 1.

### 4.5 No query rewriting/expansion exists anywhere — HyDE identified as highest-expected-value untested technique

`EfficientRAG.retrieve()` embeds `query` (the literal user-provided string) directly for both the
dense and sparse legs (`utils.py:775`, `:798`) — there is no rewriting, expansion, or HyDE-style
hypothetical-document generation anywhere in the retrieval path. This matters specifically for
this project's corpus: NFCorpus queries are lay-phrased consumer-health questions while the
corpus documents are clinical/technical PubMed-style text, a vocabulary mismatch that dense
similarity alone often under-serves and that a literal keyword/BM25 sparse leg can under-serve
even more. This is tracked as an active task, with HyDE flagged as the highest-expected-value
untested technique for this specific corpus's vocabulary gap.

### 4.6 Parent-chunk expansion is a static on/off toggle, not query-adaptive

`retrieval.fetch_parents` (`utils.py:877-903`) is a single global boolean — every query either
gets full parent-chunk expansion or none; there is no per-query decision logic at all. Research
cited elsewhere in this project's own planning indicates parent-chunk expansion measurably *hurts*
precision on single-hop factoid questions (confirmed, via `MANIFEST.md`, to be this project's
dominant NFCorpus query type) while helping multi-hop questions. A static toggle cannot capture
that split — it is either helping the multi-hop minority and hurting the single-hop majority, or
switched off and helping neither. Tracked as an active task: make expansion decision query-adaptive
rather than global.

### 4.7 No diversity-aware/redundancy-penalizing reranking (MMR or similar)

The cross-encoder rerank stage (`utils.py:847-854`) scores each candidate purely by
`(query, chunk_text)` relevance and sorts descending — there is no term in the ranking that
penalizes near-duplicate or highly-overlapping retrieved chunks. This is a plausible (lower
confidence, lower priority than HyDE) issue specifically because this project's evaluation corpus
was constructed as gold-document-plus-distractor-documents (`MANIFEST.md`'s construction method:
one gold document per query, unioned with randomly sampled distractor documents to reach 120/137
total) — a corpus shape where near-duplicate or overlapping distractor chunks ranking alongside
(or above) more diverse genuinely-relevant content is a real possibility the current reranker has
no mechanism to correct for. Tracked as an active task, explicitly lower-confidence/priority than
HyDE (§4.5).

### 4.8 CLOSED DECISION: model-routing/ensemble layer investigated and rejected

A dynamic model-routing/ensemble layer — selecting among Claude/Ollama/NeuralWatt per-query based
on measured per-task strength — was investigated as a candidate lever for this project's quality
metrics and **explicitly decided against**. The research finding: routing across models is
primarily a **cost/latency optimization**, with quality held roughly flat across the routing
choice — not a lever expected to move this project's actual quality metrics (the RAGAS/human-review
scores under active investigation in §4.1-4.7). This is a **closed decision, not an open gap** —
do not re-propose a routing/ensemble layer as a fix for the null results in §4.1 without new
evidence that contradicts this finding.

### 4.9 Generation-tier routing (reopened, scoped revision of §4.8)

**This is not a reversal of §4.8.** §4.8 correctly closed the *general* model-routing/ensemble
question: routing across Claude/Ollama/NeuralWatt per-query is primarily a cost/latency
optimization, not a lever expected to move this project's quality metrics. That finding stands.
What's reopened here is narrower: §4.8's own cited evidence (RouteLLM: ~4% average accuracy gain
per LLMRouterBench; MoA's gains attributable mostly to aggregation, not provider diversity, per
Self-MoA) was used to conclude "not a quality lever" and, from there, "not worth pursuing." That
second step doesn't follow — a 4% quality gain plus real cost/latency savings is a legitimate,
worthwhile target on its own merits, not something to dismiss without the tradeoff being
explicitly weighed. **Provenance:** this reopening was prompted by direct user pushback on §4.8's
original framing, not by new research findings that contradict §4.8's closed conclusion — worth
stating plainly, matching this project's practice of documenting why a decision was revisited, not
just what the new decision is.

Two candidate sites were evaluated.

**Site 1: Generation routing (`generate_answer()` in `evaluate_ragas.py`) — genuine RouteLLM fit,
design recommended.**

`generate_answer()` currently always uses Claude's strong tier (`_build_anthropic_client()`, §3)
for every question, regardless of difficulty. This matches RouteLLM's exact problem setup: route
easy queries to a cheap model, hard ones to the strong model, and capture the accuracy/cost
tradeoff RouteLLM's benchmark measured.

The key design insight is reuse, not new infrastructure: `EfficientRAG._classify_retrieval_need()`
(§1.2 step 1) **already computes a per-query complexity signal** — `strategy` (`"none"`/`"light"`)
and `needs_broad_context` (bool, §4.6) — for the Adaptive-RAG-lite retrieval gate, on every call
where that gate is enabled. This signal can be reused for generation routing at **zero marginal
classifier-call cost**, rather than adding a second classifier call that pays for its own
prompt/latency independently of the existing one.

Proposed routing policy:
- `needs_broad_context == True` → always strong tier (a comparative/multi-part query is exactly
  the kind of "hard" query RouteLLM's own framing routes to the strong model).
- `strategy == "none"` → cheap tier (the gate already judged this answerable from general
  knowledge with no retrieval; the strong tier's added quality has the least to work with here).
- `strategy == "light"` and not broad-context → cheap tier **only if** a separate
  `generation_routing_light_to_cheap` flag is on (default `false` — a conservative first
  increment, since `"light"` is this project's large majority bucket and mis-routing it carries
  more quality risk than the clear-cut `"none"` case).
- Classification unavailable (the adaptive gate itself is off) → strong tier, i.e. current
  behavior preserved byte-for-byte. Generation routing has no independent classification path of
  its own; it strictly rides on the existing gate's output.

Proposed new config section `generation:`, following this project's off-by-default convention for
every prior lever (Adaptive-RAG-lite, HyDE, MMR, query-adaptive parent expansion): `generation:
{generation_routing_enabled: false, generation_routing_light_to_cheap: false}`, in both
`config/config.yaml` and `load_config()`'s `default_config` dict (parity-checked, per §2's
standing practice).

Proposed integration points:
- `retrieve.py`'s `retrieve_context()` (`retrieve.py:72-100`) computes `needs_broad_context`
  internally (via `rag._last_needs_broad_context`, §1.2/CHANGELOG 2026-07-19) but currently does
  **not** include it in its returned dict — a one-line gap that needs closing before generation
  routing can read the signal at all.
- `generate_answer()` needs new optional `retrieval_result` / `config` parameters (additive,
  backward compatible — every existing caller that doesn't pass them keeps today's
  always-strong-tier behavior) to read the routing signal and select between a new
  `_build_cheap_anthropic_client()` helper (Haiku tier, mirroring the existing Haiku call sites in
  §3) and the existing `_build_anthropic_client()` (strong tier).

**Caveat, stated up front rather than left to be discovered later:** this project's own
adaptive-gate data (2026-07-19 validation, §4.1/§4.2) showed only **~20-25% of NFCorpus questions
ever classify as `"none"`** (the CHANGELOG's 2026-07-20 entry measured 25.0%/27.5%/27.5% across 3
repeats). So on this specific corpus, the addressable population for cheap-tier routing is small
unless the riskier `light_to_cheap` lever is also validated and enabled — the realistic near-term
win from the conservative default-off configuration is modest, not the full 4%-style gain cited
from RouteLLM's own benchmark population.

**Validation design (not yet run):** paired Wilcoxon + bootstrap CI on quality — reusing the
split-faithfulness fix from §4.3 (so `[G]`-tagged claims aren't penalized identically to
hallucinations under a policy this routing doesn't change) plus `answer_relevancy` and
`answer_correctness` — **and** actual measured cost (real token counts × published per-model
pricing, not an estimate) **and** wall-clock time, following the same 3-repeat /
paired-Wilcoxon-plus-bootstrap-CI methodology as §5. A real win requires both a non-regressing
quality CI **and** a nontrivial measured cost/latency reduction — quality alone, or cost alone,
is not sufficient by this project's standard.

**Site 2: Judge-side routing (NeuralWatt panel / `qwen3.5-397b` reliability) — does NOT hold up as
a router; static bypass recommended instead.**

The measured problem is real: `qwen3.5-397b` succeeded on NeuralWatt primary in only 145/240
(60.4%) of live judge calls (`/tmp/nfcorpus_eval_v2/neuralwatt_battery_v2_summary.txt`), driving a
15-hour battery runtime through its escalation chain (NeuralWatt primary → single-row Ollama-cloud
fallback → Claude tiebreaker, §1.4/CHANGELOG 2026-07-17).

But the failure pattern rules out a per-row classifier: run-to-run variation on the **identical**
40 questions produced successes ranging from 16 to 32 out of 40 across 6 runs. That spread proves
the failures are **not query-content-driven** — it's the signature of transient infra/host
flakiness, not a learnable per-row signal. No content/complexity correlate is measured or claimed
anywhere in this project's data for this failure mode. A per-row classifier built on features
already shown to be uncorrelated with the actual outcome would be "routing theater" — plausible-
looking JSON with no real predictive power behind it.

There's a mechanical objection too: `_score_one_model()` calls ragas's `evaluate()` **once per
model across the whole dataset**, not per-row (§1.4, CHANGELOG 2026-07-17's M1 fix) — there is no
natural per-row decision point to intervene at without restructuring the batching itself, which
would be a materially larger change than the reliability problem justifies.

**Recommended instead — not a router — a static, config-driven bypass:**
`neuralwatt.primary_bypass_models: []` (e.g. `["qwen3.5-397b"]`), checked in `_score_one_model()`
before attempting the NeuralWatt primary call, skipping straight to the existing
`_NEURALWATT_TO_OLLAMA_FALLBACK` path (§1.4/CHANGELOG 2026-07-17). This gets the same wall-clock
savings a router would (no more waiting out timeouts before falling back) with far less
design/validation surface, and it's justified directly by the already-measured, already-stable
~40% failure rate — there is no decision left to *learn*, it's a known number.

**Validation design (not yet run):** compare current reactive-escalation wall-clock time vs.
static-bypass wall-clock time for the same battery shape (should recover most of the 15-hour
battery's lost time), plus a paired Wilcoxon check that geometric-median faithfulness does not
regress from using Ollama's `qwen3.5:397b-cloud` as the row's judge 100% of the time instead of
~60%.

**Status: both designs are fully specified; neither has been implemented.** This section documents
the design/decision reached, not a completed feature — mirroring how §4.4's and the 2026-07-19
HyDE/MMR CHANGELOG entries distinguish "decided and specified" from "implemented" from
"statistically validated." No code, config, or testset changes were made as part of writing this
section.

---

## 5. How to verify a change didn't break anything

- **Live-testing harness.** `/tmp/nfcorpus_eval_v2/` is the established live-testing convention
  for this project: a 120-137-document NFCorpus-derived corpus (`MANIFEST.md` documents its exact
  construction — gold + distractor documents, two independently-seeded query batches unioned into
  a 40-question testset, `testset_v2_40q.json`) plus every config variant, driver log, and raw
  RAGAS/multi-judge results CSV/JSON from every experiment run against it to date. Reuse this
  testset and corpus rather than inventing a new one — its construction, seeding, and known
  reference-proxy caveat (ground truths are the gold document's own title+text, not an authored
  answer) are already documented and any new experiment stays comparable to the existing results
  in §4.1 only if it does.
- **Methodology: paired Wilcoxon signed-rank test + bootstrap 95% CI on the per-question mean
  across repeats**, exactly as used throughout §4.1 (see `stats_analysis.py`/`stats_analysis_v2.py`
  in the harness directory for the reference implementation, and
  `neuralwatt_battery_v2_summary.txt` section (c) / `adaptive_validation_summary.txt` for worked
  examples of the expected output shape). Always run **3 repeats** of each config (the harness's
  established convention) and pool paired deltas across repeats before testing — a single run's
  result is not distinguishable from same-config judge-noise re-run wobble (§4.4's noise-floor
  numbers). Report mean delta, 95% CI, Wilcoxon p-value, and rank-biserial effect size together;
  a "significant" result without the CI and effect size alongside it is not a complete report by
  this project's own standard.
- **Never trust a subagent's (or your own) self-report without independent verification.** This is
  an established, hard-earned practice in this project's history, not a suggestion: the M1 shared-
  singleton race (`docs/CHANGELOG.md`, 2026-07-17) was root-caused only after an overnight run's
  "100% NaN failures" self-report was independently checked against the actual installed `ragas`
  source; the zero-NaN guarantee (same date) was only trusted after re-running against the exact
  5-row slice that had shown the original NaN symptom and inspecting every row's score directly,
  not after reading a "should be fixed now" summary. Concretely: after any change to
  `retrieve()`/`ingest()`/the generation or evaluation paths, (a) re-run at least a small live
  slice (5-10 rows) end-to-end against a real Qdrant collection and inspect actual output —
  `points_count` after ingest, real hit payloads after retrieval, real generated text — rather than
  trusting a "ran successfully" log line; (b) if the change touches anything statistically compared
  against baseline, re-run the full 40×3 harness and recompute the Wilcoxon/bootstrap numbers
  yourself rather than accepting a reported verdict; (c) check config-key parity between
  `config/config.yaml` and `load_config()`'s defaults after adding any new config key (§2's `rrf_k`
  gap is the standing example of what an unchecked parity drift looks like).
- **Config-key parity check** is cheap and mechanical: diff `config/config.yaml`'s section keys
  against the corresponding `load_config()` `default_config` sub-dict in `scripts/utils.py`. Do
  this for any new config key introduced by the upcoming HyDE/adaptive-expansion/MMR work — it is
  exactly the kind of drift that's easy to introduce silently (see §2).
