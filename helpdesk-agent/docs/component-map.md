# AI Helpdesk Agent — Component Map

## Executive Summary

The AI Helpdesk Agent is a functional helpdesk automation service, but it is built as a single-model monolith: one Gemini model handles every AI task from ticket classification to FAQ generation. Research across the full ML landscape — from traditional statistical methods and classical ML through specialized transformer models to large language models — reveals that this approach is both over-engineered and under-engineered depending on the task. Several components (classification, sentiment, anomaly detection, deduplication) are better served by purpose-built ML models that are faster, cheaper, more interpretable, and more accurate at scale than an LLM. Conversely, tasks requiring language generation, reasoning across long context, or handling open-ended inputs genuinely need a generative model. The most urgent concrete gap is the FAQ output pipeline: articles are appended to the output Google Doc on every pipeline run with no check for existing content, which is why the doc has grown to ~400 pages. The recommended build sequence prioritizes the highest-leverage gaps first: hybrid retrieval, right-sizing the ML stack per task, semantic dedup, then deeper improvements to the feedback loop and proactive alerting.

---

## Component Map

| Layer | Component | Status | Current Model/Tool | Recommended | Dependencies |
|-------|-----------|--------|--------------------|-------------|--------------|
| **Intake** | Ticket ingestion (JSM servicedeskapi) | **Built** | Custom Python | No change | OAuth 2LO token |
| **Intake** | ADF parser (rich text → plain text) | **Built** | Custom Python | No change | — |
| **Intake** | PII scrubbing (15 regex categories) | **Built** | Regex + pseudonym maps | No change | Runs before any LLM call |
| **Intake** | Metadata signal extraction (SLA tier, request type, component, linked issue count) | **Gap** | Not ingested | Add to ingest schema | Ticket ingestion |
| **Classification** | Category / issue_type / question_type | **Built** | Single LLM (Gemini) | Fine-tuned encoder transformer (DistilBERT/RoBERTa class) or classical ML ensemble — LLM as fallback for novel/low-confidence only | PII scrub complete |
| **Classification** | Confidence thresholding + fallback routing | **Gap** | Confidence captured but unused | Route low-confidence tickets to LLM or human queue | Classification output |
| **Classification** | Metadata-enriched classification | **Gap** | Only text fields used | Include SLA tier, request type, component as structured features | Metadata ingestion |
| **Retrieval** | FAQ DB keyword search | **Built** | SQL text match | Replace with hybrid search | FAQ entries in DB |
| **Retrieval** | Confluence KB CQL search | **Built** | Confluence CQL API | Augment with semantic re-ranking | OAuth 2LO token |
| **Retrieval** | Resolved ticket search | **Built** | SQL keyword match | Replace with embedding match | ticket_classifications table |
| **Retrieval** | Atlassian doc index search | **Built** | SQL keyword match | Augment with embedding match | atlassian_docs table |
| **Retrieval** | Vector embeddings for semantic search | **Gap** | Not present | gemini-embedding-001 (Vertex AI) | Vector store required |
| **Retrieval** | Hybrid search (BM25 + vector) | **Gap** | Not present | BM25 + embedding, re-ranked by Flash | Embeddings + vector store |
| **Retrieval** | Re-ranking step | **Gap** | Not present | Flash scoring of candidates | Retrieval results |
| **Generation** | Draft response (self-service, admin, needs-info) | **Built** | Single Gemini model | Flash for self-service; Pro for admin_action | Lookup results, few-shot examples |
| **Generation** | Few-shot example selection | **Built** | Similarity score + emoji rating, static retrieval | Embedding-based retrieval (semantically matched examples) | response_examples / ai_draft_feedback |
| **Generation** | Structured output enforcement | **Gap** | Prompt asks for JSON, strip_code_fences fallback | Gemini structured output / function calling | — |
| **Generation** | Escalation detection (when not to draft) | **Partial** | Eligibility checks exist (assignee, existing response) | Confidence + category-based routing to human | Classification confidence |
| **Content Generation** | FAQ gap analysis | **Built** | Single Gemini model | Pro (reasoning-intensive) | sources.gather_all_sources() |
| **Content Generation** | FAQ article generation | **Built** | Single Gemini model | Pro (published content, quality matters) | Gap analysis output |
| **Content Generation** | Article deduplication before writing to Google Doc | **Gap** | Not present — root cause of 400-page doc | Semantic similarity check (embedding cosine similarity ≥ 0.85 threshold) before write | gemini-embedding-001 |
| **Content Generation** | Article deduplication before Confluence publish | **Gap** | No title/content check against existing pages | Title match + embedding check before POST | Confluence CQL |
| **Content Generation** | Article versioning (update vs. new) | **Gap** | INSERT OR REPLACE overwrites silently | Track version history; update existing page rather than append | generated_articles table |
| **Publishing** | Confluence v2 API publish | **Built** | generation/publisher.py | No change | OAuth write:page:confluence scope |
| **Publishing** | Google Doc export | **Built** | faq/google_docs.py | Add dedup check before append | Google Workspace API |
| **Proactive Alerting** | Ticket volume anomaly detection | **Partial** | predictions table exists; logic unknown | Unsupervised ML (Isolation Forest class) — no labeled data required, sub-second, streaming-capable | tickets table, time-series data |
| **Proactive Alerting** | Rising topic / category clustering | **Gap** | Not present | NLP clustering on ticket_classifications (NMF or embedding clustering) | ticket_classifications table |
| **Proactive Alerting** | Sentiment spike detection | **Gap** | Not present | Fine-tuned encoder transformer for real-time per-ticket sentiment; LLM only for escalation edge cases | ticket_comments table |
| **Proactive Alerting** | Alert delivery | **Gap** | Not present | Slack message to #ants-engineering or internal Jira ticket creation | Slack bot token |
| **Feedback Loop** | Emoji rating capture | **Built** | ai_draft_feedback table | No change to capture; improve usage in selection | — |
| **Feedback Loop** | Agent response similarity scoring | **Built** | difflib.SequenceMatcher (surface-level) | Embedding cosine similarity — catches semantic equivalence when agent rephrases the same answer | Any embedding model |
| **Feedback Loop** | Organic agent response harvesting | **Built** | harvest_response_examples() | No change | resolved Cloud tickets |
| **Feedback Loop** | Dynamic few-shot retrieval (by semantic similarity) | **Gap** | Static max-5 selection by score | Vector DB lookup of most semantically similar past interactions | gemini-embedding-001, vector store |
| **Feedback Loop** | JSM CSAT signal ingestion | **Gap** | Not present | servicedeskapi /request/{key}/feedback (experimental endpoint, requires no additional scope) | servicedeskapi access |
| **Orchestration** | Pipeline scheduler | **Built** | Background daemon thread, 5-min interval | No change at current scale | threading.Lock() |
| **Orchestration** | Webhook handler | **Built** | FastAPI endpoint | No change | JIRA_WEBHOOK_SECRET |
| **Orchestration** | ML stack routing layer | **Gap** | Single GEMINI_MODEL env var for everything | Per-task approach selection: classical ML / fine-tuned transformer / LLM depending on task (see ML Approach Landscape section) | config.py |
| **Orchestration** | Agent orchestration framework | **Available** | Not used | LangGraph (flow control) + Vertex AI ADK (managed runtime) — evaluate for future agentic version | Would replace background threads |

---

## Per-Layer Detail

### 1. Intake / Classification Layer

**What exists today:**
- Ticket fetch via JSM servicedeskapi (correct, stable)
- ADF parsing to plain text
- PII scrubbing (15 categories, role-separated pseudonyms)
- Gemini classification: category, issue_type, question_type, confidence (0.0–1.0)

**What's available but unused:**
- Metadata signals in every ticket that aren't fed into classification: SLA tier, request type, component, linked issue count, affect_version. Research shows these signals meaningfully improve routing accuracy — a ticket with 3 linked issues and a critical SLA tier should be classified differently than a low-priority how-to question.
- Confidence score is captured but never acted on. Low-confidence classifications (< 0.6) could be flagged for human review rather than proceeding to auto-draft.

**What's genuinely missing:**
- Metadata-enriched classification prompts
- Confidence-based routing logic (threshold → human queue)
- Classification of ticket intent vs. issue type separately (some systems use a two-step: coarse intent first, then fine-grained issue type)

**Recommended approach:**
Expand the classification prompt to include structured metadata fields. Add a confidence gate in the auto-responder: if confidence < 0.6 and issue_type is novel, skip auto-draft and flag for human review.

---

### 2. Knowledge Retrieval

**What exists today:**
- FAQ DB: SQL keyword match against `article_topic` and `title`
- Confluence KB: CQL keyword search (`text ~ "term"`)
- Resolved tickets: SQL keyword match on summary + resolution_summary
- Atlassian docs: SQL keyword match on title + URL

**What's available but unused:**
- `gemini-embedding-001` on Vertex AI — Google's current production embedding model (replaces the deprecated `text-embedding-004` which was shut down January 2026). Supports 3072 dimensions and task-type specifications: `RETRIEVAL_QUERY` for queries, `RETRIEVAL_DOCUMENT` for indexed content. Already available on the `your-gcp-project` GCP project.
- Industry consensus: hybrid search (BM25 keyword + vector semantic, combined via Reciprocal Rank Fusion) outperforms either alone. Keyword search handles exact acronyms and product names (important in this domain: "<PROJECT_KEY>", "servicedeskapi", "Jira Cloud"); vector search handles synonyms and semantic intent ("can't log in" → "SSO redirect loop").

**What's genuinely missing:**
- Any vector embedding of FAQ entries, KB articles, or resolved ticket summaries
- A vector store (options: pgvector on Postgres, Vertex AI Vector Search managed service, or SQLite with sqlite-vss extension — the last requiring no new infrastructure)
- A hybrid search implementation combining BM25 + vector
- A re-ranking step: after retrieval, a Flash call to score which candidates are actually relevant to this specific ticket

**Recommended approach:**
Add embedding generation as a post-write step in `generator.py` and `ingest/tickets.py`. Store embeddings in the existing SQLite DB via `sqlite-vss` (keeps single-pod architecture intact). Implement hybrid retrieval in `faq/lookup.py`. Add Flash re-ranking as a final filter. This is the highest-leverage improvement available — it directly fixes the most common failure mode (draft says "No matches found" when relevant content exists under different keywords).

---

### 3. Response Generation

**What exists today:**
- Single Gemini model for all drafts
- Three response types: self_service, admin_action, needs_info
- Few-shot examples: up to 5, prioritized by emoji rating + similarity score, statically selected
- Context assembly: FAQ matches + KB matches + resolved tickets + Atlassian docs (up to 4,000 chars)
- JSON response parsed with strip_code_fences fallback

**What's available but unused:**
- Gemini structured output / function calling: guarantees JSON schema compliance without needing a fallback parser. Eliminates the class of errors where Gemini wraps JSON in markdown code fences or returns malformed JSON.
- Flash for simple drafts: research confirms Flash wins decisively on structured, high-volume classification and conversational response tasks. For self_service tickets where the answer is "follow these 3 steps", Flash produces equivalent quality at significantly lower cost and latency.
- Pro for complex drafts: for admin_action tickets requiring multi-step diagnostic logic and branching, Pro's reasoning depth produces measurably better output.

**What's genuinely missing:**
- Model routing by response type (Flash for self_service/needs_info; Pro for admin_action and gap analysis)
- Structured output enforcement (function calling)
- Semantic few-shot retrieval (current selection is static; the best few-shot examples for "SSO redirect loop" may not be the 5 most highly rated overall — they're the most semantically similar to this specific ticket)

**Recommended approach:**
Add a response_type → model mapping in `config.py`. Switch to Gemini structured output for the draft response schema. Implement embedding-based few-shot retrieval as part of the hybrid retrieval improvement.

---

### 4. Proactive Alerting

**What exists today:**
- `predictions` table in `db.py`: columns for issue_type, risk_level, predicted_volume — exists but the code that writes to it is not in the explored modules (may be orphaned or minimal)
- `doc_improvements` table: documentation gap suggestions — similarly present but not prominently used

**What's available but unused:**
- The existing `ticket_classifications` table has everything needed for basic volume trend analysis: category, issue_type, classified_at. A weekly query grouping by issue_type over time is a starting point for trend detection.
- Research shows: static threshold alerts are wrong. A spike of 500 tickets on a Monday after a release is normal; the same spike at 3am on a Tuesday is a production incident. ML-based contextual baselines that account for day-of-week, recent release activity, and category distribution are the correct approach.
- Multi-factor alerting: volume spike + sentiment spike + geographic/component clustering together are more reliable signals than volume alone.
- The existing Slack bot token enables alert delivery without new infrastructure.

**What's genuinely missing:**
- Time-series volume tracking (ticket counts per issue_type per time window, stored for trend analysis)
- Any sentiment signal on incoming tickets (Flash call at classification time adds minimal cost)
- Alerting logic (threshold + baseline comparison)
- Alert delivery (Slack message to team channel, or auto-create a ANTSE tracking ticket)

**Recommended approach:**
Add sentiment scoring as a field in `ticket_classifications` (Flash call, one additional field). Add a daily aggregation job that writes volume trends per issue_type to a `ticket_trends` table. Add a simple baseline comparison (this week vs. 4-week rolling average). Alert via existing Slack integration when volume or sentiment deviates >2σ from baseline.

---

### 5. Feedback Loop

**What exists today:**
- Emoji ratings: ✅ 👤 🔧 ❌ 🔄 ❓ → stored in `ai_draft_feedback.agent_feedback`
- Similarity scoring: difflib.SequenceMatcher comparing draft vs. agent's actual response (0.0–1.0)
- Organic harvesting: `harvest_response_examples()` extracts agent responses from resolved tickets
- Few-shot prioritization: both_good > customer_good > steps_good; high-similarity pairs; organic examples

**What's available but unused:**
- JSM CSAT data: JSM sends a CSAT email to customers on ticket resolution (1–5 star rating + optional comment). This data is accessible via the servicedeskapi experimental endpoint `GET /rest/servicedeskapi/request/{key}/feedback`. The rating and comment are available. Critical limitation: CSAT comments are not standard Jira fields — they cannot be queried via JQL, and the endpoint is marked experimental by Atlassian. But the OAuth 2LO token already has the JSM read scopes needed; no new credentials required.
- Dynamic few-shot retrieval: the current system picks 5 examples based on score. The field has moved to vector-based retrieval — when drafting for a new ticket, search the feedback DB by embedding similarity to find past interactions where the same *type* of problem was handled well. This significantly improves draft quality for edge cases.
- Embedding-based similarity scoring: replacing difflib with embedding cosine similarity produces more robust similarity scores, particularly for cases where the agent rephrased the same content entirely.

**What's genuinely missing:**
- CSAT ingestion: fetch CSAT score + comment for each resolved ticket via the servicedeskapi feedback endpoint; store in a `ticket_csat` table; correlate with draft quality scores
- Vector-based few-shot retrieval (depends on the vector store from the retrieval layer)
- Using CSAT as a signal in few-shot prioritization (currently only agent behavior is tracked, not customer outcome)

**Recommended approach:**
Add a `ticket_csat` table and a nightly CSAT fetch job for recently resolved tickets. Correlate low CSAT scores with draft quality to surface which draft types lead to poor customer outcomes. Feed CSAT-weighted scoring into few-shot prioritization alongside emoji ratings.

---

### 6. Output Content Deduplication

**What exists today:**
- `UNIQUE(article_topic)` constraint: prevents duplicate topics in the DB
- `INSERT OR REPLACE`: overwrites existing articles on re-generation (no version history)
- Google Doc export: appends all draft articles to the output doc on every pipeline run, with no check for existing content in the doc → root cause of the 400-page accumulation

**The 400-page problem:**
Every pipeline run calls `write_faq_entries()` in `faq/google_docs.py`, which appends all articles marked as drafts. Since articles are regenerated (with updated content) and their topic-based dedup only prevents DB duplication, the same article gets appended again on each run. Over time this accumulates.

**What's available:**
- Exact title matching: before appending to the Google Doc, check if a section/heading with that article title already exists in the doc. The Google Docs API supports reading doc content. This is the lowest-effort fix.
- Embedding similarity threshold: for near-duplicate detection where titles differ slightly, compute `gemini-embedding-001` embedding of the new article and compare against embeddings of existing articles. Cosine similarity ≥ 0.85 = duplicate. LLMs do not generate exact duplicates (they swap synonyms, restructure sentences) so exact-hash dedup is insufficient.
- Intervention points in order of preference:
  1. **Before Google Doc write** (in `faq/google_docs.py`): check title in existing doc content — lowest cost, catches most cases
  2. **Before generation** (in `generator.py`): check if an embedding-similar article already exists — prevents wasted Gemini calls
  3. **Before Confluence publish** (in `generation/publisher.py`): CQL title match against the target space before POST

**What's genuinely missing:**
- Any dedup check before Google Doc write (the fix needed now)
- Article update logic: instead of append, check if the article already exists → update that section rather than adding a new one
- Embedding storage for existing articles to enable similarity-based dedup

**Recommended approach:**
Immediate fix (no new infrastructure): in `faq/google_docs.py`, read existing doc headings before writing; skip any article whose title already appears. Longer-term: add embedding vectors to `generated_articles` table; before generation, compute similarity to existing entries and skip if threshold exceeded.

---

### 7. ML Approach Landscape (Per Task)

**What exists today:**
- `GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")` — single LLM for every task
- All AI calls go through `utils/gemini.py` → `client.models.generate_content(model=GEMINI_MODEL, ...)`

**The core finding:** An LLM is the wrong tool for several tasks in this pipeline. At industry scale, purpose-built ML approaches — classical statistical methods, classical ML, and specialized smaller models — are faster, cheaper, more accurate, and more interpretable than using a large generative model for everything. Below is the full ML landscape per task, from lightest to heaviest approach.

---

#### Classification (category / issue_type / question_type)

| Approach | Accuracy (F1) | Latency | Cost | When to use |
|----------|--------------|---------|------|-------------|
| Rule-based (keyword patterns, regex routing) | ~70-75% | Sub-millisecond | Zero inference cost | Baseline, deterministic routing for well-known patterns |
| Classical ML (TF-IDF + ensemble) | ~85-88% | ~1-5ms | CPU only, negligible | Well-labeled data, stable category set, high volume |
| Fine-tuned encoder transformer (DistilBERT / RoBERTa class) | ~89-93% | ~10-50ms | GPU or CPU, low | Production workhorse — handles context, edge cases, sarcasm |
| LLM (Gemini / GPT-4 class) | ~90-94% | ~500-2000ms | High | Zero-shot on novel categories, when labeled data is scarce |

**Recommendation:** Classical ML (TF-IDF + ensemble) is adequate and vastly cheaper if the category taxonomy is stable and training data from resolved tickets is available. Fine-tuned encoder transformer is the industry standard at scale — 4-5% accuracy gain over classical ML on edge cases with dramatically lower cost and latency than an LLM. LLM classification is appropriate for bootstrapping (before training data exists) or for novel/ambiguous tickets that fall outside the trained taxonomy. The current system has hundreds of classified tickets already available as training data.

**Key weakness of current approach:** LLMs for classification are "context blind" to structured signals — SLA tier, request type, component, linked issue count are available in the ticket data but not fed into the classification prompt.

---

#### Sentiment Analysis

| Approach | Strength | Weakness | When to use |
|----------|----------|----------|-------------|
| Lexicon-based (VADER, TextBlob) | Zero training, zero cost, real-time | Poor on sarcasm, mixed sentiment, domain-specific language | Quick baseline, when speed matters more than accuracy |
| Fine-tuned encoder transformer (DistilBERT / RoBERTa class) | High-volume, low-latency, domain-adaptable, handles mixed sentiment | Requires labeled data, struggles with heavy sarcasm | Industry standard for real-time sentiment routing at scale |
| Aspect-Based Sentiment Analysis (ABSA) | Identifies *what* the customer is negative about, not just how negative | More complex to train | When you need to know which component/feature is frustrating users |
| LLM | Deep contextual understanding, handles sarcasm and implicit frustration | Expensive, slow for per-ticket real-time scoring | Escalation decisions for complex/ambiguous cases; not for bulk scoring |

**Recommendation:** A fine-tuned encoder transformer is the production approach. The industry pattern is: lightweight fine-tuned model for real-time routing at the perimeter; LLM only for escalation decisions where a nuanced judgment matters. Critically — if a ticket scores highly negative AND the agent already responded AND CSAT comes back low, that's a signal worth surfacing regardless of what the draft quality score says.

---

#### Anomaly Detection (ticket volume / rising issues)

| Approach | Strength | Weakness | When to use |
|----------|----------|----------|-------------|
| Static thresholds (alert if volume > N) | Simple, zero infrastructure | Fails at scale — contextual (500 tickets on a Monday is normal) | Never in production |
| Statistical time-series (ARIMA, exponential smoothing) | Good for stable seasonal patterns | Brittle for irregular spikes, slow to adapt | When you have clean historical data and predictable patterns |
| Unsupervised ML (Isolation Forest class) | Sub-second latency, minimal memory, streaming-capable, no labeled anomaly data needed, scales to thousands of queues simultaneously | Less interpretable than statistical methods | Industry AIOps standard — used in Azure ML, Elastic at enterprise scale |
| Deep learning (LSTM, Transformer-based time-series) | Captures complex temporal dependencies | Requires significant training data, expensive to operate | Useful at very high volume where simpler approaches fail |
| NLP clustering on ticket content | Detects rising *topics* before volume spikes, catches semantic shifts | Requires classification data | Complement to volume detection — catches "what is going wrong" not just "how many" |

**Recommendation:** Unsupervised ML for volume anomaly detection (Isolation Forest class) — it requires no labeled anomaly data (rare by definition), runs on CPU at sub-second latency, and is the approach used in production AIOps systems at enterprise scale. Combine with NLP topic clustering on `ticket_classifications` to detect rising issue types semantically. The `predictions` table in the DB already exists — the question is whether anything populates it.

---

#### Near-Duplicate Detection (FAQ article dedup)

| Approach | Strength | Weakness | When to use |
|----------|----------|----------|-------------|
| Exact hash (MD5 / SHA-256) | Zero false positives, zero compute | LLMs never generate identical text — this catches nothing useful here | Only for detecting literal re-runs of the same content |
| MinHash + Locality Sensitive Hashing (LSH) | O(n) vs O(n²), scales to billions of records, fast CPU, no GPU needed | Catches structural near-duplicates but misses semantic equivalence | First-pass filter before semantic check — extremely fast |
| Embedding cosine similarity | Catches semantic duplicates (same meaning, completely different words) | Slower than LSH, requires embedding model | Second-pass on candidates flagged by LSH, or small corpus |
| Hybrid (MinHash LSH → embedding similarity) | Best of both: speed at scale + semantic accuracy | Two-stage complexity | Industry standard for large-scale knowledge base dedup |

**Recommendation:** For the current scale (hundreds of articles), embedding cosine similarity alone is sufficient — compute once, store in DB, compare on each new article. If the corpus grows to tens of thousands of articles, add MinHash LSH as a first-pass filter. The key insight: LLMs rephrase synonyms and restructure sentences, so only a semantic approach catches the duplicates actually occurring in the 400-page doc.

---

#### Retrieval / Knowledge Lookup

| Approach | Strength | Weakness | When to use |
|----------|----------|----------|-------------|
| Keyword / BM25 | Exact term matching, handles product-specific acronyms (<PROJECT_KEY>, servicedeskapi), fast, no model needed | Misses synonyms and semantic intent | Always include as one leg of hybrid |
| Dense retrieval (embedding cosine similarity) | Catches "can't log in" → "SSO redirect loop", semantic intent matching | Fails on exact product names/acronyms the model hasn't seen | Always include as other leg of hybrid |
| Hybrid (BM25 + dense, combined via Reciprocal Rank Fusion) | Industry consensus for production RAG — best of both | Slightly more complex pipeline | Production standard — both failure modes are real in this domain |
| Re-ranking (lightweight cross-encoder) | Scores relevance of candidate pairs more accurately than retrieval alone | Adds latency; another model in the stack | After hybrid retrieval, before passing to generation |

**Recommendation:** Hybrid retrieval is the right target. The current system is keyword-only, which is why "No matches found" appears for tickets where the terminology differs from how the KB was written.

---

#### Response Generation

| Approach | Strength | Weakness | When to use |
|----------|----------|----------|-------------|
| Template-based (rule + slot filling) | Deterministic, zero cost, auditable | Brittle, can't handle variation | High-confidence, well-understood ticket types where the answer is always the same |
| Fine-tuned smaller generative model | Lower cost, faster, deployable on-prem | Requires training data, fixed capability ceiling | When volume is very high and draft quality from a fine-tuned model is sufficient |
| LLM (generative, large) | Handles open-ended variation, reasoning, multi-step logic, context synthesis | Expensive, slower, non-deterministic | The correct choice for drafting — generation is genuinely an open-ended language task |

**Recommendation:** LLM is the right tool for generation — this is what generative models are actually designed for. The current approach is correct in principle. The improvements are: (a) match model capability to response complexity (simple self-service vs. multi-step admin action), and (b) improve the retrieval quality feeding into generation, which has more leverage than swapping generation models.

---

#### Feedback Scoring (draft vs. agent response similarity)

| Approach | Strength | Weakness | When to use |
|----------|----------|----------|-------------|
| Sequence matching (difflib — current) | Zero cost, zero infrastructure | Surface-level — misses semantic equivalence when agent rephrases the same answer | Baseline only |
| Embedding cosine similarity | Catches "same meaning, different words" — much more robust | Requires embedding call per scored pair | Production standard for semantic similarity scoring |
| LLM-as-judge | Nuanced quality assessment, can score on multiple dimensions | Expensive, slow, adds another LLM call per draft | Use sparingly for calibration, not real-time scoring |

**Recommendation:** Replace difflib with embedding cosine similarity. This directly improves the feedback loop signal quality, which cascades into better few-shot example selection.

---

**Orchestration options:**
- **Current approach (background threads)**: appropriate for current scale. No change needed until multi-replica or higher ticket volume demands it.
- **LangGraph**: graph-based flow control, retry logic, debugging via LangGraph Studio. Best if the workflow grows into a true multi-agent system with conditional branching between specialized ML components.
- **Vertex AI Agent Builder / ADK**: managed runtime, built-in RAG, guardrails. Best for standardized production deployment at Google Cloud scale.
- **Recommendation**: do not migrate orchestration now. Right-size the ML stack per component first. Evaluate LangGraph / ADK if/when workflow complexity outgrows threading.

---

## Recommended Build Sequence

| Priority | Work | Why first | Owner |
|----------|------|-----------|-------|
| 1 | **Google Doc dedup fix** (title-match before append in `faq/google_docs.py`) | Immediate fix to the 400-page problem. No new infrastructure. | ceverson |
| 2 | **Model routing** (Flash for classification/simple drafts, Pro for generation/gap analysis) | Single env var change in config + routing dict in `utils/gemini.py`. Highest ROI per line of code. | ceverson |
| 3 | **Structured output enforcement** (Gemini function calling for JSON schemas) | Eliminates the class of JSON parse failures. Low effort, high reliability gain. | ceverson |
| 4 | **Embedding layer + hybrid retrieval** (gemini-embedding-001 + sqlite-vss + hybrid search in `faq/lookup.py`) | Highest-leverage accuracy improvement. Fixes "No matches found" failures. Unblocks semantic dedup and dynamic few-shot. | ceverson |
| 5 | **Semantic dedup before generation** (embedding similarity check in `generator.py`) | Prevents wasted Pro model calls for near-duplicate articles. Depends on embedding layer. | ceverson |
| 6 | **CSAT ingestion** (servicedeskapi feedback endpoint → `ticket_csat` table) | No new credentials. Adds customer outcome signal to the feedback loop. | ceverson |
| 7 | **Dynamic few-shot retrieval** (vector DB lookup for semantically matched examples) | Improves draft quality for edge cases. Depends on embedding layer. | ceverson |
| 8 | **Confidence-based routing** (low-confidence classification → skip auto-draft) | Reduces bad drafts on novel/ambiguous tickets. | ceverson |
| 9 | **Proactive alerting** (volume trend tracking → Slack alert on anomaly) | New capability. Depends on stable classification and ticket history. | ceverson |
| 10 | **Sentiment scoring at classification time** (Flash call, one additional field) | Enriches alerting and feedback signals. | ceverson |

---

## Open Questions

1. **`predictions` table**: What code populates it? The table exists in `db.py` but the write path wasn't found in the explored modules. If it's orphaned, this affects the proactive alerting assessment.

2. **`doc_improvements` table**: Same question. What generates doc improvement suggestions and are they surfaced anywhere?

3. **Vector store infrastructure**: `sqlite-vss` keeps the single-pod architecture intact and requires no new infrastructure. However, if ticket volume scales significantly (hundreds of thousands of embeddings), Vertex AI Vector Search is the managed alternative on the existing GCP project. Recommend starting with sqlite-vss and migrating if needed.

4. **Gemini model version**: The codebase has a discrepancy — docs say `gemini-2.0-flash-001`, code says `gemini-3.1-pro-preview`. Verify which model is actually running in production before optimizing around it.

5. **Admin approval for global Jira webhook (ANTSE-193)**: Without comment-event triggers, `/ai-lookup`, `/ai-review`, and `🤖` are only processed every 5 minutes. This affects the real-time usefulness of the feedback loop triggers. Status of ANTSE-193?

6. **Orchestration migration threshold**: At what ticket volume or workflow complexity would LangGraph / Vertex AI ADK become the right choice over the current threading model? Worth defining a trigger condition before it becomes urgent.
