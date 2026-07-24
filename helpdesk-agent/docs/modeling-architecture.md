# AI Helpdesk Agent — Full Modeling Architecture

---

## Experimentation Methodology

Before any model is selected for production, the following sequence applies to every learned component. Skipping steps leads to overfit, underperforming, or unnecessarily complex systems.

### 1. EDA (Exploratory Data Analysis)
Understand the data before choosing an approach. For each model, the specific EDA questions are listed in that model's section. General questions that apply everywhere:
- How much data actually exists? (row counts, date ranges)
- Is the distribution balanced or heavily skewed?
- What does the data look like qualitatively? (sample 50 rows manually)
- Are there data quality issues — mislabels, nulls, inconsistent formats?

### 2. Establish a Baseline
Always start with the simplest model that could possibly work. A keyword rule or a logistic regression on TF-IDF is not a placeholder — it is a real baseline that production models must beat. If the simple model is good enough, ship it.

### 3. Candidate Experiments
Run candidate approaches on a fixed held-out test set. The candidates for each model are listed in that model's section, ordered from simplest to most complex.

### 4. Evaluation Criteria
Compare candidates on:
- **Primary metric** — task-specific (F1 for classification, silhouette for clustering, false positive rate for anomaly detection). Accuracy alone is misleading on imbalanced data.
- **Inference latency** — does it meet the latency budget for its pipeline position?
- **Training cost** — GPU hours and data labeling cost.
- **Beat the baseline** — new complexity is only justified if it measurably outperforms the baseline on the held-out set.

### 5. Select and Document
Pick the winner. Record: which model won, on what metric, by how much, and what data it was trained on. The architecture doc is updated with the actual selection — the candidates table below is the pre-experiment view.

### What Does Not Need Experimentation
Deterministic and algorithmic components — BM25 indexing, MinHash fingerprinting, cosine similarity, RRF merging, SQL aggregations — are well-understood and do not require A/B testing. The implementation choices there (index library, threshold values) are tuned in production, not selected through experiments.

---

## Model 1: Ticket Classifier

**What it does:** Assigns category, issue_type, question_type, and a confidence score to every incoming ticket.

**Task type:** Multi-label supervised text classification.

**Inputs:**
- Ticket summary + description (text)
- Request type (categorical)
- Component (categorical)
- SLA tier (ordinal)
- Linked issue count (numeric)
- Affect version (categorical)

**Outputs:** `{category, issue_type, question_type, confidence}`

**EDA required before model selection:**
- Class distribution in `ticket_classifications` — are categories balanced or heavily skewed? Skewed classes require stratified sampling and weighted loss functions.
- Examples per class — categories with fewer than 50 labeled examples should not be included in the initial training set.
- Text length distribution — short tickets (1–2 sentences) and long tickets (multi-paragraph) favor different model families.
- Label quality audit — sample 50 Gemini-labeled tickets and manually verify. If label accuracy is below ~80%, clean the training set before any experiment.
- Metadata signal value — does adding SLA tier, component, and request type actually improve classification? Test with and without.

**Model selection results — ANTSE-302 (2026-05-28, n=196 holdout)**

All 8 candidates (3 individual + 5 ensembles) evaluated on the locked 196-ticket holdout set.
Primary metric: **Macro-F1** (class-imbalance-aware; see Rationale section below).

| Rank | Candidate | Accuracy | Macro-F1 | Weighted-F1 |
|------|-----------|----------|----------|-------------|
| 1 | **LR(0.3)+GB(0.2)+DistilBERT(0.5)** | **0.7245** | **0.5415** | **0.7151** |
| 2 | LR+DistilBERT (equal) | 0.6429 | 0.5391 | 0.6514 |
| 3 | DistilBERT alone | 0.6020 | 0.5346 | 0.6121 |
| 4 | LR(0.25)+DistilBERT(0.75) | 0.6276 | 0.5306 | 0.6396 |
| 5 | LR+GB+DistilBERT (equal) | 0.6786 | 0.4654 | 0.6632 |
| 6 | GB+DistilBERT (equal) | 0.6684 | 0.4488 | 0.6463 |
| 7 | LR alone | 0.5969 | 0.4433 | 0.6000 |
| 8 | GB alone | 0.6429 | 0.3946 | 0.6158 |

**Winner after Optuna OOF optimization: `LR(0.1019)+GB(0.1240)+DistilBERT(0.7741)` — weighted soft-vote ensemble.**
Implemented in `analysis/model_selection.py` (`WINNER`, `WINNER_WEIGHTS`, `WINNER_MODEL_PATH`). Weights were data-driven using Optuna TPE on 5-fold OOF predictions (see section below).

---

### Ensemble Weight Optimization — Optuna OOF (2026-05-28)

The initial ANTSE-302 candidate evaluation used hand-guessed weights (LR=0.30, GB=0.20, DistilBERT=0.50). These were replaced by Optuna TPE Bayesian optimization on out-of-fold (OOF) predictions from the training set only — the 196-ticket holdout was not touched during optimization.

**Method:** 5-fold stratified CV (`StratifiedKFold(n_splits=5, seed=42)`) generates OOF probability arrays for each model. GB probabilities are sigmoid-calibrated per fold using `FrozenEstimator` (avoids inner CV failure on the 2-sample "Other" class). Optuna TPE searches 300 trials over unnormalized weight space [0, 1]^3, normalized to sum=1 for blending.

**Script:** `analysis/ensemble_optimize.py`

| Phase | Macro-F1 | Weights |
|-------|----------|---------|
| Guessed (prior) | 0.5415 (holdout) | LR=0.300, GB=0.200, DistilBERT=0.500 |
| Optuna OOF (training set) | 0.5973 (OOF) | LR=0.1019, GB=0.1240, DistilBERT=0.7741 |
| **Optimized (holdout eval)** | **0.5742 (+0.0327)** | **LR=0.1019, GB=0.1240, DistilBERT=0.7741** |

**Per-class F1: guessed weights vs. optimized weights (196-ticket holdout)**

| Category | Guessed F1 | Optimized F1 | Delta | n |
|----------|------------|--------------|-------|---|
| Access | 0.904 | 0.902 | -0.002 | 49 |
| Configuration | 0.782 | 0.730 | -0.052 | 64 |
| Data | 0.400 | 0.467 | +0.067 | 12 |
| Integration | 0.650 | 0.651 | +0.001 | 21 |
| Notifications | 1.000 | 1.000 | +0.000 | 2 |
| Other | 0.000 | 0.000 | +0.000 | 1 |
| Performance | 0.000 | 0.286 | +0.286 | 3 |
| Permissions | 0.600 | 0.667 | +0.067 | 18 |
| UI/UX | 0.579 | 0.595 | +0.016 | 20 |
| Workflow | 0.500 | 0.444 | -0.056 | 6 |

**Interpretation:** The optimizer heavily upweights DistilBERT (0.77 vs. 0.50 guessed) because its OOF semantic embeddings provide cleaner probability estimates than the TF-IDF models for minority classes. GB and LR both shrink — LR from 0.30 to 0.10, GB from 0.20 to 0.12. This is consistent with the prior finding that "GB+DistilBERT (equal)" underperformed because GB's uncalibrated probabilities hurt the blend. With calibration (sigmoid Platt scaling) and reduced weight, GB contributes without distorting DistilBERT's signal.

The Configuration class F1 drops slightly (-0.052) because the guessed weights happened to be well-tuned for the dominant class. Macro-F1 improves because minority class gains (Performance +0.286, Permissions +0.067, Data +0.067) more than offset the majority-class cost.

---

**Winner**

`LR(0.1019)+GB(0.1240)+DistilBERT(0.7741)` wins on Macro-F1 (0.5742), beating:
- Guessed weights (LR=0.30, GB=0.20, DistilBERT=0.50) by +0.0327 Macro-F1
- Individual DistilBERT (0.5346) by +0.0396 Macro-F1
- Individual LR (0.4433) by +0.1309 Macro-F1
- Gemini zero-shot baseline (~0.350) by +0.2242 Macro-F1

Per-class F1 for winner (optimized) vs. individual DistilBERT vs. individual LR:

| Category | LR-F1 | DistilBERT-F1 | Winner-F1 | n |
|----------|-------|---------------|-----------|---|
| Access | 0.8400 | 0.8081 | **0.9023** | 49 |
| Configuration | 0.6239 | 0.5714 | **0.7302** | 64 |
| Data | 0.5000 | 0.4516 | 0.4667 | 12 |
| Integration | 0.5000 | 0.5778 | **0.6514** | 21 |
| Notifications | 0.6667 | 1.0000 | **1.0000** | 2 |
| Other | 0.0000 | 0.0000 | 0.0000 | 1 |
| Performance | 0.0000 | 0.5000 | **0.2857** | 3 |
| Permissions | 0.3871 | 0.5556 | **0.6667** | 18 |
| UI/UX | 0.4444 | 0.5333 | **0.5946** | 20 |
| Workflow | 0.4706 | 0.3478 | 0.4444 | 6 |

The optimized ensemble upweights DistilBERT (0.77 vs. 0.50 guessed) because its OOF calibrated probabilities provide better minority-class signal. The optimizer learned that GB contributes limited marginal value (0.12 weight) and LR contributes even less (0.10) — both are shrunk from the guessed weights. The main gains are in minority classes: Performance recovers from 0.000 to 0.286, Permissions improves from 0.600 to 0.667, and Data improves from 0.400 to 0.467. Configuration drops slightly (-0.052) because the guessed weights happened to favor the dominant class.

---

**Rationale for Macro-F1 as primary metric**

The 10-class distribution is heavily imbalanced: Configuration (64 tickets) and Access (49) dominate; Performance (3), Other (1), and Workflow (6) are minorities. **Weighted-F1 and accuracy are dominated by majority classes** — a model that classifies every ticket as Configuration would still show high weighted-F1. Macro-F1 weights every class equally regardless of support, making minority class failures visible. A model that fails on Performance or Other scores poorly on Macro-F1 even if overall accuracy looks acceptable.

---

**Known failure modes**

- **Performance (n=3):** F1=0.000 in winner. Three-ticket holdout is insufficient signal — the model never predicts this class. Not enough training examples to learn a reliable decision boundary. Needs more labeled data or special handling (LLM fallback for this class).
- **Other (n=1):** F1=0.000. Single-ticket holdout class. Effectively untestable at this sample size. No model can reliably detect a residual catch-all category.
- **Data (n=12):** F1=0.400 in winner, below 0.5 threshold. Short tickets with technical jargon overlap heavily with Configuration and Integration.
- **Workflow (n=6):** F1=0.500 — improved from LR (0.471) but still below threshold. Small holdout (6 tickets) means one misclassification swings F1 by ~17 points.
- **Root cause for all weak classes:** Minority class problem. The training set has proportionally few examples. `class_weight='balanced'` in LR/DistilBERT partially compensates, but the GB component does not contribute clean signal on these classes, which is why `GB+DistilBERT` (rank 6, Macro-F1=0.449) underperforms the winner.

---

**Evaluation criteria:**
- Macro-averaged F1 per class on held-out test set (not accuracy — classes are imbalanced)
- Inference latency < 100ms synchronous (all three components are CPU-only sklearn models)
- Must beat current Gemini zero-shot baseline on the same held-out set — **met: +0.1915 Macro-F1**

**Selected approach:** Weighted soft-vote ensemble `LR(0.3)+GB(0.2)+DistilBERT(0.5)`.
Source: `analysis/model_selection.py`, `WINNER` and `WINNER_MODEL_PATH` constants.
Next step (ANTSE-303): Wire winner into production classifier, replacing the Gemini classification call.

**Training data:** Resolved tickets in `ticket_classifications` — labeled by the current Gemini classifier. Existing output becomes training data. The minimum-data guard prevents training on categories with too few examples.

**Update cadence:** Automated via OpenShift CronJob — phased cadence:

| Phase | Schedule | When |
|-------|----------|------|
| Phase 1 | Every sprint (2 weeks) | Active schema changes, new form types being added |
| Phase 2 | Monthly | Classification schema stable, catching distribution drift |
| Phase 3 | Quarterly | Mature system, minimal drift |

Primary trigger at any phase: a new form type is added, or agent corrections on the same category accumulate consistently.

**Automation components:**
1. **Data extraction** — SQL query pulls resolved tickets + classifications from `ticket_classifications`, splits into train/holdout sets
2. **Training script** — runs whichever model family won the experiment on extracted data, saves weights to a PVC
3. **Evaluation gate** — compares metric on held-out set against current production model; only promotes if accuracy improves or stays within tolerance
4. **Minimum-data guard** — skips retraining for any category with fewer than N labeled examples
5. **Model swap** — rolling restart loads new weights; CronJob spins up, runs, terminates. No persistent GPU node required.

**Where it sits:** Immediately after PII scrub. Replaces the Gemini classification call for known ticket types. LLM fallback handles novel categories until the minimum-data threshold is reached.

**Dependencies:** PII-scrubbed ticket text, metadata fields from ingest.

---

## Model 2: Sentiment Scorer

**What it does:** Scores the emotional intensity and tone of each ticket — urgency, frustration level, and (with ABSA) which specific aspect the customer is frustrated about. Feeds the anomaly detector and escalation router.

**Task type:** Supervised or zero-shot text classification (sentiment / intensity).

**Inputs:** Ticket summary + description + most recent customer comment (text)

**Outputs:** `{sentiment_score: float, sentiment_label: negative/neutral/positive, intensity: low/medium/high, frustrated_aspect: str | null}`

**EDA required before model selection:**
- Do tickets actually contain readable sentiment signals? Sample 50 tickets and manually label — if IT support tickets are mostly procedural ("please reset my password"), a sentiment model adds little value.
- CSAT data availability — without CSAT labels, the only training signal is manual annotation, which is expensive.
- Agreement rate — have two people independently label the same 50 tickets. If inter-annotator agreement is below 70%, the task is ambiguous and a simple model will not help.

**Candidate approaches:**

| Approach | When to use | Tradeoffs |
|----------|-------------|-----------|
| VADER (rule-based lexicon) | Zero labeled data, quick validation | Fast, no training, well-understood — poor on domain-specific language |
| Pre-trained model (cardiffnlp/twitter-roberta-base-sentiment, distilbert-sst-2) | Initial deployment with no labeled data | Load and test immediately, may not generalize to IT ticket language |
| Domain-adapted pre-trained model | IT support sentiment models exist on HuggingFace hub | Test several before concluding fine-tuning is needed |
| CSAT-supervised fine-tune | Only after CSAT data is flowing | Correlates low CSAT with ticket text — strongest signal, but requires data |
| ABSA (Aspect-Based Sentiment) | Per-topic breakdown needed | Significantly more complex — validate that the simpler approach doesn't meet the need first |

**Evaluation criteria:**
- Agreement rate with a human-labeled sample of 50 tickets (target > 70%)
- CSAT correlation once CSAT data is available (does high frustration predict low CSAT?)

**Starting recommendation:** Load a pre-trained sentiment model and validate against a 50-ticket manual sample. If agreement rate is above 70%, deploy it. Only invest in fine-tuning after CSAT data is flowing and there is evidence the pre-trained model is systematically wrong on specific ticket patterns.

**Training data:** Pre-trained model requires no training data. Fine-tuning signal comes from CSAT-labeled resolved tickets.

**Update cadence:** No retraining initially (pre-trained). Quarterly fine-tune after CSAT data accumulates. Skip entirely until EDA confirms sentiment signals are present and actionable.

**Where it sits:** Runs at classification time. Stored as a field in `ticket_classifications`. Feeds Model 3 (anomaly detector) and Model 5 (router).

**Dependencies:** Ticket ingest complete, PII scrub complete.

---

## Model 3: Volume Anomaly Detector

**What it does:** Detects when ticket volume or category distribution is behaving abnormally relative to a learned baseline — accounting for day-of-week, time of day, and recent historical patterns. Fires an alert before a flood overwhelms the queue.

**Task type:** Unsupervised anomaly detection on time-series data.

**Inputs (per time window — e.g., 15-minute buckets):**
- Ticket count
- Per-category counts (access, configuration, etc.)
- Average sentiment intensity
- New issue_type count (categories appearing for the first time)
- Time features: hour of day, day of week, days since last deployment

**Outputs:** `{anomaly_score: float, is_anomaly: bool, contributing_signals: list}`

**EDA required before model selection:**
- How much historical data exists? Run `SELECT date(created_at), COUNT(*) FROM tickets GROUP BY date(created_at)` — need at least 90 days before any anomaly baseline is meaningful.
- Is there a day-of-week pattern? Time-of-day pattern? Plot volume by hour and by day — if patterns are strong and regular, simple statistical methods will work well.
- Are there known anomaly periods? (cutover events, incidents) — label these manually to use as a validation set.
- Is volume trending up or flat? Trending volume changes what "normal" looks like and requires drift-aware approaches.

**Candidate approaches:**

| Approach | When to use | Tradeoffs |
|----------|-------------|-----------|
| Control chart (rolling mean ± Nσ) | Strong, regular patterns, enough history | Simplest baseline — assumes Gaussian distribution, brittle on heavy tails |
| STL decomposition + residual threshold | Clear seasonal/weekly pattern | Separates trend + seasonality + residual cleanly — good complement to any detector |
| Isolation Forest | Non-Gaussian distribution, multi-feature detection | Industry standard for multivariate anomaly detection, no labeled data needed |
| One-class SVM | High-dimensional feature space | More sensitive than Isolation Forest on small datasets, slower |
| LSTM autoencoder | Strong temporal dependencies | Learns sequence context — overkill at single-queue volume, revisit if volume grows significantly |

**Evaluation criteria:**
- False positive rate on known-normal periods (target < 5%)
- True positive rate on known-anomaly periods (the manually labeled incidents)
- Alert latency — how quickly does it detect after anomaly starts?

**Starting recommendation:** Begin with a control chart on daily ticket count (rolling 7-day mean ± 2σ, segmented by hour-of-day). If false positive rate is acceptable, ship it. Graduate to Isolation Forest if the distribution is clearly non-Gaussian or if multi-feature detection (category mix, not just count) is needed.

**Training data:** Historical ticket volume derived from the `tickets` table (`created_at` timestamps + classifications). Minimum ~90 days of history for a meaningful baseline. No anomaly labels required for unsupervised approaches.

**Update cadence:** Monthly baseline refit on a 90-day sliding window. Daily inference runs continuously regardless. Deployments and known events flagged as context to suppress false positives.

**Where it sits:** Background scheduled job. Initially daily; move to near-real-time as scale grows. Reads from `tickets` table. Writes anomaly records + triggers Slack alert.

**Dependencies:** Ticket classifications (for per-category breakdown), 90+ days of historical volume data.

---

## Model 4: Topic Cluster Detector (Rising Issues)

**What it does:** Groups tickets by semantic topic to surface emerging issue themes before they become high volume — catches "what is going wrong" before it shows up in volume counts.

**Task type:** Unsupervised text clustering.

**Inputs:** Ticket summaries + descriptions from a rolling window (e.g., last 7 days)

**Outputs:** `{cluster_id, topic_label: str, ticket_count, velocity: tickets/day, example_keys: list}`

**EDA required before model selection:**
- Do tickets naturally cluster into discrete topics, or is the space continuous? A manual review of 100 tickets across a 7-day window will indicate whether clusters are real and distinct.
- Are existing classification categories (Model 1) sufficient to detect rising issues, or do sub-topic clusters matter? If classification already surfaces the pattern, this model may not add value initially.
- What time window gives stable cluster membership? (7-day, 14-day, 30-day) — too short and clusters are noisy, too long and they miss fast-rising issues.
- Vocabulary — does the ticket corpus use consistent terminology, or is the same issue described in many ways? Heavy vocabulary variation favors embedding-based approaches over TF-IDF.

**Candidate approaches:**

| Approach | When to use | Tradeoffs |
|----------|-------------|-----------|
| TF-IDF + K-means | Consistent terminology, want interpretable centroids | Fastest, no embeddings needed — requires specifying K upfront, poor on noise |
| NMF on TF-IDF | Overlapping topics, want soft assignments | Produces interpretable topic-word distributions — good for reporting |
| LDA (Latent Dirichlet Allocation) | Probabilistic topic modeling, interpretability important | Well-established, but slower than NMF and hyperparameter-sensitive |
| HDBSCAN on embeddings | Variable vocabulary, noise-tolerant clustering needed | Finds natural cluster boundaries without specifying K, handles noise — requires embedding model |
| BERTopic | Full embedding + clustering pipeline, strong topic coherence | Combines embeddings + dimensionality reduction (UMAP) + HDBSCAN — production-ready library |

**Evaluation criteria:**
- Qualitative: sample 10 tickets per cluster — do they represent a coherent topic?
- Silhouette score as a quantitative secondary signal (target > 0.3)
- Velocity detection: does the approach surface a rising cluster within 24–48 hours of it starting?

**Starting recommendation:** TF-IDF + K-means first — fast, no external dependencies, interpretable. If cluster quality is poor (noisy, overlapping, low silhouette score), move to HDBSCAN on embeddings. Only adopt BERTopic if a full embedding pipeline is already running.

**Training data:** None — fully unsupervised. Clusters emerge from ticket data. Topic labels auto-generated by passing cluster representatives to Model 9 (LLM summarizes the cluster).

**Update cadence:** Daily. Compare today's cluster distribution to the previous run — new clusters or fast-growing clusters trigger an alert.

**Where it sits:** Daily background job. Reads ticket embeddings (Model 6 output). Outputs to a `ticket_clusters` table.

**Dependencies:** Model 6 (embeddings, if using embedding-based clustering), 7+ days of classified ticket data.

---

## Model 5: Ticket Router / Confidence Gate

**What it does:** Given classifier output (confidence score + category), decides whether to proceed to auto-draft, escalate to a human, or flag for review. Traffic controller between automated and human handling.

**Task type:** Rule-based decision function initially. Can graduate to supervised binary/multi-class classification if routing outcomes are labeled over time.

**Inputs:** `{confidence: float, category: str, issue_type: str, sentiment_intensity: str, linked_issue_count: int, is_novel_issue_type: bool}`

**Outputs:** `{route: auto_draft | human_review | flag_only, reason: str}`

**EDA required before learned model:**
This component starts rule-based — no EDA needed to deploy. EDA applies only if graduating to a learned model:
- What fraction of auto-drafts are accepted vs. discarded by agents? If acceptance rate is already above 80%, the rules are working and a learned model adds nothing.
- Are there systematic patterns in discarded drafts (specific categories, specific confidence bands) that rules aren't capturing?

**Routing rules (initial — no experimentation needed):**
- confidence < 0.6 AND is_novel_issue_type → human_review
- sentiment_intensity = high AND category = access → escalate
- confidence ≥ 0.6 AND known issue_type → auto_draft

**Candidate approaches (learned, only if rules prove insufficient):**

| Approach | Notes |
|----------|-------|
| Logistic regression on routing features | Quick, interpretable — what features drive routing decisions? |
| Gradient boosted classifier | Better on threshold combinations and interaction effects |
| Decision tree | Produces human-readable rules that can be codified back into the rule-based system |

**Update cadence:** Rules updated manually as patterns emerge. Learned model retrained quarterly if enough labeled routing outcomes exist.

**Where it sits:** Between classification and auto-responder. Replaces the current hardcoded eligibility checks in `auto_responder.py`.

**Dependencies:** Model 1 (classifier), Model 2 (sentiment scorer).

---

## Model 6: Embedding Model (Shared Infrastructure)

**What it does:** Converts text (tickets, FAQ entries, KB articles, agent responses, drafts) into dense vector representations. Shared infrastructure — used by retrieval, deduplication, clustering, and feedback scoring.

**Task type:** Not a training experiment — this is an infrastructure selection choice between API vs. self-hosted, and between available pre-trained models.

**Model options:**

| Model | Hosting | Dimensions | Notes |
|-------|---------|------------|-------|
| `gemini-embedding-001` | Vertex AI API | 3072 | Current stack, task-type specs (RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT) — cost per call |
| `all-MiniLM-L6-v2` | Self-hosted CPU | 384 | 80MB, 20–60ms/request, zero per-call cost — lower accuracy on long docs |
| `BGE-base-en-v1.5` | Self-hosted CPU | 768 | Better accuracy than MiniLM, ~250MB, still CPU-viable |
| Domain fine-tuned | Self-hosted GPU | Varies | Fine-tune on in-domain ticket/KB pairs — later optimization, not day-one |

**Selection criteria:** If embedding volume is low (< 10K calls/day), Vertex AI API is fine. Self-host if cost or API dependency is a concern. Run both on a sample and compare retrieval relevance on 20 manually curated query-document pairs.

**Inputs:** Any text string

**Outputs:** Dense vector (384–3072 dimensions depending on model)

**Update cadence:** Embeddings computed once per document and stored. Re-embed when the model is upgraded. Re-embed resolved tickets on a rolling basis as resolution summaries are added.

**Where it sits:** Called by Models 4, 7, 8, 10, 11 as shared infrastructure. Embeddings stored as BLOB columns in SQLite (or `sqlite-vss` for vector similarity queries).

**Dependencies:** None. Pre-trained, no labeled data required.

---

## Model 7: Hybrid Retrieval System

**What it does:** Given an incoming ticket, finds the most relevant content across FAQ entries, KB articles, resolved tickets, and indexed docs. Replaces the current keyword-only CQL/SQL approach.

**Task type:** Information retrieval — deterministic architecture, not a training experiment. BM25 + dense + RRF is a well-established pattern.

**Model type:**
1. **BM25** (sparse retrieval) — exact keyword matching, handles domain-specific acronyms (<PROJECT_KEY>, servicedeskapi, OAuth 2LO)
2. **Dense retrieval** (embedding cosine similarity) — semantic intent matching, handles synonyms and rephrasing
3. **Reciprocal Rank Fusion (RRF)** — merges BM25 and dense results into a single ranked list
4. **Re-ranking** (optional, later) — a lightweight cross-encoder scores candidate pairs for true relevance

**Inputs:** Ticket summary + description (query). Indexed corpus: FAQ entries, KB articles, resolved ticket resolution summaries, Atlassian docs.

**Outputs:** Ranked list of `{source_type, content_snippet, url, relevance_score}`

**Tuning (not experimentation):** BM25 k1 and b parameters, RRF k constant, and the similarity threshold for dense retrieval are tuned by measuring relevance on 20–30 manually curated query-result pairs. These are threshold decisions, not model selections.

**Training data:** BM25 is unsupervised. Dense retrieval uses pre-trained embeddings (Model 6) — no training. Re-ranker uses a pre-trained cross-encoder (ms-marco class) which works without domain-specific data.

**Update cadence:** Index updated incrementally on each new article or KB publish. Full re-index monthly as a safety net, or triggered when significant KB restructuring occurs.

**Where it sits:** Replaces `faq/lookup.py`. Called by the auto-responder before drafting.

**Dependencies:** Model 6 (embeddings), indexed corpus in vector store.

---

## Model 8: Deduplication Model

**What it does:** Before a newly generated FAQ article is written to the output Google Doc or published to Confluence, determines whether semantically equivalent content already exists. This is the fix for the 400-page problem.

**Task type:** Near-duplicate detection — deterministic architecture, no training experiment needed.

**Model type:**
1. **MinHash + LSH** — probabilistic near-duplicate detection in O(n) time. Catches structural near-duplicates (same article, minor edits). Fast enough to run synchronously before every write.
2. **Embedding cosine similarity** — semantic duplicate detection. Catches "same answer, completely different wording" — the actual failure mode with LLM-generated content.

**Tuning:** The similarity threshold (e.g., cosine ≥ 0.85 = duplicate) is calibrated manually by reviewing 30–40 known-duplicate and known-distinct pairs. Not a model selection decision.

**Inputs:** New article text. Corpus: all existing articles in `generated_articles` table + existing Google Doc content.

**Outputs:** `{is_duplicate: bool, duplicate_of: article_id | null, similarity_score: float, action: skip | update | publish_new}`

**Training data:** None — algorithmic. MinHash is computed. Cosine similarity uses pre-trained embeddings (Model 6).

**Update cadence:** Runs synchronously before every article write. Embeddings for existing articles computed once and cached.

**Where it sits:** In `faq/generator.py` before `INSERT OR REPLACE`, and in `faq/google_docs.py` before append.

**Dependencies:** Model 6 (embeddings), `generated_articles` table.

---

## Model 9: Response Draft Generator

**What it does:** Given a ticket and retrieved context, generates a draft response. The one place in the pipeline where a large generative model is genuinely the right tool.

**Task type:** Generative LLM — improvement comes from prompt engineering and better retrieval/few-shot examples, not model experimentation.

**Model type:** Gemini 3.1 (current). Draft type determines model tier:
- self_service / needs_info → Flash-Lite (faster, lower cost)
- admin_action / disposition review → Pro (stronger reasoning for complex step-by-step instructions)

**"Experimentation" for this model:** Prompt architecture testing. A/B test prompt variants on 20–30 held-out tickets, measured by agent acceptance rate and similarity score (Model 11). Structured output (JSON schema enforcement), chain-of-thought, and few-shot ordering are the variables.

**Inputs:** Ticket text + retrieved context from Model 7 + few-shot examples from Model 10.

**Outputs:** `{response_type, customer_response, admin_steps, missing_info}`

**Update cadence:** Prompts updated as patterns emerge from Model 12 (CSAT + feedback data surfaces which ticket types have poor draft quality). Model version updated when new Gemini generations become available.

**Where it sits:** `faq/auto_responder.py`, called after retrieval. Also used for FAQ article generation and gap analysis (different prompts, same model infrastructure).

**Dependencies:** Models 5 (router), 7 (retrieval), 10 (few-shot examples).

---

## Model 10: Few-Shot Example Retriever

**What it does:** For each new ticket being drafted, retrieves the most semantically similar past interactions where the agent approved the draft. Replaces the current static top-5 selection.

**Task type:** Embedding-based nearest-neighbor retrieval — no training experiment. The corpus quality (gated by Model 11) is what determines performance.

**Model type:** ANN index (FAISS or similar) over Model 6 embeddings of the `ai_draft_feedback` and `response_examples` tables.

**Inputs:** Incoming ticket embedding (query). Corpus: approved draft pairs, organic agent response examples — all embedded and indexed.

**Outputs:** Top-k `{ticket_summary, draft, agent_response, response_type}` ordered by semantic similarity.

**Tuning:** k (number of examples to retrieve), similarity floor (minimum score to include in few-shot block), and whether to filter by response_type before ranking. Evaluated by measuring draft acceptance rate with different k values on held-out tickets.

**Training data:** None — fully unsupervised. Corpus grows as agents use the system. Signal quality gated by Model 11.

**Update cadence:** New examples added in real-time as feedback is captured. ANN index rebuilt nightly.

**Where it sits:** Called inside `_build_few_shot_block()` in `auto_responder.py`, before the generation call.

**Dependencies:** Model 6 (embeddings), `ai_draft_feedback` + `response_examples` tables.

---

## Model 11: Feedback Quality Scorer

**What it does:** Scores the similarity between an AI draft and the agent's actual response, determining how much the draft was used vs. rewritten. This score determines which examples graduate into the few-shot pool (Model 10).

**Task type:** Embedding cosine similarity — no training experiment. Replaces the current difflib sequence matcher.

**Model type:** Cosine similarity between draft and agent-response embeddings (Model 6). Optionally augmented by an LLM-as-judge for periodic calibration on ambiguous cases — not for real-time scoring.

**Tuning:** Threshold between feedback categories (as_is / lightly_edited / heavily_rewritten / ignored) calibrated manually on 30 known examples.

**Inputs:** `{draft_customer_response: str, agent_actual_response: str}`

**Outputs:** `{similarity_score: float, feedback_category: as_is | lightly_edited | heavily_rewritten | ignored}`

**Update cadence:** Runs on every resolved draft pair in the `capture_feedback()` pipeline cycle.

**Where it sits:** `capture_feedback()` in `auto_responder.py`. Replaces the difflib call.

**Dependencies:** Model 6 (embeddings).

---

## Model 12: CSAT Correlator

**What it does:** Pulls JSM CSAT scores for resolved tickets and correlates them with draft quality scores, response type, and agent edits. Surfaces which ticket categories and draft types lead to poor customer outcomes.

**Task type:** Statistical aggregation — not a learned model, no experimentation needed.

**Model type:** SQL aggregation + correlation analysis. CSAT score × category × feedback_category × similarity_score. Outputs a ranked table of which ticket types have poor CSAT despite good draft similarity scores — those are the gaps that prompt retraining or prompt revision.

**Inputs:** `ticket_csat` table (to be built via `GET /rest/servicedeskapi/request/{key}/feedback`), `ai_draft_feedback` table, `ticket_classifications` table.

**Outputs:** `{category, issue_type, avg_csat, avg_similarity, draft_acceptance_rate}` ranked by CSAT.

**Update cadence:** Daily aggregation job.

**Where it sits:** New reporting module. Feeds into few-shot prioritization weighting (CSAT-low draft types deprioritized in Model 10's corpus) and prompt revision decisions for Model 9.

**Dependencies:** CSAT ingestion from servicedeskapi, Models 10/11.

---

## How the Models Connect

```
Ticket arrives
    │
    ▼
[Ingest + PII Scrub]  ← deterministic, no model
    │
    ├──► [Model 1: Classifier]          ← winner of classification experiment
    │         │
    │    [Model 2: Sentiment]           ← winner of sentiment experiment (or pre-trained)
    │         │
    │    [Model 5: Router]              ← rule-based initially; learned if rules prove insufficient
    │         │
    │    ┌────┴──────────────────┐
    │  auto_draft           human_review / flag
    │    │
    │    ▼
    │  [Model 7: Hybrid Retrieval]      ← BM25 + dense (Model 6) + RRF
    │    │
    │  [Model 10: Few-Shot Retriever]   ← ANN on Model 6 embeddings
    │    │
    │  [Model 9: Draft Generator]       ← Gemini 3.1 (Flash-Lite or Pro by response type)
    │    │
    │  Draft posted as internal comment
    │    │
    │  Agent responds
    │    │
    │  [Model 11: Feedback Scorer]      ← embedding cosine similarity
    │    │
    │  [Model 12: CSAT Correlator]      ← statistical aggregation (daily)
    │
    └──► [Background: Model 3: Anomaly Detector]   ← winner of anomaly detection experiment
         [Background: Model 4: Topic Clusterer]    ← winner of clustering experiment
         [Background: Model 8: Dedup]              ← MinHash LSH + cosine similarity
```

---

## Summary Table

| # | Model | Task Type | LLM? | Experiment Needed? | Starting Candidate | Retrains? |
|---|-------|-----------|------|--------------------|--------------------|-----------|
| 1 | Ticket Classifier | Supervised classification | No | Yes — EDA then baseline vs. transformer | TF-IDF + Logistic Regression | Phased: every sprint → monthly → quarterly (automated CronJob) |
| 2 | Sentiment Scorer | Sentiment classification | No | Yes — validate pre-trained before fine-tuning | Pre-trained sentiment model | Quarterly (if CSAT data available) |
| 3 | Volume Anomaly Detector | Unsupervised time-series | No | Yes — control chart before Isolation Forest | Control chart (rolling mean ± 2σ) | Monthly rolling baseline refit |
| 4 | Topic Cluster Detector | Unsupervised clustering | No | Yes — TF-IDF+K-means before embedding-based | TF-IDF + K-means | Daily |
| 5 | Ticket Router | Rule-based / learned | No | No (rules first; experiment only if rules fail) | Threshold rules on confidence | As needed |
| 6 | Embedding Model | Pre-trained encoder | No | No — selection between API vs self-hosted | gemini-embedding-001 or all-MiniLM-L6-v2 | On model upgrade |
| 7 | Hybrid Retrieval | BM25 + dense + RRF | No | No — tuning only | BM25 + dense + RRF | Incremental on publish; full re-index monthly |
| 8 | Deduplication | MinHash + cosine | No | No — tuning only (similarity threshold) | MinHash LSH + cosine | Never |
| 9 | Draft Generator | Generative LLM | **Yes** | Prompt A/B testing only | Gemini 3.1 Pro / Flash-Lite | Never (prompts only) |
| 10 | Few-Shot Retriever | ANN retrieval | No | No — tuning only (k, similarity floor) | ANN on Model 6 embeddings | Continuous (corpus grows) |
| 11 | Feedback Scorer | Embedding cosine | No | No — threshold tuning only | Embedding cosine similarity | Never |
| 12 | CSAT Correlator | Statistical aggregation | No | No | SQL aggregation | Daily batch |

Only Model 9 requires a generative LLM. Models 1–4 require experimentation to select the right approach. Models 5–12 are deterministic or algorithmic — implementation and threshold decisions, not model-selection experiments.

---

## Training Mode, Hardware, and Cost Analysis

### Online vs. Batch Training

| # | Model | Training Mode | Why |
|---|-------|--------------|-----|
| 1 | Ticket Classifier | **Batch** | Encoder transformers suffer catastrophic forgetting — true online learning overwrites previous knowledge. Full retrain on phased schedule via automated CronJob with evaluation gate. Classical ML models (TF-IDF + LR) can be retrained from scratch in seconds on CPU. |
| 2 | Sentiment Scorer | **Batch** | Pre-trained model: no training loop. Fine-tuned model: same constraint as Model 1. |
| 3 | Volume Anomaly Detector | **Online / rolling batch** | Control chart: rolling statistics update continuously. Isolation Forest supports `partial_fit` (incremental). Baseline refits monthly. |
| 4 | Topic Cluster Detector | **Batch** | All clustering algorithms (K-means, HDBSCAN, NMF) are batch-only. Runs daily on a rolling window. |
| 5 | Ticket Router | **Batch (rules) / Online (learned)** | Rule-based: manual updates. Learned: accumulate routing labels, retrain quarterly. |
| 6 | Embedding Model | **Batch (pre-trained, rarely retrained)** | Pre-trained weights loaded once. Embeddings for new documents computed incrementally. Full re-embed only on model upgrade. |
| 7 | Hybrid Retrieval | **Online index, batch model** | BM25 + ANN index updated incrementally on new documents. Model pre-trained. Full re-index monthly. |
| 8 | Deduplication | **Online scoring, batch fingerprints** | MinHash fingerprints computed per article. Similarity scores synchronous before every write. No training loop. |
| 9 | Draft Generator | **Never retrained** | Prompt-engineered. Quality improves through better retrieval and few-shot examples. |
| 10 | Few-Shot Retriever | **Continuous (corpus grows)** | No model training — ANN index grows as approved examples are added. Index rebuilt nightly. |
| 11 | Feedback Scorer | **Never trained** | Embedding cosine similarity — no learned parameters. |
| 12 | CSAT Correlator | **Batch (daily aggregation)** | SQL aggregation runs nightly. No model, no training. |

**Key constraint on transformers:** The base encoder cannot be updated incrementally without catastrophic forgetting. Practical solution: freeze the pre-trained base, only update lightweight task-specific heads. For full retraining, LoRA (Low-Rank Adaptation) trains only a small adapter layer (~1% of parameters), dramatically reducing GPU time.

---

### CPU vs. GPU Requirements

| # | Model | CPU or GPU? | Memory | Notes |
|---|-------|-------------|--------|-------|
| 1 | Ticket Classifier | **CPU viable** (if DistilBERT wins) | ~500 MB RAM | 50–80ms/request on CPU; < 20ms with ONNX. TF-IDF + LR: negligible. GPU needed only for training, not inference. |
| 2 | Sentiment Scorer | **CPU viable** | ~500 MB RAM | Same as Model 1. Pre-trained model: no training GPU. |
| 3 | Volume Anomaly Detector | **CPU only** | < 50 MB | ~22ms inference. scikit-learn, no GPU support or need. |
| 4 | Topic Cluster Detector | **CPU only** | ~100 MB for thousands of tickets | Batch job, latency not critical. Embeddings pre-computed. |
| 5 | Ticket Router | **CPU only** | Negligible | Rule lookup or simple linear model. |
| 6 | Embedding Model | **API call** or **CPU** | No local memory (API) / ~80–250 MB (self-hosted) | all-MiniLM-L6-v2 or BGE-base run on CPU with acceptable latency. |
| 7 | Hybrid Retrieval | **CPU only** | BM25: ~0.1 GB per 100K docs. ANN: ~1–2 GB for millions of vectors | FAISS has GPU acceleration but not needed here. |
| 8 | Deduplication | **CPU only** | Few MB | Negligible. |
| 9 | Draft Generator | **API call** | No local memory | Stays external (Gemini). |
| 10 | Few-Shot Retriever | **CPU only** | Shares ANN index with Model 7 | |
| 11 | Feedback Scorer | **CPU only** | Negligible | |
| 12 | CSAT Correlator | **CPU only** | Negligible | SQL aggregation. |

**Bottom line:** No GPU node required at current ticket volume. GPU needed only for training runs (Model 1, and only if DistilBERT wins the experiment) — these run as ephemeral CronJobs, not persistent nodes.

---

### Training Cost

| # | Model | One-Time Training Cost | Recurring Cost | Notes |
|---|-------|----------------------|----------------|-------|
| 1 | Ticket Classifier | **$1–$10** (GPU hours, only if transformer wins) | **$1–$5/month** | TF-IDF + LR: $0, trains on CPU in seconds. DistilBERT: T4 instance (~$0.35–0.50/hr) for 1–2 hours with LoRA. Automated CronJob — spins up, trains, terminates. |
| 2 | Sentiment Scorer | **$0** initially | **$0–$2/month** | Pre-trained model: zero cost. Fine-tune only after CSAT data is available. |
| 3 | Volume Anomaly Detector | **$0** | **$0** | scikit-learn on CPU. Seconds to fit. |
| 4 | Topic Cluster Detector | **$0** | **$0** | K-means / HDBSCAN on CPU. No training data required. |
| 5 | Ticket Router | **$0** | **$0** | Rule-based. No training. |
| 6 | Embedding Model | **$0** (pre-trained) | **$0 if self-hosted** / API cost if Vertex AI | all-MiniLM-L6-v2 (80 MB, CPU) costs nothing beyond memory. gemini-embedding-001 ~$0.00001 per 1K tokens. |
| 7 | Hybrid Retrieval | **$0** | **$0** | Index build is compute, not training. Runs on CPU in minutes. |
| 8 | Deduplication | **$0** | **$0** | Algorithmic. No training. |
| 9 | Draft Generator | **$0** (prompt-engineered) | **Ongoing API cost** — see ROI section | |
| 10–12 | Retriever / Scorer / Correlator | **$0** | **$0** | No training required. |

**Total one-time training cost: $1–$15** (only if DistilBERT wins the classifier experiment).
**Total recurring training cost: $1–$7/month** (monthly classifier retrains on a T4 spot instance, spin up then terminate).

---

### OpenShift Deployment Cost

The current service runs as a single pod in the `ants-engineering` namespace. The ML stack additions require more memory but no additional pods initially — everything colocates.

| Component | Additional Memory | Additional CPU | Pod change? |
|-----------|------------------|----------------|-------------|
| Classifier + sentiment (Models 1, 2 — if transformer wins) | +1 GB RAM | +0.5 vCPU | No — add to existing pod |
| Anomaly detector (Model 3) | +50 MB RAM | Negligible | No |
| Cluster detector (Model 4) | +100–200 MB RAM | Negligible (batch job) | No |
| BM25 + FAISS index (Model 7) | +500 MB–1 GB RAM | Negligible at query time | No |
| MinHash dedup (Model 8) | +10 MB RAM | Negligible | No |
| Self-hosted embedding model (Model 6 alt) | +100–250 MB RAM | +0.25 vCPU | No |

**Total additional resource requirement: ~2–3 GB RAM, ~1 vCPU.**

Current deployment.yaml likely requests 512 MB–1 GB RAM. Bumping to 3–4 GB RAM covers the full ML stack on a single pod with no GPU node and no additional OpenShift infrastructure. This is a `resources.requests` change in `deploy/openshift/deployment.yaml`.

Retraining runs as a Kubernetes Job (not a Deployment) — spun up on schedule, trains, terminates. No persistent GPU node needed.

---

### ROI Analysis

#### Current Gemini API Cost (Estimated)

Based on Gemini 3.1 Pro / Flash-Lite pricing. Note: Flash-Lite pricing is estimated (fraction of Pro); exact figures should be confirmed against current Vertex AI pricing page.

| Task | Tokens/call | Model tier | Est. cost/call | At 50 tickets/day | At 500 tickets/day |
|------|-------------|------------|----------------|-------------------|-------------------|
| Classification | ~600 in / ~100 out | Flash-Lite | ~$0.0001 | ~$0.005/day | ~$0.05/day |
| Auto-draft | ~4,000 in / ~500 out | Pro | ~$0.014 | ~$0.70/day | ~$7.00/day |
| FAQ gap analysis | ~8,000 in / ~1,000 out | Pro | ~$0.028 | ~$0.028/day (once daily) | ~$0.028/day |
| FAQ generation | ~3,000 in / ~800 out | Pro | ~$0.016 | ~$0.08/day | ~$0.16/day |

*Pro pricing basis: ~$2.00/1M input tokens, ~$12.00/1M output tokens. Draft generation output cost is substantially higher than under prior Gemini 2.x pricing.*

**Estimated current total: ~$0.81–$7.24/day ($300–$2,640/year)** at 50–500 tickets/day.

Auto-draft generation dominates. Classification cost is negligible regardless of model.

#### Cost After Right-Sizing the ML Stack

| Change | Cost impact |
|--------|------------|
| Replace LLM classification with experiment winner (Model 1) | **Save ~$0.005–$0.05/day** — small direct savings, but latency drops from ~1,500ms to < 100ms |
| Replace LLM sentiment with pre-trained model (Model 2) | **Save ~$0/day** (not currently implemented) — avoids adding new LLM calls |
| Self-host embedding model (Model 6) | **Save ~$0.002–$0.01/day** — meaningful at scale |
| Add anomaly detection, dedup, clustering (Models 3, 4, 8) | **$0 added cost** — CPU only |

**The direct API cost saving is modest at current volume.** The primary ROI drivers are:

1. **Latency:** Classification drops from ~1,500ms API round-trip to < 100ms local inference. Pipeline runs faster, agents get drafts sooner.

2. **Reliability:** Removes API dependency for classification and sentiment. If Gemini API is unavailable, classification and routing continue. Only generation degrades.

3. **Accuracy at scale:** Experiment-selected classifier on domain-specific tickets outperforms zero-shot LLM on ambiguous categories. At 500 tickets/day, 4–5% F1 improvement = 20–25 better-classified tickets daily.

4. **The 400-page doc fix:** Model 8 (dedup) prevents the output doc accumulation problem. Zero API cost, immediate operational value.

5. **New capabilities at zero marginal cost:** Anomaly detection (Model 3), topic clustering (Model 4), CSAT correlation (Model 12) — none exist today, all run on CPU, all cost $0 beyond the RAM bump.

#### Break-Even

One-time training cost: ~$10 (only if DistilBERT wins classifier experiment).
Monthly retraining: ~$5/month → ~$60/year.
Additional OpenShift memory (3 GB @ Red Hat internal infrastructure): effectively $0 (no new nodes, existing cluster).

If the right-sized stack prevents one production incident per quarter caused by a misclassified ticket or a bad auto-draft — the break-even is immediate.

At scale (500+ tickets/day), API cost savings become meaningful (~$500–$1,500/year) on top of reliability and accuracy gains.

---

## Implementation Plan

### Initiative

**What:** Transform the AI Helpdesk Agent from a single-LLM monolith into a purpose-built, experiment-driven ML pipeline with hybrid retrieval, automated continuous improvement, and proactive observability.

**Why:** The current system routes every task — classification, retrieval, generation, deduplication, feedback scoring — through a single Gemini model. This creates API dependency for tasks that don't need it, slow inference where fast deterministic methods exist, and zero ability to improve without manual prompt changes. A right-sized pipeline reduces cost, improves reliability, and enables continuous improvement that doesn't require an engineer every time.

**Acceptance Criteria:**
- [ ] LLM calls limited to generation tasks only (M9). All other tasks handled by purpose-built models or algorithms.
- [ ] Automated classifier retraining running on schedule with evaluation gate.
- [ ] Hybrid retrieval replacing keyword-only search.
- [ ] Output deduplication preventing the 400-page accumulation problem.
- [ ] Proactive alerting in place for volume anomalies and rising topic clusters.
- [ ] CSAT signal closing the feedback loop from customer outcome back to draft quality.

---

### Dependency Map

| Model | Blocked by | Blocks |
|-------|-----------|--------|
| M8 — Deduplication | Nothing | Nothing |
| M6 — Embedding Model | Nothing | M4 (embedding path), M7, M8 (cosine stage), M10, M11 |
| M11 — Feedback Scorer | M6 | M10 corpus quality |
| M1 — Classifier | M6 only if DistilBERT wins | M5 (confidence scores) |
| M5 — Router (rules) | M1 | M7, M9, M10 (gating) |
| M7 — Hybrid Retrieval | M6 | M10, M9 quality |
| M10 — Few-Shot Retriever | M6, M7, M11 | M9 quality |
| M9 — Draft Generator | M7, M10 | — (already exists; this improves it) |
| M2 — Sentiment Scorer | Nothing (pre-trained) | M5 (intensity signal), M12 |
| M3 — Anomaly Detector | 90 days of ticket data (time gate, not code) | — |
| M4 — Topic Clusterer | M6 (if embedding path) | — |
| M12 — CSAT Correlator | CSAT ingestion pipeline, M11 | M10 prioritization weights |

**Critical path:** M6 → M7 → M10 → M9 improvement. Everything else is parallel or standalone.

---

### Epic Overview

Each epic is scoped to 1 sprint (2 weeks) with AI assistance. Epics only block on each other when there is a physical or deployment dependency — otherwise they run in parallel.

| Epic | MVP | Models | Sprint | Blocks on |
|------|-----|--------|--------|-----------|
| Epic 1 — Output Deduplication | MVP 0 | M8 | Sprint 1 | — |
| Epic 2 — ML Foundation | MVP 1 | M6, M11 | Sprint 1 | — |
| Epic 3 — Classification Pipeline | MVP 2 | M1, M5, CronJob | Sprint 2 | Epic 2 (model routing, embedding model) |
| Epic 4 — Retrieval and Draft Quality | MVP 3 | M7, M10, M9, templates | Sprint 2 | Epic 2 (embedding model, vector storage) |
| Epic 6 — Customer Outcome Loop | MVP 5 | M2, M12, CSAT pipeline | Sprint 2 | Epic 2 (M2 needs M6; CSAT ingest has no dep) |
| Epic 5 — Proactive Alerting | MVP 4 | M3, M4 | Sprint 3 (code) | No code dependency — deploy time-gated (90 days of embeddings after Epic 2 ships) |

---

### Epic 1 — Output Deduplication (MVP 0)

**What:** Add a two-stage deduplication gate (MinHash LSH + embedding cosine similarity) before any article is written to the `generated_articles` table or appended to the output Google Doc. Bump the pod memory allocation to prepare for the ML stack. Clean up the existing ~400-page accumulated duplicates.

**Why:** Every pipeline run currently appends all generated articles to the output doc without checking for existing content. The doc has grown to ~400 pages of mostly duplicates, making it unusable. This is an operational problem affecting every agent who refers to it. The memory bump in this epic prevents a blocking config change mid-sprint when ML models are ready to deploy in Epics 2–4.

**Acceptance Criteria:**
- [ ] No duplicate article is appended to the output Google Doc on any subsequent pipeline run.
- [ ] `generated_articles` table contains no structural near-duplicates after a pipeline run.
- [ ] Similarity threshold calibrated and documented.
- [ ] Existing Google Doc deduplicated.
- [ ] `deploy/openshift/deployment.yaml` memory request updated to 3–4 Gi.

---

**Story: MinHash structural fingerprinting**

*What:* Compute a MinHash + LSH fingerprint for each generated article before any write to `generated_articles`. Store the fingerprint alongside the article.

*Why:* O(n) structural near-duplicate detection — catches the same article with minor wording edits before the more expensive cosine check runs. Without this, structurally identical articles generated across multiple runs accumulate silently.

*Acceptance Criteria:*
- [ ] Fingerprint computed and stored on every article write.
- [ ] Structural near-duplicates (> 90% shingle overlap) flagged correctly on 10 known-duplicate test pairs.
- [ ] Fingerprint stored in `generated_articles` table. Existing articles backfilled.

---

**Story: Embedding cosine dedup gate — `faq/generator.py`**

*What:* Before `INSERT OR REPLACE` in `faq/generator.py`, compute the new article's embedding and compare against existing article embeddings. Skip the write if cosine similarity exceeds the calibrated threshold.

*Why:* LLM-generated duplicates often have completely different wording but identical meaning — MinHash won't catch these. Cosine similarity on embeddings catches semantic duplicates, which is the actual failure mode causing the 400-page problem.

*Acceptance Criteria:*
- [ ] INSERT skipped for any article whose similarity to an existing article exceeds the threshold.
- [ ] DB article count grows at expected rate on a test pipeline run (no new duplicates written).
- [ ] Note: stubs against Vertex AI embedding call until Epic 2 embedding service is live; swap is a config change.

---

**Story: Embedding cosine dedup gate — `faq/google_docs.py`**

*What:* Before appending any article to the Google Doc in `faq/google_docs.py`, embed the article and compare against embeddings of existing doc sections. Skip the append if similarity exceeds threshold.

*Why:* Fixes the output layer, not just the database layer. Both gates are needed — the DB gate prevents storage duplicates, the doc gate prevents the output document from accumulating.

*Acceptance Criteria:*
- [ ] No semantically equivalent article appended to the output doc on a pipeline run.
- [ ] Tested against a doc with 5 known near-duplicate sections — all 5 skipped.

---

**Story: Similarity threshold calibration**

*What:* Manually label 30–40 article pairs (known duplicates and known distinct). Test cosine similarity scores against the labels. Set the threshold that minimizes false positives and false negatives.

*Why:* An uncalibrated threshold suppresses legitimate new articles (too aggressive) or misses duplicates (too permissive). The threshold must be set from evidence, not intuition.

*Acceptance Criteria:*
- [ ] 30–40 pairs labeled and scores recorded.
- [ ] Threshold set with false positive rate and false negative rate documented.
- [ ] Threshold value committed to config, not hardcoded.

---

**Story: Initial Google Doc cleanup**

*What:* Write and run a one-time script that identifies existing duplicate sections in the ~400-page output doc and removes them, leaving only unique articles.

*Why:* The dedup gate prevents future accumulation but does not fix the existing doc. Agents need the doc to be usable now, not only after future runs.

*Acceptance Criteria:*
- [ ] Output doc section count before and after recorded.
- [ ] All remaining sections confirmed unique (sampled review of 20 sections).
- [ ] Script retained in the repo for future use.

---

**Story: Pod memory and CPU request bump**

*What:* Increase `resources.requests.memory` in `deploy/openshift/deployment.yaml` from current value to 3–4 Gi and add ~1 vCPU.

*Why:* The ML stack additions (classifier, sentiment model, embedding model, FAISS index) add ~2–3 GB RAM. Making this change now prevents the resource configuration from being a last-minute blocker when models are ready to deploy in Epics 2–4.

*Acceptance Criteria:*
- [ ] `deployment.yaml` updated and committed.
- [ ] Pod restarts successfully in `ants-engineering` namespace.
- [ ] No existing functionality broken. Resource limits set appropriately above requests.

---

### Epic 2 — ML Foundation Infrastructure (MVP 1)

**What:** Stand up a shared embedding service and vector storage layer. Replace the `difflib` feedback scorer with embedding cosine similarity. Configure per-task model routing so different pipeline tasks can use different Gemini model tiers.

**Why:** Every downstream epic depends on embeddings — retrieval, deduplication, clustering, few-shot selection, and feedback scoring all require a consistent embedding source. Building it once as shared infrastructure prevents N separate embedding implementations from proliferating with different models and inconsistent quality. Per-task model routing removes the single `GEMINI_MODEL` env var constraint, enabling cost optimization in later epics without code surgery.

**Acceptance Criteria:**
- [ ] Single embedding function callable from any module, model configurable without code changes.
- [ ] Embeddings stored and queryable via cosine similarity in SQLite.
- [ ] Feedback similarity scores demonstrably more accurate than difflib on a calibration set.
- [ ] Per-task model configuration in place and tested.

---

**Story: Embedding model evaluation**

*What:* Test at least two embedding options — `gemini-embedding-001` (Vertex AI API) and `all-MiniLM-L6-v2` (self-hosted CPU) — on 20 manually curated query-document pairs from the <PROJECT_KEY> corpus.

*Why:* The choice between API and self-hosted affects cost, latency, and offline reliability for every model in the pipeline. This is a one-time decision with long-lasting consequences; it must be evidence-based.

*Acceptance Criteria:*
- [ ] 20 query-document pairs curated from real ticket-to-KB matches.
- [ ] Relevance scores for both models recorded on all 20 pairs.
- [ ] Latency and cost-per-1K-embeddings measured for both.
- [ ] Decision documented with rationale. Winner committed to config.

*Subtasks:* curate test pairs; run both models; record scores; document decision

---

**Story: Shared embedding service**

*What:* Implement a single module-level function (`embed_text(text, task_type)`) used by all downstream components. Model choice is a config value, not hardcoded at each call site.

*Why:* Without a shared service, each component implements its own call — leading to inconsistent models, duplicated batching logic, and expensive model switches that require changes in many places.

*Acceptance Criteria:*
- [ ] Single importable function, handles batching, respects rate limits.
- [ ] Model configurable via environment variable.
- [ ] At least one downstream consumer (M11) using it in tests.
- [ ] Unit tests covering batching and error handling.

---

**Story: Vector storage layer**

*What:* Add BLOB embedding columns to `generated_articles`, `ai_draft_feedback`, and `response_examples` tables. Implement helper functions for store, retrieve, and top-k cosine similarity query.

*Why:* Embeddings need to be persisted so they are not recomputed on every request. Without storage, every retrieval and dedup check incurs an embedding API call — expensive and slow.

*Acceptance Criteria:*
- [ ] Embedding columns added via migration. Existing rows backfilled.
- [ ] `store_embedding(id, vector)` and `query_similar(vector, k)` functions implemented and tested.
- [ ] Cosine similarity query returns correct top-k on a test corpus of 50 articles.

---

**Story: Replace difflib feedback scorer (M11)**

*What:* Replace `difflib.SequenceMatcher` in `capture_feedback()` in `auto_responder.py` with embedding cosine similarity using the shared embedding service.

*Why:* difflib measures character-level surface overlap. It scores a heavily rewritten response as "similar" if the agent kept the same keywords. Embedding cosine similarity captures whether the semantic meaning was preserved — which is the signal that determines whether a draft was genuinely useful or completely rewritten.

*Acceptance Criteria:*
- [ ] `difflib` call removed from `capture_feedback()`.
- [ ] Cosine similarity computed using shared embedding service.
- [ ] Calibration set of 30 pairs (10 as-is, 10 lightly edited, 10 heavily rewritten) labeled and threshold set.
- [ ] Feedback categories (as_is / lightly_edited / heavily_rewritten / ignored) mapping documented.
- [ ] Existing feedback pipeline tests updated and passing.

---

**Story: Per-task model routing**

*What:* Replace the single `GEMINI_MODEL` env var in `config.py` with per-task model configuration: `GEMINI_MODEL_CLASSIFICATION` (Flash-Lite), `GEMINI_MODEL_GENERATION` (Pro), `GEMINI_MODEL_ANALYSIS` (Pro). Update each call site to use the correct config.

*Why:* The current monolith config sends every task — including cheap classification calls — through the Pro model. Once M1 replaces LLM classification, this is a zero-cost clean-up. But setting it up now means future model tier decisions are config changes, not code changes.

*Acceptance Criteria:*
- [ ] Three separate model config values in `config.py` and `.env`.
- [ ] Each call site (`classifier.py`, `auto_responder.py`, `analyzer.py`, `generator.py`) uses the correct config key.
- [ ] Existing tests pass. No regression in output quality confirmed on a sample run.

---

### Epic 3 — Classification Pipeline (MVP 2)

**What:** Run a structured ML experiment to select the best ticket classifier, replace the Gemini classification call with the experiment winner, implement the M5 rule-based router, and automate retraining via an OpenShift CronJob with an evaluation gate.

**Why:** Classification is the first decision that determines every downstream action. The current Gemini zero-shot classifier takes ~1,500ms per ticket (synchronous API call), is unavailable during API outages, and never improves without manual prompt changes. A purpose-built classifier runs in < 100ms locally, survives API outages, and retrains automatically as ticket categories evolve.

**Acceptance Criteria:**
- [ ] Experiment completed. Winner selected and rationale documented.
- [ ] Gemini classification call replaced. Known ticket types classified locally in < 100ms.
- [ ] LLM fallback active for novel ticket types below minimum-data threshold.
- [ ] M5 rule-based router in place, routing decisions logged and configurable.
- [ ] Automated CronJob running on Phase 1 schedule (every sprint).

---

**Story: EDA — `ticket_classifications` table**

*What:* Analyze the `ticket_classifications` table for class distribution, per-class example counts, text length distribution, and label quality. Manually review 50 Gemini-labeled tickets against their raw text.

*Why:* Model selection depends on data characteristics. An imbalanced class distribution requires stratified sampling and weighted loss. Categories with fewer than N examples cannot be trained on reliably. Poor label quality means any model trained on it will perform poorly regardless of architecture.

*Acceptance Criteria:*
- [ ] Per-class example counts documented.
- [ ] Minimum-data guard threshold set (categories below N excluded from first training run).
- [ ] Label quality assessment documented — agreement rate between Gemini labels and manual review of 50 tickets.
- [ ] Text length distribution documented.
- [ ] Underrepresented classes flagged with recommended handling.

*Subtasks:* SQL class distribution query; text length analysis; 50-ticket manual review; findings document

---

**Story: Gemini zero-shot baseline**

*What:* Run the current Gemini classifier on a fixed, versioned held-out test set. Record per-class F1, precision, recall, and inference latency.

*Why:* Without a baseline, there is no way to determine whether a candidate model is an improvement. The held-out set must be fixed before any training experiments begin so that results are comparable.

*Acceptance Criteria:*
- [ ] Held-out test set defined, versioned, and not used for any training.
- [ ] Per-class F1, precision, and recall documented for the current Gemini classifier.
- [ ] Latency recorded. This is the bar all candidates must beat.

---

**Story: TF-IDF + Logistic Regression experiment**

*What:* Train a TF-IDF + Logistic Regression classifier on the training split. Evaluate on the held-out test set.

*Why:* This is the simplest possible learned classifier. If it matches or beats the Gemini baseline, it should be shipped — it is faster, cheaper, needs no GPU, and requires no API dependency. Every additional model family is only justified if this one is insufficient.

*Acceptance Criteria:*
- [ ] Model trained on training split. Evaluated on held-out set.
- [ ] Per-class F1 documented. Comparison to Gemini baseline documented.
- [ ] Go/no-go decision on proceeding to gradient boosting recorded with rationale.

---

**Story: TF-IDF + Gradient Boosting experiment (conditional)**

*What:* Train a gradient boosting classifier (XGBoost or LightGBM) on TF-IDF features combined with metadata features (SLA tier, component, request type).

*Why:* Gradient boosting handles non-text metadata features better than logistic regression. Only run if LR F1 is below the acceptable threshold on the held-out set.

*Acceptance Criteria:*
- [ ] Only executed if TF-IDF + LR F1 is below threshold.
- [ ] Per-class F1 documented. Metadata feature importance recorded.
- [ ] Comparison to LR and Gemini baseline documented.
- [ ] Go/no-go on DistilBERT recorded with rationale.

---

**Story: DistilBERT fine-tune experiment (conditional)**

*What:* Fine-tune DistilBERT using LoRA on the training split. Train on a T4 GPU spot instance. Evaluate on the held-out test set.

*Why:* DistilBERT captures contextual meaning that bag-of-words approaches miss — "can't access" and "login failure" would be two different classes in TF-IDF but the same in an encoder model. Only justified if classical ML is insufficient.

*Acceptance Criteria:*
- [ ] Only executed if both TF-IDF experiments produce insufficient F1.
- [ ] Per-class F1 documented. Training time and GPU cost recorded.
- [ ] T4 spot CronJob scaffolding in `deploy/openshift/` committed.

*Subtasks:* CronJob manifest; LoRA training script; evaluation; results document

---

**Story: Model selection and documentation**

*What:* Select the winning approach from the experiments. Update this document with the actual selection — the candidates table above is replaced with the decision.

*Why:* The experiment has no value if the winner is not selected and the rationale is not recorded. Future engineers must be able to understand why this model was chosen without re-running the experiment.

*Acceptance Criteria:*
- [ ] Winner identified. Decision rationale documented (which model, on what metric, by how much over which baseline).
- [ ] This document updated to reflect the actual approach rather than "candidates."

---

**Story: Replace Gemini classification call**

*What:* Wire the experiment winner into `analysis/classifier.py`. Replace the Gemini call for known ticket types. Keep the LLM as fallback for novel categories below the minimum-data threshold.

*Why:* This is the production deployment of the experiment. Classification latency drops from ~1,500ms to < 100ms for known types. API dependency removed for the most frequent pipeline operation.

*Acceptance Criteria:*
- [ ] Winner model loaded from PVC on startup. LLM fallback active for novel types.
- [ ] Classification latency < 100ms on known types (measured, not estimated).
- [ ] Existing test suite passes. No regression in downstream draft acceptance rate confirmed over a 1-week baseline comparison.

---

**Story: M5 rule-based router**

*What:* Implement explicit routing rules in `auto_responder.py` using M1 confidence scores and M2 sentiment intensity. Replace hardcoded eligibility checks with configurable rules.

*Why:* The current system has implicit eligibility logic scattered across the auto-responder. Making it explicit — with documented thresholds and a logged routing decision per ticket — makes the system auditable and adjustable without code changes.

*Acceptance Criteria:*
- [ ] Routing decisions logged per ticket with confidence score, category, sentiment intensity, and route chosen.
- [ ] Rule thresholds configurable via env vars, not hardcoded.
- [ ] Three routes implemented: `auto_draft`, `human_review`, `flag_only`.
- [ ] Routing logic covered by unit tests.

---

**Story: Automated retraining CronJob**

*What:* Implement a Kubernetes CronJob in `deploy/openshift/` that: extracts labeled data from `ticket_classifications`, retrains the winning model, evaluates against the current production model on the held-out set, and promotes only if the new model is better. Includes minimum-data guard and evaluation gate.

*Why:* Manual retraining is not sustainable. Without automation, the classifier drifts as ticket taxonomy evolves — new form types go unrecognized, accuracy silently degrades, and nobody notices until agents start discarding drafts at higher rates.

*Acceptance Criteria:*
- [ ] CronJob manifest in `deploy/openshift/`. Runs on Phase 1 schedule (every sprint).
- [ ] Evaluation gate: new model only promoted if F1 improves or stays within tolerance on held-out set.
- [ ] Minimum-data guard: categories below threshold excluded from the training run.
- [ ] CronJob spins up, trains, terminates. No persistent GPU node.
- [ ] Retraining outcome (promoted / skipped / failed) logged.

*Subtasks:* data extraction script; training script; evaluation gate logic; model promotion; CronJob manifest; logging

---

### Epic 4 — Retrieval and Draft Quality (MVP 3)

**What:** Replace the keyword-only CQL/SQL retrieval with a hybrid system (BM25 + dense embedding + RRF). Replace static few-shot selection with ANN retrieval. Add a response template system for high-frequency ticket types. Tune M9 prompts using improved context.

**Why:** The current retrieval system is keyword-dependent — it returns nothing when a ticket uses different wording than the KB article. This is the largest single contributor to poor draft quality: bad retrieval means bad context means bad drafts. Response templates further reduce generation cost and improve consistency for well-understood patterns where the LLM should not be writing from scratch every time.

**Acceptance Criteria:**
- [ ] `faq/lookup.py` replaced with hybrid retrieval system.
- [ ] Retrieval relevance confirmed on 20–30 manually curated evaluation pairs.
- [ ] Static few-shot selection replaced with ANN retrieval. Acceptance rate compared before and after.
- [ ] Response template system live with at least 3 templates authored.
- [ ] M9 prompt variant tested and best performer deployed.

---

**Story: BM25 index**

*What:* Build a BM25 index over FAQ entries, KB articles, resolved ticket resolutions, and Atlassian docs.

*Why:* BM25 handles exact keyword matching and domain-specific acronyms (<PROJECT_KEY>, OAuth 2LO, servicedeskapi) that embedding models can fail on. It is the sparse retrieval half of the hybrid system and is required before RRF can merge results.

*Acceptance Criteria:*
- [ ] Index built over all four corpus sources. Document count per source recorded.
- [ ] Keyword queries returning relevant results on 10 manual test queries.
- [ ] Incremental update on new document publish confirmed working.

---

**Story: Dense retrieval layer**

*What:* Embed all corpus documents using the shared embedding service (Epic 2). Implement cosine similarity retrieval over the stored embeddings.

*Why:* Dense retrieval catches semantic intent — a ticket asking "can't log in" matches KB articles about "authentication failure" even with no keyword overlap. This is the gap the current system cannot fill.

*Acceptance Criteria:*
- [ ] All corpus documents embedded and stored in vector storage.
- [ ] Semantic queries returning relevant results on 10 manual test queries where keyword search returns nothing.
- [ ] Corpus re-embedding triggered on new document publish.

---

**Story: RRF merger**

*What:* Implement Reciprocal Rank Fusion to merge BM25 and dense result lists into a single ranked output.

*Why:* BM25 and dense retrieval have complementary strengths — neither alone is as good as the combination. RRF is a parameter-free fusion method that consistently outperforms either alone without requiring a learned re-ranker.

*Acceptance Criteria:*
- [ ] RRF merger implemented. k parameter configurable.
- [ ] Combined ranking evaluated on 20–30 manually curated query-result pairs.
- [ ] Fusion relevance score > BM25-only and dense-only on the evaluation set. Results documented.

---

**Story: Replace `faq/lookup.py`**

*What:* Wire hybrid retrieval into the auto-responder as the primary lookup path. Replace the `/api/faq/lookup` endpoint backend.

*Why:* This is the production deployment of the retrieval system. All downstream quality improvements — better drafts, better few-shot examples — flow from this.

*Acceptance Criteria:*
- [ ] `faq/lookup.py` replaced. Hybrid retrieval called by auto-responder.
- [ ] Existing integration tests updated and passing.
- [ ] Agent acceptance rate baseline measured in the sprint before deployment for comparison.
- [ ] Latency within acceptable bounds (< 500ms end-to-end retrieval).

---

**Story: ANN few-shot retrieval (M10)**

*What:* Replace the static top-5 selection in `_build_few_shot_block()` with ANN nearest-neighbor retrieval over embedded approved draft pairs from `ai_draft_feedback` and `response_examples`.

*Why:* Static top-5 selection ignores the content of the incoming ticket. ANN retrieval surfaces the examples semantically closest to what is being drafted — better examples produce better generation.

*Acceptance Criteria:*
- [ ] ANN index built over approved draft pairs.
- [ ] k and similarity floor tuned on 20 held-out tickets.
- [ ] Draft acceptance rate compared before and after over a 1-week window.
- [ ] Nightly index rebuild job running.

---

**Story: Response template system**

*What:* Create a `response_templates` table. Author response templates with named slots (e.g., `[STEPS]`, `[SYSTEM_NAME]`, `[KB_URL]`) for the highest-frequency ticket types. Wire into M5 and M9: if a template exists for the ticket type and confidence exceeds threshold, use template-fill mode instead of full generation.

*Why:* For well-understood ticket types (password reset, access request, permission escalation), the response structure is predictable and repeatable. Asking the LLM to generate from scratch for these is expensive, slow, and inconsistent. Template-fill mode reduces output token count significantly, lowers API cost, and produces responses that are easier for the team to maintain and improve without touching code.

*Acceptance Criteria:*
- [ ] `response_templates` table created with schema: `ticket_type`, `confidence_threshold`, `template_text`, `slot_definitions`.
- [ ] At least 3 templates authored for the highest-frequency ticket types (identified from `ticket_classifications` EDA in Epic 3).
- [ ] M5 router checks for template existence before routing to full generation.
- [ ] Template-fill mode produces correct output on 5 test tickets per template.
- [ ] Full generation fallback active when no template matches or confidence is below threshold.

*Subtasks:* schema and migration; CRUD for template management; M5 router integration; M9 template-fill mode; 3 initial templates authored; test coverage

---

**Story: M9 prompt tuning**

*What:* A/B test at least two prompt variants for full generation mode using the improved retrieval context and few-shot examples. Measure acceptance rate on 20–30 held-out tickets.

*Why:* Better inputs (retrieval + few-shot) may not automatically produce better outputs if the prompt structure is not aligned with the new context format. A targeted tuning pass captures remaining improvement.

*Acceptance Criteria:*
- [ ] At least 2 prompt variants tested on the same 20–30 held-out tickets.
- [ ] Winner selected based on acceptance rate or M11 similarity score.
- [ ] Winning prompt variant documented and deployed.

---

### Epic 5 — Proactive Alerting (MVP 4)

**What:** Implement volume anomaly detection (M3) and rising topic clustering (M4) with Slack alerting. Start with the simplest viable approach for each — graduate to more sophisticated methods only if the baseline is insufficient.

**Why:** There is currently no early warning when the queue is hit by an unusual pattern — a deployment issue, migration problem, or product bug rollout. The team discovers the pattern only after agents are already overwhelmed. Proactive alerting gives the team lead time to prepare a response, draft a known-issue article, or route additional support before the queue backs up.

**Note:** All code in this epic can be written during any sprint. The 90-day data threshold gates *deployment*, not development. Start coding in Sprint 5; deploy when the data threshold is met.

**Acceptance Criteria:**
- [ ] Slack alert fires within one detection cycle of a real volume anomaly.
- [ ] Daily topic cluster job running. New and fast-growing clusters surfaced within 24 hours.
- [ ] Both systems validated against historical data before production deployment.
- [ ] False positive rate on known-normal periods < 5%.

---

**Story: EDA — ticket volume patterns**

*What:* Query the `tickets` table for daily and hourly volume by category over the full available history. Identify day-of-week and time-of-day patterns. Label any known anomaly periods (cutover events, incidents).

*Why:* Model selection depends on whether the signal is regular and seasonal or highly variable. The EDA determines which approach is appropriate and provides a labeled validation set of known anomalies — without which there is no way to validate the detector.

*Acceptance Criteria:*
- [ ] Day-of-week and time-of-day volume distributions documented.
- [ ] At least 2 known anomaly periods identified and labeled with dates and cause.
- [ ] Feature set for M3 defined based on findings.
- [ ] 90-day data threshold confirmed met (or date when it will be met noted).

---

**Story: Control chart baseline (M3)**

*What:* Implement a rolling mean ± 2σ control chart segmented by hour-of-day and day-of-week. Flag time windows where volume exceeds the threshold.

*Why:* This is the simplest possible anomaly detector. If the volume distribution is approximately normal and patterns are regular, it works well with zero ML infrastructure.

*Acceptance Criteria:*
- [ ] Baseline computed correctly on the full historical dataset.
- [ ] Known anomaly periods flagged with TPR > 80%.
- [ ] FPR < 5% on known-normal periods. If FPR is above 5%, proceed to Isolation Forest story.

---

**Story: Slack alert on anomaly**

*What:* Implement a Slack alert that fires when M3 detects an anomaly. Message includes anomaly score, contributing category breakdown, and the affected time window.

*Why:* Detection without notification has zero operational value. The alert must reach the team lead before they discover the problem by looking at the queue.

*Acceptance Criteria:*
- [ ] Alert fires correctly on a simulated anomaly injection.
- [ ] Message format: anomaly score, category breakdown, time window, link to queue.
- [ ] Alert rate in a 7-day production window is within acceptable frequency (no spam).

---

**Story: Isolation Forest (conditional)**

*What:* Replace the control chart with Isolation Forest if Story 2 FPR exceeds 5% or if the volume distribution is clearly non-Gaussian.

*Why:* Isolation Forest handles non-Gaussian distributions and multi-feature detection (volume + category mix + sentiment) better than a control chart. Only needed if the simpler baseline is insufficient.

*Acceptance Criteria:*
- [ ] Only executed if control chart FPR > 5%.
- [ ] Evaluated on the same labeled anomaly validation set.
- [ ] FPR and TPR compared to control chart. Decision documented.

---

**Story: EDA — topic clustering**

*What:* Run a manual clustering pass on a 7-day ticket window using TF-IDF + K-means. Review resulting clusters qualitatively (sample 10 tickets per cluster). Measure silhouette score.

*Why:* Before implementing a daily job, validate that clusters are coherent and the time window gives stable results. A poor clustering algorithm produces noise alerts — the most useless kind of alert.

*Acceptance Criteria:*
- [ ] Clusters reviewed manually. Coherence assessment documented.
- [ ] Silhouette score measured. K value or HDBSCAN parameters determined.
- [ ] Time window confirmed (7-day default, or adjusted based on findings).

---

**Story: Topic clusterer daily job (M4)**

*What:* Implement a daily background job that clusters the last N days of tickets and computes a delta against the previous run to identify new or fast-growing clusters. Results written to a `ticket_clusters` table.

*Why:* Emerging issues appear as a new semantic cluster before they show up in volume counts. Detecting the pattern early gives the team time to prepare a known-issue article or route additional coverage.

*Acceptance Criteria:*
- [ ] Job runs daily without errors.
- [ ] Cluster delta computed correctly — new clusters and clusters with velocity above threshold identified.
- [ ] `ticket_clusters` table populated with cluster ID, topic label, ticket count, velocity, and example keys.

---

**Story: Rising cluster Slack alert**

*What:* Implement a Slack alert when M4 identifies a new cluster or a cluster growing above a configurable velocity threshold. Topic label generated by passing representative tickets to M9.

*Why:* Same reason as the volume anomaly alert — detection without notification has no operational value.

*Acceptance Criteria:*
- [ ] Alert fires on a simulated growing cluster.
- [ ] Message includes topic label, ticket count, velocity, and example ticket keys.
- [ ] Velocity threshold configurable. Alert rate acceptable in a 7-day production window.

---

**Story: Graduate to HDBSCAN (conditional)**

*What:* Replace K-means with HDBSCAN on embeddings if K-means cluster quality is below the threshold established in the EDA story.

*Why:* K-means requires specifying K and assumes spherical clusters. HDBSCAN finds natural cluster boundaries without K and handles noise better. Only needed if the simpler approach produces incoherent clusters.

*Acceptance Criteria:*
- [ ] Only executed if silhouette score < 0.3 or manual review finds clusters incoherent.
- [ ] HDBSCAN evaluated on same corpus. Cluster quality compared to K-means. Decision documented.

---

### Epic 6 — Customer Outcome Loop (MVP 5)

**What:** Build the CSAT ingestion pipeline, implement sentiment scoring (M2) for incoming tickets, and implement the CSAT Correlator (M12) to surface which ticket types produce poor customer outcomes. Wire findings back into few-shot prioritization.

**Why:** Agent acceptance of drafts is a leading indicator but not a customer outcome signal. An agent can accept a draft that the customer finds unhelpful. Without CSAT data, the pipeline optimizes for what agents approve — not what customers need. Closing this loop surfaces where the pipeline is systematically failing customers, regardless of what agents think of the drafts.

**Acceptance Criteria:**
- [ ] CSAT scores ingested daily for resolved tickets.
- [ ] Sentiment score present on all new tickets.
- [ ] Daily CSAT correlation report available.
- [ ] At least one actionable finding (lowest-CSAT ticket category identified and actioned) documented within the first sprint of operation.

---

**Story: CSAT ingestion pipeline**

*What:* Implement a daily job that polls `GET /rest/servicedeskapi/request/{key}/feedback` for recently resolved tickets and stores results in a new `ticket_csat` table.

*Why:* CSAT data is available via the servicedeskapi experimental endpoint with no additional OAuth scopes. Without ingesting it, the system has no customer outcome signal — only agent behavior. This story creates the data foundation the rest of the epic depends on.

*Acceptance Criteria:*
- [ ] `ticket_csat` table created: `ticket_key`, `csat_score`, `submitted_at`, `ingested_at`.
- [ ] Daily ingestion job running. CSAT scores linked to ticket keys in `ticket_classifications`.
- [ ] At least 30 days of data accumulated before M12 stories execute.

---

**Story: Pre-trained sentiment model (M2)**

*What:* Load a pre-trained sentiment model and validate on a 50-ticket manual sample against human labels. Deploy if agreement rate is above 70%.

*Why:* Sentiment intensity is an input to M5 routing (high-frustration tickets escalate) and a correlating signal in M12. Starting from a pre-trained model costs nothing and can be validated immediately against human labels — fine-tuning is only warranted if the pre-trained model is systematically wrong on this domain.

*Acceptance Criteria:*
- [ ] Pre-trained model selected and loaded via the shared embedding/inference service.
- [ ] 50-ticket manual sample labeled by a human reviewer.
- [ ] Agreement rate with human labels documented.
- [ ] If agreement rate > 70%: model deployed, scoring all new tickets.
- [ ] If agreement rate ≤ 70%: fine-tuning story created for the next sprint (conditional story below).

---

**Story: Sentiment pipeline integration**

*What:* Wire M2 into the classification pipeline — store `sentiment_score` and `sentiment_intensity` in `ticket_classifications`. Wire intensity into the M5 router's escalation check.

*Why:* A sentiment score that is not wired into any decision is just a stored number. It needs to influence routing and be available for M12 correlation to have value.

*Acceptance Criteria:*
- [ ] `sentiment_score` and `sentiment_intensity` fields in `ticket_classifications` populated for all new tickets.
- [ ] M5 router escalates high-intensity tickets in the `access` category (and any other categories identified from EDA).
- [ ] Routing change logged and configurable.

---

**Story: M12 CSAT Correlator**

*What:* Implement a daily SQL aggregation job that joins `ticket_csat`, `ai_draft_feedback`, and `ticket_classifications` to produce a ranked table of ticket types by average CSAT, agent acceptance rate, and M11 similarity score.

*Why:* This is the output that makes everything actionable. Without the ranked table, no one knows which ticket types are systematically failing customers — the team is flying blind on where to direct prompt tuning, template authoring, or retrieval improvements.

*Acceptance Criteria:*
- [ ] Daily aggregation job running. Output table queryable.
- [ ] Per-category CSAT, acceptance rate, and similarity score available.
- [ ] Top 5 worst-performing categories identified and reported in first week of operation.

---

**Story: Wire CSAT into few-shot prioritization**

*What:* Adjust M10's ANN retrieval to apply a CSAT-based weight when scoring few-shot candidates, deprioritizing draft examples from ticket categories with consistently low CSAT.

*Why:* The few-shot pool currently treats all accepted drafts as equal positive signals. An agent can accept a draft that the customer finds unhelpful. Weighting by CSAT ensures the pool surfaces examples from ticket types the system is genuinely good at.

*Acceptance Criteria:*
- [ ] CSAT weight applied to M10 example scoring. Low-CSAT category examples retrieved less frequently.
- [ ] Acceptance rate and CSAT score tracked for the sprint after deployment. Direction of change documented.

---

**Story: M2 fine-tuning (conditional)**

*What:* Fine-tune M2 on CSAT-labeled tickets if the pre-trained model agreement rate was below 70% in the deployment story.

*Why:* A general-purpose sentiment model may not generalize to IT support ticket language. If it does not, fine-tuning on in-domain data with CSAT as the label is the correct fix.

*Acceptance Criteria:*
- [ ] Only executed if pre-trained model agreement rate < 70%.
- [ ] Fine-tuned model evaluated on the same 50-ticket manual sample.
- [ ] Improvement in agreement rate documented. Decision on deployment recorded.

---

### Sprint Sequence

```
Sprint 1:   Epic 1 (Dedup) || Epic 2 (Foundation)           — no dependencies, fully parallel
Sprint 2:   Epic 3 (Classification) || Epic 4 (Retrieval)   — both unblocked by Epic 2
            || Epic 6 (CSAT — CSAT ingest has no dep;        
               M2/M12 unblocked by Epic 2)                   — all three parallel
Sprint 3:   Epic 5 (Alerting — code sprint)                 — no code dependency; deploy gate below
Sprint X:   Epic 5 deploy                                   — triggered when 90-day embedding
                                                              data threshold is met (~Sprint 7+)
```
