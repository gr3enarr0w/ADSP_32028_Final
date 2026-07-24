# AI Helpdesk Agent — Demo Script
## Red Hat Atlassian Cloud Migration Support — Stage Environment

---

## SECTION 1: THE MODEL PIPELINE

### What this is

This is not a single AI — it's a coordinated pipeline of 11 models and algorithms, each responsible for one specific job. Every ticket that comes into <PROJECT_KEY> passes through all of them in sequence. The design philosophy: local models handle volume cheaply and quickly, Gemini handles reasoning and generation.

### The full pipeline, in order

**Step 1 — Ticket Ingest**
No AI here. The system pulls from `servicedeskapi/request?requestStatus=ALL_REQUESTS` on an incremental JQL watermark — only tickets updated since the last cycle. First run is a full sync; subsequent runs are delta-only. Runs every 5 minutes.

**Step 2 — Classification (Category Routing)**

Three models run as a weighted ensemble:

| Model | Type | Where | Weight |
|---|---|---|---|
| Logistic Regression | Linear classifier | Local CPU | 10.2% |
| LightGBM | Gradient Boosting | Local CPU | 12.4% |
| all-MiniLM-L6-v2 | DistilBERT-based sentence embeddings | Local CPU (PVC) | 77.4% |

The weights are Optuna-tuned (5-fold stratified cross-validation). The ensemble predicts ticket category: Access, Permissions, Configuration, Workflow, Notifications, Data, Integration, Performance, UI/UX, Other.

If the predicted category has fewer than 10 training examples → fall back to **Gemini 2.5 Pro** for classification. This handles rare or novel categories where the local ensemble lacks signal.

Output: category + confidence score (0–1).

**Step 3 — Sentiment Scoring**

Runs in parallel with classification. Determines customer frustration level to gate escalation.

Primary: **Gemini 2.5 Pro** zero-shot — 95.7% Macro-F1 on 50-ticket labeled sample.

Fallback (if Gemini unavailable):
- **cardiffnlp/twitter-roberta-base-sentiment-latest** (RoBERTa, 500M params)
- **j-hartmann/emotion-english-distilroberta-base** (DistilRoBERTa)
- Both run locally on PVC. Soft-vote ensemble: 83.99% Macro-F1, 250ms latency.

Emotion mapping: anger/disgust/fear/sadness → NEGATIVE; joy → POSITIVE; neutral/surprise → NEUTRAL. Output: sentiment label + intensity score 0–1.

Sentiment gate: intensity ≥ 0.70 → suppress auto-draft, flag for human. Blocks ~17.8% of tickets to prevent the agent from responding to highly distressed customers.

**Step 4 — Resolution Summary (on resolved/closed tickets)**

**Gemini 2.5 Pro** reads: ticket summary + description + last 3 agent comments + Jira resolution field → produces 1–2 sentence resolution summary stored in the DB.

Purpose: used as the primary text signal for FAQ gap analysis and semantic dedup. Resolved tickets with summaries are 3x more useful for retrieval than tickets without.

8,009 resolution summaries generated across the existing corpus.

**Step 5 — Retrieval (when a response is needed)**

Two retrieval systems run in parallel and their results are fused:

- **BM25** (keyword frequency-based): fast, no API cost, handles exact terminology well
- **gemini-embedding-001** (Vertex AI, 3072-dimensional dense vectors): semantic matching, handles paraphrase and synonyms

Fusion via Reciprocal Rank Fusion (RRF) — not a simple average, weights rank position from each system.

Top candidates are re-ranked by a **cross-encoder** (cross-encoder/ms-marco-MiniLM-L-6-v2) running locally — this is more computationally expensive but significantly more accurate than bi-encoder similarity for pairwise relevance judgments.

The retrieval corpus (1,231 items) includes: generated FAQ articles, KB articles from HUB + OMEGA Confluence spaces, resolved ticket resolutions, and Atlassian documentation.

**Step 6 — Draft Generation**

**Gemini 2.5 Pro** receives:
- Ticket summary + description + latest customer reply
- Top retrieval results (KB articles, FAQ entries, past resolutions)
- 5 few-shot examples from ANN index of past approved agent responses, weighted by CSAT score
- Structured prompt variant "structured_context" (wins 5-fold CV at p < 0.05 vs baseline)

Output: structured draft with three response types:
- `self_service` — customer can resolve independently; steps provided
- `admin_action` — agent needs to take action; admin steps + customer-facing message
- `needs_info` — insufficient info; clarifying questions

The few-shot examples come from the emoji feedback loop: agents who rate drafts with ✅ 👤 🔧 ❌ 🔄 ❓ train the next generation of drafts. `both_good` examples are highest priority; `both_bad` and `wrong_type` are explicitly excluded.

**Step 7 — Self-Check**

Before posting, **Gemini 2.5 Flash Lite** (faster, cheaper) validates the draft against three criteria:
1. `type_correct` — does the response type match the ticket type?
2. `addresses_question` — does the draft actually address what the customer asked?
3. `sufficient_context` — is there specific, actionable information (not just "contact your admin")?

If any criterion fails → draft is suppressed, logged with reason, retried next cycle with different retrieval. This is the last gate. It correctly blocks drafts that Gemini generated using wrong-domain content.

**Step 8 — FAQ Gap Analysis (Saturday UTC only, or 50+ new resolved tickets)**

**Gemini 2.5 Pro** analyzes the full corpus of resolved tickets and existing KB coverage → identifies topic clusters with no FAQ coverage → generates draft articles.

Dedup before publishing:
- **MinHash** (128 permutations, Jaccard ≥ 0.80): structural similarity, fast
- **gemini-embedding-001**: semantic cosine similarity ≥ 0.95: catches same topic, different wording
- **cross-encoder/ms-marco-MiniLM-L-6-v2**: reranks borderline cases

68 articles published (stage). Target after threshold tuning: ~15 consolidated articles.

**Step 9 — KB Indexing (weekly)**

Crawls HUB and OMEGA Confluence spaces (including draft pages). Extracts plain text, builds embeddings via **gemini-embedding-001**. Stored in `kb_articles` table with embeddings cached — restart doesn't require re-embedding (warm start ~1 second vs cold 20+ minutes).

Current: 222 KB articles indexed, 1,231 total corpus items in retrieval index.

**Step 10 — Monitoring**
- **Alerting**: volume anomaly detection (control chart ±2σ) + topic clustering (HDBSCAN primary, K-means fallback) alerts via OpsGenie when ticket volume spikes or new cluster emerges
- **MRR monitor**: weekly retrieval quality snapshot — measures Mean Reciprocal Rank@10 to detect KB drift

---

## SECTION 2: THE THREE DRAFT EXAMPLES

Open: `https://stage-<YOUR_DOMAIN>.atlassian.net/browse/<PROJECT_KEY>`

### <PROJECT_KEY>-7563 — ✅ The good one

**Ticket**: Agent can't access "Quantum Leap" Confluence space  
**Response type**: `needs_info`  
**What the agent did**: Found KB content about Confluence space permissions. Generated a clarifying response asking for: space name confirmation, desired access level, and whether the user's manager has approved. Self-check passed all three criteria.  
**Why it worked**: The ticket is about Atlassian Cloud permissions — exactly what the KB covers. Retrieval found relevant articles. Draft was specific and actionable. Self-check confirmed it addressed the actual question.  
**This is what the pipeline looks like at its best.**

### <PROJECT_KEY>-7565 — ⚠️ Partially right

**Ticket**: Permissions issue, another Atlassian-context request  
**Response type**: `needs_info`  
**What the agent did**: Acknowledged the request and said "a team member will follow up." Too generic. No specific next steps. Self-check passed (technically addresses the question) but operationally wrong.  
**What it should have done**: For permissions requests that require another team to act, the correct response is explicit redirection — "Please reach out to your project lead or team manager, who can submit an access change request." The draft was vague where it should have been specific.  
**Root cause**: The legacy FAQ fallback found content that was relevant in topic but too generic in specifics. The few-shot examples for this pattern haven't been rated yet — with more emoji feedback, this improves.

### <PROJECT_KEY>-7560 — ❌ Wrong

**Ticket**: "Request access to ocp-operator-sdk GitHub repository"  
**Response type**: `admin_action`  
**What the agent generated**: Detailed GitHub admin steps — find the user's GitHub ID, add them to an org team, grant push access via GitHub's API, check branch protection rules. With source links from GitHub documentation.  
**Why this is wrong**: This helpdesk (<PROJECT_KEY>) does not manage GitHub repository permissions. The correct response is: "Please reach out to your project lead or the repository owner to request access." The agent doesn't know that GitHub repo access is outside its operational scope.  
**Why it happened**: Hybrid retrieval returned empty (no GitHub content in our KB). Legacy fallback found GitHub documentation from the `atlassian_docs` table. Gemini generated plausible admin steps using that content. Self-check evaluated the draft as technically correct (steps make sense for GitHub) but didn't know these steps aren't what this helpdesk does.  
**What fixes this**: Two paths — (1) add a corrective RAG gate that skips Gemini when retrieval finds wrong-domain content, and (2) the template system should have caught this as an `Access` category ticket and used the generic "contact your project lead" template. The template WAS available but the slot-filling was producing garbage output due to a regex bug (now fixed). Next cycle should use the template.  
**This is exactly why we're in stage.**

---

## SECTION 3: FAQ ARTICLE GENERATION + THE COSINE THRESHOLD

### How articles are generated

Open: `https://stage-<YOUR_DOMAIN>.atlassian.net/wiki/spaces/HUB/pages/120947227`

Every Saturday UTC, or sooner if 50+ new resolved tickets arrive, the system:
1. Reads every resolved ticket's resolution summary
2. Clusters them by topic using Gemini
3. Compares each cluster against existing KB coverage
4. For gaps: generates a draft FAQ article — question, answer, steps, known limitations
5. Deduplicates against all existing articles before publishing
6. Publishes to Confluence as `status: current`, restricted to the Red Hat One reviewer group

**Current state**: 68 articles published on stage, all about real ticket patterns from the real <PROJECT_KEY> corpus. Every article has a Q&A structure, with specific technical details (actual Jira field names, real Rover group references, real workflow transition names).

### The cosine threshold — live demo of the knob

The dedup threshold controls how similar two articles must be before one is suppressed as a duplicate.

**Current setting: 0.95** (95% semantic similarity = duplicate)

At 0.95, the system is relatively permissive. It let through:
- "User Access & Login"
- "User Access to Atlassian Cloud"
- "Access Management and SSO"
- "Cloud Access & Provisioning"
- "Initial Login & Account Management"

...all covering essentially the same topic from slightly different angles. An Opus-level review found 8 near-duplicate articles in the Access cluster alone.

**If we raise to 0.90** (90% similarity = duplicate): the system would have blocked most of those as too similar and published 1–2 consolidated articles instead of 8 thin ones. Industry standard for FAQ-style content is 0.90–0.95. We're currently at the permissive end.

**The tradeoff**: Higher threshold = fewer, richer articles, better retrieval precision. Lower threshold = more coverage breadth, more noise in retrieval.

**This is a tunable parameter** — raising it from 0.95 to 0.90 is a one-line change in `pipeline.yaml`. After the threshold is raised, the system would run gap analysis again (next Saturday) and generate consolidated articles. This is on the story list — team decides when.

The articles are currently restricted to the Red Hat One reviewer group. Once a reviewer approves, they remove the restriction and the article becomes publicly visible in HUB.

---

## SECTION 4: VS ATLASSIAN VIRTUAL SERVICE AGENT

### What the Virtual Service Agent is

Atlassian's built-in Virtual Service Agent (Premium/Enterprise) is a customer-facing deflection tool that intercepts requests **before they become tickets**. It operates in the customer portal, Slack (via Atlassian Assist), and Microsoft Teams. It has two modes:

1. **Intent flows** — manually configured conversation trees. You write training phrases (Atlassian recommends 20+ per intent), build a branching dialogue, and the agent matches incoming messages to those intents. Good for well-defined, repeatable processes: password reset, hardware requests, software access workflows.

2. **AI Answers** — connects to a linked Confluence knowledge base and uses Atlassian Intelligence (Gemini under the hood) to answer questions conversationally. Good for "is this documented?" questions.

### Side-by-side

| Dimension | JSM Virtual Service Agent | This Agent |
|---|---|---|
| **Who sees it** | Customer (self-service) | Agent (internal comment, invisible to customer) |
| **When it runs** | Before a ticket exists — intercepts at portal/Slack | After a ticket exists — drafts the agent's response |
| **Setup** | Manual: write 20+ training phrases per intent, build flow branches | Automatic: mines every resolved ticket, no intent configuration |
| **KB requirement** | **Must be public** — knowledge base space requires "All logged-in users" view access | KB can be restricted — our articles are gated to Red Hat One reviewers |
| **Learns from resolved tickets** | No — reads KB articles only, static until you update them | Yes — every resolved ticket's resolution summary feeds into gap analysis and few-shot examples |
| **Handles ambiguity** | Escalates if no intent match (creates a ticket) | Generates a draft with sources cited, self-checks quality before posting |
| **Skill required to configure** | Admin writes each conversation flow — ongoing maintenance as processes change | Engineers configure once, system self-maintains |
| **Output** | Customer-facing: deflects or creates ticket | Agent-facing: draft comment for agent to review, edit, and send |

### The KB visibility conflict

The Virtual Service Agent requires your linked knowledge base to be visible to **all logged-in users**. This is a hard requirement — AI Answers won't index restricted pages.

Our approach deliberately restricts new FAQ articles to the Red Hat One reviewer group until a human approves them. Running both simultaneously means:
- Either you make KB public (Virtual Agent works, our restriction model breaks)
- Or you keep KB restricted (our model works, Virtual Agent can't index it)

They pull in opposite directions on the same Confluence space.

### The conversation accounting

**What counts as an "assisted conversation"** (from Atlassian's official definition):
- *Matched conversations*: any conversation where the Virtual Agent matches an intent, whether it resolves it or escalates to an agent. Even if the customer gets routed to a human, it counts.
- *AI resolved conversations*: Virtual Agent answers using AI Answers and the customer either says it helped or abandons (auto-close triggers).

**Limit**: 1,000 assisted conversations per month — **per Atlassian Cloud site**, not per project, not per service desk. Shared across all JSM projects on `<YOUR_DOMAIN>.atlassian.net`. If you have 5 service desks all running Virtual Agent, they pull from the same 1,000 conversation pool.

Enterprise gets volume discounts on overages (list price: $0.30/conversation). Exact negotiated rate varies.

**Cost model comparison**:
- Virtual Agent: per-deflected-conversation (you pay each time a customer self-serves before creating a ticket)
- This agent: Vertex AI API costs at our GCP project level — per Gemini generation call, not per ticket. No Atlassian billing, no Rovo credit consumption. Runs independently of Atlassian's AI metering.

### Why not both simultaneously

1. **KB conflict**: visibility requirements point in opposite directions
2. **No coordination**: two AI systems answering the same domain questions with no shared signal — which one do agents rate? Which one learns from feedback?
3. **Maintenance duplication**: intent flows need ongoing tuning as processes change. Our system updates automatically from resolved tickets. Running both means maintaining two learning loops.
4. **Different problems**: Virtual Agent deflects before the ticket exists. This agent improves how agents handle the tickets that make it through. Solve one well before layering in the other.

"Evaluate each on its own merits. If Virtual Agent deflects 30% of <PROJECT_KEY> volume, and this agent cuts per-ticket handle time by 40%, they don't compete — but prove each independently first."

---

## SECTION 5: NEXT STEPS

Point to `FIXES_AND_DECISIONS.md` in the repo. The document is organized by type, not by priority — the team decides what to pursue.

**Blockers before prod** (must resolve regardless):
- Fix Confluence page restriction API (articles currently visible to all HUB members)
- Complete Privacy Impact Assessment (ANTSE-493)
- KB access control review (are any restricted pages being indexed?)
- ITOP ticket for GitHub Actions egress wildcard (ANTSE-553)
- Clean up 182 orphan draft pages

**Stories ready to build** (clear implementation, team decides when):
- Raise dedup threshold to 0.90 + multi-Q&A articles (FAQ quality overhaul)
- Add 3–5 KB articles for non-Atlassian access patterns (GitHub, OCP)
- Prod → Stage ticket sync for real-traffic model evaluation over the quarter
- Stale KB content detection and auto-archive (cloud vs DC content)
- Ingest timestamp watermark, resync API, metrics endpoint

**Spikes** (research needed first, team decides if worth pursuing):
- LLM cost/performance benchmarking after FAQ cleanup (Flash vs Pro, DeepSeek, model fine-tuning)
- Corrective RAG (skip Gemini when retrieval quality is insufficient)
- HF model startup time options (lazy load, quantization, sidecar)
- Gemini semantic response caching
- GraphRAG + multi-hop retrieval
- Autonomous ticket resolution (agentic AI)
- Real-time data access (live API calls during generation)

"Stage is running. Prod is a decision, not a technical milestone."

---

*Links used during demo:*
- Stage Jira: `https://stage-<YOUR_DOMAIN>.atlassian.net/browse/<PROJECT_KEY>-7563`
- Stage Confluence KB: `https://stage-<YOUR_DOMAIN>.atlassian.net/wiki/spaces/HUB/pages/120947227`
- Pipeline code: `github.com/agile-tech-sol/ai-helpdesk-agent`
- Decisions doc: `FIXES_AND_DECISIONS.md` in repo root
- Virtual Agent product guide: `https://www.atlassian.com/software/jira/service-management/product-guide/tips-and-tricks/virtual-agent`
