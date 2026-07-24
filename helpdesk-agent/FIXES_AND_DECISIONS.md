# Fixes and Architectural Decisions Log
<!-- Running log of bugs found, root causes, and fixes applied. Add new entries at the top. -->

---

## Calls for Action — Research Spikes & Next Steps

> **How to use this section**: Each item is a call for research or a decision needed before implementation. **Blockers** are items that must be resolved before prod — everything else is a research spike or call for action where the team decides whether and when to pursue it. Priority is not imposed on research items — that is for the team to determine after reviewing findings. Bundled items share a spike. Review existing ANTSE tickets before creating new ones.

---

### BLOCKERS — Must resolve before prod

**[STORY] FAQ-4 + PUBLISH-1: Fix Confluence page restrictions before prod**  
Pages currently published publicly to all HUB members. `_apply_page_restrictions()` returns 200 but sets empty group restrictions — v1 API payload format wrong. Also: verify publisher body format works reliably end-to-end.  
*Research needed*: Correct Confluence v1 restriction API payload. Test with group ID vs group name. Document correct format.  
*Decide*: Which group owns review? What happens after reviewer approves — auto-publish or manual?  
*Existing ticket*: None — create.

**[TASK] FAQ-3: Clean up 182 orphan draft pages + test page**    
182 draft children under `120947227` from failed generation runs. Test page `121995580` published. Not urgent for stage but must be clean before prod.  
*Action*: DELETE all `status=draft` children via Confluence API. Delete `121995580`.  
*Existing ticket*: None — create.

**[TASK] INFRA-1: ITOP ticket for *.actions.githubusercontent.com egress wildcard**  
Current CIDR workaround (20.102.0.0/17) is fragile. Human must file at red.ht/itocp-support-ticket.  
*Existing ticket*: ANTSE-553.

**[TASK] PROD-1: Privacy Impact Assessment (PIA)**  
Required before prod. AI agent reads/stores Jira ticket content including PII.  
*Existing ticket*: ANTSE-493 — assigned Clark Everson, To Do.

**[SPIKE] RESEARCH-6: KB access control — data leakage prevention**  
Currently indexing ALL HUB/OMEGA content. Restricted Confluence pages could surface in drafts to users without access. Must understand exposure before prod.  
*Research needed*: What content in HUB/OMEGA is access-controlled? Does current indexing pull restricted pages? What label/CQL filter prevents this?  
*Decide*: Label-based filter (`ai-kb`, `helpdesk-approved`) or domain centroid relevance scoring?  
*Bundle with*: KB-2.

---

### STORIES — Known implementation, ready to build

**[STORY] PROD-SYNC-1: Sync real prod tickets to stage for model evaluation**
As-is: stage has only synthetic demo tickets and old migrated data. The model is being evaluated and improved against synthetic tickets, not real production helpdesk traffic.  
What to build: A one-way sync from `<YOUR_DOMAIN>.atlassian.net` <PROJECT_KEY> → `stage-<YOUR_DOMAIN>.atlassian.net` as new tickets come in (or on a daily batch). Anonymize/scrub PII before sync. Stage becomes a real evaluation environment — agents can see actual draft quality on real tickets, emoji ratings improve the model, and the improvement cycle runs over the full quarter before prod launch.  
*Why a story not a spike*: The ingest pipeline already exists. This is a new data source pointing at prod, with a PII scrub pass, writing to the stage DB. Implementation is clear.  
*Timeframe*: Evaluate over a quarter (or less) before prod release so the model trains on real traffic patterns, not synthetic ones.  
*Dependencies*: PIA (PROD-1) must complete first — reading prod tickets requires privacy clearance.  
*Existing ticket*: None — create.

**[STORY] KB-STALE-1: Auto-detect and archive stale/irrelevant KB content**
As-is: The KB indexes all HUB/OMEGA content including articles written for Data Center (pre-migration), outdated procedures, and content that no longer applies to the current cloud-only deployment.  
What to build: A classification step in `_phase_kb_index` that evaluates each article against the current instance context (cloud-only, stage-redhat, post-migration). Articles that are DC-specific, reference deprecated tools, or cover procedures that no longer apply get flagged. Flagged articles are auto-archived (Confluence archive, not delete) and removed from `kb_articles` so they don't surface in retrieval.  
*Why a story not a spike*: The detection logic is clear — use Gemini to classify each article as cloud-relevant vs DC-specific/stale, based on a prompt that describes the current environment. The archiving mechanism (Confluence archive API + remove from kb_articles) is a known implementation.  
*Signals for staleness*: article mentions "Data Center", "server", "on-premise", "pre-migration", references products/versions no longer in use, or describes configuration paths that don't exist in cloud.  
*Existing ticket*: ANTSE-576 (detect stale KB articles) — update description to include auto-archive.

**[STORY] FAQ-1 + FAQ-2: FAQ quality overhaul**  
*Bundle*: These three must happen together — cleanup drives re-evaluation.  
- FAQ-1: Raise cosine dedup threshold 0.82 → 0.90 (industry standard for FAQ content). Also evaluate Jaccard 0.80 → 0.85. Research confirms 0.90 is sweet spot.  
- FAQ-2: Generate 3-5 Q&A pairs per article instead of 1. 68 thin articles → ~12 rich ones.  
- EVAL-1: After cleanup, rerun full eval: dedup F1/Recall, draft self-check pass rate, KB retrieval MRR@10. Measure before/after.  
*Research needed*: What threshold produces ~15 distinct articles from current corpus? What prompt structure produces the richest multi-Q&A output?  
*Decide*: Threshold value, article structure (H2 per question vs single article with sections), max articles per Saturday run.  
*Existing tickets*: None — create.

**[SPIKE] KB-1 + KB-2: KB content relevance and coverage**  
*Bundle*: Both are about what goes INTO the KB.  
- KB-1: Add 3-5 articles for non-Atlassian access patterns (GitHub org access, OCP namespace). ~40% of real tickets are external tool access; KB has zero coverage.  
- KB-2: Filter HUB/OMEGA noise — team pages, project management docs create retrieval noise. Label-based CQL filter or domain centroid scoring.  
*Research needed*: Which HUB/OMEGA pages are helpdesk-relevant? What labels exist? Is a label strategy feasible without touching every page?  
*Decide*: Manual KB articles for GitHub/OCP patterns vs. automated scraping of internal wikis. Label strategy vs. parent-page-scoped indexing.  
*Existing tickets*: ANTSE-576 (stale KB detection), ANTSE-577 (external sources).

**[SPIKE] PIPELINE-2: Corrective RAG**  
When retrieval returns empty or wrong-domain content, skip Gemini entirely. Currently wastes full API call on guaranteed self-check failure.  
*Research needed*: What cosine similarity threshold between retrieval results and ticket summary indicates "retrieval failure"? Test on real failing tickets.  
*Decide*: Threshold value, fallback behavior (template-fill vs. skip vs. ask-for-info response).  
*Existing ticket*: None — create.

**[SPIKE] EVAL-2 + EVAL-3: LLM cost/performance benchmarking after FAQ cleanup**  
*Bundle*: Run these together after KB is clean.  
- EVAL-2: Benchmark Gemini 2.5 Flash vs Pro for draft generation on real tickets. Test DeepSeek V3 for classification. Model cost-per-draft before/after cleanup. Measure Anthropic prompt caching savings.  
- EVAL-3: Re-evaluate fine-tuned embedding on cleaned corpus. Test cross-encoder reranking (ms-marco-MiniLM-L-12-v2) on top of bi-encoder.  
*Research needed*: Run evaluation harness on 100-200 real tickets across model variants. Measure self-check pass rate, draft quality score, API cost per draft.  
*Decide*: Which model for each pipeline stage in prod. Budget for fine-tuning vs. off-shelf models.  
*Existing tickets*: ANTSE-561 (embedding improvement), ANTSE-557 (human gold set).

**[SPIKE] KB-3: Verify other source pipelines are active and working**  
Atlassian public docs (505 entries in DB) — are they in retrieval or disabled end-to-end? Slack signals disabled — what would enable them? Verify ANTSE-577 scope.  
*Research needed*: Check `atlassian_docs` table — are these rows used in BM25/dense retrieval or filtered out? What would the recall improvement be if enabled?  
*Decide*: Enable atlassian_docs in retrieval now (quick win) or wait for ANTSE-577 configurable sources implementation?

---

### SPIKES — Research needed before deciding

**[SPIKE] CACHE-1 + SPEED-2: HF model startup time options**  
*Bundle*: Same goal — faster pod restart.  
- CACHE-1: Audit every expensive computation. Current gaps: HF sentiment models (~10-15 min), Gemini response caching (none), _docs list DB read on startup.  
- SPEED-2: Options for HF model startup: lazy load, INT8 quantization, TorchScript serialization, separate sentiment sidecar.  
*Research needed*: Profile exact startup time contribution per model. Test lazy loading impact on first-request latency. Evaluate quantization quality loss on <PROJECT_KEY> sentiment task.  
*Decide*: Which approach for prod (latency vs. startup time tradeoff).  
*Existing ticket*: ANTSE-563.

**[SPIKE] SPEED-1: Gemini semantic response caching**  
GenCache pattern: recurring ticket patterns (access requests, workflow issues) get cached draft structures with slot values updated for new ticket. Could reduce Gemini calls 50-80%.  
*Research needed*: What % of <PROJECT_KEY> tickets are semantically near-duplicate across weeks? Measure cache hit rate on 30 days of ticket history.  
*Decide*: Build vs. use LangChain semantic cache. Cache lifetime. How to handle slot value updates in cached drafts.

**[STORY] SPEED-3: Parallel phase execution**  
faq + kb_index don't depend on each other — can run in parallel. alerting + mrr_monitor also independent. Known architecture change with clear implementation path.  
*Profile first*: Measure actual phase durations over 5 cycles before implementing to confirm expected gain.

### STORIES — Known implementation, ready to build (continued)

**[STORY] RESPONDER-1: Auto-draft sweep metrics endpoint**  
Need `/api/sweep/stats` showing per-cycle: gate rejections by type, drafts attempted vs posted, self-check pass rate. Already partially done via logs.  
*Existing ticket*: ANTSE-582.

**[STORY] INGEST-1: Timestamp watermark for incremental ingest**  
Replace date-only watermark with full ISO datetime. Known implementation.  
*Existing ticket*: ANTSE-571.

**[STORY] INGEST-2: Resync API endpoint**  
`POST /api/admin/resync` to force full re-ingest. Known implementation.  
*Existing ticket*: ANTSE-570.

**[STORY] PIPELINE-4: Pipeline phase timing in /api/health**  
Per-phase timing visible in health endpoint. Known implementation.  
*Existing tickets*: ANTSE-580/581/583.

**[STORY] PUBLISHER-1: orjson for ingest JSON parsing**  
One-line drop-in, 3-10x faster. Known implementation.

### TASKS — Operational actions, no research or design needed

**[TASK] PIPELINE-3: Scale reinforcement learning from emoji feedback**  
Infrastructure built. Needs agent adoption — rating guide, metric tracking.

**[TASK] PROD-2: Slack app registration and /jsm-assist command**  
Blocked on Slack admin approval.  
*Existing ticket*: ANTSE-489.

**[TASK] SPEED-4: Verify Vertex AI batch size in embedding service**  
Check `_VERTEX_BATCH_SIZE` is set to 250. One-line audit.

**[TASK] MODEL-2: Human gold set for dedup calibration**  
Annotate 20-30 ticket pairs, compute Cohen's kappa.  
*Existing ticket*: ANTSE-557.

---

### SPIKES — Strategic research (long-term)

**[SPIKE] RESEARCH-1 + RESEARCH-2: GraphRAG + Multi-hop retrieval**  
*Bundle*: Both need knowledge graph infrastructure; evaluate together.  
GraphRAG builds entity/relationship graph during ingestion — surfaces related articles that vector similarity misses (SSO → Rover groups → identity provider chain). Multi-hop RAG decomposes complex cross-system tickets into 2-3 sequential retrieval steps.  
*Research needed*: PoC with 100 <PROJECT_KEY> tickets. Measure recall improvement over current BM25+dense. Estimate ingestion cost and graph storage overhead.  
*Decide*: Graph DB (Neo4j, Amazon Neptune) vs in-Postgres graph vs in-memory. Which ticket types benefit most?

**[SPIKE] RESEARCH-3: Autonomous ticket resolution**  
For high-confidence well-understood patterns, agent approves + executes without human review (provision access, update field, send notification). Reduces MTTR 30-40% in production deployments.  
*Research needed*: Which ticket categories are safe for autonomous execution? What safety controls and audit trails are required? What Jira/Confluence API actions can be safely automated?  
*Decide*: Start with read-only automation (close-with-summary) before write automation (provision access).

**[SPIKE] RESEARCH-4: Real-time data access**  
Live API calls during draft generation for data that can't wait for weekly kb_index — project admin lookups, current Rover group membership, live Confluence page state.  
*Research needed*: Which ticket types ask questions answerable only with live data? What's the latency impact of live API calls during draft generation? Rate limit exposure?  
*Decide*: Which data sources, caching TTL for live results, egress requirements.

**[SPIKE] RESEARCH-5: Modular RAG architecture**  
Decouple indexing/retrieval/generation/eval as composable building blocks. Enables A/B testing without full pipeline changes. Required for clean model benchmarking (EVAL-2/3).  
*Research needed*: What's the minimal interface contract between modules? Which current coupling points are most painful (embedding model changes, retrieval strategy swaps)?  
*Decide*: Do this as a refactor before prod or post-prod as tech debt. Is the current coupling actually painful enough to justify the work now?

**[SPIKE] MODEL-1: Improve dedup recall from 45% to 80%+**  
After FAQ cleanup, re-evaluate. Options: fine-tuned embeddings, cross-encoder reranking, expanded labeled corpus.  
*Existing ticket*: ANTSE-561.

**[SPIKE] RESPONDER-2: Remove 50-ticket cap on resolution_summary**  
Cap is intentional but revisit at prod scale. Research: what does cycle time look like at full volume with no cap?  
*Existing ticket*: ANTSE-574.

---

### MONITOR — Stage demo tracking

**[MONITOR] DEMO-1: Confirm drafts posting on demo tickets**  
Template fix + credential fix deployed. Demo tickets 7560-7581 should receive drafts next few cycles.  
*Watch*: Check `ai_draft_feedback` table after each cycle. Atlassian-topic tickets should get KB-grounded drafts. GitHub/OCP tickets use template-fill.  
*Existing ticket*: ANTSE-490.

### FAQ-2: Generate 3-5 Q&A pairs per article instead of 1
**Finding**: All 68 published articles contain a single Q&A pair. "Permissions" has 8 thin pages each answering one question — should be 1-2 consolidated articles with 3-5 questions each.  
**Proposed fix**: Change generator to produce 3-5 Q&A pairs per theme in one Gemini call, publish as one consolidated article. Expected result: ~12 rich articles covering same ground as current 68.

### FAQ-3: Clean up 182 orphan draft pages in HUB space
**Finding**: Opus audit found 182 draft child pages under parent `120947227` left over from multiple failed generation runs (wrong body format runs, draft-status runs). Not visible to users but clutters admin view.  
**Action needed**: DELETE all children of `120947227` with `status=draft` via Confluence API. Also delete test page `121995580` ("TEMP TEST DELETE ME").

### FAQ-4: Fix Confluence page restriction API — articles currently public
**Finding**: `_apply_page_restrictions()` in `generation/publisher.py` returns HTTP 200 but applies empty restrictions. All 68 published articles are visible to all HUB space members instead of Red Hat One group only.  
**Root cause**: v1 `PUT /wiki/rest/api/content/{id}/restriction` payload format is accepted but sets no restrictions. Correct format not yet identified.  
**Status**: Acceptable for stage demo (stage-<YOUR_DOMAIN>.atlassian.net is internal). Must fix before prod deploy.

### PIPELINE-1: Synthetic demo ticket generator doesn't filter corpus by domain
**Finding**: Few-shot sampling from the real <PROJECT_KEY> corpus pulled non-Atlassian tickets (GitHub repo access, RHEL/OCP system access). A prompt constraint was added as a short-term fix.  
**Proposed fix**: Pre-filter few-shot examples by embedding similarity to Atlassian Cloud domain, not just DB category label. Add RAGAS-style quality scoring post-generation, persona prompting (Jira admin / Confluence editor / integration developer personas).  
**Research findings**: Few-shot + CoT is best combination for domain-specific generation. Post-generation domain relevance scoring (cosine similarity to Atlassian Cloud domain centroid) prevents off-topic articles. Weekly drift monitoring (Chi-square test on category distribution vs real tickets) detects generator quality degradation.

### PIPELINE-2: Corrective RAG — skip Gemini when retrieval quality is insufficient
**Finding**: When hybrid retrieval returns empty and legacy fallback finds wrong-domain content, Gemini generates an off-topic draft that self-check correctly rejects. Full Gemini API call wasted on guaranteed failure.  
**Proposed fix**: Before calling Gemini, compute cosine similarity between each retrieval result and the ticket summary. If all results below a relevance threshold (e.g., 0.40), skip Gemini generation entirely and route to template-fill or skip. Log clearly: "Skipped Gemini — retrieval quality insufficient for {ticket_key}".

### PIPELINE-3: Reinforcement learning from emoji feedback needs scale
**Finding**: The emoji feedback → few-shot example pipeline (ANTSE-199/201) is implemented but has only 3 rated examples total. At scale, this will improve draft quality by training on approved examples and excluding bad ones.  
**Status**: Working as designed. Needs agent adoption — encourage agents to rate drafts with emoji to accelerate the feedback loop. Track `agent_feedback` distribution as a metric.

### PUBLISHER-1: orjson for faster JSON parsing in ingest
**Finding**: `ingest/oauth2lo.py` and Jira API response parsing uses stdlib `json`. `orjson` (Rust-backed) is 3-10x faster, one-line drop-in. Minor improvement (shaves seconds off ingest) but only meaningful optimization on the critical path.  
**Priority**: Low. Not blocking anything.

### INFRA-1: ITOP ticket for *.actions.githubusercontent.com wildcard egress
**Status**: Needs human action — file ITOP ticket at red.ht/itocp-support-ticket requesting platform team add EgressFirewall rule allowing `*.actions.githubusercontent.com` in `jira-messaging--runtime-ext`. Current CIDR workaround (20.102.0.0/17) is fragile — GitHub can reassign Azure allocations without notice.  
**Existing ticket**: ANTSE-553 tracks this.

### KB-1: KB content gap for external tool access patterns (GitHub, OCP, non-Atlassian)
**Finding**: Demo tickets and real <PROJECT_KEY> tickets include access requests for external tools (GitHub repos, OCP namespaces, RHEL systems). KB only covers Atlassian Cloud topics. Hybrid retrieval returns empty for these → legacy fallback finds wrong content → bad drafts.  
**Proposed fix**: Add 3-5 KB articles to HUB Confluence covering common non-Atlassian access request patterns: "how to request GitHub org access", "how to request OCP namespace access". These would be indexed on next weekly kb_index run and improve retrieval for ~40% of real ticket types.  
**Related**: ANTSE-577 (expanding external sources) covers configurable external source indexing long-term.

### KB-2: Mark and filter KB articles not relevant to Atlassian Cloud support
**Finding**: The KB indexes all pages in HUB and OMEGA spaces including content unrelated to helpdesk support topics (team pages, project management docs, etc.). These create retrieval noise — wrong content surfaces for helpdesk queries.  
**Proposed fix**: Add a label-based filter to kb_index CQL: only index pages with specific labels (e.g., `ai-kb`, `helpdesk-approved`) OR pages under the AI-Generated FAQ parent. Alternatively, add a relevance scoring step that drops KB articles below a cosine similarity threshold to the "Atlassian Cloud helpdesk" domain centroid.  
**Related**: ANTSE-576 (detect stale KB articles).

### KB-3: Verify other sources are correctly configured (ANTSE-577)
**Status**: Need to verify. ANTSE-577 tracks expanding FAQ RAG to configurable external sources — Slack signals, plugin docs, custom URLs via pipeline.yaml. Currently disabled pending implementation. Verify:
- Atlassian public docs (support.atlassian.com, developer.atlassian.com) — currently in `atlassian_docs` table with 505 entries, but disabled as FAQ source (future state)
- Slack signals — disabled pending quality validation
- Google Docs/Sheets/Slides — removed, Confluence is canonical KB
Need to confirm `atlassian_docs` 505 entries are being used in retrieval OR if they're disabled end-to-end.

### INGEST-1: Use ticket updated timestamp (not date) as incremental watermark
**Status**: ANTSE-571 tracks this. Currently `job_state` stores `last_run_date` as DATE only. JQL becomes `updated >= '2026-06-30'` which re-fetches all tickets updated since midnight, not since actual last run time. If pod restarts at 4pm after a 3pm cycle, re-fetches ~15hrs of tickets unnecessarily.  
**Proposed fix**: Store full ISO datetime in job_state, use precise watermark in JQL.

### INGEST-2: Resync API endpoint for stage environment refresh
**Status**: ANTSE-570 tracks this. Stage Atlassian environment refreshes every ~3 months, invalidating the incremental ingest baseline. Need `POST /api/admin/resync?project=<PROJECT_KEY>` to clear `job_state` last_run_date and trigger full re-sync.

### RESPONDER-1: Auto-draft sweep metrics endpoint
**Status**: ANTSE-582 — partially done (logging improvements deployed), metrics endpoint not yet built. Need `/api/sweep/stats` or health endpoint enrichment showing per-cycle: tickets found, gate rejections by type, drafts attempted vs posted, self-check pass rate.

### RESPONDER-2: Remove 50-ticket cap on resolution_summary per cycle
**Status**: ANTSE-574 tracks this. Currently capped at 50 tickets/cycle (~14h for 8,500 resolved tickets). Description says change `max_tickets=50` to `max_tickets=None`. Note: ANTSE-555 comments say the 50-ticket cap is intentional (keeps cycle under 100s). Revisit once cycle timing at full scale is validated.

### MODEL-1: Improve FAQ dedup recall from 45% to 80%+
**Status**: ANTSE-561 tracks this. Current Vertex AI gemini-embedding-001 dedup: F1=0.588, Recall=0.455. Score gap narrow (-0.186 separation). Approaches to investigate: fine-tuned embedding model, cross-encoder reranking, expand labeled pair corpus beyond current 348 pairs.

### MODEL-2: Dedup calibration human gold set
**Status**: ANTSE-557 tracks this. Current calibration uses word-overlap-based labels (circular — embedding model trained on lexical co-occurrence, so F1 measures self-consistency not real accuracy). Need: 20-30 human-annotated pairs, Cohen's kappa >= 0.80, hard negatives (overlap 0.20-0.40).

### PIPELINE-4: Pipeline cycle ETA and per-phase metrics
**Status**: ANTSE-583 tracks cycle ETA. Need estimate of remaining time during long-running phases (especially FAQ generation, kb_index). ANTSE-580/581 track per-phase timing in /api/health endpoint.

### PROD-1: Privacy Impact Assessment (PIA)
**Status**: ANTSE-493 — To Do, assigned to Clark Everson. Required before prod deploy. AI agent reads and stores Jira ticket content including PII fields. Need PIA completed and approved.

### PROD-2: Slack app registration and /jsm-assist command
**Status**: ANTSE-489 — New. Slack app not yet registered. /jsm-assist slash command not activated. Blocked on Slack admin approval.

### EVAL-1: Rerun full model evaluation pipeline after FAQ cleanup
**Context**: After dedup threshold raised (FAQ-1), multi-Q&A articles generated (FAQ-2), and orphan drafts cleaned (FAQ-3), the training corpus and KB will be substantially cleaner. Rerun the full evaluation pipeline to measure before/after impact:
- Dedup F1/Recall/Precision (before: 0.588/0.455/0.833)
- Draft self-check pass rate (currently ~30-40% based on observed rejections)
- End-to-end draft quality (emoji feedback distribution)
- KB retrieval MRR@10 (current baseline from mrr_monitor)
**Goal**: Establish whether cleaned FAQ corpus measurably improves draft quality and retrieval accuracy.

### EVAL-2: LLM cost/performance benchmarking — find optimal model mix
**Research findings**: 
- Gemini 2.5 Flash costs ~1/20th of Pro and handles RAG/summarization well at scale
- Anthropic prompt caching gives 90% discount on repeated system prompts — applicable to our draft generation prompts
- DeepSeek V3 is competitive for high-volume classification at fraction of Gemini cost
- Best practice: tiered model routing — Flash/cheap for classification, Pro for generation, Flash-Lite for self-check
**Proposed investigation**:
1. Benchmark Gemini 2.5 Flash vs Pro for draft generation quality on real <PROJECT_KEY> tickets
2. Test DeepSeek V3 for classification (currently using ensemble + Gemini fallback)
3. Measure Anthropic prompt caching savings on our fixed system prompts
4. Model the cost-per-draft before cleanup (bad KB, high rejection rate) vs after (clean KB, higher pass rate)
**Output**: Cost/quality matrix to inform which model to use for each pipeline stage in prod.

### EVAL-3: Fine-tuned embedding evaluation
**Context**: ANTSE-561 proposes fine-tuning gemini-embedding-001 on the labeled <PROJECT_KEY> pair corpus. After FAQ cleanup produces cleaner resolution summaries, re-evaluate whether fine-tuned embeddings improve dedup recall beyond the 45% baseline.
**Also evaluate**: cross-encoder reranking (ms-marco-MiniLM-L-12-v2) on top of bi-encoder retrieval — proven to significantly improve recall on pairwise similarity tasks.

### CACHE-1: Full cache verification and startup time audit
**Goal**: Verify every expensive computation is cached and survives pod restarts. Current cache status:
- ✅ Dense retrieval corpus embeddings (`responder_corpus_embeddings` DB table) — warm start ~1s
- ✅ KB articles with embeddings (`kb_articles.semantic_embedding`) — weekly refresh
- ✅ FAQ article embeddings (`generated_articles.semantic_embedding`)
- ✅ BM25 index — rebuilt from DB on each startup (fast, ~2s)
- ❌ HF sentiment/classifier models — still load from PVC on each restart (~10-15 min, ANTSE-563)
- ❌ Vertex AI embeddings for retrieval corpus — cached in DB but dense_retrieval._docs list rebuilt in-memory on each startup (fast now with cache, but _docs list requires a DB read of 1,231 rows)
- ❌ Gemini response caching — no semantic cache for repeated or near-identical queries
**Proposed work**: Audit each phase's cold-start cost, implement semantic response caching for Gemini calls (GenCache/LangChain pattern), profile HF model load time and explore lazy loading or model quantization.

### SPEED-1: Gemini semantic response caching
**Research**: GenCache (arxiv 2511.17565) proposes generative caching for structurally similar prompts. For this pipeline: if a ticket with similar summary was drafted last week and self-check passed, return the cached draft structure with slot values updated for the new ticket. Could save 50-80% of Gemini calls for recurring ticket patterns (access requests, workflow issues).
**Also consider**: Anthropic prompt caching — 90% cost reduction on repeated system prompt prefixes. If we use Claude for any stage, structure prompts to maximize cache hit rate.

### SPEED-2: HF model startup time (ANTSE-563)
**Current**: cardiffnlp + j-hartmann emotion models load from PVC, blocking readiness probe for 10-15 min on each pod restart.
**Options** (from scoping ticket ANTSE-563, not yet investigated):
1. Lazy load — don't load models at startup, load on first use. Pod becomes ready immediately, first sentiment call is slow.
2. Torch model quantization — INT8 reduces model size ~4x, load time ~2-3x faster
3. Pre-serialized model weights (torch.save torchscript) — skip Python class init overhead
4. Separate sentiment service — deploy cardiffnlp as a sidecar or separate lightweight deployment

### SPEED-3: Parallel phase execution for independent pipeline phases
**Current**: All 10 phases run sequentially. Some are independent: kb_index and faq phases don't depend on each other's output within the same cycle. alerting and mrr_monitor are also independent of responder.
**Proposed**: Introduce parallel phase groups:
- Group A (sequential, data-dependent): ingest → analysis → resolution_summary → feedback
- Group B (parallel): faq + kb_index (both read from DB, don't write to each other)
- Group C (sequential): responder (needs KB index complete) → export → alerting + mrr_monitor (parallel)
**Expected impact**: Reduce cycle time by ~20-30% by parallelizing kb_index (weekly, slow) with FAQ phase.

### SPEED-4: Batch Vertex AI embedding calls in dense_retrieval rebuild
**Current**: When dense_retrieval cache is cold (new docs added), embeds each document individually. Vertex AI `embed_content` supports batching up to 250 items per call.
**Current code**: `embed_batch()` in `services/embedding.py` already batches — verify `_VERTEX_BATCH_SIZE` is set to max (250) and not unnecessarily small.

### RESEARCH-1: GraphRAG / knowledge graph augmentation
**From research (2026 industry direction)**: Enterprise RAG is evolving toward GraphRAG — building a knowledge graph during ingestion (entities, relationships) alongside vector embeddings. During retrieval, graph traversal surfaces related articles that vector similarity would miss (e.g., "SSO issue" → "identity provider" → "Rover group sync"). TreeRAG and GraphRAG reduce hallucination by grounding generation in structured relationships.  
**Applicability**: <PROJECT_KEY> tickets often involve multiple related systems. A knowledge graph could connect "Confluence space permissions" → "Rover directory groups" → "identity provider sync" in ways that BM25+dense retrieval doesn't.  
**Priority**: Future research — not blocking current work.

### RESEARCH-2: Multi-hop RAG for complex tickets
**From research**: Multi-hop RAG decomposes a query, runs sequential retrieval, and aggregates results. For helpdesk: "I can't access the OCP console" might need hop 1 (OCP access procedures) + hop 2 (Rover group membership) + hop 3 (SSO login troubleshooting). Current single-hop retrieval misses cross-topic tickets.  
**Limit**: 2-3 hops before latency is prohibitive. Useful for 10-15% of tickets that span multiple systems.  
**Priority**: Medium-term after single-hop quality is solid.

### RESEARCH-3: Autonomous ticket resolution (agentic AI)
**From research**: Leading ITSM AI in 2026 goes beyond draft suggestions to autonomous resolution. Agents: diagnose root cause, take actions (provision access, update fields, send notifications), escalate only when human judgment needed. Reduces MTTR 30-40%, improves first-contact resolution.  
**Applicability**: For well-understood ticket patterns (e.g., "need access to Confluence space X") — if the agent can verify the request is valid and trigger the provisioning workflow directly via Jira API, no human review needed.  
**Current state**: We generate a draft for agent review. Next step: for high-confidence template tickets, allow the agent to approve + execute automatically.  
**Priority**: Long-term — requires workflow automation and safety review.

### RESEARCH-4: Real-time data access (live enterprise platform integration)
**From research (enterprise RAG gaps 2026)**: Basic RAG pipelines have ingestion lag — KB changes don't surface until the next kb_index cycle (weekly). Enterprise needs are shifting to real-time knowledge access: live Jira project status, live Rover group membership, live Confluence page state.  
**Applicability**: A ticket asking "who is the admin for project X" currently can't be answered because project admin data isn't in our KB. A real-time API call during draft generation would solve this.  
**Priority**: Medium-term. Requires egress to additional Atlassian APIs + response caching to avoid rate limits.

### RESEARCH-5: Modular RAG architecture (composable pipeline)
**From research**: Best-practice 2026 RAG treats retrieval, indexing, generation, and orchestration as composable building blocks that can be swapped independently. Current pipeline is tightly coupled — changing the embedding model requires touching kb_index, dense_retrieval, and dedup simultaneously.  
**Applicability**: Refactor to clearly separate: (1) indexing module, (2) retrieval module, (3) generation module, (4) evaluation module. Each pluggable with different implementations. Enables A/B testing models without full pipeline changes.  
**Priority**: Architectural improvement — plan before prod, implement during prod hardening.

### RESEARCH-6: Knowledge graph for access control (data leakage prevention)
**From research**: Enterprise RAG needs granular access controls to prevent the AI from becoming a data leakage vector. If a ticket asks about confidential project data, the agent should only surface KB content the requester is authorized to see.  
**Applicability**: Currently we index ALL HUB/OMEGA content. A restricted Confluence page could surface in a draft to a user who doesn't have access to that page.  
**Priority**: Must address before prod if any KB content is access-controlled.

### DEMO-1: Demo script docs/demo_script.md exists but no real drafts on demo tickets yet
**Status**: ANTSE-490 — In Progress. Template fix deployed, credentials fixed. Demo tickets (7560-7581) should now receive drafts. Monitor next few cycles to confirm drafts appear on demo tickets and pass self-check. Some demo tickets are about GitHub/OCP (non-Atlassian) — those will use template-fill path; Atlassian-topic tickets should get KB-grounded drafts.

---

## 2026-06-30

### FAQ Gap Analysis Running Every Cycle Despite 7-Day Gate
**Root cause**: `_phase_export()` in `core/pipeline.py` had its own ungated call to `_auto_gap_analysis()` that ran on every pipeline cycle, completely bypassing the `job_state` 7-day gate implemented in `_phase_faq()`.  
**Fix**: Removed the `_auto_gap_analysis()` call from `_phase_export()`. Gap analysis is now exclusively owned by `_phase_faq()` with the job_state gate.  
**Commit**: `1bb4f5b`

### Template [SYSTEM_NAME] Slot Producing Garbage Output
**Root cause**: `_extract_system_name()` in `plugins/responder/templates.py` concatenated `summary + description` into a single string before applying regex. The pattern `r"permission[s]? (?:for|to) (.+?)(?:\.|$|,|\n)"` matched across the boundary, capturing the description content as the system name (e.g., `"RHEL Engineering We have a Confluence space named 'RHEL Product Management' wher"`).  
**Fix**: Changed to search summary first, then fall back to description. Tests added.  
**Commit**: `019f5bb`

### Demo Tickets Generated With Non-Atlassian Topics
**Root cause**: Few-shot sampling from the real corpus pulled tickets about GitHub repository access, RHEL, OCP namespaces — not Jira/Confluence topics. Gemini generated similar off-domain tickets.  
**Fix**: Added explicit constraint to generation prompt: "tickets MUST be about Jira Cloud, Confluence Cloud, or Atlassian Cloud platform issues only."  
**Commit**: `ba9059f`

---

## 2026-06-29

### 3-Token OAuth Design
**Root cause**: Original read/write credential split failed because write-only tokens get 401 on all API calls — Atlassian API gateway requires read scope on the same token to access any resource before allowing writes.  
**Decision**: 3-token product-based design — all on the same service account:
- JSM token (20 scopes): `read:jira-work` + JSM read/write
- Jira token (26 scopes): Jira platform read/write  
- Confluence token (7 scopes): Confluence read/write  
**Commits**: `1ee320b`, `e5e11b3`

### JSM Comment Posting 401 — Scope Mismatch
**Root cause**: The JSM token's 401 on `servicedeskapi/request/{key}/comment` was not a scope issue — the token had `write:request.comment:jira-service-management`. The new write service account (5FSwaBP0...) had no <PROJECT_KEY> project access, making all API calls fail with "scope does not match."  
**Fix**: The credential routing now uses `jsm` product for servicedeskapi calls. The JSM service account needs project-level access granted in Jira admin.  
**Commits**: `1b0098f`

### GHCR Pull Secret Expiring Every 30 Days
**Root cause**: OCP image pull secret used a GitHub PAT with 30-day expiry. When it expired, pods went into `ImagePullBackOff`.  
**Fix**: CI/CD workflow now refreshes `ghcr-pull-secret` on every deploy by pulling `GHCR_PULL_TOKEN` from Bitwarden.  
**Commit**: `fd856a6`

### Dense Retrieval Index Rebuilt from Scratch on Every Restart
**Root cause**: `DenseRetriever.build()` in `plugins/responder/dense_retrieval.py` called `DELETE FROM responder_corpus_embeddings` then re-embedded all 1,231 corpus documents via Vertex AI API on every startup — causing 20+ minute warmup.  
**Fix**: Cache-aware build: compare stored text per corpus_id, only re-embed changed/new docs. Warm start now reads from DB with zero Vertex API calls. Three-phase design: READ → EMBED (no DB lock) → WRITE.  
**Commits**: `c65a101`, follow-up 3-phase split

### FAQ Gap Analysis 7-Day Gate (was running every 5 minutes)
**Root cause**: `_phase_faq()` had no `job_state` gate — ran full Gemini gap analysis + generation on every 5-minute pipeline cycle.  
**Fix**: Added hybrid gate: 7-day TTL (`run_interval_hours: 168`) OR event trigger (50+ new resolved tickets since last run). Uses `job_state` table keyed by `faq_gap_analysis:{cloud_id}`.  
**Commits**: `f195982`, `99b6961`

### Alerting Phase Hanging Pipeline Indefinitely
**Root cause**: `cluster_alert.py:75` — `generate_content()` had no timeout. If Vertex AI hung (connection reset without close), the entire pipeline blocked forever.  
**Fix**: Added `http_options=genai_types.HttpOptions(timeout=90_000)` (90s) matching the pattern used in classifier, analyzer, and generator.  
**Commit**: `c65a101`

### Embedding Service No Timeout
**Root cause**: `services/embedding.py:88` — `embed_content()` had no timeout, could hang indefinitely.  
**Fix**: Added `EmbedContentConfig(task_type=vt, http_options=genai_types.HttpOptions(timeout=90_000))`.  
**Commit**: `c65a101`

### FAQ Generation Timeout Too Short (was 90s for Gemini-2.5-Pro)
**Root cause**: `faq/generator.py` had 90s timeout. Research showed gemini-2.5-pro response times are 5-7 minutes for long-form content.  
**Fix**: Increased to 840s (14 minutes = 2× documented max).  
**Commit**: `a44c01a`

---

## 2026-06-27

### main._run_pipeline_cycle Was the Active Path (not core.pipeline)
**Root cause**: `main.py` had its own `_run_pipeline_cycle()` that was actually called at runtime. `core.pipeline._run_pipeline_cycle()` was dead code. The main.py version was missing: `resolution_summary`, `feedback`, `kb_index`, `alerting` — those 4 phases silently never ran.  
**Fix**: `main._run_pipeline_cycle()` replaced with thin shim calling `core.pipeline._run_pipeline_cycle()`. All 10 phases now registered and running.  
**Commit**: `c74d4b0`

### _phase_mrr_monitor Never Persisted last_run_date
**Root cause**: `_phase_mrr_monitor()` called `run_mrr_snapshot()` but never called `set_last_run_date`. Result: MRR snapshot ran on every cycle instead of weekly.  
**Fix**: Added `set_last_run_date(conn, _job_key, today)` after `run_mrr_snapshot()`.  
**Commit**: `d93cf3c`

### Sentiment Model Crashing on Long Tickets (>512 tokens)
**Root cause**: cardiffnlp and emotion models have 512-token limit. Without truncation, inputs > 512 tokens raised `"expanded size must match existing size"`.  
**Fix**: Added `truncation=True, max_length=512` to both model pipeline instantiations in `plugins/feedback/sentiment.py`.  
**Commit**: `dee257f`

### Classifier Workers=8 Comment/Code Mismatch
**Note**: Pod is provisioned at 8Gi specifically to support 8 workers. Comment saying "N=8 caused OOM" was written before the 8Gi bump. Workers remain at 8; comment corrected.  
**Commit**: `84a4dc8`

---

## 2026-06-26 (prior session)

### has_resolution Classify-Delete-Classify Loop
**Root cause**: Classifier always returned `has_resolution=0` in the LLM output. `_save_classification` read this from the LLM result instead of from the ticket's actual resolution field. Resolved tickets were deleted and reclassified every cycle.  
**Fix**: Changed to `int(bool(ticket_dict.get("resolution")))` — read from ticket data, not LLM output.  
**Commit**: `d6b74cd`

### confluence_pages Always Empty (ANTSE-164)
**Root cause**: `faq/sources.py` `gather_all_sources()` set `sources["confluence_pages"] = []` and never populated it. ANTSE-164 was marked Done but never implemented.  
**Fix**: Added `_fetch_confluence_pages()` with CQL pre-fetch.  
**Commit**: `9435e43`

### kb_articles Had 0 Rows
**Root cause**: `scripts/index_confluence_kb.py` existed but was never wired into the pipeline.  
**Fix**: Added `_phase_kb_index()` to `core/pipeline.py`, registered as a built-in phase with 7-day cadence.  
**Commit**: `25165da`

### All job_state Keys Must Include cloud_id
**Root cause**: Stage and prod share the same Postgres DB. job_state keys without cloud_id caused state pollution between environments.  
**Fix**: All job_state keys now include cloud_id: `ingest:{cloud_id}:{project}`, `kb_index:{cloud_id}:{space}`, `mrr_snapshot:{cloud_id}`, etc.  
**Commits**: `6e63b3a`, `1401b00`

### Dedup Threshold 0.95 Was Calibrated on Zero Article-Level Duplicate Pairs
**Root cause**: `calibrate_dedup.py` produced 0 duplicate pairs from `generated_articles` because every `article_topic` is unique. The threshold was meaningless.  
**Fix**: Changed to 0.82 for article-level dedup. (Later revised to 0.95 after discovering the ANTSE-554 Vertex AI calibration used real Gemini-as-judge labels.)  
**Commits**: `ab59704`, then reverted to `0.95` in a later session

---

## Architectural Decisions

### Why 3 OAuth Tokens
Atlassian API gateway requires read scope on the same token to authorize write operations. A write-only token gets 401 on every call. With 56 total scopes needed (43 read + 13 write) and a 50-scope limit per token, a single token is impossible. Solution: 3 product-based tokens (JSM/Jira/Confluence) each with read+write for their product, all on the same service account for consistent project access.

### Why Vertex AI for Embeddings (not MiniLM)
MiniLM calibration on JSM tickets: F1=0.54, threshold at floor (0.30). JSM tickets are short, domain-specific, semantically homogeneous — MiniLM cannot differentiate them. Vertex AI gemini-embedding-001 calibration (ANTSE-554): threshold=0.95, F1=0.588, Recall=0.455. Significantly better separation.

### Why OpsGenie Uses GenieKey Not OAuth
OpsGenie is a separate product from Atlassian's OAuth 2.0 scope system. The `write:ops-alert` and `write:ops-config` scopes are available in granular token configs but OpsGenie authentication uses `GenieKey` header auth separately. The ops-* scopes in the token don't replace GenieKey for OpsGenie API calls.

### Why Stage Only
All testing and development is against `stage-<YOUR_DOMAIN>.atlassian.net` (cloudId: `<CLOUD_ID>`). Prod is `<YOUR_DOMAIN>.atlassian.net`. State never shared between instances — all `job_state` keys include `cloud_id`.

### Why FAQ Gate Is Hybrid (TTL + Event)
Industry research (2025): TTL-only is appropriate for FAQ content (stable, staleness-tolerant). Event-driven is better for correctness-critical data. Hybrid: 7-day TTL as safety net + trigger if 50+ new resolved tickets since last run. This responds to actual data changes while preventing over-generation.

### Why Templates Don't Use KB Content
Access/Permissions tickets (the dominant demo ticket type) are about external tools (GitHub, OCP, RHEL) not covered by the Atlassian Cloud KB. Templates work from ticket context alone — `[SYSTEM_NAME]` extracted from ticket summary, `[STEPS]` from resolved ticket patterns. This is the right path for thin-KB categories; KB expansion handles Atlassian-specific topics.
