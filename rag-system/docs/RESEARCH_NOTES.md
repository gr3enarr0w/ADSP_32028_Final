# Research Notes — Candidate Fixes for the Tuned-vs-Baseline Null Result

**Status of this document:** written 2026-07-19, in response to a documentation-review finding
that a prior research investigation (six questions, conducted via parallel web research) had
never been saved anywhere in this repo — its conclusions were referenced in `docs/ARCHITECTURE.md`
§4 ("Known Issues / Open Gaps") and `docs/CHANGELOG.md`, but with no citations a reader could
independently check. This document re-derives and re-verifies each of the six findings from
scratch via web search (not by trusting the prior summary), and states plainly, per finding,
whether the citation checked out, needed correction, or could not be verified. Where a specific
number could be independently confirmed against the actual source text (not just an abstract or
a third party's paraphrase), that is noted explicitly.

**Relationship to other docs:** `docs/ARCHITECTURE.md` §4.1–§4.8 states the *engineering* findings
(what's in this codebase, what's missing, what's been decided) that motivated this research pass.
This document is the *literature verification* layer underneath those findings — it does not
restate the engineering analysis, only the external evidence for/against the techniques under
consideration.

---

## 1. Faithfulness metric mismatch for the blended `[C]`/`[G]` generation policy

**Research date:** 2026-07-19

**Claim being checked:** RAGAS-style faithfulness metrics score an answer's claims against
retrieved context as a whole, with no mechanism to distinguish an intentionally-tagged
general-knowledge claim (this project's `[G]` tag) from an actual hallucination — both simply
register as "claim not supported by context." This directly explains why `docs/ARCHITECTURE.md`
§4.3 documents the blended `[C]`/`[G]` prompt as *expected* to score lower on faithfulness by
design, not as a quality regression.

**Sources checked:**

1. **Wallat, Heuss, de Rijke, Anand — "Correctness is not Faithfulness in RAG Attributions."**
   arXiv:2412.18004 (submitted 23 Dec 2024; also at
   https://arxiv.org/abs/2412.18004, PDF at https://arxiv.org/pdf/2412.18004). **The arXiv ID in
   the original prompt is CORRECT — verified directly against arXiv's own abstract page.**
   Authors confirmed: Jonas Wallat, Maria Heuss, Maarten de Rijke, Avishek Anand (L3S Research
   Center / University of Amsterdam / TU Delft). The paper's actual argument is adjacent to, but
   not identical to, the claim above: it distinguishes citation *correctness* (does the cited
   document support the claim) from citation *faithfulness* (did the model's generation causally
   depend on the cited document, vs. "post-rationalization" — citing a document that happens to
   agree with an answer the model would have produced anyway from parametric knowledge). They
   found up to 57% of citations in their tested "RAG-optimized" model (Cohere Command-R+) were
   correct-but-not-faithful by this definition. **This is a real, relevant paper on the general
   correctness/faithfulness gap in RAG attribution, but it is not itself a study of RAGAS's
   specific metric implementation or of blended `[C]`/`[G]` tagging** — it supports the broader
   point that "sounds right" and "actually grounded" are different axes, which is the same
   underlying concern this project's Known Issue #4.3 raises, but readers should not assume this
   paper specifically analyzes RAGAS's `Faithfulness()` metric internals.

2. **DeepEval's faithfulness metric documentation** (https://deepeval.com/docs/metrics-faithfulness).
   Verified directly: DeepEval's faithfulness metric extracts individual claims from the generated
   answer via an LLM, then checks each claim against the retrieval context, scoring
   `truthful claims / total claims`. The documentation explicitly states the mechanism relevant to
   this project's concern: *"Faithfulness only rewards claims supported by the `retrieval_context`
   — a real-world truth absent from the retrieved text still counts as unfaithful."* This directly
   confirms the mechanism this project's Known Issue #4.3 describes for RAGAS-style faithfulness
   scoring generally (DeepEval and RAGAS are different libraries, but both implement the same
   "claim-vs-context" faithfulness definition) — a claim that is true, and even tagged `[G]` as
   intentionally general-knowledge by this project's own generation prompt, would still be marked
   unfaithful by this class of metric because the metric has no tag-awareness at all, only prose.

**Verdict:** Both citations check out as real, correctly identified papers/docs. The DeepEval
documentation is the more directly on-point source for the exact mechanism (claim-vs-context
scoring with no accommodation for intentionally-flagged non-context claims); the Wallat et al.
paper is real and relevant background on why "sounds grounded" and "is grounded" diverge in RAG
systems generally, but is not itself evidence specific to tag-aware faithfulness scoring or to
RAGAS's implementation. **No RAGAS-specific documentation making the identical claim explicitly
(i.e., a RAGAS maintainer statement that `Faithfulness()` cannot distinguish tagged spans) was
located during this pass** — the RAGAS-specific claim in `docs/ARCHITECTURE.md` §4.3 is this
project's own code-level observation (confirmed directly against
`evaluate_ragas.py:2160-2171`/`:995-1030`/`:805-849`, per that section), not an externally
published RAGAS-team statement, and should continue to be understood that way.

---

## 2. Statistical power of the N=40 tuned-vs-baseline comparison, and published hybrid+rerank gains on NFCorpus/BEIR

**Research date:** 2026-07-19

This finding has two independent parts: (a) an internal statistical-power characterization of
this project's own N=40 harness, and (b) external literature on whether hybrid+rerank gains on
NFCorpus/BEIR are typically small anyway. Both are treated separately below since only (b) is
externally verifiable.

### 2a. Statistical power of N=40 — NOT an externally-sourced citation, flagged as such

The "N=40 is roughly 30-100x underpowered" framing in the prompt this document responds to is a
**paraphrase of this project's own internal analysis**, not a citation to an external paper —
`docs/ARCHITECTURE.md` §4.4 states the underpowering as "roughly one to two orders of magnitude"
(i.e., ~10x-100x), derived from comparing the observed effect sizes (~0.0004-0.02 absolute, see
§4.1's table) against the measured same-config judge-noise floor (std-dev 0.0038-0.0146 across
repeats) and the observed 95% CI half-widths (~0.05-0.15). **This document did not locate, and
was not asked to locate, an external published power-analysis source for this specific number —
it is this project's own derived statistic, verifiable by reading `docs/ARCHITECTURE.md` §4.1/§4.4
and the underlying files in `/tmp/nfcorpus_eval_v2/` directly, not by checking an external
citation.** The "30-100x" phrasing in the original research summary should be read as a rough
restatement of ARCHITECTURE.md's own "one to two orders of magnitude," not a more precise or
independently-sourced number — treat the two as the same claim, not corroborating evidence from
two directions. General background on statistical power confirms the underlying logic is sound in
direction: effect sizes on the order of 0.02 (small in absolute terms) require sample sizes far
beyond 40 to reach conventional 80% power at typical alpha=0.05 thresholds, and importantly the
*relevant* variance for this test is the per-question variance across the 40 different questions
(implied by the wide ~0.05-0.15 CI half-widths), not the much smaller same-question repeat-noise
floor (0.0038-0.0146) — these are two different, non-interchangeable noise sources, and conflating
them would understate how underpowered the design actually is.

### 2b. Published NFCorpus/BEIR hybrid+rerank gains — externally verified, with a major caveat on source quality

1. **Dense-vs-hybrid recall on NFCorpus (~17.6% vs. 17.4%).** Verified via direct web search:
   this exact figure pair appears in **Abraham Itzhak Weinberg, "Hybrid Dense-Sparse Retrieval for
   High-Recall Information Retrieval"** (ResearchGate preprint, DOI:10.13140/RG.2.2.23909.46562,
   dated January 2026, https://www.researchgate.net/publication/399428523). The paper reports,
   specifically for BEIR NFCorpus (3,633 documents, 100 queries): dense-only recall 17.6%,
   hybrid-α=0.7 recall 17.4% — i.e., hybrid search provides **no improvement, and a very slight
   regression**, on NFCorpus specifically, in contrast to the same paper's headline MS MARCO result
   (dense 13.9% → hybrid 80.8% Recall@10, a 580% relative gain). The paper attributes NFCorpus's
   flat/negative hybrid result to the corpus's small size and highly specialized medical
   vocabulary reducing the complementary benefit hybrid search usually provides. **Important
   caveat: this is a ResearchGate preprint by a single author (affiliation listed as "AI-WEINBERG,
   AI Experts," not a university or established research lab), explicitly marked "not yet peer
   reviewed."** The specific numbers were confirmed by directly locating them in search results
   describing the paper's content, but this document was not able to fetch and read the full PDF
   directly (ResearchGate returned HTTP 403 to automated fetch), so the number is corroborated by
   multiple independent search-result descriptions of the same paper rather than by this document's
   own read of the primary source. Treat this as suggestive, not authoritative, evidence — a single
   unreviewed preprint is a weak citation for a load-bearing architectural claim.

2. **LLM-reranking ablation on NFCorpus (~+0.1% gain).** Verified directly by downloading and
   text-extracting the source PDF (not just reading a search snippet): **R. Mabubasha, Navuluri
   Sindhu, MuppalaVenkata Pujitha, Jonnalagadda Abhishek Vardhan — "Component-Level Evaluation of
   Cascaded Retrieval and Reranking in RAG: Ablations Across Financial, Biomedical, and Medical
   Domains,"** *International Research Journal of Engineering and Technology (IRJET)*, Volume 13,
   Issue 4, April 2026 (https://www.irjet.net/archives/V13/i4/IRJET-V13I04289.pdf). The exact
   sentence, confirmed by direct PDF text extraction: *"LLM reranking added smaller and more
   variable improvements. On FiQA it pushed nDCG@10 up a further 7.7% over the cross-encoder; on
   SciFact, 4.8%. On NFCorpus it contributed almost nothing (+0.1%)."* This is a **confirmed exact
   match to the number in the original research summary.** The same paper also separately measured
   cross-encoder reranking as contributing much more (+35.9% on NFCorpus), consistent with this
   project's own `retrieval.rerank: true` default carrying most of the reranking value while the
   heavier LLM-based reranking step this project has never implemented would likely add little on
   this specific corpus. **Important caveat on source quality:** IRJET is a low-prestige,
   non-selective engineering journal; the paper's listed authors are one assistant professor and
   three undergraduate students at a regional Indian engineering college, published in April
   2026 — this is not a peer-reviewed venue with the rigor of, e.g., SIGIR/EMNLP/ACL, and the
   number should be treated as one data point from a small, non-peer-reviewed ablation study, not
   as an established benchmark result.

**Verdict:** Both specific numbers (17.6%/17.4% dense-vs-hybrid, +0.1% LLM-rerank gain) were
**found and confirmed to match** what the original research summary claimed — this is a genuine
success in re-verification, not a case of numbers being fabricated or misremembered. However,
**both source papers are non-peer-reviewed, low-visibility sources** (an unreviewed ResearchGate
preprint by an independent author, and a student-authored paper in a non-selective journal) rather
than established BEIR-benchmark literature from a recognized IR research group. This is a
meaningful caveat: the *direction* of the finding (hybrid and LLM-reranking gains are small to
negligible on NFCorpus specifically, given its small size and specialized vocabulary) is plausible
and consistent with NFCorpus's well-documented shallow-judgment-coverage problem (per Elastic's
search-relevance blog, NFCorpus has only ~35% qrel coverage vs. TREC-COVID's >90%, a a separately
well-established property of this specific BEIR dataset), but readers should not treat the specific
percentages as citable from an authoritative benchmark paper — they are directionally supportive,
not rock-solid, evidence.

---

## 3. HyDE (Hypothetical Document Embeddings)

**Research date:** 2026-07-19

**Claim being checked:** HyDE shows large effect sizes on corpora similar to NFCorpus (the BEIR
technical/scientific cluster), making it the highest-expected-value untested retrieval technique
for this project (per `docs/ARCHITECTURE.md` §4.5's stated rationale: NFCorpus queries are
lay-phrased while its corpus is clinical/PubMed-style, a vocabulary-mismatch problem HyDE's
hypothetical-document generation step is specifically designed to bridge).

**Source verified:** **Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan — "Precise Zero-Shot Dense
Retrieval without Relevance Labels."** arXiv:2212.10496 (submitted 20 Dec 2022;
https://arxiv.org/abs/2212.10496), later published at ACL 2023 (Proceedings of the 61st Annual
Meeting of the ACL, pp. 1762-1777, ACL Anthology: https://aclanthology.org/2023.acl-long.99/).
**The arXiv ID in the original prompt is CORRECT — verified directly against arXiv's own abstract
page and the ACL Anthology record.** Authors and mechanism confirmed: given a query, an
instruction-following LLM (e.g., InstructGPT) generates a hypothetical answer document (which may
contain factual errors); an unsupervised contrastive encoder (e.g., Contriever) embeds that
hypothetical document; the resulting vector is used to retrieve real, similar documents from the
corpus, with the encoder's dense-embedding step filtering out the hypothetical document's
inaccuracies while preserving its relevance-pattern signal. The paper reports HyDE
significantly outperforming the unsupervised Contriever baseline and performing comparably to
fine-tuned dense retrievers across multiple tasks and languages.

**Verdict:** Citation fully confirmed — correct arXiv ID, correct authors, correct mechanism
description, correct venue (ACL 2023). **One caveat for this project specifically:** this
document's search did not find a source specifically quantifying HyDE's effect size on NFCorpus
or the BEIR "technical/scientific" cluster by name — the original HyDE paper's own BEIR evaluation
(per its abstract/summary) emphasizes web search, QA, and fact verification tasks across multiple
languages, not a specific NFCorpus-vs-other-BEIR-subsets breakdown. The claim that HyDE shows
"large effect sizes on corpora similar to NFCorpus" should be understood as a reasonable
extrapolation from HyDE's general strong zero-shot results and this project's own
vocabulary-mismatch reasoning (laid out independently in `docs/ARCHITECTURE.md` §4.5), not as a
direct quote or table value from the HyDE paper itself pinned to NFCorpus. This document did not
locate a specific NFCorpus HyDE ablation number to cite here; if one is needed before committing
engineering time to a HyDE implementation, that would be a good follow-up search.

---

## 4. MMR (Maximal Marginal Relevance) diversity reranking

**Research date:** 2026-07-19

**Claim being checked:** MMR addresses a real, documented gap in this project's current
cross-encoder reranker (per `docs/ARCHITECTURE.md` §4.7: `utils.py:847-854` scores each candidate
purely on `(query, chunk_text)` relevance with no redundancy penalty), which matters specifically
because this project's NFCorpus-derived eval corpus was constructed with distractor documents that
could plausibly cluster near-duplicate content around a query's gold document.

**Sources verified:**

1. **Carbonell, J. & Goldstein, J. — "The Use of MMR, Diversity-Based Reranking for Reordering
   Documents and Producing Summaries."** SIGIR '98: Proceedings of the 21st Annual International
   ACM SIGIR Conference on Research and Development in Information Retrieval, Melbourne, Australia,
   24-28 August 1998, pp. 335-336 (ACM DL: https://dl.acm.org/doi/10.1145/290941.291025; PDF via
   CMU: https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf).
   **The citation (Carbonell & Goldstein, 1998, SIGIR) is CORRECT — verified against both the ACM
   Digital Library record and the original CMU-hosted PDF.** Authors: Jaime G. Carbonell and Jade
   Goldstein, Language Technologies Institute, Carnegie Mellon University. The paper introduces
   MMR to balance query relevance against information novelty when reranking retrieved documents
   or selecting summary passages, explicitly to reduce redundancy in the returned set — this is
   exactly the mechanism §4.7 identifies as missing from this project's cross-encoder-only
   reranking stage. Highly cited (over 1,800 citations per ACM DL), a foundational and
   uncontroversial reference for this technique.

2. **2025 RAG-specific diversity study, arXiv:2502.09017.** Verified: **Zhichao Wang, Bin Bi,
   Yanqi Luo, Sitaram Asur, Claire Na Cheng — "Diversity Enhances an LLM's Performance in RAG and
   Long-context Task."** Submitted 13 Feb 2025 (v1), revised 7 Apr 2025 (v2)
   (https://arxiv.org/abs/2502.09017). **The arXiv ID in the original prompt is CORRECT.** The
   paper builds directly on MMR and Farthest Point Sampling (FPS) to inject diversity into
   content selection for RAG QA and long-context summarization, and explicitly argues its
   lightweight MMR/FPS approach is preferable to LLM-based reranking for diversity on cost/latency
   grounds — directly relevant to this project's own reasoning in §4.7 that LLM-based diversity
   reranking would be a heavier, likely lower-priority alternative to a lightweight MMR pass.

**Verdict:** Both citations fully confirmed — correct authors, correct venue, correct arXiv ID for
the 2025 paper, and both are substantively on-point for the gap this project has identified (a
cross-encoder reranker with no redundancy/diversity term). No inaccuracies found in this finding.

---

## 5. Model routing/ensemble architectures (closed decision, per `docs/ARCHITECTURE.md` §4.8)

**Research date:** 2026-07-19

**Claim being checked:** RouteLLM-style dynamic model routing is primarily a cost/latency
optimization with quality held roughly flat; Mixture-of-Agents (MoA) shows larger quality gains,
but those gains are attributable to output aggregation itself rather than genuine cross-provider
model diversity (per the Self-MoA finding). This is the evidentiary basis for
`docs/ARCHITECTURE.md` §4.8's explicit, closed decision **not** to pursue a model-routing/ensemble
layer as a lever for this project's quality metrics.

**Sources verified:**

1. **RouteLLM.** **Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E.
   Gonzalez, M Waleed Kadous, Ion Stoica — "RouteLLM: Learning to Route LLMs with Preference
   Data."** arXiv:2406.18665 (submitted 26 Jun 2024; https://arxiv.org/abs/2406.18665).
   **The arXiv ID in the original prompt is CORRECT.** Confirmed: the paper's own framing is
   explicitly a cost/quality trade-off optimization — routing between a stronger and weaker LLM to
   reduce cost (reported: >2x cost reduction in some cases, and the released open-source routers
   reportedly cut cost up to 85% while retaining ~95% of GPT-4-level quality on benchmarks like MT
   Bench) **without compromising response quality**, i.e., quality is preserved/held flat by
   design (that is the paper's stated goal), not improved as a side effect of routing itself. This
   matches the "primarily a cost/latency optimization, quality roughly flat" characterization.

2. **Mixture-of-Agents (MoA).** **Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, James Zou
   — "Mixture-of-Agents Enhances Large Language Model Capabilities."** arXiv:2406.04692
   (submitted 7 Jun 2024; https://arxiv.org/abs/2406.04692), later a Spotlight at ICLR 2025.
   **The arXiv ID in the original prompt is CORRECT.** Confirmed: MoA layers multiple LLM
   "proposer" agents whose outputs are synthesized by subsequent layers, reporting large quality
   gains (65.1% vs. GPT-4 Omni's 57.5% on AlpacaEval 2.0 using only open-source models).

3. **Self-MoA (the paper that reframes MoA's gains as aggregation-driven, not
   diversity-driven).** **Wenzhe Li, Yong Lin, Mengzhou Xia, Chi Jin — "Rethinking
   Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?"** arXiv:2502.00674
   (submitted 2 Feb 2025; https://arxiv.org/abs/2502.00674), accepted at ICLR 2025.
   **The arXiv ID in the original prompt is CORRECT.** Confirmed directly: this paper's central
   finding is that **Self-MoA — aggregating multiple outputs sampled from a single top-performing
   model, with no cross-model diversity at all — outperforms standard MoA that mixes different
   LLMs**, in a majority of tested scenarios (reported: 6.6% improvement over standard MoA on
   AlpacaEval 2.0, ~3.8% average improvement across MMLU/CRUX/MATH). The paper's own explanation is
   a quality-diversity trade-off: mixing in weaker models to gain diversity often lowers the
   ensemble's average output quality more than the diversity itself helps, so aggregation of
   *any* multiple outputs (from one or many models) is doing most of the work MoA gets credit for,
   not provider/model diversity specifically. **This directly and precisely supports the claim in
   the original prompt** — this is a strong, well-confirmed citation for the exact point being
   made.

**Verdict:** All three arXiv IDs confirmed correct, all three papers' actual findings match the
characterization in `docs/ARCHITECTURE.md` §4.8 and the original research summary closely. This is
the strongest-verified of the six findings — no corrections, no unfindable claims, no source-quality
caveats (all three are from recognized research groups / accepted at ICLR). The "closed decision"
framing in `docs/ARCHITECTURE.md` §4.8 (routing/ensembling is not expected to move this project's
quality metrics) is well-supported by this literature.

---

## 6. Chunking granularity: hierarchical parent-child trade-off and the "Chroma Research" faithfulness-invalid-rate claim

**Research date:** 2026-07-19

**Claim being checked:** Two separate sub-claims were bundled in the original prompt: (a)
hierarchical parent-child chunk expansion trades retrieval precision for context, helping
multi-hop questions but hurting single-hop factoid questions (directly relevant to
`docs/ARCHITECTURE.md` §4.6's finding that this project's `retrieval.fetch_parents` toggle is
global/static, not query-adaptive, and that single-hop factoid questions are this project's
dominant NFCorpus query type); and (b) a specific finding, attributed to "Chroma Research," that
RAGAS faithfulness has a documented ~44% invalid-score rate in chunking-ablation contexts.

**What was found — sub-claim (b) is MISATTRIBUTED, flagged as an inaccurate citation:**

The ~44% invalid-faithfulness-score figure is real and was located, but **it is not a Chroma
Research finding.** It comes from: **Valentin J. J. Kreileder, Johannes Reisinger, Andreas
Fischer — "Evaluating Chunking Strategies for Retrieval-Augmented Generation on Academic Texts."**
arXiv:2607.01852 (https://arxiv.org/abs/2607.01852 / https://arxiv.org/html/2607.01852v1). This
paper evaluated fixed-size, recursive, and cluster-based chunking on long academic theses using
the RAGAS framework, and reported: faithfulness computation **failed (timed out or returned
invalid values) in 44% of cases**, versus only 2-3% failure for other RAGAS metrics — broken down
as 47% for recursive chunking, 42% for cluster-based, 44% for fixed-size chunking. The paper's own
conclusion is that "RAGAS-based faithfulness shows limited reliability in this setup." This part
of the claim is **confirmed accurate in substance and number** — but attributed to the wrong
source. **This document independently checked Chroma's own actual chunking-evaluation technical
report** (https://www.trychroma.com/research/evaluating-chunking) directly via fetch, and confirmed
it does **not** use RAGAS or faithfulness scoring at all, and does **not** mention a 44%
invalid-score rate anywhere — Chroma's real study instead uses token-level IR metrics (Recall,
Precision, Precision-Ω, and a custom IoU metric) computed against LLM-generated reference excerpts,
with no LLM-as-judge faithfulness scoring in the loop at all. **The "Chroma Research" label in the
original research summary is incorrect** — the actual source is the unrelated Kreileder/Reisinger/
Fischer paper (arXiv:2607.01852), and the two studies use entirely different, non-overlapping
methodologies. Anyone citing the 44% figure going forward should cite arXiv:2607.01852 by name, not
"Chroma Research."

**Sub-claim (a) — hierarchical chunking's single-hop-vs-multi-hop trade-off:** this document did
not locate a specific external paper cited for this exact claim in the original prompt (none was
given to verify), and a targeted search for this specific framing did not surface one canonical
source distinct from general chunking-strategy literature (e.g., the general finding that smaller
chunks favor precision/specific-fact retrieval while larger/expanded context favors multi-hop
reasoning is common across several recent chunking papers surfaced during this research pass —
e.g., arXiv:2607.01852 itself, and other 2025-2026 chunking-ablation papers — but no single paper
was identified as *the* source for this specific single-hop/multi-hop framing as applied to
parent-child hierarchical retrieval specifically). **This sub-claim should be treated as this
project's own synthesis/inference** (as `docs/ARCHITECTURE.md` §4.6 itself frames it — "research
cited elsewhere in this project's own planning," not a specific paper this document was asked to
verify), not as an independently pinned-down external citation. If a specific source is needed for
this sub-claim before acting on it, that would need a separate, more targeted literature search.

**Verdict:** The specific number (~44% invalid-score rate) is **confirmed accurate**, but its
**source attribution is wrong** — it is not from Chroma Research, and Chroma's actual published
chunking research uses a completely different, non-RAGAS methodology. This is the one finding in
this document where a citation in the original prompt is **flagged as actively incorrect**, not
merely unverifiable or approximate. The correct citation is Kreileder, Reisinger, and Fischer,
arXiv:2607.01852.

---

## Summary table

| # | Finding | Core citation(s) | Verified? |
|---|---|---|---|
| 1 | Faithfulness metric mismatch | Wallat et al. arXiv:2412.18004; DeepEval docs | ID/authors confirmed correct; paper is relevant background, not a direct RAGAS-mechanism study — the RAGAS-specific claim is this project's own code-level finding, not an external citation |
| 2a | N=40 statistical power | (internal to this project) | Not an external citation — this is `docs/ARCHITECTURE.md` §4.4's own derived statistic; the "30-100x" phrasing is a paraphrase of that section's "one to two orders of magnitude" |
| 2b | NFCorpus dense-vs-hybrid (17.6%/17.4%), LLM-rerank gain (+0.1%) | Weinberg (ResearchGate preprint, unreviewed); Mabubasha et al., IRJET 2026 | Numbers confirmed accurate (one via direct PDF extraction); **both sources are non-peer-reviewed / low-prestige — treat as suggestive, not authoritative** |
| 3 | HyDE | Gao et al. arXiv:2212.10496 | Fully confirmed (ID, authors, venue); NFCorpus-specific effect size not directly quoted in the source, this project's own extrapolation |
| 4 | MMR diversity reranking | Carbonell & Goldstein 1998 SIGIR; Wang et al. arXiv:2502.09017 | Fully confirmed, both citations accurate and on-point |
| 5 | Model routing/ensembles | RouteLLM arXiv:2406.18665; MoA arXiv:2406.04692; Self-MoA arXiv:2502.00674 | Fully confirmed, all three IDs correct, all three papers' findings match the claim precisely — strongest-verified finding in this document |
| 6 | Chunking granularity / 44% invalid faithfulness rate | **Misattributed to "Chroma Research"** — actual source is Kreileder/Reisinger/Fischer arXiv:2607.01852 | Number confirmed accurate; **source attribution confirmed WRONG**, flagged explicitly; single-hop/multi-hop sub-claim not independently sourced (project's own synthesis) |
