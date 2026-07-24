#!/usr/bin/env python3
"""
RAGAS Evaluation for the Efficient Hermes RAG skill.

Evaluates retrieval + generation quality using RAGAS's current
EvaluationDataset / SingleTurnSample schema and five RAG metrics:
context_precision, context_recall, faithfulness, answer_relevancy, and
answer_correctness.

GENERATION POLICY (attribution-tagged blending, not strict refusal):
generate_answer() no longer instructs the model to answer ONLY from
retrieved context and refuse otherwise. That strict policy was measured to
cause a 75% refusal rate on a hard corpus whenever retrieved context was
thin or tangential — driving RAG's answer_relevancy score far below a
plain no-retrieval (closed-book) baseline, because a refusal is never a
relevant answer even when the retrieval itself was reasonable (see
generate_closed_book_answer() / run_closed_book_evaluation() for that
baseline). The fix: generate_answer() now asks the model to write a
complete, helpful answer that uses the retrieved CONTEXT wherever it
applies (tagged inline with "[C]") and permits supplementing with the
model's own general knowledge when the context is missing, incomplete, or
only tangentially related (tagged inline with "[G]"), while still
forbidding fabricated specifics attributed to [C] and requiring a trailing
"Grounding: ..." line summarizing how context-based the answer was.

IMPORTANT — faithfulness is EXPECTED to drop under this policy, and that is
not a regression. faithfulness measures how much of the answer is
attributable to the retrieved context; a blended answer that legitimately
draws on general knowledge for gaps in the context will faithfully score
lower on that axis by design. Don't "fix" a lower faithfulness number back
to the old strict-refusal prompt — track faithfulness alongside
answer_correctness and answer_relevancy instead: if answer_correctness and
answer_relevancy are healthy while faithfulness is lower than in a
strict-refusal run, that's the blending policy working as intended, not a
quality bug. answer_correctness was added specifically to catch the failure
mode a faithfulness-only view would miss (a well-grounded but wrong answer,
or a well-attributed but unhelpful one).

Usage:
    # Smoke test (no args) — dummy retrieval/generation, real evaluate() call
    # only if ANTHROPIC_API_KEY is set (otherwise prints a message and exits).
    python scripts/evaluate_ragas.py

    # Real evaluation against a live collection
    python scripts/evaluate_ragas.py \\
        --testset testset.json \\
        --collection my_collection \\
        --config config/config.yaml \\
        --generate

    testset.json shape: [{"question": "...", "ground_truths": ["..."]}, ...]
    ("ground_truth": "..." — a single string — is also accepted and is
    normalized into a one-item "ground_truths" list.)

Answer generation (--generate) uses the Anthropic Python SDK when
ANTHROPIC_API_KEY is set — the natural fit for this Hermes/Claude-ecosystem
skill, since RAGAS needs *some* LLM to produce the "response" text that the
faithfulness, answer_relevancy, and answer_correctness metrics grade. If
ANTHROPIC_API_KEY is not set but ANTHROPIC_VERTEX_PROJECT_ID is (Claude
accessed via Google Vertex AI using Application Default Credentials, no
Anthropic API key needed), it falls back to `anthropic.AnthropicVertex`.
Without either, generation is skipped and the three generation-dependent
metrics are dropped from the run, printing a clear message; context_precision
and context_recall (which only need the retrieved contexts + reference, not a
generated response) still run as long as an evaluator ("judge") LLM is
available.

The RAGAS judge LLM defaults to direct Anthropic (via langchain-anthropic,
ChatAnthropic) when ANTHROPIC_API_KEY is set, else falls back to Claude on
Vertex AI (via langchain-google-vertexai, ChatAnthropicVertex) when
ANTHROPIC_VERTEX_PROJECT_ID is set. The judge embeddings model (needed only
for answer_relevancy) defaults to OpenAI (via langchain-openai) when
OPENAI_API_KEY is set, else falls back to Vertex AI embeddings (via
langchain-google-vertexai, VertexAIEmbeddings) using the same GCP project —
both LLM and embeddings fallbacks require no new API key, only Google
Application Default Credentials (`gcloud auth application-default login`)
already being configured. Both are overridable via the `llm=`/`embeddings=`
parameters on run_ragas_evaluation().

Vertex-specific env vars (only used when falling back to Vertex):
    ANTHROPIC_VERTEX_PROJECT_ID   GCP project ID (required for the fallback)
    ANTHROPIC_VERTEX_REGION       Vertex region for Claude (default: us-east5;
                                   Claude on Vertex is only available in a
                                   subset of regions)
    ANTHROPIC_VERTEX_MODEL        Claude model on Vertex (default:
                                   claude-sonnet-4-5@20250929)
    VERTEX_EMBEDDING_MODEL        Vertex embedding model (default:
                                   gemini-embedding-001 — note some Vertex
                                   projects restrict text-embedding-* models
                                   via org policy; gemini-embedding-001 is
                                   confirmed to work under such policies)

MULTI-JUDGE FAITHFULNESS CROSS-CHECK (--multi-judge-crosscheck):
A second-opinion capability for the faithfulness metric specifically,
motivated by the fact that a single-judge score (Claude, whether direct or
via Vertex) can't distinguish "the answer is genuinely unfaithful" from
"this particular judge model is miscalibrated/biased on this corpus". A
Gemini-based second judge was considered and explicitly rejected (the
readily available Gemini model version was stale enough to be an unreliable
comparison point). Instead, this uses THREE independent-family judge models
served locally via Ollama — glm-5.2:cloud, kimi-k2.7-code:cloud, and
minimax-m3:cloud — so a triangulated cross-check of the original Claude
faithfulness score doesn't depend on any single vendor's judge behavior.
Requires Ollama running locally (default http://localhost:11434, override
with OLLAMA_BASE_URL) with all three models pulled — `ollama list` should
show glm-5.2:cloud, kimi-k2.7-code:cloud, and minimax-m3:cloud. See
_ollama_judge_llm() and run_multi_judge_faithfulness_crosscheck(). This
capability is implemented but has intentionally NOT been run yet — it
requires the three Ollama cloud models to be pulled locally, which is a
separate manual step.

JUDGE-INDEPENDENT HUMAN REVIEW TIER (--export-human-review / --summarize-human-review):
Every quality signal in this module so far -- RAGAS's five metrics, the
closed-book arm, the Claude web_search arm, and even the multi-judge Ollama
cross-check above -- is graded by an LLM judge of one flavor or another.
That is a real gap: this session alone surfaced a rejected stale Gemini
judge and several RAGAS metric deltas that a statistical rigor pass found
"not distinguishable from judge noise" at reasonable sample sizes. None of
that is fixed by adding yet another LLM judge. This capability adds the one
signal that is NOT an LLM judge at all: a human spot-checks a small sample
of (question, retrieved_contexts, generated_answer) triples and assigns one
of three labels, using this rubric (also in docs/HUMAN_REVIEW_RUBRIC.md and
the HUMAN_REVIEW_RUBRIC constant below -- do not invent a different one):

    - Grounded: every factual claim in the answer is traceable to the
      retrieved context.
    - Partially grounded: some claims are supported by context, others are
      unsupported/inferred/from general knowledge. Because generate_answer()
      already tags claims inline as [C] (context) or [G] (general
      knowledge), a "Partially grounded" label should naturally correlate
      with answers containing both tags -- this is a real, checkable
      prediction, not just a formatting exercise (see
      summarize_human_review()'s optional RAGAS cross-tabulation).
    - Not grounded: the answer contradicts the retrieved context, or
      ignores it entirely -- e.g. it answers from general knowledge when
      relevant context WAS available and should have been used, or it
      hallucinates specifics not present anywhere.

export_for_human_review() reuses _make_rag_retrieve_func()/generate_answer()
(or any caller-supplied equivalents) to produce the triples and writes them
to a CSV with empty human_label/human_comment columns for a human to fill
in by hand. summarize_human_review() reads a hand-labeled copy of that CSV
back in and prints a label distribution, plus (given an optional saved
RAGAS per-question results CSV) a cross-tabulation of mean RAGAS scores per
human label -- e.g. checking whether "Partially grounded" rows really do
score lower on faithfulness than "Grounded" rows, which is a
rubric-derived prediction the human labels can actually be checked
against, independent of whatever any single LLM judge said.

Requires: pip install -r requirements.txt
"""

import argparse
import ast
import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    Faithfulness,
    answer_relevancy,
    answer_similarity,
    answer_correctness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    AnswerSimilarity,
    AnswerCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Default judge models for --multi-judge-crosscheck: three independent
# model families served locally via Ollama, chosen specifically so no two
# share a training lineage with each other or with the primary Claude judge.
DEFAULT_OLLAMA_JUDGE_MODELS = ["glm-5.2:cloud", "kimi-k2.7-code:cloud", "minimax-m3:cloud"]

# CONFIRMED, live-verified — NeuralWatt model catalog.
#
# NeuralWatt is a hosted inference platform confirmed to expose an OpenAI-API-compatible endpoint
# at https://api.neuralwatt.com/v1. All three model IDs below are confirmed to exist and respond
# correctly: this catalog backed a completed live battery of 720 real judge calls (3 models x 6
# CSV files x 40 rows, see docs/CHANGELOG.md and /tmp/nfcorpus_eval_v2/neuralwatt_battery_v2_summary.txt)
# run via run_neuralwatt_multi_judge_consensus(), with a zero-NaN escalation chain (NeuralWatt ->
# Ollama fallback -> Claude tiebreaker) resolving every row. "qwen3.5-397b" is confirmed to have a
# real ~40% failure rate on direct NeuralWatt calls (29.6% ollama_fallback + 10.0%
# claude_tiebreaker_escalation across the battery) — this is an observed, live-verified fact about
# that model's reliability, not a hypothetical concern the escalation chain merely guards against.
#
# Selection rationale (confirmed against live access):
#   - "glm-5.2-short": reasoning-capable, 200K context, $1.45/$4.50 per M input/output tokens,
#     ~1.59 Wh/request. Chosen over the full "glm-5.2" (1048K context) for generation+judging
#     because it retains the same "Reasoning" capability tag at a near-identical price point but
#     lower per-request energy, and 200K context is already well beyond this project's actual
#     chunk/context sizes (see config/config.yaml's chunking defaults) -- the extra 1048K context
#     of the full variant buys nothing here.
#   - "kimi-k2.7-code": reasoning + native JSON mode, intended for judging. Native JSON mode may
#     resolve the RAGAS schema-validation NaN issue more reliably than Ollama's generic
#     `format="json"` flag (see _ollama_judge_llm() callers / DEFAULT_OLLAMA_JUDGE_MODELS).
#   - "qwen3.5-397b": reasoning + JSON mode, intended as a backup/third multi-judge consensus
#     member, replacing "minimax-m3" from DEFAULT_OLLAMA_JUDGE_MODELS since NeuralWatt does not
#     offer a minimax-m3 model.
NEURALWATT_MODELS: Dict[str, Dict[str, Any]] = {
    "glm-5.2-short": {
        "capabilities": ["reasoning"],
        "context_window": 200_000,
        "price_per_m_input_tokens_usd": 1.45,
        "price_per_m_output_tokens_usd": 4.50,
        "energy_wh_per_request": 1.59,
        "intended_use": "generation + judging",
    },
    "kimi-k2.7-code": {
        "capabilities": ["reasoning", "json_mode"],
        "intended_use": "judging (native JSON mode)",
    },
    "qwen3.5-397b": {
        "capabilities": ["reasoning", "json_mode"],
        "intended_use": "backup/third multi-judge consensus member (replaces minimax-m3)",
    },
}

# Per-judge Ollama-cloud fallback used by _score_one_model() (run_neuralwatt_multi_judge_consensus())
# when a NeuralWatt judge call fails (timeout, rate-limit, or any other exception). Added directly
# in response to live TimeoutError instability observed against api.neuralwatt.com (see
# docs/CHANGELOG.md's dated entry) -- rather than a failed NeuralWatt call simply becoming a NaN/
# error dict for that model, it gets ONE retry via the equivalent Ollama-hosted cloud model before
# giving up. User-confirmed exact Ollama cloud model names (not independently verified against
# `ollama list` by this change -- same caveat as DEFAULT_OLLAMA_JUDGE_MODELS elsewhere in this
# file: confirm these are pulled locally before relying on this fallback path for real).
_NEURALWATT_TO_OLLAMA_FALLBACK: Dict[str, str] = {
    "glm-5.2-short": "glm-5.2:cloud",
    "kimi-k2.7-code": "kimi-k2.7-code:cloud",
    "qwen3.5-397b": "qwen3.5:397b-cloud",
}

# CONCURRENCY CONSTANTS for the per-call-site `run_config=RunConfig(max_workers=...)` passed to
# ragas's evaluate() (see M2/S4 in docs/CHANGELOG.md's dated entry on this). Neither evaluate()
# call site below used to pass run_config= at all, silently defaulting to ragas's own
# RunConfig(max_workers=16) -- fine for a single sequential judge, but dangerous once multiple
# judges' evaluate() calls run concurrently in their own threads (see
# run_neuralwatt_multi_judge_consensus()).
#
# _NEURALWATT_JUDGE_MAX_WORKERS: max concurrent requests INSIDE a single NeuralWatt judge's own
# evaluate() call (used by _score_one_model() below).
#
# CONFIRMED: NeuralWatt's rate limit is 3 concurrent requests ACCOUNT-WIDE (not per-model) --
# verified directly with the user, not an assumption. This design allocates all 3 slots across
# the panel: the OUTER ThreadPoolExecutor(max_workers=concurrency) in
# run_neuralwatt_multi_judge_consensus() (default concurrency=3) gives one slot per judge model
# -- genuine 3-way panel parallelism, the whole point of the `concurrency` parameter -- and this
# INNER RunConfig(max_workers=1) ensures each judge thread only ever uses its own single
# allocated slot at a time (rather than ALSO fanning out internally, e.g. to ragas's own default
# of 16, which would blow the 3-slot account-wide budget many times over on its own). Net effect:
# total concurrent NeuralWatt requests never exceeds 3, matching the account-wide limit exactly,
# with all 3 judge models still running genuinely concurrently with each other. Do NOT raise this
# above 1 without re-deriving the total-concurrency math above -- the 3-slot budget is fixed by
# NeuralWatt's account plan, not a guess.
_NEURALWATT_JUDGE_MAX_WORKERS = 1

# _CLAUDE_TIEBREAKER_MAX_WORKERS: max concurrent requests inside _claude_tiebreaker_faithfulness()'s
# own evaluate() call (which always scores exactly one sample, so this is a no-op today either
# way). Kept as its OWN named constant, separate from _NEURALWATT_JUDGE_MAX_WORKERS above, purely
# so that if the tiebreaker loop is ever parallelized/batched in the future (see
# run_neuralwatt_multi_judge_consensus()'s per-question loop, currently sequential), that change
# doesn't silently inherit a value that was tuned for a completely different provider's
# (NeuralWatt's) rate limit.
_CLAUDE_TIEBREAKER_MAX_WORKERS = 1

# _OLLAMA_JUDGE_MAX_WORKERS: used by run_multi_judge_faithfulness_crosscheck()'s per-model loop
# (sequential across models, one Ollama judge at a time -- see that function's docstring). Set
# to ragas's own historical default (16) rather than a smaller number: Ollama here is a locally
# hosted server, not a rate-limited hosted API like NeuralWatt, and no evidence of an Ollama-side
# concurrency ceiling has been observed (unlike NeuralWatt's documented per-model 3-slot cap).
# Made explicit (rather than just omitting run_config=) purely so this evaluate() call site is
# consistent with the other two above about not silently relying on an implicit default.
_OLLAMA_JUDGE_MAX_WORKERS = 16

# The three valid values for the "human_label" column produced by
# export_for_human_review() / consumed by summarize_human_review(). This is
# the canonical rubric for this project's judge-independent human review
# tier (see the module docstring's "JUDGE-INDEPENDENT HUMAN REVIEW TIER"
# section and docs/HUMAN_REVIEW_RUBRIC.md) — do not invent a different
# rubric or different label strings; every consumer of the CSV compares
# against these exact strings.
HUMAN_REVIEW_LABELS = ["Grounded", "Partially grounded", "Not grounded"]

HUMAN_REVIEW_RUBRIC = """\
GROUNDING LABEL RUBRIC (judge-independent, human-assigned)

For each (question, retrieved_contexts, generated_answer) triple, assign
exactly one of the following three labels to the "human_label" column:

  Grounded
      Every factual claim in the answer is traceable to the retrieved
      context.

  Partially grounded
      Some claims are supported by context, others are unsupported /
      inferred / from general knowledge. Because generate_answer() tags
      claims inline as [C] (context-derived) or [G] (general-knowledge),
      a "Partially grounded" label should naturally correlate with answers
      containing BOTH [C] and [G] tags — if it doesn't, that's worth a
      comment.

  Not grounded
      The answer contradicts the retrieved context, or ignores it
      entirely — e.g. it answers from general knowledge when relevant
      context WAS available and should have been used, or it hallucinates
      specifics (numbers, names, dates) not present anywhere in the
      retrieved context.

Use the "human_comment" column freely for anything qualitative: what
worked, what should be improved, whether the [C]/[G] tagging in the answer
matches your own read of what's grounded vs. not, retrieval misses, etc.

Use these EXACT three strings in "human_label" (case-sensitive, no
trailing punctuation) so downstream tooling can parse them:
    Grounded | Partially grounded | Not grounded
"""

sys.path.append(str(Path(__file__).parent))


def _extract_contexts(retrieval_result: Any) -> List[str]:
    """Defensively pull a List[str] of contexts out of whatever
    rag_retrieve_func returned.

    Handles:
      - scripts/retrieve.py's retrieve_context() output shape:
        {"raw_results": [{"text": "..."}, ...], ...}
      - a plain {"contexts": [...]} dict
      - a plain List[str]
      - a list of dicts with a "text" key
    """
    if isinstance(retrieval_result, dict):
        raw_results = retrieval_result.get("raw_results")
        if raw_results is not None:
            return [
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw_results
            ]
        contexts = retrieval_result.get("contexts")
        if contexts is not None:
            return [str(c) for c in contexts]
        return []

    if isinstance(retrieval_result, list):
        return [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in retrieval_result
        ]

    return []


def _build_anthropic_client(model_override: Optional[str] = None):
    """Build an Anthropic client + model id, preferring the direct API
    (ANTHROPIC_API_KEY) and falling back to Claude on Vertex AI
    (ANTHROPIC_VERTEX_PROJECT_ID, via Application Default Credentials, no
    API key needed). Returns (client, model, mode) or (None, None, None) if
    neither is available.

    Args:
        model_override: when given, use this model id instead of the normal
            default-selection logic below, on whichever branch (direct/
            Vertex) ends up being used. Added additively for the
            closed-book-vs-RAG-vs-model-tier comparison (see
            /tmp/nfcorpus_eval_v2/model_tier_comparison_progress.log) so
            callers can force a specific Claude model tier (Sonnet/Haiku/
            Opus) per call without touching any global/env state. Defaults
            to None, which reproduces the exact prior behavior (direct API:
            "claude-opus-4-8"; Vertex: ANTHROPIC_VERTEX_MODEL or its
            hardcoded default) for every existing caller that doesn't pass
            it — same additive-parameter pattern as generate_answer()'s
            `verify_and_repair` and _neuralwatt_llm()'s explicit
            `model_name`.
    """
    try:
        import anthropic
    except ImportError:
        print(
            "The 'anthropic' package is not installed (pip install -r requirements.txt) — "
            "skipping answer generation."
        )
        return None, None, None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        model = model_override or "claude-opus-4-8"
        return anthropic.Anthropic(api_key=api_key), model, "direct"

    vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if vertex_project:
        region = os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5")
        model = model_override or os.environ.get("ANTHROPIC_VERTEX_MODEL", "claude-sonnet-4-5@20250929")
        try:
            client = anthropic.AnthropicVertex(project_id=vertex_project, region=region)
            return client, model, "vertex"
        except Exception as exc:
            print(f"Could not build AnthropicVertex client ({exc}) — skipping answer generation.")
            return None, None, None

    print(
        "Neither ANTHROPIC_API_KEY nor ANTHROPIC_VERTEX_PROJECT_ID is set — "
        "skipping answer generation for this item."
    )
    return None, None, None


def _default_judge_llm():
    """Build the default RAGAS judge LLM: direct Anthropic (ANTHROPIC_API_KEY)
    if available, else Claude on Vertex AI (ANTHROPIC_VERTEX_PROJECT_ID) via
    Application Default Credentials — no API key needed for the Vertex path.
    Returns a LangchainLLMWrapper, or None (with a printed message) if neither
    is configured.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            print(
                "langchain-anthropic is not installed (pip install -r requirements.txt) — "
                "cannot build the default RAGAS judge LLM. Pass `llm=` explicitly, or install "
                "the package."
            )
            return None
        return LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-5", max_tokens=4096))

    vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if vertex_project:
        try:
            from langchain_google_vertexai.model_garden import ChatAnthropicVertex
        except ImportError:
            print(
                "langchain-google-vertexai is not installed (pip install -r requirements.txt) — "
                "cannot build the Vertex-based RAGAS judge LLM. Pass `llm=` explicitly, or "
                "install the package."
            )
            return None
        region = os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5")
        model = os.environ.get("ANTHROPIC_VERTEX_MODEL", "claude-sonnet-4-5@20250929")
        try:
            return LangchainLLMWrapper(
                ChatAnthropicVertex(
                    model_name=model, project=vertex_project, location=region, max_tokens=4096
                )
            )
        except Exception as exc:
            print(f"Could not build the Vertex-based RAGAS judge LLM ({exc}).")
            return None

    print(
        "Neither ANTHROPIC_API_KEY nor ANTHROPIC_VERTEX_PROJECT_ID is set — cannot build the "
        "default RAGAS judge LLM. Set one, or pass a pre-configured `llm=` argument."
    )
    return None


def _default_judge_embeddings():
    """Build the default RAGAS judge embeddings (used only by answer_relevancy):
    OpenAI (OPENAI_API_KEY) if available, else Vertex AI embeddings
    (ANTHROPIC_VERTEX_PROJECT_ID) via Application Default Credentials — no new
    API key needed for the Vertex path. Returns a LangchainEmbeddingsWrapper,
    or None (with a printed message) if neither is configured.
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            return LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        except Exception as exc:
            print(f"Could not build the OpenAI embeddings backend ({exc}).")

    vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if vertex_project:
        try:
            from langchain_google_vertexai import VertexAIEmbeddings
            model = os.environ.get("VERTEX_EMBEDDING_MODEL", "gemini-embedding-001")
            return LangchainEmbeddingsWrapper(
                VertexAIEmbeddings(model_name=model, project=vertex_project)
            )
        except Exception as exc:
            print(f"Could not build the Vertex-based embeddings backend ({exc}).")
            return None

    print(
        "Skipping answer_relevancy metric — neither OPENAI_API_KEY nor "
        "ANTHROPIC_VERTEX_PROJECT_ID is set. Install/configure one, or pass `embeddings=` "
        "explicitly."
    )
    return None


def _ollama_judge_llm(model_name: str, num_ctx: int = 200000):
    """Build a RAGAS judge LLM backed by a local Ollama model, for the
    --multi-judge-crosscheck / run_multi_judge_faithfulness_crosscheck()
    capability (see module docstring).

    `num_ctx` maps directly to ChatOllama's own `num_ctx` constructor field
    ("Sets the size of the context window used to generate the next token.",
    default 2048 per langchain_ollama.chat_models.ChatOllama) — verified by
    reading the installed langchain-ollama 0.3.10 source
    (langchain_ollama/chat_models.py) rather than guessed. A high default
    (200000) is used here because faithfulness judging concatenates the full
    retrieved_contexts + response for each sample, which can be long; Ollama
    silently truncates/drops older context once num_ctx is exceeded rather
    than erroring, so under-sizing it would silently corrupt judgments.

    base_url defaults to Ollama's standard local endpoint
    (http://localhost:11434) and is overridable via OLLAMA_BASE_URL, mirroring
    the env-var-driven configuration pattern used by _default_judge_llm() /
    _default_judge_embeddings() elsewhere in this module.

    This function only constructs the wrapper — it never calls .invoke() or
    otherwise makes a network request. Requires `langchain-ollama` to be
    installed (pinned to ==0.3.10 in requirements.txt; see that file's
    comment for why the exact pin matters) and an Ollama server actually
    running with `model_name` pulled for this to work when later invoked via
    ragas.evaluate().
    """
    from langchain_ollama import ChatOllama

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return LangchainLLMWrapper(ChatOllama(model=model_name, base_url=base_url, num_ctx=num_ctx))


def _neuralwatt_llm(model_name: str, temperature: float = 0.1) -> LangchainLLMWrapper:
    """Build a RAGAS judge/generation LLM backed by NeuralWatt, a hosted inference platform
    CONFIRMED to be OpenAI-API-compatible.

    LIVE-VERIFIED: this function IS wired into and actively used by
    run_neuralwatt_multi_judge_consensus() (see that function's docstring), which has been run
    end-to-end against the real NeuralWatt endpoint -- a completed 3-model x 6-file x 40-row
    battery (720 real judge calls, zero NaN results via the escalation chain; see
    docs/CHANGELOG.md and /tmp/nfcorpus_eval_v2/neuralwatt_battery_v2_summary.txt). It is still
    NOT wired into run_multi_judge_faithfulness_crosscheck() (the Ollama-only cross-check
    function) or exposed via `--neuralwatt-judge-models` -- that is a separate, still-unbuilt
    extension point (see that flag's help text and run_multi_judge_faithfulness_crosscheck()'s
    docstring), not a gap in NeuralWatt access itself.

    base_url: defaults to `https://api.neuralwatt.com/v1`, overridable via NEURALWATT_BASE_URL.
    THIS DEFAULT URL IS CONFIRMED WORKING -- verified via the live battery referenced above, not
    a placeholder guess.

    api_key: read from the NEURALWATT_API_KEY environment variable. Unlike
    _default_judge_llm()/_default_judge_embeddings() elsewhere in this module (which fall back
    to an alternate provider when a credential is missing), this raises a RuntimeError instead
    of silently substituting a fake/placeholder key or falling back to another provider --
    NeuralWatt has no established fallback path in this project yet, so a missing key should be a
    loud, actionable failure rather than a silent wrong-provider substitution.

    Args:
        model_name: a NeuralWatt model ID, e.g. one of NEURALWATT_MODELS's keys
            ("glm-5.2-short", "kimi-k2.7-code", "qwen3.5-397b").
        temperature: forwarded to ChatOpenAI (default 0.1, matching this module's other judge
            builders' preference for low-temperature, low-variance judging).

    Returns:
        A LangchainLLMWrapper wrapping a langchain_openai.ChatOpenAI client configured for
        NeuralWatt -- constructed only; this function never calls .invoke() or otherwise makes a
        network request.

    Raises:
        RuntimeError: if NEURALWATT_API_KEY is not set.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print(
            "langchain-openai is not installed (pip install -r requirements.txt) — cannot build "
            "the NeuralWatt judge LLM."
        )
        raise

    api_key = os.environ.get("NEURALWATT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEURALWATT_API_KEY is not set. Set it to a valid NeuralWatt API key before calling "
            "_neuralwatt_llm() (e.g. `export NEURALWATT_API_KEY=...`). This function will not "
            "silently fall back to a fake key or another provider."
        )

    # CONFIRMED working default -- verified via the live 3-model x 6-file x 40-row battery (see
    # docs/CHANGELOG.md and /tmp/nfcorpus_eval_v2/neuralwatt_battery_v2_summary.txt).
    # NeuralWatt's OpenAI-API-compatibility is likewise confirmed by that same live battery.
    base_url = os.environ.get("NEURALWATT_BASE_URL", "https://api.neuralwatt.com/v1")

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            # S3 mitigation: request_timeout/max_retries were previously unset, so a hung
            # request had no client-side deadline and a single transient failure was never
            # retried. 90s + 2 retries is a mitigation for observed Cloudflare 524 gateway
            # timeouts on at least one NeuralWatt model, not a full fix for genuine backend
            # instability -- a model that is actually down will still eventually surface as a
            # failed judge (see _score_one_model()'s broad except), just after fewer wasted
            # minutes than an unbounded hang would cost.
            request_timeout=90,
            max_retries=2,
        )
    )


def _load_faithfulness_dataset_from_csv(csv_path: str) -> "EvaluationDataset":
    """Shared CSV -> EvaluationDataset loader for the multi-judge faithfulness
    cross-check functions (run_multi_judge_faithfulness_crosscheck() and
    run_neuralwatt_multi_judge_consensus()).

    Factored out because both functions need the exact same thing: read a
    previously saved per-question RAGAS results CSV (columns "user_input",
    "retrieved_contexts", "response", "reference"), parse the stringified
    "retrieved_contexts" list column back via ast.literal_eval, and build a
    ragas EvaluationDataset of SingleTurnSamples from it. Neither function's
    surrounding flow (which judges to build, how to aggregate results) is
    shared, so only this loading step is pulled out rather than merging the
    two functions wholesale.

    Raises:
        ValueError: if csv_path is missing any required column.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    required_columns = ["user_input", "retrieved_contexts", "response", "reference"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s) {missing}. Expected a CSV saved from a "
            "RAGAS EvaluationResult (result.to_pandas()) via run_ragas_evaluation() or similar."
        )

    samples = []
    for _, row in df.iterrows():
        retrieved_contexts = row["retrieved_contexts"]
        if isinstance(retrieved_contexts, str):
            try:
                retrieved_contexts = ast.literal_eval(retrieved_contexts)
            except (ValueError, SyntaxError):
                retrieved_contexts = [retrieved_contexts]
        if not isinstance(retrieved_contexts, list):
            retrieved_contexts = list(retrieved_contexts) if retrieved_contexts else []

        samples.append(
            SingleTurnSample(
                user_input=row["user_input"],
                retrieved_contexts=[str(c) for c in retrieved_contexts],
                response=str(row["response"]) if not pd.isna(row["response"]) else "",
                reference=str(row["reference"]) if not pd.isna(row["reference"]) else "",
            )
        )

    return EvaluationDataset(samples=samples)


def run_multi_judge_faithfulness_crosscheck(
    csv_path: str,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Re-judge a saved per-question RAGAS results CSV's faithfulness metric
    using several independent local Ollama models, as a cross-check against
    the original (Claude) judge's faithfulness column.

    This is a NEW, separate capability from run_ragas_evaluation() /
    run_closed_book_evaluation() / run_claude_websearch_evaluation() above —
    it does not generate any new answers or run retrieval; it re-scores
    faithfulness only, on already-generated (user_input, retrieved_contexts,
    response, reference) rows, using a different judge each time. See the
    module docstring's "MULTI-JUDGE FAITHFULNESS CROSS-CHECK" section for the
    motivation (Gemini was tried and rejected as a second judge for being a
    stale model version; this uses three independent-family Ollama models
    instead).

    Args:
        csv_path: Path to a CSV previously saved from a RAGAS
            EvaluationResult (e.g. via `result.to_pandas().to_csv(...)`),
            expected to contain at least the columns "user_input",
            "retrieved_contexts", "response", and "reference". "retrieved_contexts"
            is expected to be a stringified Python list (pandas' default CSV
            serialization of a list-typed column), parsed back via
            ast.literal_eval. Any existing metric columns (e.g. a
            "faithfulness" column from the original Claude judge run) are
            left untouched and simply available for comparison by the caller.
        model_names: Ollama model tags to cross-check with. Defaults to
            DEFAULT_OLLAMA_JUDGE_MODELS (glm-5.2:cloud, kimi-k2.7-code:cloud,
            minimax-m3:cloud).

    NOT-YET-WIRED-HERE NEURALWATT INTEGRATION (not implemented in THIS function, default behavior
    unchanged): this function is hardcoded to build each judge via _ollama_judge_llm(model_name).
    NeuralWatt API access and its OpenAI-API-compatibility are CONFIRMED and live-verified (see
    NEURALWATT_MODELS / _neuralwatt_llm() / run_neuralwatt_multi_judge_consensus(), the latter of
    which already runs a real NeuralWatt-backed multi-judge panel end-to-end) -- what remains
    unbuilt is specifically folding a NeuralWatt-backed member into *this* Ollama-only function's
    judge list, e.g.:

        results = run_multi_judge_faithfulness_crosscheck(
            csv_path,
            model_names=[*DEFAULT_OLLAMA_JUDGE_MODELS, "glm-5.2-short"],
        )
        # ...and inside the loop, dispatch on a naming convention or an explicit
        # judge_builder=lambda name: _neuralwatt_llm(name) if name in NEURALWATT_MODELS
        #                            else _ollama_judge_llm(name)

    This is deliberately NOT implemented here: doing so would require this function to either
    guess which models belong to which provider or take on a new parameter, and both changes are
    out of scope while run_neuralwatt_multi_judge_consensus() already serves the live NeuralWatt
    multi-judge use case as its own standalone function. See the `--neuralwatt-judge-models` CLI
    flag (main()) for the corresponding documented-but-inert extension point into this function
    specifically.

    Returns:
        Dict keyed by model_name, where each value is itself a dict:
            {"result": <ragas EvaluationResult or None>, "error": <str or None>}
        so that one model's failure (e.g. that model isn't pulled locally, or
        Ollama isn't running) doesn't prevent the others from being scored.
        Structured this way (independent per-model try/except, no shared
        state across iterations) specifically so callers can parallelize the
        per-model calls later (e.g. via concurrent.futures) without needing
        to restructure this function — each iteration's evaluate() call is
        already fully self-contained.

    NOTE: this function is complete and ready to run, but per this task's
    scope it has intentionally not been invoked/tested against a live Ollama
    server. Before running for real, confirm `ollama list` shows all of
    model_names pulled.
    """
    if model_names is None:
        model_names = DEFAULT_OLLAMA_JUDGE_MODELS

    dataset = _load_faithfulness_dataset_from_csv(csv_path)

    results: Dict[str, Any] = {}
    for model_name in model_names:
        try:
            judge_llm = _ollama_judge_llm(model_name)
            # M1 fix: construct a FRESH Faithfulness() instance per iteration instead of
            # passing the shared module-level `faithfulness` singleton imported at the top of
            # this file. ragas's evaluate() mutates `metric.llm` in place as part of run setup
            # (a check-then-act race under concurrency); this loop runs sequentially today so
            # the race can't actually fire yet, but it shares the identical bug pattern as
            # _score_one_model()/_claude_tiebreaker_faithfulness() (see docs/CHANGELOG.md) and
            # this function's own docstring already documents parallelization as a plausible
            # future extension -- fixing it now avoids silently reintroducing the same race the
            # moment someone adds a ThreadPoolExecutor here.
            result = evaluate(
                dataset=dataset,
                metrics=[Faithfulness()],
                llm=judge_llm,
                run_config=RunConfig(max_workers=_OLLAMA_JUDGE_MAX_WORKERS),
            )
            results[model_name] = {"result": result, "error": None}
        except Exception as exc:  # noqa: BLE001 - one bad judge shouldn't kill the others
            results[model_name] = {"result": None, "error": str(exc)}

    return results


def _geometric_median(values: List[float], tol: float = 1e-8, max_iter: int = 500) -> float:
    """Compute the geometric median (a.k.a. L1 median / spatial median) of a small set of
    scalar judge scores via Weiszfeld's algorithm (Weiszfeld, "Sur le point pour lequel la
    somme des distances de n points donnes est minimum," 1937; modernized convergence
    treatment in Vardi & Zhang, "The multivariate L1-median and associated data depth,"
    PNAS 97(4):8442-8447, 2000). The geometric median is the point that minimizes the SUM
    OF (unsquared) DISTANCES to every input point -- unlike the arithmetic mean (which
    minimizes squared distance and is pulled toward outliers) it is a robust estimator of
    central tendency. This is exactly why RoPoLL (arXiv:2606.30931) recommends geometric-
    median aggregation over mean/majority-vote when combining scores from multiple LLM
    judges: one miscalibrated/adversarial judge should not dominate the aggregate the way
    it would under a mean.

    NOTE on the 1-D case used here: for purely scalar inputs, the point minimizing the sum
    of absolute distances is mathematically IDENTICAL to the ordinary median -- this is a
    real, well-known property of the geometric median in one dimension, not a shortcut
    substitution dressed up with a fancier name. This function still performs Weiszfeld's
    explicit iterative re-weighting (rather than just calling `statistics.median()`)
    because (a) that's the actual named algorithm this task asked for, with a citable
    source, and (b) the same implementation generalizes correctly if this is ever extended
    to jointly aggregate multiple metrics per judge as a multi-dimensional point (e.g.
    (faithfulness, answer_relevancy) pairs), at which point mean and median no longer
    coincide with the geometric median but this iteration still would.

    Weiszfeld's update rule: starting from an initial estimate y (here, the arithmetic
    mean), repeatedly re-weight each point x_i by 1/|x_i - y| and replace y with the
    resulting weighted average. Repeat until the estimate moves less than `tol` between
    iterations or `max_iter` is reached.

    Args:
        values: the scalar scores to aggregate (e.g. one judge model's score per question).
        tol: convergence threshold on the change in the estimate between iterations.
        max_iter: safety cap on iterations (Weiszfeld's algorithm converges quickly for
            small n, but is not guaranteed to converge in pathological cases).

    Returns:
        The geometric median as a float. For 0 values this raises ValueError; for exactly
        1 value it is returned unchanged; for 2+ identical values that value is returned.

    Raises:
        ValueError: if `values` is empty.
    """
    pts = [float(v) for v in values]
    if not pts:
        raise ValueError("_geometric_median requires at least one value")
    if len(pts) == 1:
        return pts[0]

    y = sum(pts) / len(pts)  # Weiszfeld initializes from the mean.
    for _ in range(max_iter):
        distances = [abs(x - y) for x in pts]
        if any(d < 1e-12 for d in distances):
            # y coincides (within float tolerance) with one of the input points, where
            # Weiszfeld's classic update is undefined (division by zero in the 1/|x_i - y|
            # weight). Returning that point directly is the standard, documented handling
            # of this edge case (see Vardi & Zhang, 2000).
            return pts[distances.index(min(distances))]
        weights = [1.0 / d for d in distances]
        total_weight = sum(weights)
        y_new = sum(w * x for w, x in zip(weights, pts)) / total_weight
        if abs(y_new - y) < tol:
            return y_new
        y = y_new
    return y


def _claude_tiebreaker_faithfulness(sample: "SingleTurnSample") -> Optional[float]:
    """Score a single (user_input, retrieved_contexts, response, reference) sample's
    faithfulness using the SAME default Claude judge machinery already used elsewhere in
    this module (_default_judge_llm(), which prefers direct ANTHROPIC_API_KEY and falls
    back to Claude on Vertex AI via ANTHROPIC_VERTEX_PROJECT_ID -- see that function's
    docstring), rather than inventing a new prompt/parsing path. Used as the tiebreaker
    for rows run_neuralwatt_multi_judge_consensus() flags as "high disagreement" among the
    three NeuralWatt judges, per the ProofAgent Harness "Consensus Agent" pattern of
    escalating high-spread metrics to an additional independent juror.

    Returns:
        The faithfulness score (float, 0-1) as a plain Python float, or None if no Claude
        judge LLM is configured (matching _default_judge_llm()'s own None-on-missing-
        credentials convention) or if the evaluate() call fails for any reason -- a failed
        tiebreaker should not crash the whole consensus run.
    """
    judge_llm = _default_judge_llm()
    if judge_llm is None:
        return None

    try:
        dataset = EvaluationDataset(samples=[sample])
        # M1 fix: a fresh Faithfulness() instance, not the shared module-level `faithfulness`
        # singleton -- this function is called from run_neuralwatt_multi_judge_consensus()'s
        # per-question loop while that same module-level object may simultaneously be in use
        # by a concurrently-running _score_one_model() worker thread (or a future parallelized
        # call to this very function), and ragas mutates `metric.llm` in place as part of run
        # setup (see docs/CHANGELOG.md's dated entry on this race).
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness()],
            llm=judge_llm,
            # M3: own named constant (_CLAUDE_TIEBREAKER_MAX_WORKERS), deliberately NOT
            # _NEURALWATT_JUDGE_MAX_WORKERS -- see that constant's module-level comment. A
            # single-sample evaluate() call only ever does one unit of work regardless of this
            # value, so this is a no-op today; it exists so a future change that batches
            # multiple flagged rows into one evaluate() call doesn't silently inherit a
            # constant tuned for NeuralWatt's rate limit instead of Claude's.
            run_config=RunConfig(max_workers=_CLAUDE_TIEBREAKER_MAX_WORKERS),
        )
        df = result.to_pandas()
        return float(df["faithfulness"].iloc[0])
    except Exception as exc:  # noqa: BLE001 - tiebreaker is best-effort, never fatal
        print(f"_claude_tiebreaker_faithfulness: evaluate() failed ({exc}) — no tiebreaker score.")
        return None


def run_neuralwatt_multi_judge_consensus(
    csv_path: str,
    model_names: Optional[List[str]] = None,
    concurrency: int = 3,
    disagreement_threshold: float = 0.3,
    unanimous_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Re-judge a saved per-question RAGAS results CSV's faithfulness metric using three
    independent NeuralWatt-hosted judge models IN PARALLEL, then aggregate the three
    scores per question via geometric median and flag rows for disagreement or
    suspicious unanimity — the NeuralWatt counterpart to
    run_multi_judge_faithfulness_crosscheck() (which uses local Ollama models), extended
    with the consensus-aggregation logic that function's docstring documents as a
    deliberately-deferred future extension.

    This does NOT generate any new answers or run retrieval; like the Ollama version, it
    re-scores faithfulness only, on already-generated (user_input, retrieved_contexts,
    response, reference) rows (see _load_faithfulness_dataset_from_csv(), shared with the
    Ollama version).

    PARALLELISM: each of the 3 judge models' evaluate() call runs in its own worker via
    concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) — this is a genuinely
    new mechanism, not a match to an existing one: run_multi_judge_faithfulness_crosscheck()
    above actually runs its per-model loop sequentially today (its docstring's "Returns"
    section notes the per-model try/except was structured to make later parallelization
    via concurrent.futures easy, but that parallelization was never added). `concurrency`
    defaults to 3 to match len(model_names) — i.e. all three judges run genuinely
    concurrently by default.

    NeuralWatt's rate limit is CONFIRMED to be 3 concurrent requests ACCOUNT-WIDE (not
    per-model — verified directly with the user, not an assumption). This design allocates
    all 3 of those slots across the panel: the OUTER ThreadPoolExecutor(max_workers=concurrency)
    here gives one slot per judge model, and each judge's OWN evaluate() call is internally
    capped to RunConfig(max_workers=_NEURALWATT_JUDGE_MAX_WORKERS) (=1) inside
    _score_one_model() so it only ever uses its one allocated slot at a time rather than also
    fanning out internally. Net effect: total concurrent NeuralWatt requests never exceeds 3,
    exactly matching the account-wide limit, while all 3 judge models still run genuinely
    concurrently with each other — see _NEURALWATT_JUDGE_MAX_WORKERS's module-level comment for
    the full reasoning. `concurrency` remains a parameter so it can be capped lower still if
    ever needed, but should not be raised above 3 without first getting NeuralWatt to raise the
    account-wide limit itself.

    CONSENSUS AGGREGATION (per question, per metric — here just "faithfulness"):
      1. Collect all 3 judges' faithfulness scores for that question.
      2. Compute their geometric median via _geometric_median() (Weiszfeld's algorithm;
         see that function's docstring for the citation and why geometric median over
         mean/majority-vote — RoPoLL, arXiv:2606.30931).
      3. Compute the spread delta_m = max(scores) - min(scores) (the ProofAgent Harness
         "Consensus Agent" pattern's disagreement measure).
      4. Flag "high_disagreement" = delta_m > disagreement_threshold (default 0.3 on the
         0-1 faithfulness scale).
      5. ALSO flag "suspiciously_unanimous" = delta_m < unanimous_tolerance (default
         0.05) — per Kohli 2026 (arXiv:2605.29800, "Nine Judges, Two Effective Votes"),
         unanimous agreement across judges can reflect a shared/correlated blind spot
         rather than extra reliability, and should be surfaced for a human to sanity-check
         rather than trusted unconditionally just because all judges agree. Note
         "high_disagreement" and "suspiciously_unanimous" are mutually exclusive given
         disagreement_threshold > unanimous_tolerance (the default 0.3 > 0.05); if a
         caller passes thresholds where that no longer holds, both flags are computed
         independently and could theoretically both be False without contradiction, but
         cannot both be True for the same row. This mutual-exclusivity claim only holds
         when at least 2 judges produced a valid score for that row (len(valid_scores) >=
         2) — with 0 or 1 valid scores there is no real agreement/disagreement to measure,
         so delta_m is None and BOTH flags are always False (see point 6.5 below; this was
         a real bug (fixed) prior to this note being added — 1 valid score used to force
         suspiciously_unanimous=True unconditionally).
      6. For "high_disagreement" rows OR rows where every judge failed (0 valid scores,
         "all_judges_failed"), calls _claude_tiebreaker_faithfulness() to get a 4th,
         independent opinion from this project's existing default Claude judge — the
         row's dict then carries all 4 individual scores (3 NeuralWatt + Claude) plus the
         geometric median and both flags, so a human reviewer sees every opinion at once
         rather than just the aggregate. An all-judges-failed row is exactly the case most
         in need of a tiebreaker, and would otherwise never get one (delta_m is None there,
         so high_disagreement alone is False) — see "all_judges_failed" in Returns below.
      6.5. When fewer than 2 judges produced a valid score, "delta_m" is None (not 0.0) and
         both "high_disagreement" and "suspiciously_unanimous" are False — there is no
         agreement/disagreement to measure with 0 or 1 data points, and treating 1 valid
         score as automatically "unanimous" was a bug this function used to have.

    Args:
        csv_path: path to a CSV previously saved from a RAGAS EvaluationResult (e.g. via
            `result.to_pandas().to_csv(...)`), same shape required by
            run_multi_judge_faithfulness_crosscheck() / _load_faithfulness_dataset_from_csv().
        model_names: NeuralWatt model IDs to use as the 3-judge panel. Defaults to
            ["glm-5.2-short", "kimi-k2.7-code", "qwen3.5-397b"] (all three keys of
            NEURALWATT_MODELS).
        concurrency: max worker threads for the parallel per-model evaluate() calls.
            Defaults to 3 (one per default judge model).
        disagreement_threshold: delta_m above which a row is flagged "high_disagreement"
            and gets a Claude tiebreaker. Default 0.3 on the 0-1 faithfulness scale.
        unanimous_tolerance: delta_m below which a row is flagged "suspiciously_unanimous".
            Default 0.05.

    OLLAMA-CLOUD FALLBACK: if a judge model's NeuralWatt evaluate() call fails for any reason
    (timeout, rate-limit, outage, etc. — see _score_one_model()), ONE retry is attempted via the
    equivalent Ollama-hosted cloud model (_NEURALWATT_TO_OLLAMA_FALLBACK) before that model is
    reported as failed. The returned per-model dict's "judge_used" field ("neuralwatt" |
    "ollama_fallback" | None) records which provider actually produced the score, so a caller can
    tell "NeuralWatt succeeded outright" apart from "had to fall back to Ollama" — relevant to the
    cross-provider reliability comparison this panel exists to make. On a successful fallback, the
    original NeuralWatt failure is preserved in "neuralwatt_error"/"neuralwatt_error_type" rather
    than being discarded, so the fallback never silently hides which provider actually had the
    problem.

    Returns:
        Dict with two keys:
          "per_model": Dict[model_name, {"result": <ragas EvaluationResult or None>,
              "error": <str or None>, "error_type": <"config" | "runtime" | None>,
              "judge_used": <"neuralwatt" | "ollama_fallback" | None>,
              "neuralwatt_error": <str, only present when judge_used == "ollama_fallback" or
                  the fallback itself also failed>,
              "neuralwatt_error_type": <"config" | "runtime", same presence condition>}] — the
              base "result"/"error" shape matches run_multi_judge_faithfulness_crosscheck()'s
              return value, so a model that failed (bad model ID, NeuralWatt outage, etc.)
              doesn't prevent the others from being scored or reported; "error_type",
              "judge_used", and the "neuralwatt_error*" keys are additions specific to this
              function's NeuralWatt-plus-fallback design.
          "per_question": List[Dict] (one dict per CSV row, in original row order), each
              with keys "user_input", "neuralwatt_scores" (Dict[model_name, float or
              None]), "geometric_median" (float or None if 0 models succeeded for that
              row), "delta_m" (float or None — None when fewer than 2 models produced a
              valid score for that row, since there is nothing to compute a spread over),
              "high_disagreement" (bool — always False when delta_m is None),
              "suspiciously_unanimous" (bool — always False when delta_m is None; a single
              valid score is NOT treated as unanimous agreement), "all_judges_failed"
              (bool — True when 0 of the 3 NeuralWatt judges produced a valid score for
              this row), and "claude_tiebreaker" (float or None — populated, i.e.
              non-None, for rows where high_disagreement OR all_judges_failed is True AND
              the tiebreaker call itself succeeded).

    NOTE: this function has been run live end-to-end, including the Claude tiebreaker path -- a
    completed 3-model x 6-file x 40-row battery (720 real judge calls, zero NaN results via the
    escalation chain) is documented in docs/CHANGELOG.md and
    /tmp/nfcorpus_eval_v2/neuralwatt_battery_v2_summary.txt. NEURALWATT_API_KEY must still be set
    (see _neuralwatt_llm()) to run this again; NeuralWatt's OpenAI-API-compatibility is confirmed,
    not assumed.
    """
    import pandas as pd

    if model_names is None:
        model_names = ["glm-5.2-short", "kimi-k2.7-code", "qwen3.5-397b"]

    dataset = _load_faithfulness_dataset_from_csv(csv_path)
    samples = dataset.samples

    def _score_one_model(model_name: str) -> Tuple[str, Dict[str, Any]]:
        def _run_faithfulness(judge_llm, max_workers: int):
            # Shared by both the primary NeuralWatt attempt and the Ollama fallback below --
            # both need the same M1 fix (a fresh Faithfulness() instance, never the shared
            # module-level `faithfulness` singleton) since this function runs concurrently
            # (one call per judge model, inside the ThreadPoolExecutor below) and ragas's
            # evaluate() mutates `metric.llm` in place as part of run setup. Passing the SAME
            # shared object into 3 simultaneous evaluate() calls (one per judge, each with a
            # different `llm=`) is a cross-thread check-then-act race on that mutation. This
            # was reproduced live: a full run at concurrency=3 with the shared singleton
            # returned 100% NaN faithfulness for all 3 NeuralWatt models (see
            # docs/CHANGELOG.md's dated entry).
            return evaluate(
                dataset=dataset,
                metrics=[Faithfulness()],
                llm=judge_llm,
                run_config=RunConfig(max_workers=max_workers),
            )

        try:
            judge_llm = _neuralwatt_llm(model_name)
            # M2/S4: pass an explicit run_config= instead of relying on ragas's own default
            # RunConfig(max_workers=16). NeuralWatt's rate limit is CONFIRMED to be 3 concurrent
            # requests account-wide (not per-model, per direct user confirmation) -- max_workers=1
            # here means each judge thread processes its own dataset's rows one at a time,
            # combined with the OUTER ThreadPoolExecutor(max_workers=concurrency) below (one
            # thread per judge model, default 3) this allocates exactly one of NeuralWatt's 3
            # account-wide slots to each judge, so TOTAL concurrent NeuralWatt requests never
            # exceeds 3 -- see _NEURALWATT_JUDGE_MAX_WORKERS's module-level comment for the full
            # reasoning on why this specific split was chosen.
            result = _run_faithfulness(judge_llm, _NEURALWATT_JUDGE_MAX_WORKERS)
            return model_name, {
                "result": result,
                "error": None,
                "error_type": None,
                "judge_used": "neuralwatt",
            }
        except Exception as exc:  # noqa: BLE001 - one bad judge shouldn't kill the others
            # M4/S2 fix: distinguish a loud, actionable config error (e.g. NEURALWATT_API_KEY
            # unset -- _neuralwatt_llm() raises RuntimeError specifically for this, by design,
            # per its own docstring) from a transient runtime/network/API failure. Without this,
            # a simple missing env var and 3 independent API outages look identical in the
            # returned error dict, and a caller/log reader has no signal which one-line fix (set
            # an env var) vs. which multi-model outage they're actually looking at.
            neuralwatt_error = str(exc)
            neuralwatt_error_type = "config" if isinstance(exc, RuntimeError) else "runtime"

            # Ollama-cloud fallback: added directly in response to live TimeoutError
            # instability observed against api.neuralwatt.com during verification (see
            # docs/CHANGELOG.md's dated entry) -- rather than immediately giving up on this
            # judge model, attempt ONE retry via the equivalent Ollama-hosted cloud model
            # (_NEURALWATT_TO_OLLAMA_FALLBACK), reusing the SAME _ollama_judge_llm() builder
            # already used by run_multi_judge_faithfulness_crosscheck() (no duplicated
            # client-construction logic). Uses _OLLAMA_JUDGE_MAX_WORKERS (not
            # _NEURALWATT_JUDGE_MAX_WORKERS) for its run_config since Ollama is local and not
            # subject to NeuralWatt's 3-slot account-wide limit. "judge_used" in the returned
            # dict records which provider actually produced the score, and
            # "neuralwatt_error"/"neuralwatt_error_type" preserve the original failure even on
            # a successful fallback, so the aggregation/reporting layer (and a human reading
            # the output) can distinguish "NeuralWatt succeeded outright" from "had to fall
            # back to Ollama" -- this matters for the cross-provider reliability comparison the
            # multi-judge panel exists to make, and keeps the M4/S2 config-vs-runtime signal
            # visible even when the fallback masks the failure from the caller's immediate
            # perspective.
            fallback_model = _NEURALWATT_TO_OLLAMA_FALLBACK.get(model_name)
            if fallback_model is not None:
                try:
                    fallback_llm = _ollama_judge_llm(fallback_model)
                    result = _run_faithfulness(fallback_llm, _OLLAMA_JUDGE_MAX_WORKERS)
                    return model_name, {
                        "result": result,
                        "error": None,
                        "error_type": None,
                        "judge_used": "ollama_fallback",
                        "neuralwatt_error": neuralwatt_error,
                        "neuralwatt_error_type": neuralwatt_error_type,
                    }
                except Exception as fallback_exc:  # noqa: BLE001 - fallback is best-effort too
                    return model_name, {
                        "result": None,
                        "error": (
                            f"NeuralWatt failed ({neuralwatt_error}); Ollama fallback "
                            f"({fallback_model}) also failed ({fallback_exc})"
                        ),
                        "error_type": "runtime",
                        "judge_used": None,
                        "neuralwatt_error": neuralwatt_error,
                        "neuralwatt_error_type": neuralwatt_error_type,
                    }

            # No fallback mapped for this model_name (shouldn't happen for the 3 default
            # model names -- only reachable if a caller passes a custom model_names list with
            # a model not in _NEURALWATT_TO_OLLAMA_FALLBACK). The broad `except Exception`
            # itself is kept intentionally (consistent with
            # run_multi_judge_faithfulness_crosscheck()'s sibling design: "one bad judge
            # shouldn't kill the others") -- the resulting dict still distinguishes config vs.
            # runtime failures per M4/S2.
            return model_name, {
                "result": None,
                "error": neuralwatt_error,
                "error_type": neuralwatt_error_type,
                "judge_used": None,
            }

    per_model: Dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, len(model_names))) as pool:
        futures = [pool.submit(_score_one_model, name) for name in model_names]
        for future in concurrent.futures.as_completed(futures):
            model_name, outcome = future.result()
            per_model[model_name] = outcome

    # Pull each successful model's per-question faithfulness column, in the SAME row
    # order as `samples`/the source CSV (ragas preserves input sample order in
    # to_pandas()), so scores can be zipped together by row index below.
    per_model_scores: Dict[str, List[Optional[float]]] = {}
    for model_name, outcome in per_model.items():
        if outcome["error"] is not None:
            per_model_scores[model_name] = [None] * len(samples)
            continue
        col = outcome["result"].to_pandas()["faithfulness"]
        per_model_scores[model_name] = [
            None if pd.isna(v) else float(v) for v in col.tolist()
        ]

    # --- ZERO-NaN GUARANTEE: per-row escalation for silent ragas-internal NaNs ---
    #
    # ragas's evaluate() has its OWN internal per-row retry logic. When it exhausts those
    # retries for a specific row, it does NOT raise -- it just leaves NaN in that row's score
    # and returns normally. The per-model exception handling in _score_one_model() above (the
    # whole-dataset Ollama-cloud fallback) can only ever catch an exception from the WHOLE
    # evaluate() call, so it NEVER catches these individual silent per-row NaNs -- confirmed
    # live: a 5-row NeuralWatt run had 2/5 rows NaN for "qwen3.5-397b" even with that fallback
    # active, because no exception was ever raised for it to catch.
    #
    # Fix: for every row that is STILL None (NaN) after the block above, retry ONLY that row
    # (never the whole batch) through an escalation chain, in priority order:
    #   1. NeuralWatt succeeded on the whole dataset but this row came back NaN -> retry this
    #      one row via the Ollama-cloud fallback model for this judge (same
    #      _NEURALWATT_TO_OLLAMA_FALLBACK mapping the whole-dataset fallback already uses).
    #      Built as a fresh single-row EvaluationDataset containing just this one
    #      SingleTurnSample and evaluated on its own -- mirrors
    #      _claude_tiebreaker_faithfulness()'s existing one-sample pattern rather than
    #      inventing a new one.
    #   2. If the whole dataset ALREADY fell back to Ollama (this row was NaN there too), or
    #      the whole model failed outright (nothing to retry with), or the single-row Ollama
    #      retry in step 1 ALSO comes back NaN/fails: escalate to the Claude tiebreaker
    #      (_claude_tiebreaker_faithfulness()) as the final backstop.
    #   3. If even the Claude tiebreaker fails (e.g. no ANTHROPIC_API_KEY/Vertex access): the
    #      row is loudly logged and flagged "unrecoverable" (score stays None,
    #      "score_unrecoverable": True on the per-question row) -- an acceptable final failure
    #      mode, but a VISIBLE one, never a silently-reintroduced NaN.
    #
    # End state: the only way a row's score can still be None after this loop is if NeuralWatt,
    # its Ollama-cloud fallback, AND the Claude tiebreaker all failed for that row -- everything
    # else gets a real number. `row_judge_used[model_name][i]` records the full escalation-chain
    # outcome for that specific (model, row): "neuralwatt" | "ollama_fallback" |
    # "claude_tiebreaker_escalation" | "unrecoverable".
    row_judge_used: Dict[str, List[Optional[str]]] = {
        model_name: [
            (per_model[model_name].get("judge_used") or "unrecoverable")
            if per_model_scores[model_name][i] is not None
            else None
            for i in range(len(samples))
        ]
        for model_name in model_names
    }

    for model_name in model_names:
        fallback_model = _NEURALWATT_TO_OLLAMA_FALLBACK.get(model_name)
        whole_dataset_judge = per_model[model_name].get("judge_used")

        for i, sample in enumerate(samples):
            if per_model_scores[model_name][i] is not None:
                continue  # already has a real score -- nothing to escalate

            escalated_score: Optional[float] = None
            escalated_via: Optional[str] = None

            # Step 1: single-row Ollama-cloud retry. Only attempted when the WHOLE dataset
            # ran on NeuralWatt primary (this row is a genuine ragas-internal silent NaN, not
            # already an Ollama result) -- if the whole dataset already fell back to Ollama,
            # this row already got its Ollama shot and came back NaN there too; retrying the
            # identical model+row combination again would not change the outcome.
            if fallback_model is not None and whole_dataset_judge == "neuralwatt":
                try:
                    ollama_llm = _ollama_judge_llm(fallback_model)
                    single_row_dataset = EvaluationDataset(samples=[sample])
                    # Fresh Faithfulness() instance per M1 (never the shared module-level
                    # singleton) -- see the fix note on _run_faithfulness() above.
                    single_result = evaluate(
                        dataset=single_row_dataset,
                        metrics=[Faithfulness()],
                        llm=ollama_llm,
                        run_config=RunConfig(max_workers=_OLLAMA_JUDGE_MAX_WORKERS),
                    )
                    val = single_result.to_pandas()["faithfulness"].iloc[0]
                    if not pd.isna(val):
                        escalated_score = float(val)
                        escalated_via = "ollama_fallback"
                except Exception as exc:  # noqa: BLE001 - row-level retry is best-effort
                    print(
                        f"NaN row escalation: single-row Ollama fallback ({fallback_model}) "
                        f"for {model_name!r} row {i} failed ({exc}) -- escalating to the "
                        "Claude tiebreaker."
                    )

            # Step 2/3: Claude tiebreaker as the final backstop, or loud "unrecoverable".
            if escalated_score is None:
                claude_score = _claude_tiebreaker_faithfulness(sample)
                if claude_score is not None:
                    escalated_score = claude_score
                    escalated_via = "claude_tiebreaker_escalation"
                else:
                    escalated_via = "unrecoverable"
                    print(
                        f"UNRECOVERABLE NaN: {model_name!r} row {i} could not be scored by "
                        "NeuralWatt, its Ollama-cloud fallback, or the Claude tiebreaker "
                        f"(score_unrecoverable=True). user_input: {sample.user_input[:80]!r}"
                    )

            per_model_scores[model_name][i] = escalated_score
            row_judge_used[model_name][i] = escalated_via

    per_question: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        row_scores = {
            model_name: per_model_scores[model_name][i] for model_name in model_names
        }
        row_judges = {
            model_name: row_judge_used[model_name][i] for model_name in model_names
        }
        valid_scores = [v for v in row_scores.values() if v is not None]

        geometric_median = _geometric_median(valid_scores) if valid_scores else None

        # Post-escalation: with the zero-NaN guarantee above, this should only ever be True
        # if EVERY judge's full escalation chain (NeuralWatt -> Ollama fallback -> Claude
        # tiebreaker) failed for this row -- an extreme, and extremely rare, edge case.
        all_judges_failed = len(valid_scores) == 0

        if len(valid_scores) >= 2:
            delta_m = max(valid_scores) - min(valid_scores)
            high_disagreement = delta_m > disagreement_threshold
            suspiciously_unanimous = delta_m < unanimous_tolerance
        else:
            # S5 fix: with fewer than 2 valid scores there is no actual agreement or
            # disagreement to measure. The old code hardcoded delta_m=0.0 for exactly 1 valid
            # score, which made suspiciously_unanimous unconditionally True (any positive
            # unanimous_tolerance exceeds 0.0) even though a single judge cannot "agree" with
            # itself -- semantically wrong, and post-fix (once M1-M4 stop suppressing real
            # NeuralWatt failures into false single-survivor rows) likely to occur often, since
            # "qwen3.5-397b" has a genuinely high real-world failure rate independent of these
            # bugs. delta_m=None here (distinct from a real delta_m of 0.0, which means genuine
            # unanimous agreement across >=2 judges) makes "insufficient data to assess
            # agreement" distinguishable from "these judges genuinely agreed".
            delta_m = None
            high_disagreement = False
            suspiciously_unanimous = False

        # S6 fix: a row where the ENTIRE panel failed (0 valid scores) is the row MOST in need
        # of a tiebreaker, but high_disagreement is False here (delta_m is None, and
        # `delta_m is not None and delta_m > threshold` short-circuits to False) -- without this
        # explicit OR such a row would never get a Claude tiebreaker and would silently blend
        # into "boring agreement" statistics instead of being visible as its own failure mode.
        # (This is the ORIGINAL disagreement/all-failed-triggered tiebreaker, kept as-is and
        # independent of the per-row NaN-escalation tiebreaker calls above -- a row can get a
        # per-model NaN-escalation tiebreaker score for one judge AND this separate consensus
        # tiebreaker if the row also qualifies as high_disagreement/all_judges_failed.)
        needs_tiebreaker = high_disagreement or all_judges_failed

        claude_tiebreaker = None
        if needs_tiebreaker:
            claude_tiebreaker = _claude_tiebreaker_faithfulness(sample)

        per_question.append(
            {
                "user_input": sample.user_input,
                "neuralwatt_scores": row_scores,
                "judge_used": row_judges,
                "score_unrecoverable": any(v == "unrecoverable" for v in row_judges.values()),
                "geometric_median": geometric_median,
                "delta_m": delta_m,
                "high_disagreement": high_disagreement,
                "suspiciously_unanimous": suspiciously_unanimous,
                "all_judges_failed": all_judges_failed,
                "claude_tiebreaker": claude_tiebreaker,
            }
        )

    return {"per_model": per_model, "per_question": per_question}


def export_for_human_review(
    testset: List[Dict],
    rag_retrieve_func: Callable[[str], Any],
    llm_generate_func: Callable[[str, List[str]], str],
    output_csv_path: str,
    max_context_chars: int = 4000,
) -> str:
    """Run retrieval + generation over `testset` and export a CSV for the
    judge-INDEPENDENT human review tier (see the module docstring's
    "JUDGE-INDEPENDENT HUMAN REVIEW TIER" section and
    docs/HUMAN_REVIEW_RUBRIC.md for the full rubric).

    This deliberately reuses the exact same retrieval/generation machinery
    as run_ragas_evaluation() (rag_retrieve_func / llm_generate_func,
    typically _make_rag_retrieve_func(...) and generate_answer()) rather
    than duplicating it, so the triples a human reviews are the same ones
    an LLM judge would have scored — the point is a second, independent
    read on the SAME outputs, not a different pipeline.

    Args:
        testset: List of {"question": str, "ground_truths": [str, ...]}.
        rag_retrieve_func: called as rag_retrieve_func(question) -> retrieval
            result (same shape _extract_contexts() already handles).
        llm_generate_func: called as llm_generate_func(question, contexts) ->
            answer string.
        output_csv_path: where to write the CSV. Columns: "question",
            "retrieved_contexts" (each context joined with a "---"
            separator for readability, truncated to max_context_chars),
            "generated_answer", "human_label" (empty — a human fills this
            in with one of HUMAN_REVIEW_LABELS), "human_comment" (empty —
            free-text notes on what worked / what should improve).
        max_context_chars: truncate the joined retrieved_contexts string to
            this many characters (default 4000) so the CSV stays readable
            in a spreadsheet viewer; full contexts are rarely needed for a
            grounding spot-check and very long ones make the sheet unusable.

    Returns:
        output_csv_path, for convenience chaining.
    """
    import csv

    rows = []
    for item in testset:
        question = item["question"]
        retrieval_result = rag_retrieve_func(question)
        contexts = _extract_contexts(retrieval_result)
        answer = llm_generate_func(question, contexts) or ""

        joined_contexts = "\n\n---\n\n".join(contexts) if contexts else "(no context retrieved)"
        if len(joined_contexts) > max_context_chars:
            joined_contexts = joined_contexts[:max_context_chars] + " …[truncated]"

        rows.append(
            {
                "question": question,
                "retrieved_contexts": joined_contexts,
                "generated_answer": answer,
                "human_label": "",
                "human_comment": "",
            }
        )

    fieldnames = ["question", "retrieved_contexts", "generated_answer", "human_label", "human_comment"]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        # A leading "#"-prefixed line documenting the exact valid human_label
        # values, so a reviewer opening the raw CSV sees the rubric summary
        # even without docs/HUMAN_REVIEW_RUBRIC.md open. pandas.read_csv(...,
        # comment="#") (used by summarize_human_review()) skips this line
        # automatically, so it never pollutes the parsed data.
        f.write(
            "# human_label must be exactly one of: "
            + " | ".join(HUMAN_REVIEW_LABELS)
            + " -- see docs/HUMAN_REVIEW_RUBRIC.md for the full rubric.\n"
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) for human review to {output_csv_path}")
    print(f"Valid human_label values: {' | '.join(HUMAN_REVIEW_LABELS)}")
    print("Full rubric: docs/HUMAN_REVIEW_RUBRIC.md")
    return output_csv_path


def summarize_human_review(csv_path: str, ragas_csv_path: Optional[str] = None) -> None:
    """Read back a hand-labeled copy of an export_for_human_review() CSV and
    print a summary: count/percentage per label, plus (if `ragas_csv_path`
    is given) a cross-tabulation of mean RAGAS scores per human label.

    This is the other half of the judge-INDEPENDENT human review tier (see
    the module docstring) — it doesn't just format the human's labels, it
    checks a real, rubric-derived prediction: "Partially grounded" answers
    should score lower on `faithfulness` than "Grounded" answers, since
    faithfulness measures exactly the thing the rubric's middle label calls
    out (mixed [C]/[G] support). If that prediction fails, it's flagged
    rather than silently reported as if everything lined up.

    Args:
        csv_path: path to a CSV in the shape written by
            export_for_human_review(), with "human_label" (and optionally
            "human_comment") now filled in by a human reviewer. Rows with
            an empty/missing human_label are counted as unlabeled and
            excluded from the label distribution and any cross-tab.
        ragas_csv_path: optional path to a saved RAGAS per-question results
            CSV (e.g. from `result.to_pandas().to_csv(...)` after
            run_ragas_evaluation()), expected to have a "user_input" (or
            "question") column plus one or more of "faithfulness",
            "answer_relevancy", "answer_correctness", "context_precision",
            "context_recall". Rows are matched to the human-review CSV by
            exact question-text match. If omitted, only the label
            distribution is printed.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, comment="#")
    if "human_label" not in df.columns:
        raise ValueError(f"{csv_path} has no 'human_label' column — is this an "
                          "export_for_human_review() CSV?")

    df["human_label"] = df["human_label"].fillna("").astype(str).str.strip()
    labeled = df[df["human_label"] != ""].copy()
    unlabeled_count = len(df) - len(labeled)

    print(f"\nHuman review summary for {csv_path}")
    print("=" * 72)
    print(f"Total rows: {len(df)}  |  Labeled: {len(labeled)}  |  Unlabeled: {unlabeled_count}")

    invalid = labeled[~labeled["human_label"].isin(HUMAN_REVIEW_LABELS)]
    if not invalid.empty:
        print(
            f"\nWARNING: {len(invalid)} row(s) have a human_label that does not match one of the "
            f"rubric's exact strings ({' | '.join(HUMAN_REVIEW_LABELS)}). These are still counted "
            "below under their literal text, but check for typos:"
        )
        for _, row in invalid.iterrows():
            question_preview = str(row["question"])[:60]
            print(f"  - {question_preview!r}: human_label={row['human_label']!r}")

    if labeled.empty:
        print("\nNo labeled rows yet — nothing further to summarize.")
        return

    counts = labeled["human_label"].value_counts()
    print("\nLabel distribution:")
    for label in HUMAN_REVIEW_LABELS:
        n = int(counts.get(label, 0))
        pct = 100.0 * n / len(labeled)
        print(f"  {label:<20} {n:>4}  ({pct:5.1f}%)")
    for label in sorted(set(counts.index) - set(HUMAN_REVIEW_LABELS)):
        n = int(counts[label])
        pct = 100.0 * n / len(labeled)
        print(f"  {label!r:<20} {n:>4}  ({pct:5.1f}%)  <- not a rubric label, see warning above")

    if not ragas_csv_path:
        return

    try:
        ragas_df = pd.read_csv(ragas_csv_path)
    except OSError as exc:
        print(f"\nCould not read --compare-ragas CSV {ragas_csv_path}: {exc}")
        return

    question_col = "user_input" if "user_input" in ragas_df.columns else (
        "question" if "question" in ragas_df.columns else None
    )
    if question_col is None:
        print(
            f"\n{ragas_csv_path} has neither a 'user_input' nor 'question' column — cannot "
            "cross-tabulate against RAGAS scores."
        )
        return

    merged = labeled.merge(
        ragas_df, left_on="question", right_on=question_col, how="inner", suffixes=("", "_ragas")
    )
    if merged.empty:
        print(
            f"\nNo matching questions found between {csv_path} and {ragas_csv_path} — cannot "
            "cross-tabulate. (Questions are matched by exact text, so re-wording between runs "
            "will break the match.)"
        )
        return

    metric_cols = [
        c
        for c in [
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "context_precision",
            "context_recall",
        ]
        if c in merged.columns
    ]
    if not metric_cols:
        print(f"\n{ragas_csv_path} has none of the expected RAGAS metric columns — cannot cross-tabulate.")
        return

    print(
        f"\nCross-tab of mean RAGAS scores by human_label, from {ragas_csv_path} "
        f"({len(merged)}/{len(labeled)} labeled questions matched):"
    )
    grouped = merged.groupby("human_label")[metric_cols].mean()
    print(grouped.to_string())

    # The one rubric-derived prediction this function can actually check
    # itself against: faithfulness specifically measures how much of the
    # answer is attributable to retrieved context, which is exactly what
    # distinguishes "Partially grounded" from "Grounded" in the rubric. If
    # the human labels are tracking the same thing the judge is measuring,
    # "Partially grounded" should average lower faithfulness than
    # "Grounded". This is a real check, not a tautology — it can fail, and
    # if it does that's evidence the human rubric and the LLM judge are
    # measuring different things (worth investigating), not just noise to
    # explain away.
    if "faithfulness" in metric_cols and {"Partially grounded", "Grounded"} <= set(grouped.index):
        pg = grouped.loc["Partially grounded", "faithfulness"]
        g = grouped.loc["Grounded", "faithfulness"]
        print()
        if pg < g:
            print(
                f"Sanity check PASSED: mean faithfulness for 'Partially grounded' ({pg:.3f}) is "
                f"lower than for 'Grounded' ({g:.3f}), as the rubric predicts."
            )
        else:
            print(
                f"Sanity check FLAGGED: mean faithfulness for 'Partially grounded' ({pg:.3f}) is "
                f"NOT lower than for 'Grounded' ({g:.3f}) — this contradicts the rubric's "
                "expectation and may indicate the human rubric and the LLM judge are diverging on "
                "this sample. Worth a closer look before trusting either signal blindly."
            )


_C_TAG_CLAIM_RE = re.compile(r"(?:(?<=[.!?\n])\s*|\A\s*)([^.!?\n]*?\[C\])", re.MULTILINE)
_G_TAG_CLAIM_RE = re.compile(r"(?:(?<=[.!?\n])\s*|\A\s*)([^.!?\n]*?\[G\])", re.MULTILINE)


def _extract_c_tagged_claims(answer: str) -> List[Tuple[str, int, int]]:
    """Find every `[C]`-tagged claim in `answer` produced by generate_answer()'s
    blended [C]/[G] attribution prompt.

    A "claim" is defined as the span of text from the end of the previous
    sentence (a `.`/`!`/`?`/newline) or the start of the string, up to and
    including the literal "[C]" tag itself — this deliberately does not try
    to include trailing punctuation after the tag, since the prompt's own
    example ("claim text [C].") puts the tag before the sentence-final
    period. Matches are non-overlapping and returned in document order.

    Returns a list of (claim_text, start, end) tuples where `claim_text`
    includes the trailing "[C]" and `start`/`end` are character offsets into
    `answer` (usable for later targeted string surgery via slicing).
    """
    return [(m.group(1), m.start(1), m.end(1)) for m in _C_TAG_CLAIM_RE.finditer(answer)]


def _extract_g_tagged_claims(answer: str) -> List[Tuple[str, int, int]]:
    """Sibling of _extract_c_tagged_claims() for `[G]` (general-knowledge)
    claims, added for compute_split_faithfulness() (see that function's
    docstring for the motivation): the faithfulness metric only ever needed
    [C]-tagged content extracted, but scoring [G]-tagged content asks a
    completely different question ("is this true/reasonable on its own
    merits?", not "is this in the retrieved context?"), so it needed its own
    extractor rather than being folded into a single "extract everything"
    function that then has to be told which tag it found.

    Same claim-span definition as _extract_c_tagged_claims() (from the
    previous sentence boundary or string start, through the literal "[G]"
    tag), same non-overlapping / document-order guarantees, same
    (claim_text, start, end) return shape. Deliberately kept as a separate
    function rather than parameterizing _extract_c_tagged_claims() with a
    `tag` argument: that function is already called elsewhere
    (verify_and_repair_answer()) with a fixed no-argument signature, and
    changing it would be a gratuitous ripple for a two-line regex swap.
    """
    return [(m.group(1), m.start(1), m.end(1)) for m in _G_TAG_CLAIM_RE.finditer(answer)]


def verify_and_repair_answer(question: str, contexts: List[str], draft_answer: str) -> str:
    """Generation-side verify-then-repair step for the blended [C]/[G] prompt's
    faithfulness regression.

    BACKGROUND: an A/B test on this project measured that generate_answer()'s
    blended [C]/[G] prompt (see that function's docstring) fixed a
    75%-refusal-rate / answer_relevancy problem versus the old strict
    "context-only" prompt, but caused a real faithfulness regression (-0.30
    to -0.35 absolute, p<0.001) relative to the strict prompt. Root-causing
    that regression (manual inspection of low-scoring samples, not just the
    aggregate number) found it was driven mostly by the model's prose STYLE
    around [C]-tagged claims -- markdown headers, expository/synthesized
    framing that restates or lightly embellishes a context-derived fact
    rather than extracting it near-verbatim -- tripping RAGAS's sentence-level
    faithfulness judge, NOT by the underlying content actually being
    ungrounded. That distinction matters: the honest fix is to tighten how
    [C]-tagged sentences are phrased, not to revert to the strict prompt (which
    reintroduces the refusal-rate regression) or to silently accept a lower
    faithfulness number as "expected" when some of it is a fixable style
    artifact.

    This is directly informed by the Self-Correcting RAG paper's ablation
    finding (arXiv:2604.10734): the authors found that improving
    retrieval/context quality ALONE left faithfulness flat, but adding a
    review-and-correct step on the GENERATED OUTPUT (not the input) measurably
    fixed faithfulness (their AP metric moved 0.58 -> 0.85). This function is
    that step, applied to generate_answer()'s output specifically.

    WHAT IT DOES:
      1. Extracts every [C]-tagged claim from `draft_answer` via
         _extract_c_tagged_claims() (reuses the existing blended-prompt
         tagging format -- no new tagging scheme introduced).
      2. If there are no [C]-tagged claims (fully [G] or untagged answer),
         returns `draft_answer` unchanged -- there is nothing to verify.
      3. Batches ALL claims from this one answer into a SINGLE verification
         LLM call (never one call per claim -- see cost note below), asking a
         cheap/fast judge model whether each claim is actually directly
         traceable to `contexts`.
      4. For each claim the judge marks unsupported, applies one of two
         repairs, per the judge's own recommendation:
           - "retag": the claim is judged to be general-knowledge content
             that was mistagged as [C] -- flip its tag to [G] with the
             claim text otherwise unchanged.
           - "rewrite": the claim genuinely drew on context but embellished/
             synthesized beyond it -- replace it with a tighter, more literal
             extraction the judge supplies (still tagged [C]).
      5. Returns the repaired answer. If the repair call fails outright
         (network error, unparseable response) this fails SAFE: it logs a
         message and returns `draft_answer` unmodified rather than raising or
         silently corrupting the answer.

    COST: exactly one extra LLM call per generate_answer() invocation when
    the draft contains at least one [C]-tagged claim (zero extra calls
    otherwise) -- all claims are batched into that single call regardless of
    how many there are, specifically to keep this practical for a full
    evaluation run rather than one verification call per sentence.

    Args:
        question: the original question (given to the judge for context on
            what the claims are answering).
        contexts: the same retrieved contexts passed to generate_answer().
        draft_answer: generate_answer()'s raw output, before repair.

    Returns:
        The repaired answer string, or `draft_answer` unchanged if there was
        nothing to repair or the repair call could not be completed.
    """
    claims = _extract_c_tagged_claims(draft_answer)
    if not claims:
        return draft_answer

    client, model, mode = _build_anthropic_client()
    if client is None:
        return draft_answer

    # Use a cheap/fast judge model for this verification pass wherever
    # possible -- it's a lightweight support/no-support classification, not
    # generation, and this call runs once per answer regardless of claim
    # count. On the direct Anthropic API, Haiku is available and is the
    # deliberate choice for this. On Vertex, the deployed Claude model is
    # whatever ANTHROPIC_VERTEX_MODEL names (often not Haiku), so we simply
    # reuse the generation model there rather than guessing at a Haiku
    # deployment that may not exist in that GCP project.
    judge_model = "claude-haiku-4-5" if mode == "direct" else model

    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "(no context retrieved)"
    numbered_claims = "\n".join(
        f"{idx}. {claim_text[:-len('[C]')].strip()}" for idx, (claim_text, _start, _end) in enumerate(claims, start=1)
    )

    verify_prompt = (
        "You are fact-checking sentences that an assistant tagged as directly supported by "
        "retrieved CONTEXT (tagged [C]) when answering a QUESTION. For each numbered claim "
        "below, decide whether it is ACTUALLY directly traceable to the CONTEXT -- not just "
        "topically related, but a claim whose specific content (facts, numbers, names, "
        "relationships) genuinely appears in or is a direct paraphrase of the CONTEXT.\n\n"
        "CONTEXT:\n"
        f"{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        "CLAIMS TO VERIFY (originally tagged [C]):\n"
        f"{numbered_claims}\n\n"
        "For each claim, respond with one JSON object. If the claim IS directly supported by "
        "the CONTEXT, set \"supported\": true and omit \"repair\". If it is NOT directly "
        "supported, set \"supported\": false and choose exactly one repair:\n"
        "  - \"retag\": the claim is accurate but is really general knowledge, not something "
        "the CONTEXT actually states -- it was mistagged.\n"
        "  - \"rewrite\": the claim draws on the CONTEXT but adds embellishment, synthesis, or "
        "specifics not actually present -- provide \"corrected_text\": a tighter sentence that "
        "stays as close as possible to the CONTEXT's actual wording, with no fabricated "
        "specifics.\n\n"
        "Respond with ONLY a JSON array, one object per claim, in claim order, with keys "
        "\"index\" (1-based, matching the numbering above), \"supported\" (bool), \"repair\" "
        "(\"retag\" | \"rewrite\" | omitted when supported), and \"corrected_text\" (only for "
        "\"rewrite\"). No prose, no markdown fences, just the JSON array."
    )

    try:
        response = client.messages.create(
            model=judge_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": verify_prompt}],
        )
        raw_text = next((block.text for block in response.content if block.type == "text"), "")
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            print("verify_and_repair_answer: judge response had no JSON array — leaving answer unrepaired.")
            return draft_answer
        verdicts = json.loads(match.group(0))
    except Exception as exc:  # noqa: BLE001 - repair is best-effort, never fatal
        print(f"verify_and_repair_answer: verification call failed ({exc}) — leaving answer unrepaired.")
        return draft_answer

    verdicts_by_index = {}
    for v in verdicts:
        try:
            verdicts_by_index[int(v["index"])] = v
        except (KeyError, TypeError, ValueError):
            continue

    # Apply repairs back-to-front so earlier spans' offsets stay valid as we
    # slice-and-splice the string.
    repaired = draft_answer
    for idx in range(len(claims), 0, -1):
        verdict = verdicts_by_index.get(idx)
        if not verdict or verdict.get("supported", True):
            continue

        claim_text, start, end = claims[idx - 1]
        repair = verdict.get("repair")

        if repair == "retag":
            replacement = claim_text[: -len("[C]")] + "[G]"
        elif repair == "rewrite":
            corrected = str(verdict.get("corrected_text") or "").strip()
            if not corrected:
                continue
            replacement = f"{corrected} [C]"
        else:
            continue

        repaired = repaired[:start] + replacement + repaired[end:]

    return repaired


def _score_g_claims_groundedness(question: str, g_claims: List[Tuple[str, int, int]]) -> Optional[float]:
    """Cheap LLM call: for a batch of `[G]`-tagged (general-knowledge) claims from one
    answer, ask "is this actually true/reasonable, independent of the retrieved context?"
    and return a single 0-1 groundedness score for the batch (the mean of the per-claim
    verdicts the judge returns).

    WHY A SEPARATE CALL FROM RAGAS'S FAITHFULNESS METRIC: RAGAS's Faithfulness metric asks
    "is this claim supported BY THE RETRIEVED CONTEXT" -- that question is meaningless for a
    [G]-tagged claim, which this project's generate_answer() prompt explicitly permits and
    instructs the model to draw from *outside* the context on purpose (see that function's
    docstring and docs/ARCHITECTURE.md Known Issue #4.3). Asking RAGAS's context-attribution
    question about a claim that was never supposed to be context-attributed produces a
    meaningless answer either way ("unsupported" is the *correct*, by-design outcome for a
    [G] claim). What's actually worth checking for a [G] claim is a different, ordinary
    factual-correctness question -- "is this true?" -- which is what this function asks,
    using a cheap Haiku-class model (reusing _build_anthropic_client()'s provider selection,
    same pattern as review_and_distill_context()'s per-sentence relevance scoring and
    verify_and_repair_answer()'s claim-verification call).

    FAIL-OPEN BY DESIGN (documented here deliberately, not just in the caller): this is a
    secondary DIAGNOSTIC metric about the *generator's* general-knowledge claims, not a hard
    quality gate anything currently blocks on. A parse failure, network error, or missing
    Anthropic credentials returns 1.0 ("assume true") rather than 0.0 ("assume false") or
    None/NaN, because 0.0 would misleadingly read as "these claims are false/hallucinated" on
    a dashboard when the actual situation is just "the checker itself didn't run" -- the two
    have very different implications for a reader skimming a results CSV, and conflating them
    would be worse than a slightly-too-generous default. This mirrors this project's other
    documented fail-open choices (e.g. the Adaptive-RAG-lite gate's
    adaptive_retrieval_fallback_strategy, which fails open to "do retrieval anyway" rather
    than silently skipping it) -- when a secondary/diagnostic check can't run, degrade to "did
    nothing," never to a loud false negative.

    Args:
        question: the original question (given to the judge for context on what the claims
            are answering).
        g_claims: the exact (claim_text, start, end) tuples from _extract_g_tagged_claims() --
            must be non-empty; callers are responsible for the "zero [G] claims" case (see
            compute_split_faithfulness(), which returns None/"no_g_claims" for that case
            instead of calling this function at all).

    Returns:
        A float in [0, 1] (the mean of the judge's per-claim 0/1 or 0-1 verdicts), or 1.0
        (fail-open) if no Anthropic client is configured or the verification call/parse fails
        for any reason. Never returns None -- a caller that wants to distinguish "fail-open
        1.0" from "a real 1.0" should treat any exception path as fail-open by construction
        (this function does not currently expose that distinction; see
        compute_split_faithfulness()'s "g_only_reason" field for that signal instead).
    """
    client, model, mode = _build_anthropic_client()
    if client is None:
        return 1.0  # Fail open -- see docstring.

    judge_model = "claude-haiku-4-5" if mode == "direct" else model

    numbered_claims = "\n".join(
        f"{idx}. {claim_text[:-len('[G]')].strip()}"
        for idx, (claim_text, _start, _end) in enumerate(g_claims, start=1)
    )

    prompt = (
        "The following claims were tagged by an assistant as general knowledge (NOT drawn "
        "from any retrieved document) while answering a question. For each claim, judge "
        "ONLY whether it is actually true/reasonable/accurate on its own merits -- do NOT "
        "penalize it for lacking a document citation; that is expected and intentional for "
        "these claims.\n\n"
        f"QUESTION (for context only): {question}\n\n"
        "CLAIMS TO JUDGE (tagged [G], general knowledge):\n"
        f"{numbered_claims}\n\n"
        "Respond with ONLY a JSON array, one object per claim, in claim order, with keys "
        "\"index\" (1-based) and \"true\" (bool: is this claim actually true/reasonable?). "
        "No prose, no markdown fences, just the JSON array."
    )

    try:
        response = client.messages.create(
            model=judge_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = next((block.text for block in response.content if block.type == "text"), "")
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            print(
                "_score_g_claims_groundedness: judge response had no JSON array -- "
                "failing open (assume true, score=1.0)."
            )
            return 1.0
        verdicts = json.loads(match.group(0))
        verdicts_by_index = {}
        for v in verdicts:
            try:
                verdicts_by_index[int(v["index"])] = bool(v["true"])
            except (KeyError, TypeError, ValueError):
                continue
        if not verdicts_by_index:
            return 1.0
        scores = [1.0 if verdicts_by_index.get(i, True) else 0.0 for i in range(1, len(g_claims) + 1)]
        return sum(scores) / len(scores)
    except Exception as exc:  # noqa: BLE001 - diagnostic metric, must fail open, never fatal
        print(
            f"_score_g_claims_groundedness: verification call failed ({exc}) -- failing open "
            "(assume true, score=1.0)."
        )
        return 1.0


# --- answer_relevancy noncommittal-zero-gate fix (measurement-only, see
# _strip_context_gap_disclaimer()'s docstring) ---------------------------------------------

# Cheap pre-filter: if NONE of these (case-insensitive) substrings appear anywhere in the
# answer, skip the LLM call entirely and return it unchanged -- a context-gap disclaimer or
# closing hedge, by construction, always references the source material and/or explicitly
# flags a limitation, so an answer containing none of these words has nothing for the LLM
# pass to find. This is a recall-preserving cost guard, not a replacement for the LLM
# check: it only ever SKIPS the call (fails open to "unchanged"), never substitutes its own
# judgment for a positive strip decision.
_DISCLAIMER_PREFILTER_RE = re.compile(
    r"\b(context|documents?|passages?|sources?|research(?:\s+is\s+needed)?|"
    r"cannot\s+confirm|can't\s+confirm|would\s+need|further\s+research|"
    r"additional\s+(?:sources|information|research|data)|beyond\s+what|"
    r"no\s+information|not\s+mention|doesn't\s+(?:contain|address|discuss|mention)|"
    r"does\s+not\s+(?:contain|address|discuss|mention)|don't\s+find|"
    r"grounding\s*:)\b",
    re.I,
)

_DISCLAIMER_STRIP_PROMPT = """Below is an AI-generated answer to a question, produced by a \
RAG pipeline that blends retrieved-context claims (tagged [C]) with general-knowledge \
claims (tagged [G]).

Your job: remove ONLY the meta-commentary / hedging sentences that talk ABOUT the process \
of answering or the limitations of the available source material -- NOT the substantive \
factual content itself. Examples of sentences to REMOVE:
- Opening disclaimers: "I don't find any information about X in the provided context.", \
"The context does not address X.", "I've reviewed the documents and none of them mention X."
- Transition/meta sentences: "However, I can provide relevant information based on what's \
available.", "I can address what information is available and supplement with general \
knowledge."
- Closing hedges/limitations: "I cannot confirm this without specific research data.", "I \
would need additional sources to answer this specifically.", "More research is needed to \
establish this.", "To answer your question properly, I would need access to X."
- A trailing "Grounding: ..." labeling line, if present.

KEEP everything else completely UNCHANGED, verbatim -- every factual claim, every [C]/[G] \
tag, every heading, every list item. Do not rewrite, rephrase, summarize, or add anything. \
Only DELETE the disqualified sentences described above.

If there is nothing to remove, return the answer completely unchanged.

Return ONLY the resulting answer text. No preamble, no explanation, no markdown code fences.

ANSWER:
{answer}
"""


def _strip_context_gap_disclaimer(answer: str) -> str:
    """Strip context-gap disclaimer / hedging sentences from `answer`, for use ONLY as an
    input to RAGAS's answer_relevancy metric -- never for faithfulness/context_precision/
    context_recall/answer_correctness, and never for whatever is actually shown/returned to
    a real caller (see this project's `--generate` CLI output and every other RAGAS metric,
    which all continue to see generate_answer()'s full, unmodified output).

    THE PROBLEM THIS FIXES (see docs/CHANGELOG.md's dated entry and the diagnostic pass
    that root-caused it): RAGAS's `ResponseRelevancy._calculate_score()` (installed
    ragas/metrics/_answer_relevance.py) computes
    `score = cosine_sim.mean() * int(not all_noncommittal)` -- a binary zero-gate. If ITS
    OWN judge-LLM few-shot-primed classifier (whose only example of "noncommittal" is an
    evasive "I don't know about the groundbreaking feature... as I am unaware of
    information beyond 2022" -- see ResponseRelevancePrompt.examples) flags an answer as
    noncommittal, the ENTIRE relevancy score becomes 0.0 regardless of how good the rest of
    the answer is. This project's generate_answer() prompt (see its own docstring) tells
    the model to honestly say when CONTEXT doesn't cover the question before optionally
    supplementing with [G]-tagged general knowledge -- exactly the kind of opening/closing
    that superficially resembles RAGAS's "I don't know" trigger example, even when the
    answer that follows/precedes it is a complete, relevant, well-grounded response.
    Confirmed via a live diagnostic pass against this project's actual saved results CSVs
    (/tmp/nfcorpus_eval_v2/results_ragHyde_sonnet_r1.csv and siblings): 25-38% of RAG
    answers hit this zero-gate vs. only 2-3% of closed-book answers, and the nonzero-only
    means for BOTH arms cluster at ~0.80-0.82 -- statistically indistinguishable -- meaning
    the "RAG scores dramatically lower on relevancy" finding this session started from was
    a metric artifact of this zero-gate, not a real quality gap. Stripping [C]/[G]
    attribution tags was directly tested and RULED OUT as the cause (+0.012 change,
    noise-level); it is specifically disclaimer/hedge phrasing that trips the gate. See also
    RAGAS GitHub issue #1475 (community reports of the same zero-gate behavior on honest
    "context doesn't cover this" answers).

    THIS IS A MEASUREMENT FIX, NOT A GENERATION-POLICY CHANGE: generate_answer()'s prompt
    is untouched. The model should keep producing the same real answers, disclaimers
    included -- that's correct, honest behavior. This function only changes what text is
    handed to RAGAS's answer_relevancy calculation.

    DETECTION METHOD -- a cheap LLM classifier/editor call, NOT a pure regex heuristic.
    A first implementation of this fix used a deterministic three-way keyword-co-occurrence
    regex (context/documents/sources + a negation + an absence word) that stripped only a
    LEADING sentence run. It was LIVE-TESTED against RAGAS's actual installed judge
    (ResponseRelevancePrompt, via _default_judge_llm()) on the exact 5 zero-scored rows this
    diagnostic pass had already identified in
    /tmp/nfcorpus_eval_v2/results_ragHyde_sonnet_r1.csv, and FAILED to flip
    `noncommittal` to 0 for ANY of them (5/5 still noncommittal=1 after the leading-only
    strip). Manual inspection of the FULL answer text (not just the truncated opening
    quoted in the original diagnostic) showed why: the actual trigger is very often a
    TRAILING hedge/limitation sentence near the END of the answer -- "I cannot confirm this
    without specific research data.", "I would need additional sources beyond what's
    provided here.", "To answer your question properly, I would need access to Dr.
    Jenkins' actual statements..." -- and sometimes a non-disclaimer TRANSITION sentence
    right after the opening ("However, I can provide relevant information based on what's
    available:") that reads as evasive meta-commentary even though it isn't a literal
    "context doesn't have X" statement. This phrasing is far more heterogeneous in
    location and shape than the opening-only pattern the original regex targeted, and no
    reliable regex was found that generalized to both ends without unacceptable false-
    positive risk on genuine content sentences (e.g. sentences that legitimately discuss
    what evidence IS and ISN'T available as part of a substantive answer, like the
    high-scoring "Stool Size and Breast Cancer Risk" row's opening, which also uses hedge-
    adjacent language but scored 0.90 and must NOT be touched). Switching to an LLM
    editor call -- explicitly told to delete ONLY meta-commentary/hedging sentences
    (opening OR closing) while preserving every substantive claim, tag, and heading
    verbatim -- was then live-tested against the SAME 5 rows and flipped noncommittal to 0
    for 4 of 5 (the remaining row's content was about a genuinely different sub-topic than
    asked -- prostate cancer research answering a breast-cancer question -- which is a real
    topical mismatch, not a phrasing artifact, and this fix correctly leaves that one
    alone). This matches the task's own suggested robustness rationale for an LLM-based
    classifier over a regex, once the regex was shown empirically insufficient.

    COST GUARD: a cheap keyword pre-filter (`_DISCLAIMER_PREFILTER_RE`) skips the LLM call
    entirely for answers containing none of a small set of trigger substrings (context/
    documents/sources, "cannot confirm", "would need", "further research", "grounding:",
    etc.) -- this only ever skips a call (failing open to "unchanged"), it never overrides
    a positive LLM strip decision, so it cannot introduce a false strip, only reduce cost
    for answers that plainly have nothing to check.

    SAFETY BOUNDS on the LLM's edit: the model is instructed to ONLY delete, never rewrite/
    add/summarize. As a mechanical guard against a botched edit (e.g. the model
    paraphrasing instead of deleting, or over-deleting until almost nothing is left), the
    result is discarded (falls back to the original answer, unchanged) if its length falls
    outside [30%, 105%] of the original's length -- a legitimate delete-only edit should
    never grow the text and should rarely remove more than ~70% of it.

    FAILS OPEN (returns `answer` completely unchanged) on ANY error (no Anthropic client
    configured, call/parse failure, empty response, or a result outside the safety bounds
    above) -- same convention as every other fail-open function in this module
    (review_and_distill_context(), _score_g_claims_groundedness()).

    Args:
        answer: generate_answer()'s (or generate_closed_book_answer()'s) raw output
            string, [C]/[G] tags and all.

    Returns:
        The answer with any context-gap disclaimer/hedging sentence(s) removed (leading,
        trailing, or both), or the original `answer` unchanged if the pre-filter found no
        trigger words, the LLM found nothing to remove, the input is empty/whitespace-only,
        or anything above raised/violated the safety bounds.
    """
    try:
        if not answer or not answer.strip():
            return answer

        if not _DISCLAIMER_PREFILTER_RE.search(answer):
            return answer

        client, model, mode = _build_anthropic_client()
        if client is None:
            return answer

        judge_model = "claude-haiku-4-5" if mode == "direct" else model

        response = client.messages.create(
            model=judge_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": _DISCLAIMER_STRIP_PROMPT.format(answer=answer)}],
        )
        stripped = next((b.text for b in response.content if b.type == "text"), "").strip()

        if not stripped:
            return answer

        original_len = len(answer)
        if not (0.30 * original_len <= len(stripped) <= 1.05 * original_len):
            print(
                "_strip_context_gap_disclaimer: LLM result length outside safety bounds "
                f"({len(stripped)} vs original {original_len}) -- failing open (returning "
                "original answer unchanged)."
            )
            return answer

        return stripped
    except Exception as exc:  # noqa: BLE001 - measurement helper, must fail open, never fatal
        print(
            f"_strip_context_gap_disclaimer: detection/stripping failed ({exc}) -- failing "
            "open (returning original answer unchanged)."
        )
        return answer


def compute_split_faithfulness(
    question: str,
    contexts: List[str],
    answer: str,
    precomputed_full_faithfulness: Optional[float] = None,
) -> Dict[str, Any]:
    """Bespoke split-scoring fix for RAGAS's faithfulness/[C]-[G] metric-policy mismatch
    (docs/ARCHITECTURE.md Known Issue #4.3; also this module's own docstring and
    generate_answer()'s docstring).

    THE PROBLEM THIS FIXES: RAGAS's Faithfulness metric decomposes the WHOLE answer string
    into claims and scores every one of them against retrieved_contexts, with no awareness
    of generate_answer()'s [C]/[G] attribution tags. A [G]-tagged claim -- general knowledge
    the prompt explicitly *permits and asks the model to tag honestly* when context is thin
    -- is then scored by RAGAS's sentence-level judge exactly like an unattributed
    hallucination, because the judge only sees prose, not tag semantics. This is not a
    hypothesis: DeepEval's own faithfulness-metric documentation explicitly describes this
    exact failure mode (claims correctly flagged "outside the retrieval context" penalize a
    generator for deliberately-out-of-context content that was never supposed to be
    attributed to the context in the first place); Wallat et al. ("Real-World Summarization:
    When Evaluation Metrics Encounter Real-World Constraints" -- the split-scoring citation
    for this fix, arXiv:2412.18004) separately argues that faithfulness/attribution metrics
    of this shape conflate CORRECTNESS with actual CONTEXT-DERIVATION, and that no off-the-
    shelf RAGAS-style variant handles mixed-provenance answers (part grounded, part
    deliberately not) correctly. This project's own manual root-causing of a -0.30 to -0.35
    absolute faithfulness regression after the [C]/[G] blended prompt shipped (see
    verify_and_repair_answer()'s docstring and docs/CHANGELOG.md) is consistent with exactly
    this mechanism.

    THE FIX -- split scoring, not a single patched-over number:
      1. faithfulness_c_only: extract ONLY the [C]-tagged claims (_extract_c_tagged_claims(),
         already used by verify_and_repair_answer() -- reused as-is, no new extraction logic
         needed for this half), reconstruct a "context-only view" of the answer as just the
         concatenated [C]-tagged claim text, and run RAGAS's real Faithfulness metric on THAT
         string instead of the full answer. This measures exactly what the [C]/[G] policy
         actually intends to be faithful: the portion the model claimed was context-derived.
         A [G]-tagged claim can no longer drag this number down, because it was never in the
         string being scored.
      2. groundedness_g_only: extract the [G]-tagged claims (_extract_g_tagged_claims(), the
         new sibling function added alongside this fix) and score them via a cheap secondary
         LLM call (_score_g_claims_groundedness()) asking "is this actually true/reasonable,
         independent of the provided context?" -- i.e. ordinary factual correctness, not
         context-attribution, since attribution is not what a [G] claim is claiming. This
         call fails OPEN (returns 1.0) on any error -- see _score_g_claims_groundedness()'s
         docstring for why a secondary diagnostic metric should never silently read as "these
         claims are false" just because the checker itself broke.
      3. faithfulness_full_answer_legacy: the ORIGINAL whole-answer RAGAS faithfulness score,
         preserved verbatim (not removed, not replaced) so every existing historical
         tuned-vs-baseline comparison in this project (docs/ARCHITECTURE.md Known Issue #4.1,
         built entirely on the legacy full-answer metric) remains comparable going forward.
         If the caller already has this score (e.g. run_ragas_evaluation() computed it as
         part of its own main evaluate() call, on the exact same dataset/judge), pass it in
         via `precomputed_full_faithfulness` to avoid a redundant evaluate() call; otherwise
         this function computes it itself with a fresh single-sample Faithfulness() evaluate()
         call, so compute_split_faithfulness() is fully usable standalone (e.g. against one
         row pulled out of an existing results CSV, with no run_ragas_evaluation() in sight).

    EDGE CASE -- zero [C]-tagged claims (fully [G] or untagged answer): faithfulness_c_only is
    None (NOT 0.0). A 0.0 would look identical to "every claim was checked and found
    unfaithful," which is a real, different, and worse-sounding condition than "there was
    nothing to check." This follows the same "no-data vs. bad-score" distinction this project
    already established for run_neuralwatt_multi_judge_consensus()'s delta_m/
    all_judges_failed/"unrecoverable" flagging (docs/CHANGELOG.md, 2026-07-17 entries) --
    None/NaN means "not applicable," a real 0.0 means "applicable and it scored zero." The
    "c_only_reason" field in the returned dict spells out WHY it's None ("no_c_claims" vs. a
    judge/network failure) rather than leaving the reader to guess. The symmetric case (zero
    [G]-tagged claims) similarly returns groundedness_g_only=None with g_only_reason=
    "no_g_claims" -- there is nothing to check for correctness-independent-of-context if the
    model made no such claims.

    Args:
        question: the original question.
        contexts: the same retrieved context strings passed to generate_answer().
        answer: generate_answer()'s raw output (with [C]/[G] tags intact -- do NOT pass an
            already-stripped or already-repaired-and-retagged answer through this unless that
            is genuinely the version you want scored; verify_and_repair_answer()'s output is
            a legitimate input here too, just a different "answer" than the pre-repair draft).
        precomputed_full_faithfulness: optional. If the caller already computed whole-answer
            RAGAS faithfulness for this exact (question, contexts, answer) triple (e.g.
            run_ragas_evaluation()'s own main evaluate() call), pass it here to skip a
            redundant evaluate() call for faithfulness_full_answer_legacy. If None, this
            function computes it itself.

    Returns:
        {
            "faithfulness_c_only": float in [0,1], or None if c_claim_count == 0 or the
                evaluate() call failed (see "c_only_reason" for which),
            "groundedness_g_only": float in [0,1] (fail-open default 1.0 on error), or None
                if g_claim_count == 0 (see "g_only_reason"),
            "faithfulness_full_answer_legacy": float in [0,1], or None if it could not be
                computed (no judge LLM configured, or the evaluate() call failed) -- kept for
                backward comparison against every prior result in this project that used the
                unmodified whole-answer metric,
            "c_claim_count": int, number of [C]-tagged claims found,
            "g_claim_count": int, number of [G]-tagged claims found,
            "c_only_reason": None if faithfulness_c_only is a real score, else one of
                "no_c_claims" | "no_judge_llm" | "ragas_call_failed",
            "g_only_reason": None if groundedness_g_only is a real (non-fail-open) score,
                else one of "no_g_claims" | "fail_open_no_client" | "fail_open_error" (the
                latter two both mean "scored 1.0 because the checker itself couldn't run,
                not because the claims were verified true"),
        }
    """
    c_claims = _extract_c_tagged_claims(answer)
    g_claims = _extract_g_tagged_claims(answer)
    c_claim_count = len(c_claims)
    g_claim_count = len(g_claims)

    result: Dict[str, Any] = {
        "faithfulness_c_only": None,
        "groundedness_g_only": None,
        "faithfulness_full_answer_legacy": precomputed_full_faithfulness,
        "c_claim_count": c_claim_count,
        "g_claim_count": g_claim_count,
        "c_only_reason": None,
        "g_only_reason": None,
    }

    judge_llm = None
    if precomputed_full_faithfulness is None or c_claim_count > 0:
        # Only bother building a judge LLM if we actually need one below (either to compute
        # the legacy score ourselves, or to score the [C]-only view).
        judge_llm = _default_judge_llm()

    # --- faithfulness_full_answer_legacy: reuse if given, else compute fresh. ---
    if precomputed_full_faithfulness is None:
        if judge_llm is None:
            result["c_only_reason"] = "no_judge_llm" if c_claim_count > 0 else "no_c_claims"
            if c_claim_count == 0:
                result["faithfulness_c_only"] = None
            result["groundedness_g_only"] = (
                _score_g_claims_groundedness(question, g_claims) if g_claim_count > 0 else None
            )
            result["g_only_reason"] = None if g_claim_count > 0 else "no_g_claims"
            return result
        try:
            full_sample = SingleTurnSample(
                user_input=question, retrieved_contexts=contexts, response=answer, reference=""
            )
            full_result = evaluate(
                dataset=EvaluationDataset(samples=[full_sample]),
                metrics=[Faithfulness()],
                llm=judge_llm,
                run_config=RunConfig(max_workers=1),
            )
            import pandas as pd  # local import, matches this module's existing lazy-pandas convention

            full_score = full_result.to_pandas()["faithfulness"].iloc[0]
            result["faithfulness_full_answer_legacy"] = None if pd.isna(full_score) else float(full_score)
        except Exception as exc:  # noqa: BLE001 - this is a comparison metric, not a hard gate
            print(f"compute_split_faithfulness: legacy full-answer faithfulness failed ({exc}).")
            result["faithfulness_full_answer_legacy"] = None

    # --- faithfulness_c_only: RAGAS Faithfulness, but on a [C]-claims-only reconstruction. ---
    if c_claim_count == 0:
        result["faithfulness_c_only"] = None
        result["c_only_reason"] = "no_c_claims"
    else:
        c_only_answer = " ".join(claim_text for claim_text, _start, _end in c_claims)
        try:
            if judge_llm is None:
                judge_llm = _default_judge_llm()
            if judge_llm is None:
                result["c_only_reason"] = "no_judge_llm"
            else:
                c_sample = SingleTurnSample(
                    user_input=question,
                    retrieved_contexts=contexts,
                    response=c_only_answer,
                    reference="",
                )
                c_result = evaluate(
                    dataset=EvaluationDataset(samples=[c_sample]),
                    metrics=[Faithfulness()],
                    llm=judge_llm,
                    run_config=RunConfig(max_workers=1),
                )
                import pandas as pd  # local import, matches this module's existing lazy-pandas convention

                c_score = c_result.to_pandas()["faithfulness"].iloc[0]
                result["faithfulness_c_only"] = None if pd.isna(c_score) else float(c_score)
        except Exception as exc:  # noqa: BLE001 - this is a comparison metric, not a hard gate
            print(f"compute_split_faithfulness: c-only faithfulness failed ({exc}).")
            result["faithfulness_c_only"] = None
            result["c_only_reason"] = "ragas_call_failed"

    # --- groundedness_g_only: cheap correctness-only LLM check, fails open to 1.0. ---
    if g_claim_count == 0:
        result["groundedness_g_only"] = None
        result["g_only_reason"] = "no_g_claims"
    else:
        result["groundedness_g_only"] = _score_g_claims_groundedness(question, g_claims)
        result["g_only_reason"] = None

    return result


def compute_disclaimer_stripped_answer_relevancy(
    user_inputs: List[str],
    responses: List[str],
    existing_relevancy_scores: List[Optional[float]],
    llm=None,
    embeddings=None,
) -> List[Optional[float]]:
    """Re-score RAGAS's answer_relevancy metric on a disclaimer-stripped view of each
    response, as a SECOND, separate evaluate() call alongside the main one -- the fix for
    the noncommittal zero-gate documented in _strip_context_gap_disclaimer()'s docstring
    (docs/ARCHITECTURE.md Known Issue, docs/CHANGELOG.md's dated entry for this fix).

    WHY A SEPARATE evaluate() CALL, NOT A DIFFERENT `response` ON THE SAME SAMPLE: RAGAS's
    `SingleTurnSample`/`evaluate()` scores every metric passed to one `evaluate()` call from
    the SAME sample object's fields -- there is no per-metric override of `response` within
    a single call. To get answer_relevancy scored on different text than faithfulness/
    context_precision/context_recall/answer_correctness see, a second `evaluate()` call
    with different `SingleTurnSample.response` values is required. This mirrors
    compute_split_faithfulness()'s own established convention in this exact module (its
    faithfulness_c_only score is likewise computed via a second, separate evaluate() call
    on a reconstructed view of the answer, not by mutating the main call's samples).

    COST-CONSCIOUS BATCHING: only rows where _strip_context_gap_disclaimer() actually
    changed the text are re-scored; rows with no detected disclaimer reuse
    `existing_relevancy_scores[i]` unchanged (stripping is a no-op there, so the score
    would not meaningfully differ, and re-running RAGAS's stochastic n=3
    question-generation sampling on identical input text would just add cost/latency for a
    number expected to land in the same place). Changed rows are batched into ONE
    evaluate() call (not one call per row), matching this module's existing "batch, don't
    loop LLM calls per row" convention (see _score_sentence_relevance()'s docstring in
    utils.py for the same principle applied to context review).

    Args:
        user_inputs: the question/user_input string for each row (same order as
            `responses`/`existing_relevancy_scores`).
        responses: the FULL, unmodified generate_answer()/generate_closed_book_answer()
            output for each row -- this function strips disclaimers internally; do not
            pre-strip before calling.
        existing_relevancy_scores: the already-computed whole-answer "answer_relevancy"
            score for each row, from the caller's own main evaluate() call (may contain
            None/NaN-as-None entries; those are passed through unchanged for unmodified
            rows). Must be the same length as `user_inputs`/`responses`.
        llm: optional pre-configured RAGAS/LangChain judge LLM. Defaults the same way as
            run_ragas_evaluation() (via _default_judge_llm()) if None.
        embeddings: optional pre-configured RAGAS/LangChain embeddings backend. Defaults
            the same way as run_ragas_evaluation() (via _default_judge_embeddings()) if
            None.

    Returns:
        A list the same length as `user_inputs`, where each entry is either the
        disclaimer-stripped-and-rescored answer_relevancy (float in [0, 1]) for a row whose
        disclaimer was actually stripped, or the corresponding `existing_relevancy_scores[i]`
        passed through unchanged otherwise (including for a row where re-scoring itself
        failed -- fails open to the already-known whole-answer score rather than to None,
        since that score is a real, already-paid-for measurement of the same answer).
    """
    n = len(user_inputs)
    results: List[Optional[float]] = list(existing_relevancy_scores)
    if len(results) != n or len(responses) != n:
        print(
            "compute_disclaimer_stripped_answer_relevancy: input length mismatch -- "
            "failing open (returning existing_relevancy_scores unchanged)."
        )
        return list(existing_relevancy_scores)

    stripped_texts = [_strip_context_gap_disclaimer(r or "") for r in responses]
    changed_indices = [i for i in range(n) if stripped_texts[i] != (responses[i] or "")]

    if not changed_indices:
        return results

    evaluator_llm = llm if llm is not None else _default_judge_llm()
    evaluator_embeddings = embeddings if embeddings is not None else _default_judge_embeddings()
    if evaluator_llm is None or evaluator_embeddings is None:
        print(
            "compute_disclaimer_stripped_answer_relevancy: no judge LLM/embeddings available "
            "-- failing open (returning existing_relevancy_scores unchanged)."
        )
        return results

    try:
        changed_samples = [
            SingleTurnSample(
                user_input=user_inputs[i],
                response=stripped_texts[i],
                reference="",
            )
            for i in changed_indices
        ]
        # Fresh AnswerRelevancy() instance, never the shared module-level `answer_relevancy`
        # singleton -- same shared-mutable-metric-object race this module already guards
        # against everywhere else (see run_ragas_evaluation()'s own comment on this).
        rescored = evaluate(
            dataset=EvaluationDataset(samples=changed_samples),
            metrics=[AnswerRelevancy()],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        import pandas as pd  # local import, matches this module's existing lazy-pandas convention

        rescored_df = rescored.to_pandas()
        for pos, orig_idx in enumerate(changed_indices):
            score = rescored_df["answer_relevancy"].iloc[pos]
            results[orig_idx] = None if pd.isna(score) else float(score)
    except Exception as exc:  # noqa: BLE001 - this is a comparison metric, not a hard gate
        print(
            f"compute_disclaimer_stripped_answer_relevancy: rescoring evaluate() call failed "
            f"({exc}) -- failing open (returning existing_relevancy_scores unchanged for all "
            "rows in this batch)."
        )
        return list(existing_relevancy_scores)

    return results


def generate_answer(
    question: str,
    contexts: List[str],
    verify_and_repair: bool = False,
    model_override: Optional[str] = None,
) -> str:
    """Generate an answer for `question` given `contexts`, using the
    Anthropic Python SDK (direct API, or Claude on Vertex AI as a fallback
    when only ANTHROPIC_VERTEX_PROJECT_ID is available).

    This is the natural fit for this Hermes/Claude-ecosystem skill: the
    skill already assumes a Claude-based LLM downstream, so answer
    generation for RAGAS's faithfulness / answer_relevancy / answer_correctness
    metrics uses the `anthropic` SDK directly.

    Prompt policy: attribution-tagged blending, NOT strict "answer only from
    context, refuse otherwise". The prior strict-refusal prompt measured a
    75% refusal rate on a hard corpus whenever retrieved context was thin or
    tangential, which tanked answer_relevancy well below the no-retrieval
    closed-book baseline (see generate_closed_book_answer()) even when
    retrieval itself was working reasonably. The current prompt asks the
    model to use retrieved CONTEXT wherever it applies (tagged inline with
    "[C]") and permits supplementing with general knowledge when the context
    is thin (tagged inline with "[G]"), while still forbidding fabricated
    specifics under a [C] tag. Expect faithfulness to read lower than under
    the old strict prompt — that's the intended tradeoff, not a bug; judge
    generation quality using answer_correctness and answer_relevancy
    alongside faithfulness, not faithfulness alone.

    Returns "" (and prints a message) if no Anthropic access (direct or
    Vertex) is configured, or the `anthropic` package is not installed —
    callers should treat an empty string as "generation was skipped", not as
    a real answer.

    Args:
        question: the question to answer.
        contexts: retrieved context strings.
        verify_and_repair: when True, runs verify_and_repair_answer() on the
            draft output before returning (see that function's docstring for
            what it does and why). Defaults to False so existing callers/
            tests see unchanged behavior; pass True (or use the CLI's
            `--verify-repair` flag) to opt into the extra verification call.
        model_override: forwarded to _build_anthropic_client() — forces a
            specific generation model (e.g. a Sonnet/Haiku/Opus Vertex model
            id) instead of the normal default selection. Defaults to None
            (unchanged prior behavior). Added for the RAG-vs-closed-book x
            model-tier comparison; see _build_anthropic_client()'s docstring.
    """
    client, model, _mode = _build_anthropic_client(model_override=model_override)
    if client is None:
        return ""

    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "(no context retrieved)"
    prompt = (
        "You are answering a question using retrieved context plus your own domain knowledge.\n\n"
        "CONTEXT:\n"
        f"{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        "Instructions:\n"
        "1. First, check the CONTEXT for information relevant to the question.\n"
        "2. Write your answer using CONTEXT wherever it applies. Tag each context-derived "
        "claim inline with [C].\n"
        "3. If the CONTEXT is missing, incomplete, or only tangentially related, you MAY "
        "supplement with your own general knowledge to give a complete, helpful answer — but "
        "tag each such claim inline with [G].\n"
        "4. If CONTEXT and your general knowledge conflict, say so explicitly and prefer the "
        "CONTEXT, noting the discrepancy.\n"
        "5. Never fabricate specifics (numbers, dosages, names, dates) attributed to [C] that "
        "are not actually in the CONTEXT. If you don't know something even from general "
        "knowledge, say so — don't guess.\n"
        "6. End with one line: \"Grounding: <fully context-based | partially context-based | "
        "general-knowledge-based>\" summarizing which mode this answer used.\n\n"
        "Answer:"
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    draft_answer = next((block.text for block in response.content if block.type == "text"), "")

    if verify_and_repair and draft_answer:
        return verify_and_repair_answer(question, contexts, draft_answer)
    return draft_answer


def generate_closed_book_answer(question: str, model_override: Optional[str] = None) -> str:
    """Generate an answer for `question` using ONLY the model's own
    parametric knowledge — zero retrieved context.

    This is a deliberately SEPARATE function from generate_answer(): that
    function's prompt tells the model "answer using ONLY the provided
    context... if the context does not contain the answer, say so
    explicitly", which is the wrong instruction for a closed-book condition
    (passing it empty context would make the model refuse rather than
    actually draw on its training knowledge). This function's prompt just
    asks the question directly and explicitly invites the model to answer
    from what it already knows.

    Reuses the same Claude-on-Vertex/direct-Anthropic client selection as
    generate_answer() (_build_anthropic_client()) — only the prompt differs.

    Returns "" (and prints a message) if no Anthropic access (direct or
    Vertex) is configured, or the `anthropic` package is not installed —
    callers should treat an empty string as "generation was skipped", not as
    a real answer.

    Args:
        model_override: forwarded to _build_anthropic_client() — forces a
            specific generation model instead of the normal default
            selection. Defaults to None (unchanged prior behavior). Added
            for the RAG-vs-closed-book x model-tier comparison; see
            _build_anthropic_client()'s docstring.
    """
    client, model, _mode = _build_anthropic_client(model_override=model_override)
    if client is None:
        return ""

    prompt = (
        "Answer the following question as best you can, using your own knowledge. "
        "Do not mention that you lack access to external documents or a knowledge "
        "base — just answer directly, as a knowledgeable expert would.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((block.text for block in response.content if block.type == "text"), "")


def run_closed_book_evaluation(
    testset: List[Dict],
    llm_generate_func: Optional[Callable[[str], str]] = None,
    llm=None,
    embeddings=None,
    compute_disclaimer_stripped_relevancy: bool = True,
    return_dataframe: bool = False,
):
    """Run a no-RAG / closed-book evaluation: for each testset question,
    generate an answer using ONLY the model's parametric knowledge (zero
    retrieved context), then score it with metrics that do NOT require
    retrieved_contexts:

      - answer_relevancy: compares the generated answer against the
        question itself (no context or reference needed) — this metric
        works unmodified in a no-retrieval setting.
      - answer_similarity (ragas's embeddings-only "AnswerSimilarity" /
        SemanticSimilarity metric): cosine similarity, in embedding space,
        between the generated answer and the reference (the testset's
        ground_truths[0]). Chosen over ragas's LLM-based
        `answer_correctness` because that metric decomposes the reference
        into atomic statements and classifies each as TP/FP/FN — a step
        designed for concise, authored reference *answers*. This project's
        testset (NFCorpus) has no authored answers; "ground_truths" is the
        full text of the closest matching source document, used as a
        reference-proxy. Running statement-level TP/FP/FN classification
        against a full document (rather than a short answer) is both much
        more expensive (many more LLM calls) and noisier (the model
        penalizes an answer for every document detail it didn't restate,
        which isn't a meaningful notion of "correctness" here). Embedding
        cosine similarity avoids both problems and still directly answers
        the question this arm cares about: does the closed-book answer's
        meaning overlap with the reference content, at all.

    This intentionally does NOT run context_precision / context_recall
    (they require retrieved_contexts, which don't exist in a no-retrieval
    condition) or faithfulness (it measures whether the answer is grounded
    in the retrieved context, which is meaningless when there is none).

    Args:
        testset: List of {"question": str, "ground_truths": [str, ...]}.
        llm_generate_func: called as llm_generate_func(question) -> answer
            string. Defaults to generate_closed_book_answer.
        llm: optional pre-configured RAGAS/LangChain judge LLM (used by
            answer_relevancy's underlying question-generation step).
            Defaults the same way as run_ragas_evaluation.
        embeddings: optional pre-configured RAGAS/LangChain embeddings
            backend (used by both answer_relevancy and answer_similarity).
            Defaults the same way as run_ragas_evaluation.
        compute_disclaimer_stripped_relevancy: when True (default) AND
            return_dataframe is True, also runs
            compute_disclaimer_stripped_answer_relevancy() and adds
            "answer_relevancy_disclaimer_stripped" as an extra column on the
            returned DataFrame — the fix for RAGAS's answer_relevancy
            noncommittal zero-gate (see _strip_context_gap_disclaimer()'s
            docstring). The EXISTING "answer_relevancy" column (whole-answer,
            unmodified) is never removed or altered — kept for backward
            comparability with every prior measurement. Has no effect at all
            when return_dataframe is False (the extra evaluate() call is
            skipped entirely in that case, so old callers pay zero added
            cost/latency for this parameter's default).
        return_dataframe: when True, returns (result, df) instead of just
            `result` — see "Returns" below. Defaults to False so every
            EXISTING caller of this function (this module's main(), and the
            /tmp/nfcorpus_eval_v2/ driver scripts that predate this fix, all
            of which do `result = run_closed_book_evaluation(...)` then call
            `result.to_pandas()`) keeps working completely unchanged. Pass
            True to opt into the new disclaimer-stripped-relevancy column.

    Returns:
        If return_dataframe is False (default): the ragas EvaluationResult,
        or None if evaluation could not run — byte-identical to this
        function's behavior before this fix.
        If return_dataframe is True: a tuple (result, df), where df is
        result.to_pandas() with "answer_relevancy_disclaimer_stripped" added
        (when compute_disclaimer_stripped_relevancy is True and
        "answer_relevancy" was actually scored), or None, None if evaluation
        could not run.
    """
    if llm_generate_func is None:
        llm_generate_func = generate_closed_book_answer

    samples = []
    generated_answers = []
    any_answer_generated = False
    for item in testset:
        question = item["question"]
        ground_truths = item.get("ground_truths") or []

        answer = llm_generate_func(question) or ""
        if answer.strip():
            any_answer_generated = True
        generated_answers.append(answer)

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                reference=ground_truths[0] if ground_truths else "",
            )
        )

    dataset = EvaluationDataset(samples=samples)

    if not any_answer_generated:
        print(
            "No closed-book answers were generated (no Anthropic access configured) — "
            "cannot score. Set ANTHROPIC_API_KEY or ANTHROPIC_VERTEX_PROJECT_ID."
        )
        return (None, None) if return_dataframe else None

    evaluator_llm = llm if llm is not None else _default_judge_llm()
    if evaluator_llm is None:
        return (None, None) if return_dataframe else None

    evaluator_embeddings = embeddings if embeddings is not None else _default_judge_embeddings()
    if evaluator_embeddings is None:
        print("No embeddings backend available — cannot score answer_relevancy/answer_similarity.")
        return (None, None) if return_dataframe else None

    # Fresh instances, never the shared module-level `answer_relevancy`/`answer_similarity`
    # singletons (see the 2026-07-17 NeuralWatt shared-singleton-race fix in
    # docs/CHANGELOG.md, M1: ragas's evaluate() mutates `metric.llm = llm` / `metric.embeddings
    # = embeddings` in place on the metric object it's given, so two concurrent evaluate()
    # calls sharing one metric object race on that mutation). This function is called
    # concurrently across (arm, tier) cells by
    # /tmp/nfcorpus_eval_v2/run_model_tier_comparison.py's ThreadPoolExecutor.
    metrics = [AnswerRelevancy(), AnswerSimilarity()]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    print(result.to_pandas())

    if not return_dataframe:
        return result

    df = result.to_pandas()
    if compute_disclaimer_stripped_relevancy and "answer_relevancy" in df.columns:
        df["answer_relevancy_disclaimer_stripped"] = compute_disclaimer_stripped_answer_relevancy(
            user_inputs=[s.user_input for s in samples],
            responses=[s.response for s in samples],
            existing_relevancy_scores=df["answer_relevancy"].tolist(),
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
    return result, df


def generate_claude_websearch_answer(question: str) -> Dict[str, Any]:
    """Generate an answer for `question` using Claude's own native web_search
    tool on Vertex AI (anthropic.AnthropicVertex) — zero custom retrieval, no
    Qdrant, no NFCorpus context. Claude issues its own search queries and
    grounds its answer in whatever it finds live on the web.

    This is a comparison ARM, not part of the RAG pipeline under test: it
    exists to answer "how does this skill's tuned hybrid+rerank retrieval
    compare to just letting Claude search the web itself?"

    Returns {"answer": str, "contexts": List[str]} where contexts are built
    from each `web_search_tool_result` content block's title/url (one
    "TITLE (URL)" string per web result actually surfaced to the model) —
    this is the model's OWN retrieval, captured only so faithfulness can be
    scored against what Claude actually grounded on. Never conflate these
    with NFCorpus gold passages; see run_claude_websearch_evaluation() for
    why context_precision/context_recall are never computed for this arm.

    Requires ANTHROPIC_VERTEX_PROJECT_ID (this function always uses
    AnthropicVertex directly rather than falling back to the direct API,
    since that's how this project is configured and the web_search tool
    works the same way on both). Returns {"answer": "", "contexts": []} (with
    a printed message) if that's not set, the `anthropic` package is
    missing, the client can't be built, or the API call fails — callers
    should treat an empty answer as "generation was skipped/failed", not a
    real (non-)answer.
    """
    try:
        import anthropic
    except ImportError:
        print(
            "The 'anthropic' package is not installed (pip install -r requirements.txt) — "
            "skipping Claude web_search generation."
        )
        return {"answer": "", "contexts": []}

    vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not vertex_project:
        print("ANTHROPIC_VERTEX_PROJECT_ID is not set — skipping Claude web_search generation.")
        return {"answer": "", "contexts": []}

    region = os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5")
    model = os.environ.get("ANTHROPIC_VERTEX_MODEL", "claude-sonnet-4-5@20250929")

    try:
        client = anthropic.AnthropicVertex(project_id=vertex_project, region=region)
    except Exception as exc:
        print(f"Could not build AnthropicVertex client ({exc}) — skipping Claude web_search generation.")
        return {"answer": "", "contexts": []}

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            messages=[{"role": "user", "content": question}],
        )
    except Exception as exc:
        print(f"Claude web_search call failed for question {question!r}: {exc}")
        return {"answer": "", "contexts": []}

    answer_parts: List[str] = []
    contexts: List[str] = []
    for block in response.content:
        if block.type == "text":
            answer_parts.append(block.text)
        elif block.type == "web_search_tool_result":
            content = block.content
            items = content if isinstance(content, list) else []
            for item in items:
                if isinstance(item, dict):
                    title = item.get("title")
                    url = item.get("url")
                else:
                    title = getattr(item, "title", None)
                    url = getattr(item, "url", None)
                if title or url:
                    contexts.append(f"{title or '(no title)'} ({url or '(no url)'})")

    return {"answer": "".join(answer_parts), "contexts": contexts}


def run_claude_websearch_evaluation(
    testset: List[Dict],
    llm_generate_func: Optional[Callable[[str], Dict[str, Any]]] = None,
    llm=None,
    embeddings=None,
):
    """Run the Claude-web_search comparison arm: no custom retrieval, no
    Qdrant collection, no NFCorpus context — just Claude's own native
    web_search tool grounding (see generate_claude_websearch_answer()).

    Scores ONLY answer_correctness, answer_relevancy, and faithfulness — the
    same corpus-agnostic metric set used for run_closed_book_evaluation's
    generation-only condition. This is a deliberate choice, not an
    oversight: NFCorpus's ground_truths are reference-proxy passages drawn
    from a closed PubMed-style corpus (see MANIFEST.md's "Reference-Proxy
    Caveat"), while Claude's web_search grounds in live web pages that are,
    by construction, a completely different source set. Computing
    context_precision/context_recall against NFCorpus gold passages here
    would just measure "did Claude happen to cite PubMed", which is not a
    meaningful retrieval-quality signal for a web-grounded arm. faithfulness
    is instead computed against each answer's OWN web_search contexts
    (title+url of whatever Claude actually cited that turn), matching how
    run_closed_book_evaluation is scored against no context at all rather
    than NFCorpus's.

    Args:
        testset: List of {"question": str, "ground_truths": [str, ...]}.
        llm_generate_func: called as llm_generate_func(question) ->
            {"answer": str, "contexts": [str, ...]}. Defaults to
            generate_claude_websearch_answer.
        llm / embeddings: optional pre-configured RAGAS/LangChain judge
            LLM/embeddings backend. Default the same way as
            run_ragas_evaluation.

    Returns:
        (ragas EvaluationResult or None, raw_records) where raw_records is a
        List[Dict] of {"question", "answer", "contexts"} for every testset
        item (populated even if scoring itself could not run, so callers can
        still inspect/save what was generated).
    """
    if llm_generate_func is None:
        llm_generate_func = generate_claude_websearch_answer

    samples = []
    raw_records: List[Dict[str, Any]] = []
    any_answer_generated = False
    for item in testset:
        question = item["question"]
        ground_truths = item.get("ground_truths") or []

        gen = llm_generate_func(question) or {}
        answer = gen.get("answer", "") or ""
        contexts = gen.get("contexts") or []
        if answer.strip():
            any_answer_generated = True

        raw_records.append({"question": question, "answer": answer, "contexts": contexts})

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts if contexts else [""],
                response=answer,
                reference=ground_truths[0] if ground_truths else "",
            )
        )

    dataset = EvaluationDataset(samples=samples)

    if not any_answer_generated:
        print(
            "No Claude web_search answers were generated (ANTHROPIC_VERTEX_PROJECT_ID unset, "
            "or all calls failed) — cannot score."
        )
        return None, raw_records

    evaluator_llm = llm if llm is not None else _default_judge_llm()
    if evaluator_llm is None:
        return None, raw_records

    evaluator_embeddings = embeddings if embeddings is not None else _default_judge_embeddings()
    if evaluator_embeddings is None:
        print("No embeddings backend available — cannot score answer_relevancy/answer_correctness.")
        return None, raw_records

    metrics = [faithfulness, answer_relevancy, answer_correctness]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    print(result.to_pandas())
    return result, raw_records


def run_ragas_evaluation(
    testset: List[Dict],
    rag_retrieve_func: Optional[Callable[[str], Any]],
    llm_generate_func: Optional[Callable[[str, List[str]], str]],
    llm=None,
    embeddings=None,
    compute_split_faithfulness_metrics: bool = True,
    compute_disclaimer_stripped_relevancy: bool = True,
):
    """Run a real RAGAS evaluation over `testset`.

    Args:
        testset: List of {"question": str, "ground_truths": [str, ...]}
            (kept as-is — this is the module's existing public input shape).
        rag_retrieve_func: called as rag_retrieve_func(question) -> retrieval
            result. Expected to be compatible with scripts/retrieve.py's
            retrieve_context() output (a dict with "raw_results") or a plain
            List[str] of context strings; both are handled defensively.
        llm_generate_func: called as llm_generate_func(question, contexts) ->
            answer string.
        llm: optional pre-configured RAGAS/LangChain judge LLM. Defaults to
            LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-5")) when
            ANTHROPIC_API_KEY is set, else falls back to Claude on Vertex AI
            (LangchainLLMWrapper(ChatAnthropicVertex(...))) when
            ANTHROPIC_VERTEX_PROJECT_ID is set (via Application Default
            Credentials, no API key needed).
        embeddings: optional pre-configured RAGAS/LangChain embeddings
            backend (used only by answer_relevancy). Defaults to
            LangchainEmbeddingsWrapper(OpenAIEmbeddings()) when OPENAI_API_KEY
            is set, else falls back to Vertex AI embeddings
            (LangchainEmbeddingsWrapper(VertexAIEmbeddings(...))) when
            ANTHROPIC_VERTEX_PROJECT_ID is set.
        compute_split_faithfulness_metrics: when True (default) and at least
            one answer was generated, also runs compute_split_faithfulness()
            per question and adds its fields as extra columns on the returned
            DataFrame (see "Returns" below) — this is the fix for RAGAS's
            faithfulness/[C]-[G] metric-policy mismatch (docs/ARCHITECTURE.md
            Known Issue #4.3; compute_split_faithfulness()'s own docstring
            has the full citation/motivation). Set False to skip it (e.g. to
            keep a run's cost/latency identical to before this fix). The
            EXISTING "faithfulness" column (whole-answer RAGAS faithfulness)
            is never removed or altered by this — it stays exactly as before,
            for comparability against every prior historical result that used
            it. Adds one extra RAGAS evaluate() call per question with at
            least one [C]-tagged claim (for faithfulness_c_only) plus one
            extra cheap LLM call per question with at least one [G]-tagged
            claim (for groundedness_g_only) — the legacy full-answer score
            is NOT recomputed, it's reused directly from this function's own
            main evaluate() call.
        compute_disclaimer_stripped_relevancy: when True (default) and
            "answer_relevancy" was actually scored, also runs
            compute_disclaimer_stripped_answer_relevancy() and adds
            "answer_relevancy_disclaimer_stripped" as an extra column on the
            returned DataFrame — the fix for RAGAS's answer_relevancy
            noncommittal zero-gate (see _strip_context_gap_disclaimer()'s
            docstring for the full root-cause/citation). The EXISTING
            "answer_relevancy" column (whole-answer, unmodified) is never
            removed or altered by this — kept for backward comparability
            with every prior measurement. Set False to skip it (e.g. to keep
            a run's cost/latency identical to before this fix). Only adds a
            second evaluate() call for rows where a disclaimer was actually
            detected and stripped (see that function's docstring for why);
            rows with no disclaimer reuse the existing "answer_relevancy"
            value directly, at zero extra cost.

    Returns:
        A tuple (result, df):
          - result: the ragas EvaluationResult from the main evaluate() call
            (unchanged shape/content from before this fix), or None if
            evaluation could not run (missing callables, or no evaluator LLM
            available) — in that case df is also None.
          - df: result.to_pandas() with the split-faithfulness columns added
            (when compute_split_faithfulness_metrics is True and any answer
            was generated): "faithfulness_c_only", "groundedness_g_only",
            "faithfulness_full_answer_legacy" (identical values to the
            existing "faithfulness" column — kept as an explicit, self-
            documenting alias so a saved CSV doesn't require cross-
            referencing two different column names to know which is which),
            "c_claim_count", "g_claim_count", "c_only_reason",
            "g_only_reason" (see compute_split_faithfulness()'s docstring for
            what each means, especially the None-vs-0.0 "not applicable" vs.
            "applicable and scored zero" distinction). If
            compute_split_faithfulness_metrics is False, or no answers were
            generated, or evaluation failed, df is result.to_pandas() (or
            None) with no split-faithfulness columns added. Independently,
            df also gets "answer_relevancy_disclaimer_stripped" added when
            compute_disclaimer_stripped_relevancy is True and
            "answer_relevancy" was scored (see that parameter's docstring).

        NOTE (return-shape change): earlier versions of this function
        returned just `result`. Every call site inside this module (only
        main(), which discards the return value) has been updated for the
        new (result, df) shape; a caller outside this module using the old
        single-value return would need updating too.
    """
    if rag_retrieve_func is None or llm_generate_func is None:
        print("Provide real rag_retrieve_func and llm_generate_func to run a live evaluation.")
        print(
            "Metrics tracked once wired up: context_precision, context_recall, "
            "faithfulness, answer_relevancy, answer_correctness."
        )
        print("Compare 'light' mode scores vs 'full' mode to prove efficiency without quality loss.")
        return None, None

    samples = []
    any_answer_generated = False
    for item in testset:
        question = item["question"]
        ground_truths = item.get("ground_truths") or []

        retrieval_result = rag_retrieve_func(question)
        contexts = _extract_contexts(retrieval_result)

        answer = llm_generate_func(question, contexts) or ""
        if answer.strip():
            any_answer_generated = True

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=ground_truths[0] if ground_truths else "",
            )
        )

    dataset = EvaluationDataset(samples=samples)

    # --- Evaluator ("judge") LLM ---
    if llm is not None:
        evaluator_llm = llm
    else:
        evaluator_llm = _default_judge_llm()
        if evaluator_llm is None:
            return None, None

    # --- Metrics: always include the context-only metrics; add the
    # generation-dependent ones only if we actually have generated answers.
    # Fresh instances, never the shared module-level `context_precision`/`context_recall`/
    # `faithfulness`/`answer_relevancy`/`answer_correctness` singletons (see the 2026-07-17
    # NeuralWatt shared-singleton-race fix in docs/CHANGELOG.md, M1: ragas's evaluate()
    # mutates `metric.llm = llm` / `metric.embeddings = embeddings` in place on the metric
    # object it's given, so two concurrent evaluate() calls sharing one metric object race on
    # that mutation). This function is called concurrently across (arm, tier) cells by
    # /tmp/nfcorpus_eval_v2/run_model_tier_comparison.py's ThreadPoolExecutor. ---
    metrics = [ContextPrecision(), ContextRecall()]

    if not any_answer_generated:
        print(
            "No generated answers available (ANTHROPIC_API_KEY unset or --generate not passed) "
            "— skipping generation-dependent metrics: faithfulness, answer_relevancy, "
            "answer_correctness."
        )
        evaluator_embeddings = embeddings
    else:
        metrics.append(Faithfulness())

        if embeddings is not None:
            evaluator_embeddings = embeddings
            metrics.append(AnswerRelevancy())
            metrics.append(AnswerCorrectness())
        else:
            evaluator_embeddings = _default_judge_embeddings()
            if evaluator_embeddings is not None:
                metrics.append(AnswerRelevancy())
                metrics.append(AnswerCorrectness())

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    df = result.to_pandas()

    # --- Split-faithfulness fix (docs/ARCHITECTURE.md Known Issue #4.3): an ADDITIONAL
    # reported metric set, never a replacement for the existing "faithfulness" column above
    # (that column is untouched — every prior tuned-vs-baseline comparison in this project
    # depends on it staying exactly as it was). Only runs when there's an answer to split at
    # all ("faithfulness" in metrics implies any_answer_generated was True above).
    if compute_split_faithfulness_metrics and "faithfulness" in df.columns:
        import pandas as pd  # local import, matches this module's existing lazy-pandas convention

        # PARALLELIZED (was a plain sequential for-loop): each row's compute_split_faithfulness()
        # call is an independent, self-contained unit of work -- it builds its own fresh judge_llm/
        # Faithfulness() instance per call (no shared mutable metric object across rows, unlike the
        # NeuralWatt multi-judge panel's shared-singleton hazard documented elsewhere in this file),
        # and this project's judge here is Claude (direct Anthropic or Vertex AI), NOT NeuralWatt --
        # the NeuralWatt-specific 3-concurrent-request account-wide rate limit and the careful
        # max_workers=1-per-thread pattern used by run_neuralwatt_multi_judge_consensus() do NOT
        # apply to this Claude/Anthropic-based path. Earlier eval runs this session used 9-way
        # concurrency for this exact judge/generation path, so ThreadPoolExecutor(max_workers=9) is
        # used here too, submitting all rows at once and collecting results back into their
        # original row order (never assume completion order == submission order).
        precomputed_list: List[Optional[float]] = [
            None if pd.isna(v) else float(v) for v in df["faithfulness"].tolist()
        ]

        def _split_one(i: int) -> Tuple[int, Dict[str, Any]]:
            sample = samples[i]
            return i, compute_split_faithfulness(
                sample.user_input,
                sample.retrieved_contexts,
                sample.response,
                precomputed_full_faithfulness=precomputed_list[i],
            )

        split_by_index: Dict[int, Dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
            futures = [pool.submit(_split_one, i) for i in range(len(samples))]
            for future in concurrent.futures.as_completed(futures):
                i, split = future.result()
                split_by_index[i] = split

        c_only_scores: List[Optional[float]] = []
        g_only_scores: List[Optional[float]] = []
        legacy_scores: List[Optional[float]] = []
        c_counts: List[int] = []
        g_counts: List[int] = []
        c_reasons: List[Optional[str]] = []
        g_reasons: List[Optional[str]] = []

        for i in range(len(samples)):
            split = split_by_index[i]
            c_only_scores.append(split["faithfulness_c_only"])
            g_only_scores.append(split["groundedness_g_only"])
            legacy_scores.append(split["faithfulness_full_answer_legacy"])
            c_counts.append(split["c_claim_count"])
            g_counts.append(split["g_claim_count"])
            c_reasons.append(split["c_only_reason"])
            g_reasons.append(split["g_only_reason"])

        df["faithfulness_c_only"] = c_only_scores
        df["groundedness_g_only"] = g_only_scores
        df["faithfulness_full_answer_legacy"] = legacy_scores
        df["c_claim_count"] = c_counts
        df["g_claim_count"] = g_counts
        df["c_only_reason"] = c_reasons
        df["g_only_reason"] = g_reasons

    # --- answer_relevancy noncommittal-zero-gate fix: an ADDITIONAL reported column,
    # never a replacement for the existing "answer_relevancy" column above (see
    # _strip_context_gap_disclaimer()'s and compute_disclaimer_stripped_answer_relevancy()'s
    # docstrings for the full root-cause and why this runs as a second evaluate() call
    # rather than mutating the main call's samples). Only runs when answer_relevancy was
    # actually scored in the main call above.
    if compute_disclaimer_stripped_relevancy and "answer_relevancy" in df.columns:
        df["answer_relevancy_disclaimer_stripped"] = compute_disclaimer_stripped_answer_relevancy(
            user_inputs=[s.user_input for s in samples],
            responses=[s.response for s in samples],
            existing_relevancy_scores=df["answer_relevancy"].tolist(),
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

    print(df)
    return result, df


def _normalize_testset_item(item: Dict) -> Dict:
    """Normalize a testset item to the module's canonical shape:
    {"question": str, "ground_truths": [str, ...]}.

    Accepts a "ground_truth" (singular string) key as a convenience for
    hand-written CLI testset JSON files and wraps it into "ground_truths".
    """
    item = dict(item)
    if "ground_truths" not in item:
        if "ground_truth" in item:
            item["ground_truths"] = [item["ground_truth"]]
        else:
            item["ground_truths"] = []
    return item


def _dummy_retrieve(question: str) -> Dict[str, Any]:
    """No-op retrieval used for the standalone smoke test."""
    return {
        "raw_results": [
            {
                "text": (
                    "Hierarchical chunking splits documents into large parent chunks for "
                    "context and smaller child chunks for precise retrieval, reducing "
                    "storage while preserving semantic completeness."
                )
            }
        ]
    }


def _dummy_generate(question: str, contexts: List[str]) -> str:
    """No-op generation used for the standalone smoke test."""
    return (
        "It reduces storage while preserving context by using small child chunks for "
        "search and larger parents for full context."
    )


def _make_rag_retrieve_func(collection: str, config: Dict) -> Callable[[str], Any]:
    """Build a rag_retrieve_func bound to a collection/config, backed by
    scripts/retrieve.py's retrieve_context()."""
    from retrieve import retrieve_context

    def _fn(question: str) -> Any:
        return retrieve_context(question, collection, config=config)

    return _fn


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation (context_precision, context_recall, "
        "faithfulness, answer_relevancy, answer_correctness) against the Efficient Hermes RAG."
    )
    parser.add_argument(
        "--testset",
        help='Path to a JSON file of [{"question": "...", "ground_truths": ["..."]}, ...]',
    )
    parser.add_argument("--collection", help="Qdrant collection name to retrieve from")
    parser.add_argument("--config", help="Path to config.yaml (defaults to built-in defaults)")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate answers with the Anthropic SDK (requires ANTHROPIC_API_KEY) so "
        "faithfulness/answer_relevancy/answer_correctness can run",
    )
    parser.add_argument(
        "--verify-repair",
        action="store_true",
        help="Enable the generation-side verify-then-repair step (verify_and_repair_answer()) "
        "on top of --generate: batches every [C]-tagged claim in each draft answer into one "
        "extra LLM call that checks it's actually traceable to the retrieved context, and "
        "retags/rewrites any claim that isn't. Off by default (see docs/CHANGELOG.md); only "
        "takes effect when --generate is also passed. Adds exactly one extra LLM call per "
        "answer that has at least one [C]-tagged claim.",
    )
    parser.add_argument(
        "--closed-book",
        action="store_true",
        help="No-RAG / closed-book mode: skip retrieval entirely and generate answers from "
        "the model's own parametric knowledge only. Scores with answer_relevancy + "
        "answer_similarity (no context-based metrics, since there are no retrieved_contexts). "
        "Requires --testset; --collection/--config are ignored.",
    )
    parser.add_argument(
        "--claude-websearch",
        action="store_true",
        help="Comparison-arm mode: no custom retrieval, use Claude's own native web_search "
        "tool (AnthropicVertex, requires ANTHROPIC_VERTEX_PROJECT_ID) for both retrieval and "
        "generation. Scores with answer_correctness, answer_relevancy, and faithfulness "
        "(against the arm's own web_search contexts, not NFCorpus gold passages). Requires "
        "--testset; --collection/--config are ignored.",
    )
    parser.add_argument(
        "--multi-judge-crosscheck",
        metavar="CSV_PATH",
        help="Re-judge faithfulness only, on a previously saved per-question RAGAS results CSV "
        "(columns: user_input, retrieved_contexts, response, reference, plus any existing "
        "metric columns), using three independent local Ollama judge models "
        f"({', '.join(DEFAULT_OLLAMA_JUDGE_MODELS)}) as a cross-check against the original "
        "judge's faithfulness column. Requires Ollama running locally with those models pulled "
        "(`ollama list`); see the module docstring's MULTI-JUDGE FAITHFULNESS CROSS-CHECK "
        "section. Ignores --testset/--collection/--config/--generate/--closed-book/"
        "--claude-websearch.",
    )
    parser.add_argument(
        "--neuralwatt-multi-judge-crosscheck",
        metavar="CSV_PATH",
        help="Re-judge faithfulness only, on a previously saved per-question RAGAS results CSV "
        "(same shape as --multi-judge-crosscheck), using three NeuralWatt-hosted judge models "
        "(glm-5.2-short, kimi-k2.7-code, qwen3.5-397b) run IN PARALLEL, then aggregate via "
        "geometric median and flag high-disagreement / suspiciously-unanimous rows, with a "
        "Claude tiebreaker for flagged rows. Requires NEURALWATT_API_KEY. See "
        "run_neuralwatt_multi_judge_consensus(). Ignores --testset/--collection/--config/"
        "--generate/--closed-book/--claude-websearch/--multi-judge-crosscheck.",
    )
    parser.add_argument(
        "--neuralwatt-judge-models",
        metavar="MODEL_IDS",
        help="DOCUMENTED EXTENSION POINT, NOT YET IMPLEMENTED into --multi-judge-crosscheck "
        "specifically. Comma-separated NeuralWatt model "
        f"IDs (see NEURALWATT_MODELS: {', '.join(NEURALWATT_MODELS)}) that WOULD be used as "
        "additional --multi-judge-crosscheck members via _neuralwatt_llm() -- NeuralWatt API "
        "access and its OpenAI-API-compatibility are already CONFIRMED and live-verified (see "
        "--neuralwatt-multi-judge-crosscheck, which already runs a real NeuralWatt panel); what's "
        "unbuilt here is specifically folding a NeuralWatt member into the Ollama-only "
        "--multi-judge-crosscheck function. Passing this flag today "
        "does NOT run an evaluation and makes NO network call of any kind — it only prints what "
        "would happen, and exits. See _neuralwatt_llm() and "
        "run_multi_judge_faithfulness_crosscheck()'s docstring for the wiring this would require.",
    )
    parser.add_argument(
        "--export-human-review",
        metavar="OUTPUT_CSV",
        help="Judge-INDEPENDENT sanity check: run retrieval+generation over --testset/"
        "--collection and export a CSV of (question, retrieved_contexts, generated_answer) rows "
        "for a human to hand-label with the Grounded / Partially grounded / Not grounded rubric "
        "(see docs/HUMAN_REVIEW_RUBRIC.md and the module docstring). Requires --testset and "
        "--collection; generation always runs (via generate_answer()) regardless of --generate.",
    )
    parser.add_argument(
        "--summarize-human-review",
        metavar="FILLED_CSV",
        help="Read back a hand-labeled CSV produced by --export-human-review and print a "
        "label count/percentage summary. Combine with --compare-ragas to cross-tabulate mean "
        "RAGAS scores per human label (e.g. checking that 'Partially grounded' correlates with "
        "lower faithfulness, as the rubric predicts). Ignores --testset/--collection/--config/"
        "--generate/--closed-book/--claude-websearch/--multi-judge-crosscheck.",
    )
    parser.add_argument(
        "--compare-ragas",
        metavar="RAGAS_CSV",
        help="Optional companion to --summarize-human-review: a saved RAGAS per-question "
        "results CSV (e.g. result.to_pandas().to_csv(...) from run_ragas_evaluation()) to "
        "cross-tabulate against the human review labels. Ignored without --summarize-human-review.",
    )
    args = parser.parse_args()

    if args.neuralwatt_judge_models:
        requested = [m.strip() for m in args.neuralwatt_judge_models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in NEURALWATT_MODELS]
        print(
            "--neuralwatt-judge-models is a documented extension point into "
            "--multi-judge-crosscheck specifically, not a working integration there yet -- "
            "making NO network call of any kind (including to any neuralwatt.com domain) and "
            "NOT running an evaluation. (NeuralWatt access itself is confirmed and live-verified "
            "-- see --neuralwatt-multi-judge-crosscheck, which already runs a real panel.)"
        )
        print(f"Requested NeuralWatt model(s): {', '.join(requested) if requested else '(none)'}")
        if unknown:
            print(
                f"Note: {', '.join(unknown)} not found in NEURALWATT_MODELS "
                f"({', '.join(NEURALWATT_MODELS)}) — this is a confirmed, live-verified model "
                "catalog, so an unrecognized ID here is most likely a typo rather than a stale "
                "listing."
            )
        print(
            "NeuralWatt API access and its OpenAI-API-compatibility are already CONFIRMED and "
            "live-verified (see run_neuralwatt_multi_judge_consensus()). What remains unbuilt is "
            "specifically folding these models into run_multi_judge_faithfulness_crosscheck() as "
            "additional judge members -- see that function's docstring's NeuralWatt-integration "
            "section for the exact shape. This flag intentionally stops here."
        )
        return

    if args.summarize_human_review:
        summarize_human_review(args.summarize_human_review, args.compare_ragas)
        return

    if args.multi_judge_crosscheck:
        import pandas as pd

        original_df = pd.read_csv(args.multi_judge_crosscheck)
        crosscheck_results = run_multi_judge_faithfulness_crosscheck(args.multi_judge_crosscheck)

        print(f"\nMulti-judge faithfulness cross-check for {args.multi_judge_crosscheck}")
        print("=" * 72)

        comparison = pd.DataFrame(
            {"user_input": original_df["user_input"]}
        )
        if "faithfulness" in original_df.columns:
            comparison["claude_faithfulness"] = original_df["faithfulness"]
        else:
            print(
                "Note: original CSV has no 'faithfulness' column to compare against — showing "
                "Ollama judge scores only."
            )

        for model_name, outcome in crosscheck_results.items():
            if outcome["error"] is not None:
                print(f"\n[{model_name}] FAILED: {outcome['error']}")
                continue
            model_df = outcome["result"].to_pandas()
            comparison[f"{model_name}_faithfulness"] = model_df["faithfulness"]

        print(comparison.to_string(index=False))
        return

    if args.neuralwatt_multi_judge_crosscheck:
        consensus = run_neuralwatt_multi_judge_consensus(args.neuralwatt_multi_judge_crosscheck)
        per_question = consensus["per_question"]
        per_model = consensus["per_model"]

        print(f"\nNeuralWatt multi-judge consensus for {args.neuralwatt_multi_judge_crosscheck}")
        print("=" * 72)

        for model_name, outcome in per_model.items():
            if outcome["error"] is not None:
                print(f"[{model_name}] FAILED: {outcome['error']}")
            elif outcome.get("judge_used") == "ollama_fallback":
                print(
                    f"[{model_name}] NeuralWatt failed ({outcome.get('neuralwatt_error')}) — "
                    f"scored via Ollama-cloud fallback instead."
                )

        model_names = list(per_model.keys())
        for model_name in model_names:
            scores = [
                row["neuralwatt_scores"][model_name]
                for row in per_question
                if row["neuralwatt_scores"].get(model_name) is not None
            ]
            mean_score = sum(scores) / len(scores) if scores else None
            print(
                f"Mean {model_name} faithfulness: "
                f"{mean_score:.3f}" if mean_score is not None else f"Mean {model_name} faithfulness: n/a"
            )

        medians = [row["geometric_median"] for row in per_question if row["geometric_median"] is not None]
        mean_median = sum(medians) / len(medians) if medians else None
        print(
            f"Mean geometric-median faithfulness: {mean_median:.3f}"
            if mean_median is not None
            else "Mean geometric-median faithfulness: n/a"
        )

        high_disagreement_count = sum(1 for row in per_question if row["high_disagreement"])
        suspiciously_unanimous_count = sum(1 for row in per_question if row["suspiciously_unanimous"])
        all_judges_failed_count = sum(1 for row in per_question if row["all_judges_failed"])
        print(f"High-disagreement rows: {high_disagreement_count} / {len(per_question)}")
        print(f"Suspiciously-unanimous rows: {suspiciously_unanimous_count} / {len(per_question)}")
        print(f"All-judges-failed rows: {all_judges_failed_count} / {len(per_question)}")

        if high_disagreement_count or all_judges_failed_count:
            print(
                "\nHigh-disagreement / all-judges-failed rows "
                "(all judge scores + geometric median + Claude tiebreaker):"
            )
            for row in per_question:
                if not (row["high_disagreement"] or row["all_judges_failed"]):
                    continue
                question_preview = str(row["user_input"])[:60]
                print(f"  - {question_preview!r} (all_judges_failed={row['all_judges_failed']})")
                print(f"    scores: {row['neuralwatt_scores']}")
                print(f"    geometric_median: {row['geometric_median']}  delta_m: {row['delta_m']}")
                print(f"    claude_tiebreaker: {row['claude_tiebreaker']}")
        return

    if args.closed_book and not args.testset:
        print("--closed-book requires --testset.")
        sys.exit(1)

    if args.claude_websearch and not args.testset:
        print("--claude-websearch requires --testset.")
        sys.exit(1)

    if args.export_human_review and not (args.testset and args.collection):
        print("--export-human-review requires both --testset and --collection.")
        sys.exit(1)

    if args.testset:
        try:
            with open(args.testset, "r") as f:
                raw_testset = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read --testset {args.testset}: {exc}")
            sys.exit(1)
        testset = [_normalize_testset_item(item) for item in raw_testset]
    else:
        testset = [
            {
                "question": "What is the main benefit of hierarchical chunking in RAG?",
                "ground_truths": [
                    "It reduces storage while preserving context by using small child "
                    "chunks for search and larger parents for full context."
                ],
            }
        ]

    if args.closed_book:
        run_closed_book_evaluation(testset)
        return

    if args.claude_websearch:
        run_claude_websearch_evaluation(testset)
        return

    if args.export_human_review:
        from utils import load_config

        config = load_config(args.config)
        rag_retrieve_func = _make_rag_retrieve_func(args.collection, config)
        export_for_human_review(testset, rag_retrieve_func, generate_answer, args.export_human_review)
        return

    if args.testset or args.collection:
        if not (args.testset and args.collection):
            print("Both --testset and --collection are required together for a live evaluation.")
            sys.exit(1)

        from utils import load_config

        config = load_config(args.config)
        rag_retrieve_func = _make_rag_retrieve_func(args.collection, config)
        if args.generate:
            if args.verify_repair:
                llm_generate_func = lambda question, contexts: generate_answer(  # noqa: E731
                    question, contexts, verify_and_repair=True
                )
            else:
                llm_generate_func = generate_answer
        else:
            print("--generate not passed — answers will not be generated for this run.")
            llm_generate_func = lambda question, contexts: ""  # noqa: E731

        run_ragas_evaluation(testset, rag_retrieve_func, llm_generate_func)
    else:
        print(
            "No --testset/--collection provided — running a synthetic smoke test with dummy "
            "retrieval/generation. Pass --testset and --collection for a real evaluation."
        )
        run_ragas_evaluation(testset, _dummy_retrieve, _dummy_generate)


if __name__ == "__main__":
    main()
