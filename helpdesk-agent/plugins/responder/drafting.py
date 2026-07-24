"""LLM draft construction — Gemini prompt, response classification, ticket context.

Prompt tuning results (5-fold stratified CV on ai_draft_feedback ground truth):
==============================================================================
Variant designs:
  A baseline           — original ticket + reference material prompt
  B structured_context — XML tags for KB/FAQ/tickets/docs + ticket
  C chain_of_thought   — classify ticket type/resolution pattern before drafting
  D lean_context       — top-3 hits per source (vs top-5 baseline)
  E xml_plus_cot       — XML tags + chain of thought reasoning
  F lean_plus_cot      — top-3 hits + chain of thought reasoning
  G kitchen_sink       — XML tags + top-3 hits + chain of thought reasoning

variant            | mean ± std judge | p vs baseline | n    | rt_acc | len p50/p95 | lat p95
-------------------|------------------|---------------|------|--------|-------------|--------
baseline           | 0.503 ± 0.266    | —             | 1595 | 0.763  | 222/480     | 37.03s
structured_context | 0.541 ± 0.279    | 0.0001        | 1595 | 0.744  | 222/510     | 30.35s
chain_of_thought   | 0.510 ± 0.275    | 0.5182        | 1595 | 0.757  | 222/486     | 37.19s
lean_context       | 0.510 ± 0.273    | 0.5061        | 1595 | 0.759  | 222/479     | 36.21s
xml_plus_cot       | 0.535 ± 0.288    | 0.0012        | 1595 | 0.748  | 220/494     | 29.38s
lean_plus_cot      | 0.510 ± 0.275    | 0.5133        | 1595 | 0.750  | 222/487     | 37.35s
kitchen_sink       | 0.531 ± 0.286    | 0.0054        | 1595 | 0.752  | 221/499     | 31.38s

Winner: structured_context (structured_context beat baseline at p < 0.05)
Evaluated: 2026-06-16; folds=5; stratify=response_type; random_state=42
Welch's t-test (equal_var=False) vs baseline; alpha=0.05 required for promotion.
Fold means ± std:
  baseline: [0.489±0.251, 0.499±0.266, 0.522±0.268, 0.500±0.275, 0.506±0.269]
  structured_context: [0.529±0.276, 0.527±0.285, 0.539±0.279, 0.557±0.273, 0.551±0.280]
  chain_of_thought: [0.512±0.261, 0.494±0.280, 0.512±0.275, 0.506±0.269, 0.524±0.286]
  lean_context: [0.500±0.266, 0.493±0.278, 0.519±0.268, 0.522±0.264, 0.514±0.286]
  xml_plus_cot: [0.520±0.281, 0.525±0.292, 0.539±0.273, 0.547±0.296, 0.544±0.293]
  lean_plus_cot: [0.495±0.271, 0.498±0.285, 0.513±0.267, 0.516±0.271, 0.526±0.279]
  kitchen_sink: [0.514±0.275, 0.519±0.293, 0.526±0.289, 0.563±0.284, 0.530±0.287]
==============================================================================
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold

from config import (
    GEMINI_MODEL_ANALYSIS, DOC_SOURCE_DOMAINS, DOC_CACHE_DAYS,
    apply_cloud_terminology,
)
from core.pipeline import get_plugin_config
from core.genai import get_genai_client
from db import get_db_conn
from ingest.oauth2lo import get_cloud_base_url

from .ann_fewshot import ANNFewShotIndex
from .jira_comments import _jira_request
from .feedback import get_few_shot_examples, get_organic_examples

log = logging.getLogger(__name__)

PROMPT_VARIANT_BASELINE = "baseline"
PROMPT_VARIANT_STRUCTURED = "structured_context"
PROMPT_VARIANT_COT = "chain_of_thought"
PROMPT_VARIANT_LEAN = "lean_context"
PROMPT_VARIANT_XML_COT = "xml_plus_cot"
PROMPT_VARIANT_LEAN_COT = "lean_plus_cot"
PROMPT_VARIANT_KITCHEN_SINK = "kitchen_sink"

PROMPT_VARIANTS = (
    PROMPT_VARIANT_BASELINE,
    PROMPT_VARIANT_STRUCTURED,
    PROMPT_VARIANT_COT,
    PROMPT_VARIANT_LEAN,
    PROMPT_VARIANT_XML_COT,
    PROMPT_VARIANT_LEAN_COT,
    PROMPT_VARIANT_KITCHEN_SINK,
)

SUPPORTED_RESPONSE_TYPES = (
    "self_service",
    "admin_action",
    "needs_info",
)

DEFAULT_MAX_PER_SOURCE = 5
LEAN_MAX_PER_SOURCE = 3
CV_RANDOM_STATE = 42
CV_N_SPLITS = 5
CV_SIGNIFICANCE_ALPHA = 0.05
CHECKPOINT_SCHEMA_VERSION = 3
CHECKPOINT_METRIC = "llm_judge_dual_v1"

# Deployed prompt variant used by _draft_response().
WINNING_PROMPT_VARIANT = 'structured_context'

_COT_PREAMBLE = """First, identify the ticket type (self_service / admin_action / needs_info) and the likely resolution pattern. Then draft your response.

"""

_CLASSIFY_INSTRUCTIONS = """STEP 1 — CLASSIFY the response type:

- "self_service": The user can resolve this themselves with documentation or steps they can perform with their own permissions (e.g. adjusting their own settings, following a guide, using a feature differently).
- "admin_action": An administrator or someone with elevated project/org roles needs to make changes on behalf of the user (e.g. modifying project settings, changing permission schemes, updating workflow configurations, adding groups to roles). The user cannot do this themselves.
- "needs_info": The ticket does not contain enough information to determine the solution or take action. Specific details are needed from the customer before proceeding.

IMPORTANT CONTEXT:
- Our team uses the Atlassian Cloud platform, Rovo MCP Server, and related tools successfully. When customers report issues with these tools, the problem is typically on THEIR side (configuration, permissions, setup).
- There are no OAuth tokens on personal Atlassian accounts. API tokens are the personal auth method.
- Many issues involve BOTH admin and customer actions. Classify based on what the PRIMARY fix requires.

STEP 2 — DRAFT the response. ALWAYS produce BOTH technician_steps AND customer_response:

TECHNICIAN_STEPS (always required — internal diagnostic notes for the support agent):
  - Numbered diagnostic and action steps for the agent/admin to investigate
  - Be specific — include exact navigation paths (e.g. "admin.atlassian.com > Security > API tokens")
  - ALWAYS include branching logic: "If X is already enabled/done, then check Y"
  - Cover multiple scenarios — don't assume the first check will be the fix
  - Include when to escalate
  - CITE YOUR SOURCES: For each step, include the URL from the reference material that supports it. Format: "See: [title](URL)" at the end of the step.
  - Example format:
    1. Check setting A at [path]. If disabled, enable it. See: [Configure SLAs](https://support.atlassian.com/...)
    2. If A was already enabled, check setting B at [path]. See: [Manage Permissions](https://support.atlassian.com/...)
    3. If both are correct, the issue is likely [C] — escalate to [team].

CUSTOMER_RESPONSE (always required — what gets sent to the customer):
  - This is a BRIEF acknowledgment, NOT detailed technical steps
  - For "admin_action" or issues being investigated: Simply acknowledge the issue and confirm the team is looking into it (e.g. "We are looking into this and will follow up with you shortly.")
  - For "self_service" where the user truly can fix it themselves: Provide clear numbered steps with inline links to source documentation
  - For "needs_info": Acknowledge the issue, list specific missing info, explain why it's needed
  - The customer response should NEVER contain the admin/technician diagnostic steps
  - Keep it concise — 1-3 sentences for acknowledgments, up to 300 words for self-service steps

FORMATTING RULES (all types):
- Be warm, professional, and concise
- If the solution involves steps for the customer, use a numbered list
- MANDATORY: Embed source links inline using markdown format [Title](URL). Use the URLs provided in the REFERENCE MATERIAL above. Every instruction or claim MUST cite its source URL.

FORBIDDEN PATTERNS (Negative Constraints):
1. Do NOT mention that this is an AI-drafted response.
2. Do NOT reference internal ticket numbers or internal tools.
3. Do NOT include any sign-off, greeting, or signature.
4. Do NOT invent missing values; output 'unknown' if data is absent.

TERMINOLOGY: In Jira Cloud, "projects" have been renamed to "spaces". Always use "space" / "spaces" instead of "project" / "projects". Exception: keep "project key" and "project category" as-is.

Return valid JSON only (no markdown fencing):
{{
  "response_type": "self_service" | "admin_action" | "needs_info",
  "customer_response": "<brief acknowledgment OR self-service steps for the customer — include source links>",
  "admin_steps": "<diagnostic and action steps for the technician — include source links and branching logic>",
  "missing_info": "<summary of what's missing, or null if not needs_info>"
}}"""


def _fetch_doc_content(url: str) -> str | None:
    """Fetch and cache doc page content from configured source domains.

    Only fetches from DOC_SOURCE_DOMAINS allowlist. Returns plain text
    content (HTML stripped), or None if not fetchable.
    """
    from urllib.parse import urlparse
    from datetime import datetime, timezone, timedelta

    parsed = urlparse(url)
    if parsed.hostname not in DOC_SOURCE_DOMAINS:
        return None

    # Check cache
    with get_db_conn() as conn:
        cached = conn.execute(
            "SELECT content, fetched_at FROM doc_content_cache WHERE url = ?",
            (url,),
        ).fetchone()
        if cached:
            fetched = datetime.fromisoformat(cached["fetched_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - fetched < timedelta(days=DOC_CACHE_DAYS):
                return cached["content"]

    # Fetch fresh
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "JSM-Modeling-Bot/1.0 (internal tooling)"
        })
        if resp.status_code != 200:
            return None

        # Strip HTML to plain text
        text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:3000]

        # Cache it
        now = datetime.now(timezone.utc).isoformat()
        with get_db_conn() as conn:
            conn.execute(
                """INSERT INTO doc_content_cache (url, content, fetched_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT (url) DO UPDATE SET content=EXCLUDED.content, fetched_at=EXCLUDED.fetched_at""",
                (url, text, now),
            )

        return text
    except Exception as e:
        log.debug("Could not fetch doc content from %s: %s", url, e)
        return None


def _build_static_few_shot_block() -> str:
    """Build few-shot examples block for the Gemini prompt.

    Pulls up to 5 total examples, prioritizing draft-feedback pairs
    (which have correction signal) and filling remaining slots with
    organic agent response examples from resolved tickets.
    Returns empty string during cold start with no examples.
    """
    examples = []
    for rt in ("self_service", "admin_action", "needs_info"):
        examples.extend(get_few_shot_examples(rt, limit=2))
        if len(examples) >= 5:
            break

    remaining = 5 - len(examples)
    if remaining > 0:
        organic = get_organic_examples(response_type=None, limit=remaining)
        examples.extend(organic)

    if not examples:
        return ""

    lines = ["EXAMPLES FROM PAST RESPONSES (these show the style our agents prefer):\n"]
    for ex in examples[:5]:
        rt_label = ex["response_type"].upper().replace("_", " ")
        if "draft_customer_response" in ex:
            # Draft-feedback pair
            lines.append(f"[{rt_label} example]")
            lines.append(f"AI Draft JSON:\n{{\n  \"response_type\": \"{ex['response_type']}\",\n  \"customer_response\": {repr(ex['draft_customer_response'][:500])},\n  \"admin_steps\": \"<diagnostic steps here>\"\n}}")
            lines.append(f"Agent's Final Version: {ex['actual_response'][:500]}")
        else:
            # Organic agent response
            lines.append(f"[{rt_label} example]")
            lines.append(f"Agent Response JSON:\n{{\n  \"response_type\": \"{ex['response_type']}\",\n  \"customer_response\": {repr(ex['agent_response'][:500])},\n  \"admin_steps\": \"<diagnostic steps here>\"\n}}")
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def _build_few_shot_block(ticket_text: str) -> str:
    """Build the few-shot block using ANN retrieval, with a static fallback."""
    cfg = get_plugin_config("responder")
    k = int(cfg.get("fewshot_k", 5))
    similarity_floor = float(cfg.get("fewshot_similarity_floor", 0.5))

    examples = ANNFewShotIndex.retrieve(
        ticket_text,
        k=k,
        similarity_floor=similarity_floor,
    )
    if not examples:
        if ANNFewShotIndex.is_empty():
            log.warning(
                "Few-shot ANN index is empty - falling back to static top-5 examples"
            )
        else:
            log.warning(
                "Few-shot ANN retrieval returned no matches above %.2f - falling back to static top-5 examples",
                similarity_floor,
            )
        return _build_static_few_shot_block()

    lines = [
        "EXAMPLES FROM PAST RESPONSES (these are semantically similar approved examples):\n"
    ]
    for example_text, score in examples[:k]:
        lines.append(f"[ANN match score: {score:.3f}]")
        lines.append(example_text[:1500])
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def _max_per_source_for_variant(variant: str) -> int:
    if variant in (PROMPT_VARIANT_LEAN, PROMPT_VARIANT_LEAN_COT, PROMPT_VARIANT_KITCHEN_SINK):
        return LEAN_MAX_PER_SOURCE
    return DEFAULT_MAX_PER_SOURCE


def _build_baseline_context(matches: dict) -> str:
    """Preserve the original baseline retrieval ordering and limits."""
    context_parts: list[str] = []

    for faq in matches.get("faq_matches", []):
        context_parts.append(f"FAQ: {faq.get('title', '')}\n{faq.get('body_html', '')[:1000]}")

    fetched_count = 0
    for kb in matches.get("kb_matches", []):
        title = kb.get("title", "")
        url = kb.get("url", "")
        entry = f"KB Article: {title}\nURL: {url}"
        if url and fetched_count < 3:
            content = _fetch_doc_content(url)
            if content:
                entry += f"\nContent: {content[:1500]}"
                fetched_count += 1
        context_parts.append(entry)

    for ticket in matches.get("ticket_matches", [])[:3]:
        summary = ticket.get("summary", "")
        resolution = ticket.get("resolution_summary", "")
        context_parts.append(f"Resolved Ticket: {summary}\nResolution: {resolution}")

    for doc in matches.get("atlassian_matches", []):
        title = doc.get("title", "")
        url = doc.get("url", "")
        entry = f"Official Atlassian Documentation: {title}\nURL: {url}"
        if url and fetched_count < 3:
            content = _fetch_doc_content(url)
            if content:
                entry += f"\nContent: {content[:1500]}"
                fetched_count += 1
        context_parts.append(entry)

    if not context_parts:
        return ""
    return "\n\n---\n\n".join(context_parts)


def _build_match_sections(matches: dict, *, max_per_source: int = DEFAULT_MAX_PER_SOURCE) -> dict[str, list[str]]:
    """Build labeled retrieval sections from lookup matches."""
    sections: dict[str, list[str]] = {
        "kb_articles": [],
        "faq_entries": [],
        "similar_tickets": [],
        "atlassian_docs": [],
    }

    fetched_count = 0
    for kb in matches.get("kb_matches", [])[:max_per_source]:
        title = kb.get("title", "")
        url = kb.get("url", "")
        entry = f"KB Article: {title}\nURL: {url}"
        if url and fetched_count < 3:
            content = _fetch_doc_content(url)
            if content:
                entry += f"\nContent: {content[:1500]}"
                fetched_count += 1
        sections["kb_articles"].append(entry)

    for faq in matches.get("faq_matches", [])[:max_per_source]:
        sections["faq_entries"].append(
            f"FAQ: {faq.get('title', '')}\n{faq.get('body_html', '')[:1000]}"
        )

    for ticket in matches.get("ticket_matches", [])[:max_per_source]:
        summary = ticket.get("summary", "")
        resolution = ticket.get("resolution_summary", "")
        sections["similar_tickets"].append(
            f"Resolved Ticket: {summary}\nResolution: {resolution}"
        )

    for doc in matches.get("atlassian_matches", [])[:max_per_source]:
        title = doc.get("title", "")
        url = doc.get("url", "")
        entry = f"Official Atlassian Documentation: {title}\nURL: {url}"
        if url and fetched_count < 3:
            content = _fetch_doc_content(url)
            if content:
                entry += f"\nContent: {content[:1500]}"
                fetched_count += 1
        sections["atlassian_docs"].append(entry)

    return sections


def _sections_have_content(sections: dict[str, list[str]]) -> bool:
    return any(sections[key] for key in sections)


def _join_section_lines(lines: list[str]) -> str:
    return "\n\n---\n\n".join(lines)


def _format_baseline_context(sections: dict[str, list[str]]) -> str:
    parts: list[str] = []
    parts.extend(sections["faq_entries"])
    parts.extend(sections["kb_articles"])
    parts.extend(sections["similar_tickets"])
    parts.extend(sections["atlassian_docs"])
    return _join_section_lines(parts)


def _format_structured_context(
    sections: dict[str, list[str]],
    ticket_summary: str,
    ticket_description: str,
) -> str:
    blocks: list[str] = []

    if sections["kb_articles"]:
        blocks.append(
            "<kb_articles>\n" + _join_section_lines(sections["kb_articles"]) + "\n</kb_articles>"
        )
    if sections["faq_entries"]:
        blocks.append(
            "<faq_entries>\n" + _join_section_lines(sections["faq_entries"]) + "\n</faq_entries>"
        )
    if sections["similar_tickets"]:
        blocks.append(
            "<resolved_tickets>\n"
            + _join_section_lines(sections["similar_tickets"])
            + "\n</resolved_tickets>"
        )
    if sections["atlassian_docs"]:
        blocks.append(
            "<atlassian_docs>\n"
            + _join_section_lines(sections["atlassian_docs"])
            + "\n</atlassian_docs>"
        )

    blocks.append(
        "<ticket>\n"
        f"Summary: {ticket_summary}\n"
        f"Description: {(ticket_description or '')[:3000]}\n"
        "</ticket>"
    )
    return "\n\n".join(blocks)


def _build_draft_prompt(
    variant: str,
    ticket_summary: str,
    ticket_description: str,
    matches: dict,
    *,
    few_shot_block: str | None = None,
) -> str | None:
    """Build the Gemini prompt for a given tuning variant."""
    ticket_text = f"{ticket_summary}\n\n{ticket_description or ''}".strip()
    if few_shot_block is None:
        few_shot_block = _build_few_shot_block(ticket_text)

    preamble = (
        "You are a friendly, knowledgeable IT help desk agent for the Atlassian Cloud migration team.\n"
        "A user has submitted a support ticket. Analyze the ticket and reference material, "
        "then classify and draft the appropriate response.\n\n"
    )

    cot_block = _COT_PREAMBLE if variant in (
        PROMPT_VARIANT_COT, PROMPT_VARIANT_XML_COT, PROMPT_VARIANT_LEAN_COT, PROMPT_VARIANT_KITCHEN_SINK
    ) else ""

    if variant in (PROMPT_VARIANT_BASELINE, PROMPT_VARIANT_COT):
        context = _build_baseline_context(matches)
        if not context:
            return None
        return (
            f"{preamble}"
            f"TICKET:\n"
            f"Summary: {ticket_summary}\n"
            f"Description: {(ticket_description or '')[:3000]}\n\n"
            f"REFERENCE MATERIAL (from FAQ, KB articles, previously resolved tickets, "
            f"and official Atlassian documentation):\n"
            f"{context[:4000]}\n\n"
            f"{few_shot_block}{cot_block}{_CLASSIFY_INSTRUCTIONS}"
        )

    if variant in (PROMPT_VARIANT_LEAN, PROMPT_VARIANT_LEAN_COT):
        max_per_source = _max_per_source_for_variant(variant)
        sections = _build_match_sections(matches, max_per_source=max_per_source)
        if not _sections_have_content(sections):
            return None
        context = _format_baseline_context(sections)[:4000]
        return (
            f"{preamble}"
            f"TICKET:\n"
            f"Summary: {ticket_summary}\n"
            f"Description: {(ticket_description or '')[:3000]}\n\n"
            f"REFERENCE MATERIAL (from FAQ, KB articles, previously resolved tickets, "
            f"and official Atlassian documentation):\n"
            f"{context}\n\n"
            f"{few_shot_block}{cot_block}{_CLASSIFY_INSTRUCTIONS}"
        )

    if variant in (PROMPT_VARIANT_STRUCTURED, PROMPT_VARIANT_XML_COT, PROMPT_VARIANT_KITCHEN_SINK):
        max_per_source = _max_per_source_for_variant(variant)
        sections = _build_match_sections(matches, max_per_source=max_per_source)
        if not _sections_have_content(sections):
            return None
        context = _format_structured_context(
            sections, ticket_summary, ticket_description
        )[:4000]
        return (
            f"{preamble}{context}\n\n"
            f"{few_shot_block}{cot_block}{_CLASSIFY_INSTRUCTIONS}"
        )

    raise ValueError(f"Unhandled prompt variant: {variant!r}")


def _extract_source_urls(text: str | None) -> list[str]:
    """Extract all http/https URLs from draft text, preserving insertion order."""
    return list(dict.fromkeys(re.findall(r'https?://[^\s\)\]"]+', text or "")))


def _invoke_gemini_draft(prompt: str) -> dict | None:
    """Call Gemini and normalize the structured draft response."""
    result, _latency = _invoke_gemini_draft_timed(prompt)
    return result


def _invoke_gemini_draft_timed(prompt: str) -> tuple[dict | None, float]:
    """Call Gemini and return (draft_dict, latency_seconds)."""
    started = time.perf_counter()
    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL_ANALYSIS,
            contents=prompt,
        )

        from utils.gemini import parse_json_response

        result = parse_json_response(response.text)

        if isinstance(result, list):
            result = result[0] if result else None
        if not isinstance(result, dict) or "response_type" not in result:
            log.warning("Could not parse structured response, falling back to plain text")
            return {
                "response_type": "self_service",
                "customer_response": response.text.strip(),
                "admin_steps": None,
                "missing_info": None,
                "sources": _extract_source_urls(response.text.strip()),
            }, time.perf_counter() - started

        rt = result.get("response_type", "self_service").lower().replace(" ", "_")
        if rt not in ("self_service", "admin_action", "needs_info"):
            rt = "self_service"
        result["response_type"] = rt

        for key in ("customer_response", "admin_steps", "missing_info"):
            if result.get(key):
                val = result[key]
                if isinstance(val, list):
                    val = "\n".join(str(item) for item in val)
                elif not isinstance(val, str):
                    val = str(val)
                result[key] = apply_cloud_terminology(val)

        result["sources"] = _extract_source_urls(
            result.get("customer_response")
        ) + _extract_source_urls(result.get("admin_steps"))

        return result, time.perf_counter() - started
    except Exception as e:
        log.error("Failed to draft response: %s", e)
        return None, time.perf_counter() - started


def _percentile(values: list[float | int], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, pct))


def _draft_response_with_variant_timed(
    variant: str,
    ticket_summary: str,
    ticket_description: str,
    matches: dict,
    *,
    few_shot_block: str | None = None,
) -> tuple[dict | None, float]:
    """Generate a draft using a specific prompt-tuning variant; return (draft, latency_s)."""
    if not matches.get("found"):
        return None, 0.0

    prompt = _build_draft_prompt(
        variant,
        ticket_summary,
        ticket_description,
        matches,
        few_shot_block=few_shot_block,
    )
    if not prompt:
        return None, 0.0
    return _invoke_gemini_draft_timed(prompt)


def _draft_response_with_variant(
    variant: str,
    ticket_summary: str,
    ticket_description: str,
    matches: dict,
    *,
    few_shot_block: str | None = None,
) -> dict | None:
    """Generate a draft using a specific prompt-tuning variant."""
    draft, _latency = _draft_response_with_variant_timed(
        variant,
        ticket_summary,
        ticket_description,
        matches,
        few_shot_block=few_shot_block,
    )
    return draft


def _draft_response(ticket_summary: str, ticket_description: str, matches: dict) -> dict | None:
    """Use Gemini to classify the response type and draft accordingly.

    Returns a dict with:
        response_type: "self_service" | "admin_action" | "needs_info"
        customer_response: str — the draft to send to the customer
        admin_steps: str | None — step-by-step admin instructions (admin_action only)
        missing_info: str | None — what info is needed (needs_info only)
    Returns None if no draft could be produced.
    """
    return _draft_response_with_variant(
        WINNING_PROMPT_VARIANT, ticket_summary, ticket_description, matches
    )


@dataclass
class PromptVariantFoldMetrics:
    fold: int
    judge_scores: list[float] = field(default_factory=list)
    response_type_hits: list[bool] = field(default_factory=list)
    draft_lengths: list[int] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)


@dataclass
class PromptVariantCVResult:
    variant: str
    mean_judge_score: float
    std_judge_score: float
    fold_means: list[float]
    fold_stds: list[float]
    p_value_vs_baseline: float | None
    n_samples: int
    response_type_accuracy: float
    draft_length_p50: float
    draft_length_p95: float
    latency_p95: float


def _default_checkpoint_file(
    *,
    variants: tuple[str, ...],
    n_splits: int,
    random_state: int,
) -> str:
    variant_slug = "-".join(variants)
    safe_slug = re.sub(r"[^a-z0-9_-]+", "-", variant_slug.lower()).strip("-")
    return (
        f"m9_checkpoint_v{CHECKPOINT_SCHEMA_VERSION}_{CHECKPOINT_METRIC}_"
        f"{n_splits}fold_seed{random_state}_{safe_slug}.jsonl"
    )


def _checkpoint_metadata(
    *,
    variants: tuple[str, ...],
    n_splits: int,
    random_state: int,
) -> dict[str, object]:
    return {
        "record_type": "meta",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metric": CHECKPOINT_METRIC,
        "variants": list(variants),
        "n_splits": n_splits,
        "random_state": random_state,
    }


def load_ground_truth_corpus() -> list[dict]:
    """Load labeled rows from ai_draft_feedback for prompt tuning."""
    with get_db_conn() as conn:
        rows = conn.execute(
            """
            SELECT id AS feedback_row_id, ticket_key, response_type, draft_customer_response,
                   actual_response, similarity_score, agent_feedback,
                   feedback_category, captured_at
            FROM ai_draft_feedback
            WHERE actual_response IS NOT NULL
              AND similarity_score IS NOT NULL
              AND response_type IN ('self_service', 'admin_action', 'needs_info')
            ORDER BY ticket_key, captured_at, id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def build_stratified_folds(
    corpus: list[dict],
    *,
    n_splits: int = CV_N_SPLITS,
    random_state: int = CV_RANDOM_STATE,
) -> list[tuple[list[int], list[int]]]:
    """Return (train_indices, test_indices) for stratified k-fold CV."""
    if len(corpus) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} labeled rows for {n_splits}-fold CV, got {len(corpus)}"
        )

    labels = [row["response_type"] for row in corpus]
    unsupported = sorted({label for label in labels if label not in SUPPORTED_RESPONSE_TYPES})
    if unsupported:
        raise ValueError(
            "Unsupported response_type values in corpus: "
            f"{unsupported}. Expected only {SUPPORTED_RESPONSE_TYPES}."
        )
    label_counts = Counter(labels)
    if any(count < n_splits for count in label_counts.values()):
        raise ValueError(
            "Each response_type needs at least "
            f"{n_splits} samples for stratified CV; counts={dict(label_counts)}"
        )

    indices = np.arange(len(corpus))
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return [
        (train_idx.tolist(), test_idx.tolist())
        for train_idx, test_idx in splitter.split(indices, labels)
    ]


def _normalize_judge_score(score: object) -> float | None:
    """Normalize a 1-5 judge score to 0.0-1.0."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value < 1.0 or value > 5.0:
        return None
    return max(0.0, min(1.0, (value - 1.0) / 4.0))


def evaluate_draft_with_llm(
    customer_response: str,
    admin_steps: str | None,
    actual_response: str,
    *,
    summary: str = "",
    description: str = "",
    expected_response_type: str | None = None,
) -> float:
    """LLM-as-a-Judge evaluation of customer and internal draft quality.

    Returns the average normalized score across the customer-facing response
    and the internal admin steps, each graded on a 1-5 scale with different
    rubrics.
    """
    from config import GEMINI_FLASH_MODEL

    customer_response = customer_response or ""
    admin_steps = admin_steps or ""
    actual_response = actual_response or ""

    if not customer_response.strip() or not actual_response.strip():
        return 0.0

    prompt = f"""You are an expert evaluator for IT help-desk response quality.
Evaluate the AI draft against the ticket context and the ground-truth agent response.

Return ONLY valid JSON with this exact shape:
{{
  "customer_score": <integer 1-5>,
  "admin_score": <integer 1-5>,
  "customer_rationale": "<short reason>",
  "admin_rationale": "<short reason>"
}}

Score the two fields separately with different rubrics:

1. customer_score
- Grade ONLY the customer-facing response.
- Focus on accuracy vs the ground-truth response, clarity, tone, empathy, and whether it gives the customer the right next step.
- This text should be friendly and suitable to send externally.

2. admin_score
- Grade ONLY the internal admin/technician steps.
- Focus on diagnostic usefulness, actionability, branching logic, escalation guidance, and source-grounded troubleshooting.
- This text is internal and does NOT need a friendly customer tone.
- Judge it against the ticket context and expected response type, not against customer-facing writing style.

Ticket Summary:
{summary}

Ticket Description:
{description}

Expected Response Type:
{expected_response_type or "unknown"}

Ground Truth Agent Response:
{actual_response}

AI Draft Customer Response:
{customer_response}

AI Draft Admin Steps:
{admin_steps or "<empty>"}
"""
    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        from utils.gemini import parse_json_response

        result = parse_json_response(response.text)
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            result = {}

        customer_score = _normalize_judge_score(result.get("customer_score"))
        admin_score = _normalize_judge_score(result.get("admin_score"))

        scores = [score for score in (customer_score, admin_score) if score is not None]
        if not scores:
            return 0.0
        return float(np.mean(scores))
    except Exception as e:
        log.error("LLM evaluation failed: %s", e)
        return 0.0


def _get_ticket_context_from_db(ticket_key: str) -> tuple[str, str] | None:
    with get_db_conn() as conn:
        ticket = conn.execute(
            "SELECT summary, description FROM tickets WHERE ticket_key = ?",
            (ticket_key,),
        ).fetchone()
    if ticket:
        return ticket["summary"], ticket["description"] or ""
    return None


def _lookup_matches_for_ticket(summary: str, description: str) -> dict:
    from .lookup import lookup

    matches = lookup(summary)
    if not matches.get("found") and description:
        words = " ".join(description.split()[:20])
        matches = lookup(words)
    return matches


def _evaluate_variant_row(
    row: dict,
    variant: str,
    fold_idx: int,
    *,
    timed_draft_fn: Callable[[str, str, str, dict], tuple[dict | None, float]] | None = None,
    draft_fn: Callable[[str, str, str, dict], dict | None] | None = None,
    summary: str | None = None,
    description: str | None = None,
    matches: dict | None = None,
    few_shot_block: str | None = None,
) -> tuple[int, str, float | None, bool | None, int | None, float | None]:
    """Evaluate one corpus row for a variant; returns fold metrics tuple."""
    if summary is None or description is None or matches is None:
        ctx = _get_ticket_context_from_db(row["ticket_key"])
        if not ctx:
            log.warning("Skipping %s — ticket context unavailable", row["ticket_key"])
            return fold_idx, variant, None, None, None, None

        summary, description = ctx
        matches = _lookup_matches_for_ticket(summary, description)
        if not matches.get("found"):
            log.warning("Skipping %s — lookup returned no matches", row["ticket_key"])
            return fold_idx, variant, None, None, None, None

    latency = 0.0
    if timed_draft_fn is not None:
        try:
            draft, latency = timed_draft_fn(
                variant, summary, description, matches, few_shot_block=few_shot_block
            )
        except TypeError:
            draft, latency = timed_draft_fn(variant, summary, description, matches)
    else:
        if draft_fn is None:
            draft = None
        else:
            try:
                draft = draft_fn(
                    variant,
                    summary,
                    description,
                    matches,
                    few_shot_block=few_shot_block,
                )
            except TypeError:
                draft = draft_fn(variant, summary, description, matches)

    if not draft:
        return fold_idx, variant, None, None, None, latency or None

    judge_score = evaluate_draft_with_llm(
        draft.get("customer_response", ""),
        draft.get("admin_steps"),
        row["actual_response"],
        summary=summary,
        description=description,
        expected_response_type=row.get("response_type"),
    )
    rt_hit = draft.get("response_type") == row["response_type"]
    draft_len = len(draft.get("customer_response", "") or "")
    return fold_idx, variant, judge_score, rt_hit, draft_len, latency or None


def evaluate_prompt_variants(
    corpus: list[dict] | None = None,
    *,
    variants: tuple[str, ...] = PROMPT_VARIANTS,
    n_splits: int = CV_N_SPLITS,
    random_state: int = CV_RANDOM_STATE,
    draft_fn: Callable[[str, str, str, dict], dict | None] | None = None,
    timed_draft_fn: Callable[[str, str, str, dict], tuple[dict | None, float]] | None = None,
    workers: int = 1,
    checkpoint_file: str | None = None,
) -> dict[str, PromptVariantCVResult]:
    """Run stratified k-fold CV comparing prompt variants against ground truth."""
    if corpus is None:
        corpus = load_ground_truth_corpus()
    if not corpus:
        raise ValueError("Ground truth corpus is empty")

    folds = build_stratified_folds(
        corpus, n_splits=n_splits, random_state=random_state
    )
    if timed_draft_fn is None and draft_fn is None:
        if workers > 1:
            timed_draft_fn = _draft_response_with_variant_timed
        else:
            draft_fn = _draft_response_with_variant

    import json

    per_variant_folds: dict[str, list[PromptVariantFoldMetrics]] = {
        variant: [
            PromptVariantFoldMetrics(fold=fold_idx) for fold_idx in range(len(folds))
        ]
        for variant in variants
    }

    checkpoint_file = checkpoint_file or _default_checkpoint_file(
        variants=variants,
        n_splits=n_splits,
        random_state=random_state,
    )
    checkpoint_path = Path(checkpoint_file)
    metadata = _checkpoint_metadata(
        variants=variants,
        n_splits=n_splits,
        random_state=random_state,
    )
    completed_variants: set[tuple[int, int, str]] = set()

    if checkpoint_path.exists():
        log.info("Loading checkpoint from %s", checkpoint_path)
        with checkpoint_path.open("r") as f:
            first_record: dict[str, object] | None = None
            for line in f:
                if not line.strip():
                    continue
                first_record = json.loads(line)
                break

        if first_record != metadata:
            raise ValueError(
                "Checkpoint file is incompatible with this experiment configuration. "
                f"Delete or rename '{checkpoint_path}' and rerun."
            )

        with checkpoint_path.open("r") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except Exception as exc:
                    log.warning("Failed to parse checkpoint line %d: %s", line_number, exc)
                    continue
                if record.get("record_type") == "meta":
                    continue
                if record.get("record_type") != "result":
                    log.warning("Skipping unknown checkpoint record at line %d", line_number)
                    continue

                feedback_row_id = record["feedback_row_id"]
                for res in record["results"]:
                    fold_idx, variant, judge_score, rt_hit, draft_len, latency = res
                    variant_key = (feedback_row_id, fold_idx, variant)
                    if variant_key in completed_variants:
                        continue
                    completed_variants.add(variant_key)
                    if judge_score is None:
                        continue
                    fold_metrics = per_variant_folds[variant][fold_idx]
                    fold_metrics.judge_scores.append(judge_score)
                    if rt_hit is not None:
                        fold_metrics.response_type_hits.append(rt_hit)
                    if draft_len is not None:
                        fold_metrics.draft_lengths.append(draft_len)
                    if latency is not None:
                        fold_metrics.latencies.append(latency)
        log.info("Resumed %d completed variant tasks from checkpoint", len(completed_variants))
    else:
        with checkpoint_path.open("w") as f:
            f.write(json.dumps(metadata) + "\n")

    tasks: list[tuple[dict, str, int]] = []
    for fold_idx, (_train_idx, test_idx) in enumerate(folds):
        for idx in test_idx:
            row = corpus[idx]
            pending_variants = [
                variant
                for variant in variants
                if (row["feedback_row_id"], fold_idx, variant) not in completed_variants
            ]
            for variant in pending_variants:
                tasks.append((row, variant, fold_idx))

    def _run_task(
        task: tuple[dict, str, int]
    ) -> tuple[
        int,
        str,
        int,
        list[tuple[int, str, float | None, bool | None, int | None, float | None]],
    ]:
        row, variant, fold_idx = task
        ticket_key = row["ticket_key"]
        feedback_row_id = row["feedback_row_id"]
        ctx = _get_ticket_context_from_db(ticket_key)
        if not ctx:
            log.warning("Skipping %s — ticket context unavailable", ticket_key)
            return feedback_row_id, ticket_key, fold_idx, [(fold_idx, variant, None, None, None, None)]

        summary, description = ctx
        matches = _lookup_matches_for_ticket(summary, description)
        if not matches.get("found"):
            log.warning("Skipping %s — lookup returned no matches", ticket_key)
            return feedback_row_id, ticket_key, fold_idx, [(fold_idx, variant, None, None, None, None)]

        ticket_text = f"{summary}\n\n{description or ''}".strip()
        few_shot_block = _build_few_shot_block(ticket_text)

        result = _evaluate_variant_row(
            row,
            variant,
            fold_idx,
            timed_draft_fn=timed_draft_fn,
            draft_fn=draft_fn,
            summary=summary,
            description=description,
            matches=matches,
            few_shot_block=few_shot_block,
        )
        return feedback_row_id, ticket_key, fold_idx, [result]

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_task, task) for task in tasks]
            with checkpoint_path.open("a") as f_out:
                for future in as_completed(futures):
                    try:
                        feedback_row_id, ticket_key, fold_idx, results_list = future.result()
                        for _f_idx, variant, judge_score, rt_hit, draft_len, latency in results_list:
                            if judge_score is None:
                                continue
                            fold_metrics = per_variant_folds[variant][_f_idx]
                            fold_metrics.judge_scores.append(judge_score)
                            if rt_hit is not None:
                                fold_metrics.response_type_hits.append(rt_hit)
                            if draft_len is not None:
                                fold_metrics.draft_lengths.append(draft_len)
                            if latency is not None:
                                fold_metrics.latencies.append(latency)
                        
                        record = {
                            "record_type": "result",
                            "feedback_row_id": feedback_row_id,
                            "ticket_key": ticket_key,
                            "fold_idx": fold_idx,
                            "results": results_list,
                        }
                        f_out.write(json.dumps(record) + "\n")
                        f_out.flush()
                    except Exception as e:
                        log.error("Task failed: %s", e)
    else:
        with checkpoint_path.open("a") as f_out:
            for task in tasks:
                feedback_row_id, ticket_key, fold_idx, results_list = _run_task(task)
                for _f_idx, variant, judge_score, rt_hit, draft_len, latency in results_list:
                    if judge_score is None:
                        continue
                    fold_metrics = per_variant_folds[variant][_f_idx]
                    fold_metrics.judge_scores.append(judge_score)
                    if rt_hit is not None:
                        fold_metrics.response_type_hits.append(rt_hit)
                    if draft_len is not None:
                        fold_metrics.draft_lengths.append(draft_len)
                    if latency is not None:
                        fold_metrics.latencies.append(latency)
                
                record = {
                    "record_type": "result",
                    "feedback_row_id": feedback_row_id,
                    "ticket_key": ticket_key,
                    "fold_idx": fold_idx,
                    "results": results_list,
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

    results: dict[str, PromptVariantCVResult] = {}
    baseline_scores: list[float] | None = None

    for variant in variants:
        fold_means: list[float] = []
        fold_stds: list[float] = []
        all_scores: list[float] = []
        all_hits: list[bool] = []
        all_lengths: list[int] = []
        all_latencies: list[float] = []

        for fold_metrics in per_variant_folds[variant]:
            if fold_metrics.judge_scores:
                fold_means.append(float(np.mean(fold_metrics.judge_scores)))
                fold_stds.append(float(np.std(fold_metrics.judge_scores)))
            else:
                fold_means.append(float("nan"))
                fold_stds.append(float("nan"))
            all_scores.extend(fold_metrics.judge_scores)
            all_hits.extend(fold_metrics.response_type_hits)
            all_lengths.extend(fold_metrics.draft_lengths)
            all_latencies.extend(fold_metrics.latencies)

        if variant == PROMPT_VARIANT_BASELINE:
            baseline_scores = all_scores

        mean_score = float(np.mean(all_scores)) if all_scores else float("nan")
        std_score = float(np.std(all_scores)) if all_scores else float("nan")
        rt_acc = float(np.mean(all_hits)) if all_hits else float("nan")

        p_value: float | None = None
        if (
            variant != PROMPT_VARIANT_BASELINE
            and baseline_scores
            and all_scores
            and len(baseline_scores) > 1
            and len(all_scores) > 1
        ):
            _, p_value = ttest_ind(all_scores, baseline_scores, equal_var=False)

        results[variant] = PromptVariantCVResult(
            variant=variant,
            mean_judge_score=mean_score,
            std_judge_score=std_score,
            fold_means=fold_means,
            fold_stds=fold_stds,
            p_value_vs_baseline=p_value,
            n_samples=len(all_scores),
            response_type_accuracy=rt_acc,
            draft_length_p50=_percentile(all_lengths, 50),
            draft_length_p95=_percentile(all_lengths, 95),
            latency_p95=_percentile(all_latencies, 95),
        )

    return results


def select_winning_variant(
    results: dict[str, PromptVariantCVResult],
    *,
    alpha: float = CV_SIGNIFICANCE_ALPHA,
) -> str:
    """Pick the best variant that significantly beats baseline, else baseline."""
    baseline = results.get(PROMPT_VARIANT_BASELINE)
    if not baseline or np.isnan(baseline.mean_judge_score):
        return PROMPT_VARIANT_BASELINE

    winner = PROMPT_VARIANT_BASELINE
    best_mean = baseline.mean_judge_score

    for variant in PROMPT_VARIANTS:
        if variant == PROMPT_VARIANT_BASELINE:
            continue
        result = results.get(variant)
        if not result or np.isnan(result.mean_judge_score):
            continue
        if (
            result.p_value_vs_baseline is not None
            and result.p_value_vs_baseline < alpha
            and result.mean_judge_score > baseline.mean_judge_score
            and result.mean_judge_score > best_mean
        ):
            winner = variant
            best_mean = result.mean_judge_score

    return winner


def format_prompt_tuning_results_comment(
    results: dict[str, PromptVariantCVResult],
    *,
    winner: str,
    evaluated_at: str,
) -> str:
    """Format the module-level results table comment block."""
    lines = [
        "Prompt tuning results (5-fold stratified CV on ai_draft_feedback ground truth):",
        "==============================================================================",
        "Variant designs:",
        "  A baseline           — original ticket + reference material prompt",
        "  B structured_context — XML tags for KB/FAQ/tickets/docs + ticket",
        "  C chain_of_thought   — classify ticket type/resolution pattern before drafting",
        "  D lean_context       — top-3 hits per source (vs top-5 baseline)",
        "  E xml_plus_cot       — XML tags + chain of thought reasoning",
        "  F lean_plus_cot      — top-3 hits + chain of thought reasoning",
        "  G kitchen_sink       — XML tags + top-3 hits + chain of thought reasoning",
        "",
        "variant            | mean ± std judge | p vs baseline | n    | rt_acc | len p50/p95 | lat p95",
        "-------------------|------------------|---------------|------|--------|-------------|--------",
    ]

    for variant in PROMPT_VARIANTS:
        result = results.get(variant)
        if not result:
            lines.append(
                f"{variant:<18} | N/A ± N/A        | N/A           | 0    | N/A    | N/A/N/A     | N/A"
            )
            continue

        mean_str = (
            f"{result.mean_judge_score:.3f}"
            if not np.isnan(result.mean_judge_score)
            else "N/A"
        )
        std_str = (
            f"{result.std_judge_score:.3f}"
            if not np.isnan(result.std_judge_score)
            else "N/A"
        )
        p_str = (
            f"{result.p_value_vs_baseline:.4f}"
            if result.p_value_vs_baseline is not None
            else "—"
            if variant == PROMPT_VARIANT_BASELINE
            else "N/A"
        )
        rt_str = (
            f"{result.response_type_accuracy:.3f}"
            if not np.isnan(result.response_type_accuracy)
            else "N/A"
        )
        len_str = (
            f"{result.draft_length_p50:.0f}/{result.draft_length_p95:.0f}"
            if not np.isnan(result.draft_length_p50)
            else "N/A/N/A"
        )
        lat_str = (
            f"{result.latency_p95:.2f}s"
            if not np.isnan(result.latency_p95)
            else "N/A"
        )
        lines.append(
            f"{variant:<18} | {mean_str} ± {std_str:<6} | {p_str:<13} | "
            f"{result.n_samples:<4} | {rt_str:<6} | {len_str:<11} | {lat_str}"
        )

    if winner == PROMPT_VARIANT_BASELINE:
        reason = "no variant beat baseline at p < 0.05"
    else:
        reason = f"{winner} beat baseline at p < 0.05"
    lines.extend(
        [
            "",
            f"Winner: {winner} ({reason})",
            f"Evaluated: {evaluated_at}; folds={CV_N_SPLITS}; "
            f"stratify=response_type; random_state={CV_RANDOM_STATE}",
            "Welch's t-test (equal_var=False) vs baseline; alpha=0.05 required for promotion.",
            "Fold means ± std:",
        ]
    )
    for variant in PROMPT_VARIANTS:
        result = results.get(variant)
        if not result:
            continue
        fold_summary = ", ".join(
            f"{m:.3f}±{s:.3f}"
            for m, s in zip(result.fold_means, result.fold_stds)
            if not np.isnan(m)
        )
        lines.append(f"  {variant}: [{fold_summary}]")
    lines.append("==============================================================================")
    return "\n".join(lines)


def apply_prompt_tuning_results_to_module(
    results: dict[str, PromptVariantCVResult],
    *,
    winner: str,
    evaluated_at: str,
    module_path: str | None = None,
) -> None:
    """Update drafting.py docstring table and WINNING_PROMPT_VARIANT."""
    from pathlib import Path

    path = Path(module_path or Path(__file__))
    text = path.read_text()
    comment_body = format_prompt_tuning_results_comment(
        results, winner=winner, evaluated_at=evaluated_at
    )
    docstring = f'"""LLM draft construction — Gemini prompt, response classification, ticket context.\n\n{comment_body}\n"""'
    text = re.sub(
        r'"""LLM draft construction — Gemini prompt, response classification, ticket context\..*?"""',
        docstring,
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"^WINNING_PROMPT_VARIANT = .*$",
        f"WINNING_PROMPT_VARIANT = {winner!r}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    path.write_text(text)
    set_winning_prompt_variant(winner)


def run_prompt_variant_cv(
    *,
    variants: tuple[str, ...] = PROMPT_VARIANTS,
    draft_fn: Callable[[str, str, str, dict], dict | None] | None = None,
    timed_draft_fn: Callable[[str, str, str, dict], tuple[dict | None, float]] | None = None,
    workers: int = 1,
    apply_winner: bool = True,
    evaluated_at: str | None = None,
) -> tuple[str, dict[str, PromptVariantCVResult]]:
    """Run CV, optionally deploy winner, and return (winner, results)."""
    from datetime import date

    corpus = load_ground_truth_corpus()
    if not corpus:
        raise ValueError("Ground truth corpus is empty")

    results = evaluate_prompt_variants(
        corpus,
        variants=variants,
        draft_fn=draft_fn,
        timed_draft_fn=timed_draft_fn,
        workers=workers,
    )
    winner = select_winning_variant(results)
    if apply_winner:
        apply_prompt_tuning_results_to_module(
            results,
            winner=winner,
            evaluated_at=evaluated_at or date.today().isoformat(),
        )
    return winner, results


def set_winning_prompt_variant(variant: str) -> None:
    """Set deployed prompt variant used by live drafting."""
    global WINNING_PROMPT_VARIANT
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unsupported prompt variant: {variant}")
    WINNING_PROMPT_VARIANT = variant


def _get_ticket_context(ticket_key: str) -> tuple[str, str] | None:
    """Fetch ticket summary and description from DB or Jira API.

    Returns (summary, description) or None if unavailable.
    """
    with get_db_conn() as conn:
        ticket = conn.execute(
            "SELECT summary, description FROM tickets WHERE ticket_key = ?",
            (ticket_key,),
        ).fetchone()

    if ticket:
        return ticket["summary"], ticket["description"] or ""

    # Not in DB — fetch from JSM API
    try:
        base = get_cloud_base_url("jsm")
        resp = _jira_request(
            "get",
            f"{base}/rest/servicedeskapi/request/{ticket_key}",
            "jsm",
            params={"expand": "requestFieldValues"},
        )
        if resp.status_code != 200:
            log.warning("Could not fetch %s from JSM: %d", ticket_key, resp.status_code)
            return None

        data = resp.json()
        summary = data.get("summary", "")
        desc = ""
        for field in data.get("requestFieldValues", []):
            if field.get("fieldId") == "description":
                val = field.get("value", "")
                if isinstance(val, dict):
                    from ingest.tickets import _extract_adf_text
                    val = _extract_adf_text(val)
                desc = str(val)
                break
        return summary, desc
    except Exception as e:
        log.warning("Could not fetch %s: %s", ticket_key, e)
        return None


def _get_latest_customer_reply(ticket_key: str) -> str | None:
    """Fetch the most recent public customer comment on a ticket.

    Walks comments in reverse to find the latest public comment
    that is NOT from an agent (i.e., not internal). Returns the
    text or None.
    """
    from ingest.tickets import _extract_adf_text

    try:
        base = get_cloud_base_url("jsm")

        # Get reporter from DB
        reporter_id = None
        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT reporter_id FROM tickets WHERE ticket_key = ?", (ticket_key,)
            ).fetchone()
            if row:
                reporter_id = row["reporter_id"]

        resp = _jira_request(
            "get",
            f"{base}/rest/servicedeskapi/request/{ticket_key}/comment",
            "jsm",
            params={"limit": 50},
        )
        if resp.status_code != 200:
            return None

        comments = resp.json().get("values", [])
        for comment in reversed(comments):
            if not comment.get("public", True):
                continue  # skip internal comments

            author_id = comment.get("author", {}).get("accountId", "")
            if reporter_id and author_id != reporter_id:
                continue  # skip agent public comments, want customer reply

            body = comment.get("body", "")
            text = _extract_adf_text(body) if isinstance(body, dict) else str(body)
            if text.strip():
                return text.strip()

        return None
    except Exception as e:
        log.debug("Could not fetch customer reply for %s: %s", ticket_key, e)
        return None


def _lookup_and_draft(ticket_key: str, summary: str, desc: str, force: bool = False) -> bool:
    """Shared logic: FAQ lookup -> Gemini draft -> post internal comment.

    Returns True if a draft was posted. Skips if a pending draft already
    exists unless force=True (e.g., triggered via /ai-lookup).
    """
    from .gates import _has_pending_draft
    from .jira_comments import _post_draft_comment, _post_self_check_and_decide
    from .feedback import _store_draft_record
    from .lookup import lookup

    if not force and _has_pending_draft(ticket_key):
        log.info("Ticket %s already has a pending AI draft — skipping", ticket_key)
        return False

    matches = lookup(summary)
    if not matches.get("found"):
        if desc:
            words = " ".join(desc.split()[:20])
            matches = lookup(words)

    from .templates import try_template_draft

    draft = try_template_draft(ticket_key, summary, desc, matches)
    if draft:
        log.info("Using template-fill mode for %s", ticket_key)
    elif matches.get("found"):
        draft = _draft_response(summary, desc, matches)
    else:
        log.info("No matches found for %s and no template match — skipping auto-response", ticket_key)
        return False
    if not draft:
        total_matches = sum(len(v) for v in matches.values() if isinstance(v, list))
        log.warning(
            "_draft_response returned None for %s despite %d matches — Gemini returned nothing",
            ticket_key,
            total_matches,
        )
        return False

    passed, reason = _post_self_check_and_decide(summary, desc, draft)
    if not passed:
        log.info("Draft for %s failed self-check: %s — not posting", ticket_key, reason)
        return False
    log.debug("Self-check passed for %s: %s", ticket_key, reason)

    log.info("Drafted %s response for %s", draft["response_type"], ticket_key)

    comment_id = _post_draft_comment(ticket_key, draft)
    if not comment_id:
        log.warning(
            "Auto-draft sweep: %s — _post_draft_comment returned no comment_id"
            " (HTTP error or missing id in response) — draft record not stored",
            ticket_key,
        )
        return False
    try:
        _store_draft_record(ticket_key, comment_id, draft)
    except Exception as store_err:
        log.error(
            "Auto-draft sweep: %s — comment posted (id=%s) but _store_draft_record failed: %s"
            " — next sweep will re-draft this ticket",
            ticket_key,
            comment_id,
            store_err,
        )
    return True
