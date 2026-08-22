"""Single-provider research orchestrator with smart routing and escalation."""

import asyncio
import logging
import os
import re
from urllib.parse import urlparse

import usage_tracker
from providers.base import BaseProvider, SearchResult

log = logging.getLogger(__name__)

MODE_PRIMARY = {
    "quick":    "brave",
    "code":     "tavily",
    "academic": "gemini",
    "company":  "linkup",
    "people":   "tavily",
    "news":     "newsdata",
    "deep":     "gemini",
    "general":  "brave",
}

MODE_FALLBACK = {
    "quick":    "tavily",
    "code":     "exa",
    "academic": "exa",
    "company":  "exa",
    "people":   "exa",
    "news":     "brave",
    "deep":     "exa",
    "general":  "linkup",
}

QUESTION_WORDS = {"what", "who", "when", "where", "how", "why", "is", "are", "does", "did", "can", "will"}
CODE_KEYWORDS = {
    "code", "programming", "api", "sdk", "library", "framework", "github", "stackoverflow",
    "debug", "error", "react", "python", "javascript", "typescript", "rust", "go", "java",
    "how to use", "implement", "function", "module", "package", "npm", "pip", "import",
    "docker", "kubernetes", "terraform", "ansible", "nextjs", "django", "fastapi", "flask",
}
ACADEMIC_KEYWORDS = {"research", "paper", "study", "journal", "arxiv", "academic", "peer-reviewed", "thesis", "survey"}
NEWS_KEYWORDS = {"news", "latest", "today", "yesterday", "breaking", "announced", "launched", "released", "update"}
COMPANY_KEYWORDS = {"company", "startup", "founded", "ceo", "funding", "valuation", "revenue", "employees", "stock"}
PEOPLE_KEYWORDS = {"who is", "founder", "author", "researcher", "biography", "profile", "person"}
DEEP_KEYWORDS = {"deep dive", "thoroughly", "analyze", "investigate", "comprehensive", "in-depth"}

PROVIDER_KEY_ENV = {
    "exa": "EXA_API_KEY",
    "tavily": "TAVILY_API_KEY",
    "brave": "BRAVE_SEARCH_API_KEY",
    "linkup": "LINKUP_API_KEY",
    "newsdata": "NEWSDATA_API_KEY",
    "gemini": "GOOGLE_SERVICE_ACCOUNT_JSON",
}


def _provider_configured(name: str) -> bool:
    """Return whether the provider has the credential it requires."""
    key_name = PROVIDER_KEY_ENV.get(name)
    return key_name is None or bool(os.environ.get(key_name))


def _keyword_score(query: str, keywords: set[str]) -> int:
    score = 0
    for kw in keywords:
        if " " in kw:
            if kw in query:
                score += 1
        else:
            if re.search(rf"\b{re.escape(kw)}\b", query):
                score += 1
    return score


def classify_query(query: str) -> str:
    q = query.lower().strip()
    words = q.split()

    if re.match(r"^who is\b", q):
        return "people"
    if re.match(r"^what company\b", q) or re.match(r"^tell me about .+ company", q):
        return "company"

    for kw in DEEP_KEYWORDS:
        if kw in q:
            return "deep"

    if len(words) <= 8 and words[0] in QUESTION_WORDS:
        if _keyword_score(q, CODE_KEYWORDS) == 0 and _keyword_score(q, COMPANY_KEYWORDS) == 0 and _keyword_score(q, PEOPLE_KEYWORDS) == 0:
            return "quick"

    code_score = _keyword_score(q, CODE_KEYWORDS)
    academic_score = _keyword_score(q, ACADEMIC_KEYWORDS)
    news_score = _keyword_score(q, NEWS_KEYWORDS)
    company_score = _keyword_score(q, COMPANY_KEYWORDS)
    people_score = _keyword_score(q, PEOPLE_KEYWORDS)

    scores = {
        "code": code_score,
        "academic": academic_score,
        "news": news_score,
        "company": company_score,
        "people": people_score,
    }
    best_mode = max(scores, key=scores.get)
    if scores[best_mode] > 0:
        return best_mode

    if len(words) <= 8 and words[0] in QUESTION_WORDS:
        return "quick"

    return "general"


def _init_provider(name: str) -> BaseProvider | None:
    from providers import ALL_PROVIDERS
    cls = ALL_PROVIDERS.get(name)
    if cls is None:
        return None
    try:
        return cls()
    except Exception as e:
        log.warning("Failed to init %s: %s", name, e)
        return None


def _select_provider(mode: str, exclude: list[str] | None = None) -> tuple[str, str]:
    """Select best available provider for mode, respecting usage limits and exclusions."""
    exclude = set(exclude or [])
    usage_data = usage_tracker.load()

    primary_name = MODE_PRIMARY.get(mode, "tavily")
    fallback_name = MODE_FALLBACK.get(mode, "tavily")

    candidates = [primary_name, fallback_name, "gemini", "tavily", "brave", "linkup", "newsdata", "exa"]

    for name in candidates:
        if name in exclude:
            continue
        if not _provider_configured(name):
            continue
        if usage_tracker.is_near_limit(name, usage_data):
            continue
        return name, "primary" if name == primary_name else "fallback"

    remaining = [n for n in candidates if n not in exclude and _provider_configured(n)]
    if remaining:
        return remaining[0], "last-resort"
    return "tavily", "last-resort"


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    seen = {}
    for r in results:
        key = ""
        if r.url:
            parsed = urlparse(r.url)
            key = parsed.netloc + parsed.path.rstrip("/")
        key = key or r.title
        if not key:
            continue
        if key in seen:
            if r.score > seen[key].score or (len(r.snippet) > len(seen[key].snippet) and r.score >= seen[key].score):
                seen[key] = r
        else:
            seen[key] = r
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)


async def _call_provider(provider: BaseProvider, query: str, mode: str, max_results: int) -> list[SearchResult]:
    try:
        if mode == "deep":
            return await asyncio.wait_for(provider.deep_search(query, max_results), timeout=300)
        if mode == "code" and hasattr(provider, "code_search"):
            return await asyncio.wait_for(provider.code_search(query, max_results), timeout=20)
        if mode == "company" and hasattr(provider, "company_search"):
            return await asyncio.wait_for(provider.company_search(query, max_results), timeout=20)
        if mode == "people" and hasattr(provider, "people_search"):
            return await asyncio.wait_for(provider.people_search(query, max_results), timeout=20)
        if mode == "news" and hasattr(provider, "news_search"):
            return await asyncio.wait_for(provider.news_search(query, max_results), timeout=15)
        if mode == "academic" and hasattr(provider, "academic_search"):
            return await asyncio.wait_for(provider.academic_search(query, max_results), timeout=60)
        return await asyncio.wait_for(provider.search(query, max_results), timeout=15)
    except Exception as e:
        log.warning("%s failed: %s", provider.name, e)
        return []


async def search(query: str, mode: str = "auto", max_results: int = 10) -> dict:
    if mode == "auto":
        mode = classify_query(query)

    provider_name, selection_reason = _select_provider(mode)
    provider = _init_provider(provider_name)

    if provider is None:
        provider_name, _ = _select_provider(mode, exclude=[provider_name])
        provider = _init_provider(provider_name)
        if provider is None:
            return {"error": "No providers available", "mode": mode}

    results = await _call_provider(provider, query, mode, max_results)
    usage_tracker.increment(provider_name)

    answer = None
    for r in results:
        if r.raw and r.raw.get("answer"):
            answer = r.raw["answer"]
            break

    usage_summary = usage_tracker.get_usage_summary()
    can_escalate = len(results) < 3
    escalation_hint = None
    if can_escalate:
        fallback, _ = _select_provider(mode, exclude=[provider_name])
        escalation_hint = f"{len(results)} results found. Say 'escalate' to try {fallback} for more."

    return {
        "answer": answer,
        "sources": [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "provider": r.provider, "score": r.score}
            for r in results
        ],
        "provider_used": provider_name,
        "mode": mode,
        "usage": usage_summary["usage"],
        "warnings": usage_summary["warnings"],
        "can_escalate": can_escalate,
        "escalation_hint": escalation_hint,
    }


async def escalate(query: str, mode: str = "auto", exclude_providers: list[str] | None = None, max_results: int = 10) -> dict:
    if mode == "auto":
        mode = classify_query(query)

    exclude = exclude_providers or []
    provider_name, _ = _select_provider(mode, exclude=exclude)
    provider = _init_provider(provider_name)

    if provider is None or provider_name in exclude:
        return {"error": f"No additional providers available (tried: {exclude})", "mode": mode}

    results = await _call_provider(provider, query, mode, max_results)
    usage_tracker.increment(provider_name)

    answer = None
    for r in results:
        if r.raw and r.raw.get("answer"):
            answer = r.raw["answer"]
            break

    usage_summary = usage_tracker.get_usage_summary()

    return {
        "answer": answer,
        "sources": [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "provider": r.provider, "score": r.score}
            for r in results
        ],
        "provider_used": provider_name,
        "mode": mode,
        "previously_tried": exclude,
        "usage": usage_summary["usage"],
        "warnings": usage_summary["warnings"],
    }
