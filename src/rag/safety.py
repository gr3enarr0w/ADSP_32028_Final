"""
safety.py — the safety layer the prompts already promise (Final deliverable).

`prompts/system_assistant.md` tells the assistant to "respect the domain
allowlist" and to refuse unsafe chemical advice, and
`prompts/answerer_critic.md` gives the Critic an `unsafe` verdict field. Until
now both were *only* instructions to a model: there was no allowlist anywhere
in the codebase, and nothing checked a hazardous combination deterministically.
A safety rule that exists solely inside a prompt is a safety rule that
disappears the moment the model has an off day.

This module makes both enforceable in code:

* `filter_web_results()` — drops live results whose host is not on the
  allowlist, and reports what it dropped and why. Applied in
  `rag.nodes.retriever_node` before anything reaches the Answerer, so an
  off-allowlist domain can never become a citation.
* `hazard_flags()` — a deterministic check for the chemical combinations the
  system prompt names, independent of the model's judgement. The Router's
  `safety_flags` and the Critic's `unsafe` verdict remain in play; this is the
  floor beneath them, not a replacement.

**On robots.txt / ToS:** we do not crawl. `web.search` reaches the web only
through vendor search APIs (Brave, Tavily, Exa, …) under their terms of
service, and we surface their result URLs to the user rather than fetching,
scraping, or storing page bodies. That is the honest scope of the
robots.txt-compliance requirement here: there is no fetch to gate. If a future
version fetches a result page directly, it must check that host's robots.txt
first — and this docstring is the reminder.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, Optional
from urllib.parse import urlparse

# Retailers, manufacturers and standards bodies plausible for household-cleaning
# product facts. Deliberately conservative: an unknown host is dropped, not
# kept, because a citation is an implicit endorsement of its source.
DEFAULT_ALLOWLIST = (
    "amazon.com", "walmart.com", "target.com", "costco.com", "samsclub.com",
    "homedepot.com", "lowes.com", "wayfair.com", "kroger.com", "staples.com",
    "webstaurantstore.com", "grove.co", "thrivemarket.com", "iherb.com",
    "weiman.com", "seventhgeneration.com", "mrsmeyers.com", "methodhome.com",
    "ecover.com", "biokleenhome.com", "betterlifeclean.com", "cloroxpro.com",
    "epa.gov", "nih.gov", "cdc.gov", "consumerreports.org",
)

# Chemical combinations the system prompt names as hazardous, plus the two
# other classic household ones. Each entry: (flag, [term groups]) — a flag
# fires when at least one term from every group is present.
_HAZARD_RULES: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("bleach_ammonia", (("bleach", "hypochlorite", "clorox"),
                        ("ammonia", "ammonium hydroxide", "windex"))),
    ("bleach_acid", (("bleach", "hypochlorite", "clorox"),
                     ("vinegar", "acetic acid", "muriatic", "hydrochloric",
                      "toilet bowl acid", "descaler"))),
    ("bleach_rubbing_alcohol", (("bleach", "hypochlorite"),
                                ("rubbing alcohol", "isopropyl"))),
    ("peroxide_vinegar", (("hydrogen peroxide",), ("vinegar", "acetic acid"))),
)


def get_allowlist() -> tuple[str, ...]:
    """The active domain allowlist.

    Overridable with `WEB_SEARCH_ALLOWLIST` (comma-separated hosts) so a demo
    or a different catalog vertical can widen it without a code change — the
    same env-driven convention as everything in `rag.config`. Setting it to
    `*` disables filtering entirely, which is logged loudly by
    `filter_web_results()` rather than happening quietly.
    """
    raw = os.environ.get("WEB_SEARCH_ALLOWLIST", "").strip()
    if not raw:
        return DEFAULT_ALLOWLIST
    return tuple(d.strip().lower() for d in raw.split(",") if d.strip())


def host_of(url: Optional[str]) -> str:
    """Registrable-ish host for `url`, lowercased, with a leading `www.` removed."""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


def is_allowed(url: Optional[str], allowlist: Optional[Iterable[str]] = None) -> bool:
    """True if `url`'s host is the allowlist entry or a subdomain of one.

    Subdomain matching is on a label boundary: `smile.amazon.com` matches
    `amazon.com`, but `amazon.com.phishing.example` does not.
    """
    allow = tuple(allowlist) if allowlist is not None else get_allowlist()
    if "*" in allow:
        return True
    host = host_of(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in allow)


def filter_web_results(results: list[dict],
                       allowlist: Optional[Iterable[str]] = None) -> tuple[list[dict], list[dict]]:
    """Split live search results into (allowed, blocked).

    Args:
        results: the `results` list from a `web.search` response.
        allowlist: override the active allowlist (mostly for tests).

    Returns:
        `(kept, blocked)`. Each blocked entry is the original dict plus a
        `blocked_reason` key, so the step log can show *what* was dropped
        rather than silently shrinking the result count.
    """
    allow = tuple(allowlist) if allowlist is not None else get_allowlist()
    kept, blocked = [], []
    for r in results or []:
        url = r.get("url")
        if is_allowed(url, allow):
            kept.append(r)
        else:
            reason = ("no url on result" if not url
                      else f"host {host_of(url)!r} not on the allowlist")
            blocked.append({**r, "blocked_reason": reason})
    return kept, blocked


def hazard_flags(*texts: Optional[str]) -> list[str]:
    """Deterministic hazardous-combination flags across the given text(s).

    Case-insensitive substring matching over the combined text. Returns flag
    names (e.g. `"bleach_ammonia"`) in rule order; an empty list means nothing
    fired. Intentionally simple and over-eager rather than clever: a false
    positive costs a caution sentence, a false negative costs a chlorine-gas
    recipe.
    """
    blob = " ".join(t for t in texts if t).lower()
    if not blob:
        return []
    flags = []
    for flag, groups in _HAZARD_RULES:
        if all(any(re.search(r"\b" + re.escape(term), blob) for term in group)
               for group in groups):
            flags.append(flag)
    return flags


HAZARD_CAUTION = (
    "I won't help combine those — mixing them releases toxic gas. Use one "
    "product at a time on a rinsed surface, with the room ventilated."
)


def caution_for(flags: Iterable[str]) -> str:
    """The spoken caution for a set of hazard flags ('' when there are none)."""
    return HAZARD_CAUTION if list(flags) else ""
