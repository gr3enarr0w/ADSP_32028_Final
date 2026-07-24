"""Usage tracker for search provider API calls.

Persists call counts to ~/.claude/research-tool-usage.json.
Monthly providers reset on month change. One-time providers (Exa) never reset.
"""

import json
import os
from datetime import datetime
from pathlib import Path

USAGE_FILE = Path.home() / ".claude" / "research-tool-usage.json"

PROVIDER_LIMITS = {
    "exa":    {"limit": 1400, "type": "one-time"},
    "tavily": {"limit": 1000, "type": "monthly"},
    "brave":  {"limit": 1000, "type": "monthly"},
    "linkup":   {"limit": 1000, "type": "monthly"},
    "newsdata": {"limit": 6000, "type": "monthly"},
    "gemini":   {"limit": None, "type": "unlimited"},
}

WARN_THRESHOLD = 0.80
SHIFT_THRESHOLD = 0.95


def _current_month() -> str:
    return datetime.now().strftime("%Y-%m")


def _default_data() -> dict:
    return {
        "month": _current_month(),
        "providers": {
            name: {"calls": 0, "limit": info["limit"], "type": info["type"]}
            for name, info in PROVIDER_LIMITS.items()
        },
    }


def load() -> dict:
    if not USAGE_FILE.exists():
        return _default_data()
    try:
        data = json.loads(USAGE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return _default_data()

    current = _current_month()
    if data.get("month") != current:
        for name, info in data.get("providers", {}).items():
            if info.get("type") == "monthly":
                info["calls"] = 0
        data["month"] = current

    for name, defaults in PROVIDER_LIMITS.items():
        if name not in data.get("providers", {}):
            data.setdefault("providers", {})[name] = {
                "calls": 0, "limit": defaults["limit"], "type": defaults["type"],
            }

    return data


def save(data: dict) -> None:
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(data, indent=2) + "\n")


def increment(provider: str) -> dict:
    data = load()
    prov = data["providers"].get(provider)
    if prov:
        prov["calls"] = prov.get("calls", 0) + 1
    save(data)
    return data


def get_usage_summary(data: dict | None = None) -> dict:
    if data is None:
        data = load()
    summary = {}
    warnings = []
    for name, info in data.get("providers", {}).items():
        calls = info.get("calls", 0)
        limit = info.get("limit")
        ptype = info.get("type", "monthly")

        if limit is None:
            summary[name] = f"{calls} (unlimited)"
            continue

        pct = calls / limit if limit > 0 else 0
        label = "one-time" if ptype == "one-time" else "monthly"
        summary[name] = f"{calls}/{limit} ({label})"

        if pct >= SHIFT_THRESHOLD:
            warnings.append(f"{name}: {calls}/{limit} — NEAR LIMIT, auto-shifting to fallback")
        elif pct >= WARN_THRESHOLD:
            warnings.append(f"{name}: {calls}/{limit} — approaching limit (80%+)")

    return {"usage": summary, "warnings": warnings if warnings else None}


def is_near_limit(provider: str, data: dict | None = None) -> bool:
    if data is None:
        data = load()
    info = data.get("providers", {}).get(provider, {})
    limit = info.get("limit")
    if limit is None:
        return False
    calls = info.get("calls", 0)
    return (calls / limit) >= SHIFT_THRESHOLD if limit > 0 else False


def get_remaining(provider: str, data: dict | None = None) -> int | None:
    if data is None:
        data = load()
    info = data.get("providers", {}).get(provider, {})
    limit = info.get("limit")
    if limit is None:
        return None
    return max(0, limit - info.get("calls", 0))
