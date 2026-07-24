"""Shared routing utilities used by analysis/router.py and plugins/responder/gates.py.

Kept in core/ to avoid a circular dependency between the analysis layer and the
responder plugin. Neither module should import directly from the other.
"""

INTENSITY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def is_access_related(category: str | None, issue_type: str | None) -> bool:
    """Return True when the ticket category or issue type is access-related."""
    if (category or "").strip().lower() == "access":
        return True
    return "access" in (issue_type or "").lower()
