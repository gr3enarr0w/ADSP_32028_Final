"""Configuration management — loads .env and exposes settings."""

import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

log = logging.getLogger(__name__)

# ── Cloud (Atlassian Cloud — post-migration, sole instance) ──
CLOUD_URL = os.getenv("JSM_CLOUD_URL", "").rstrip("/")

# ── Cloud OAuth 2LO — 3-token product-based design ──
# Each token is self-contained (read + write) for its product family.
# All 3 credentials must be on the same service account for consistent project access.
JSM_CLIENT_ID = os.getenv("JSM_CLIENT_ID", "")
JSM_CLIENT_SECRET = os.getenv("JSM_CLIENT_SECRET", "")
JIRA_CLIENT_ID = os.getenv("JIRA_CLIENT_ID", "")
JIRA_CLIENT_SECRET = os.getenv("JIRA_CLIENT_SECRET", "")
CONFLUENCE_CLIENT_ID = os.getenv("CONFLUENCE_CLIENT_ID", "")
CONFLUENCE_CLIENT_SECRET = os.getenv("CONFLUENCE_CLIENT_SECRET", "")

# Legacy aliases for any code not yet migrated
ATLASSIAN_OAUTH_CLIENT_ID = JSM_CLIENT_ID
ATLASSIAN_OAUTH_CLIENT_SECRET = JSM_CLIENT_SECRET
JIRA_WRITE_CLIENT_ID = JIRA_CLIENT_ID
JIRA_WRITE_CLIENT_SECRET = JIRA_CLIENT_SECRET

# ── Projects and defaults ──
PROJECT_KEYS = [k.strip() for k in os.getenv("PROJECT_KEYS", "<PROJECT_KEY>").split(",") if k.strip()]
DEFAULT_AFFECT_VERSION = os.getenv("JSM_AFFECT_VERSION", "")

# ── Gemini AI (Vertex AI via service account) ──
_sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
GOOGLE_SERVICE_ACCOUNT_JSON = str(
    Path(_sa_path) if Path(_sa_path).is_absolute() else _PROJECT_ROOT / _sa_path
)
GEMINI_PROJECT = os.getenv("GEMINI_PROJECT", "your-gcp-project")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "global")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_MODEL_CLASSIFICATION = os.getenv("GEMINI_MODEL_CLASSIFICATION", GEMINI_MODEL)
GEMINI_MODEL_GENERATION = os.getenv("GEMINI_MODEL_GENERATION", GEMINI_MODEL)
GEMINI_MODEL_ANALYSIS = os.getenv("GEMINI_MODEL_ANALYSIS", GEMINI_MODEL)

# ── Confluence ──
CONFLUENCE_KB_SPACE = os.getenv("CONFLUENCE_KB_SPACE", "HUB")
CONFLUENCE_PARENT_PAGE_ID = os.getenv("CONFLUENCE_PARENT_PAGE_ID")

# ── Google Sheets ──
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

# ── Slack (channels) ──
SLACK_CHANNELS = [c.strip() for c in os.getenv("SLACK_CHANNELS", "").split(",") if c.strip()]

# ── Link traversal — discovers projects from ticket links if env not set ──
LINK_FOLLOW_PROJECTS = [p.strip() for p in os.getenv("LINK_FOLLOW_PROJECTS", "").split(",") if p.strip()]

# ── FAQ Service ──
FAQ_SOURCE_DOC_IDS = [d.strip() for d in os.getenv("FAQ_SOURCE_DOC_IDS",
    "1Nj4CCLHvARHNnOwZLW5bWGw8uBCmYruMvboXMn2Td4I,"
    "1tpf_XJB1MiqkwFSKI02fa_KWkXFsarN6zVfj1-jl2g8"
).split(",") if d.strip()]
FAQ_OUTPUT_DOC_ID = os.getenv("FAQ_OUTPUT_DOC_ID", "1ny6lz3h_JCAdVoHU-Pb2czYq6Gbw0WOQwy9tK8kXhEo")
FAQ_SLIDES_ID = os.getenv("FAQ_SLIDES_ID", "1oyg-2MNzRUcmd_ibtmiEqNYBfJtwBY57tQxfGTugnSQ")
FAQ_CONFLUENCE_SPACES = [s.strip() for s in os.getenv("FAQ_CONFLUENCE_SPACES", "HUB,OMEGA").split(",") if s.strip()]
CLOUD_CUTOVER_DATE = os.getenv("CLOUD_CUTOVER_DATE", "2026-03-16")

# ── Deep Doc Research ──
DOC_SOURCE_DOMAINS = [d.strip() for d in os.getenv("DOC_SOURCE_DOMAINS",
    "support.atlassian.com,developer.atlassian.com").split(",") if d.strip()]
DOC_CACHE_DAYS = int(os.getenv("DOC_CACHE_DAYS", "7"))

# ── FAQ Documentation Sources (Google Sheets) ──
FAQ_SOURCE_SHEET_IDS = [s.strip() for s in os.getenv("FAQ_SOURCE_SHEET_IDS",
    "1an47jZKTBqgZ6luWVu_V6k8xuQkFRlJ4a_9H7O6TQ10"
).split(",") if s.strip()]

# ── Embedding model ──
# Default: gemini-embedding-001 (Vertex AI, 3072-dim).
# Fallback: sentence-transformers/all-MiniLM-L6-v2 (self-hosted, 384-dim).
# MiniLM achieved only F1=0.54 with threshold stuck at floor 0.30 on JSM
# tickets (ANTSE-554). Set EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# to revert to the old model.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))

# ── Dedup thresholds ──
DEDUP_JACCARD_THRESHOLD = float(os.getenv("DEDUP_JACCARD_THRESHOLD", "0.80"))
# Cosine threshold: env var overrides calibration result, which overrides 0.75 fallback.
# Re-calibrate via: python -m faq.threshold_calibration --save
_COSINE_THRESHOLD_FALLBACK = 0.75

def _load_calibrated_cosine_threshold() -> float:
    env_val = os.getenv("DEDUP_COSINE_THRESHOLD")
    if env_val is not None:
        return float(env_val)
    try:
        from faq.threshold_calibration import load_calibration_result
        result = load_calibration_result()
        if result is not None:
            if result.model_id != EMBEDDING_MODEL:
                log.warning(
                    "Calibration model '%s' differs from active model '%s' — "
                    "threshold %.2f may be unreliable; re-run: python -m faq.threshold_calibration --save",
                    result.model_id,
                    EMBEDDING_MODEL,
                    result.optimal_threshold,
                )
            return result.optimal_threshold
    except Exception as exc:
        log.warning(
            "Dedup calibration load failed (%s) — using fallback threshold %.2f",
            exc,
            _COSINE_THRESHOLD_FALLBACK,
        )
    return _COSINE_THRESHOLD_FALLBACK

DEDUP_COSINE_THRESHOLD = _load_calibrated_cosine_threshold()

# ── FAQ API auth ──
FAQ_API_TOKEN = os.getenv("FAQ_API_TOKEN", "")

# ── OpsGenie / JSM Operations (proactive alerting — ANTSE-315 / ANTSE-319) ──
OPSGENIE_API_KEY = os.getenv("OPSGENIE_API_KEY", "")
OPSGENIE_TOKEN = os.getenv("OPSGENIE_TOKEN", "")    # Atlassian service account API token (ATSTT...)
OPSGENIE_EMAIL = os.getenv("OPSGENIE_EMAIL", "")    # Service account email for Basic Auth

# ── Slack ──
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_XOXC_TOKEN = os.getenv("SLACK_XOXC_TOKEN", "")
SLACK_XOXD_TOKEN = os.getenv("SLACK_XOXD_TOKEN", "")

# ── Auto-draft gates ──
AGE_GATE_HOURS = int(os.getenv("AGE_GATE_HOURS", "4"))
DRAFT_SELF_CHECK = os.getenv("DRAFT_SELF_CHECK", "true").lower() in ("1", "true")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash-lite")
AUTO_DRAFT_NOISE_PATTERNS = [
    p.strip() for p in os.getenv("AUTO_DRAFT_NOISE_PATTERNS", "").split(",") if p.strip()
]

# ── Auto-responder ──
# Account IDs for instant webhook-triggered drafts on issue_created
AUTO_RESPOND_ASSIGNEES = [a.strip() for a in os.getenv("AUTO_RESPOND_ASSIGNEES",
    "5f7dd332837bb8006863d4d3"  # ceverson
).split(",") if a.strip()]
# When true, pipeline sweep drafts ALL open tickets regardless of assignee
AUTO_DRAFT_ALL = os.getenv("AUTO_DRAFT_ALL", "true").lower() in ("1", "true")

# ── Atlassian Cloud Documentation (fallback source) ──
ATLASSIAN_DOC_URLS = [u.strip() for u in os.getenv("ATLASSIAN_DOC_URLS",
    "https://support.atlassian.com/jira-software-cloud/resources/,"
    "https://support.atlassian.com/confluence-cloud/resources/,"
    "https://support.atlassian.com/jira-service-management-cloud/resources/,"
    "https://developer.atlassian.com/cloud/jira/platform/,"
    "https://developer.atlassian.com/cloud/jira/service-desk/,"
    "https://developer.atlassian.com/cloud/confluence/,"
    "https://developer.atlassian.com/cloud/admin/"
).split(",") if u.strip()]
ATLASSIAN_DOCS_REFRESH_DAYS = int(os.getenv("ATLASSIAN_DOCS_REFRESH_DAYS", "7"))

# ── GitHub Documentation Repos (doc files indexed for lookups) ──
GITHUB_DOC_REPOS = [r.strip() for r in os.getenv("GITHUB_DOC_REPOS",
    "atlassian/atlassian-mcp-server"
).split(",") if r.strip()]

# ── Cloud Terminology (DC → Cloud renames) ──
# Applied to all generated FAQ and auto-responder output.
# Each tuple: (pattern, replacement). Patterns are case-insensitive word-boundary matches.
CLOUD_TERMINOLOGY = [
    (r"\bprojects?\b", "space", "spaces"),  # (pattern, singular, plural)
]

# Pre-compiled for performance
_CLOUD_TERM_REPLACEMENTS: list[tuple[re.Pattern, str, str]] = []
for _pat, _singular, _plural in CLOUD_TERMINOLOGY:
    _CLOUD_TERM_REPLACEMENTS.append((
        re.compile(_pat + r"s", re.IGNORECASE),  # plural form first
        _plural,
    ))
    _CLOUD_TERM_REPLACEMENTS.append((
        re.compile(_pat.rstrip(r"s?\b") + r"\b", re.IGNORECASE),  # singular
        _singular,
    ))

# Simpler approach: explicit case-aware replacements
_TERM_MAP = {
    "Project": "Space",
    "project": "space",
    "Projects": "Spaces",
    "projects": "spaces",
    "PROJECT": "SPACE",
}
# Contexts where "project" should NOT be replaced (e.g., "Jira project key", "project-level")
_TERM_PRESERVE = re.compile(
    r"project[- ]?(?:key|id|lead|type|categor|template|scheme|avatar)",
    re.IGNORECASE,
)


def apply_cloud_terminology(text: str) -> str:
    """Replace DC terminology with Cloud equivalents in generated text.

    Handles case preservation and skips technical contexts where the
    original term is still correct (e.g., API field names, project keys).
    """
    if not text:
        return text

    # Protect preserved contexts with placeholders
    preserved = []
    def _protect(m):
        preserved.append(m.group(0))
        return f"\x00TERM{len(preserved) - 1}\x00"

    text = _TERM_PRESERVE.sub(_protect, text)

    # Apply replacements using word boundaries
    for old, new in _TERM_MAP.items():
        text = re.sub(rf"\b{old}\b", new, text)

    # Restore preserved terms
    for i, original in enumerate(preserved):
        text = text.replace(f"\x00TERM{i}\x00", original)

    return text


# ── Validation ──
_GOOGLE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{10,80}$")


def validate_google_id(doc_id: str) -> bool:
    """Validate a Google Doc/Slides/Sheets ID format."""
    if not doc_id or not _GOOGLE_ID_PATTERN.match(doc_id):
        raise ValueError(f"Invalid Google document ID format: {doc_id!r}")
    return True


def mask_id(value: str) -> str:
    """Mask a sensitive ID for logging (show first 4 and last 4 chars)."""
    if len(value) <= 12:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
