"""
Core utilities for Efficient RAG with Qdrant in Hermes.
Implements hierarchical parent-child, quantization-aware ingestion/retrieval,
contextual summaries, hybrid (dense + sparse RRF) search, cross-encoder
reranking, and metadata handling.
Storage-optimized by default (light mode).
"""

import os
import re
import json
import math
import uuid
import hashlib
import subprocess
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
    BinaryQuantization, ScalarQuantization,
    ScalarQuantizationConfig, BinaryQuantizationConfig,
    Prefetch, FusionQuery, Fusion, SparseVectorParams, SparseVector, Document
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

# Optional imports (graceful fallback)
try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError, FileNotDecryptedError, PyPdfError
except ImportError:
    PdfReader = None
    PdfReadError = FileNotDecryptedError = PyPdfError = Exception

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None


# Directory names excluded anywhere in a path during `--recursive` local
# ingestion (scripts/ingest.py). These are near-universal noise for a
# "personal knowledge" corpus — dependency trees, VCS internals, build
# artifacts, and caches — never real authored content worth chunking/
# embedding. Exposed as a CLI-overridable default (`--exclude-dirs`) rather
# than hardcoded, but this list is what a bare `--recursive` invocation uses
# so a first-time user pointing this skill at a real personal directory
# doesn't reproduce the same noise problem this default list was written to
# solve.
DEFAULT_EXCLUDE_DIRS = [
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "dist", "build", "target", ".cache",
    # Added after a real dry-run scan of ~/Documents found a full local
    # conda environment (~/Documents/Repositories/wip/.conda — Python
    # stdlib + site-packages, 1,517 .py files, ~140MB) sitting inside a
    # personal directory. Same noise category as .venv/venv: an installed
    # package library, not authored knowledge.
    ".conda",
]

# Directory-name SUFFIXES excluded anywhere in a path, matched independently
# of DEFAULT_EXCLUDE_DIRS (which is exact-name matching only). Python wheel/
# package metadata directories are named after the package+version (e.g.
# `sentence_transformers-2.2.2.dist-info/`), so they can't be captured by a
# fixed exclude-dir name list — a suffix match is required instead.
DEFAULT_EXCLUDE_DIR_SUFFIXES = [".dist-info", ".egg-info"]

# File extensions treated as real "knowledge" for `--recursive` local
# ingestion by default: prose/document formats plus plain-text source/config
# formats. Deliberately an ALLOW-list (not a deny-list) — this is what keeps
# noise extensions like `.pyc`/`.pyi`/`.so`/`.dat`, and binary/image/archive
# assets (`.png`, `.jpg`, `.zip`, `.dmg`, etc.), out without needing to name
# every possible noise extension individually. `.json`/large `.csv`/`.xlsx`
# data dumps are intentionally NOT blanket-included here (see
# DEFAULT_MAX_FILE_SIZE_MB) — those are handled by the size guard below
# rather than by extension, since a small config file and a multi-GB data
# export can share an extension.
DEFAULT_INCLUDE_EXTENSIONS = [
    # documents
    ".md", ".markdown", ".txt", ".pdf", ".docx", ".csv", ".xlsx", ".html",
    # source code / plain-text config
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt", ".cs",
    ".sql", ".yml", ".yaml",
]

# Files larger than this are skipped even if their extension is on the
# include list — guards against a "config file" extension (`.csv`, `.xlsx`,
# even `.md`/`.txt`) actually being a huge data dump that isn't meant to be
# chunked/embedded as prose. 20MB comfortably fits real documents/source
# files while excluding multi-hundred-MB+ exports.
DEFAULT_MAX_FILE_SIZE_MB = 20.0

# ---------------------------------------------------------------------------
# WORK-TOPIC EXCLUSIONS — SPECIFIC TO THIS USER, NOT A GENERIC SKILL DEFAULT.
# ---------------------------------------------------------------------------
# Unlike DEFAULT_EXCLUDE_DIRS above (generic noise — dependency trees, VCS
# internals, build artifacts — that ANY user of this skill would want
# excluded), the two lists below name THIS USER's own confirmed Example Organization /
# Third-Party Service work-topic repositories and file-name patterns under
# ~/Development, found via manual inspection on 2026-07-20 ahead of a real
# personal-corpus ingestion run (see docs/CHANGELOG.md's dated entry). They
# are deliberately kept OUT of DEFAULT_EXCLUDE_DIRS / DEFAULT_INCLUDE_EXTENSIONS
# so they are never silently applied for other callers/other directories
# (e.g. ~/Downloads, ~/Documents) or for other users of this skill — they
# are opt-in only, via scripts/ingest.py's `--work-topic-excludes` flag
# (which merges WORK_TOPIC_EXCLUDE_DIRS into --exclude-dirs and
# WORK_TOPIC_EXCLUDE_FILE_PATTERNS into --exclude-patterns).
WORK_TOPIC_EXCLUDE_DIRS = [
    # NOTE: a literal "secrets" directory-name guard used to live here, but
    # that made a pure security control (excluding secrets/ regardless of
    # work/personal content scope) silently ride along with the opt-in
    # --work-topic-excludes flag — dropping that flag (e.g. for a personal-
    # only ingestion run that still wants Example Organization/Third-Party Service docs) would have
    # also silently dropped the secrets guard. Fixed 2026-07-21: the
    # secrets/ directory-name guard now lives unconditionally in
    # `_is_excluded_dir()` below, independent of exclude_dirs/this list, so
    # it is always active regardless of --work-topic-excludes or any other
    # flag. See `_is_excluded_dir()`'s docstring.
    #
    # Confirmed Example Organization / Third-Party Service work-topic repositories under
    # ~/Development (manual inspection, 2026-07-20).
    "atlassian_cloud_cleanup",
    "atlassian_cloud_migration_user_mapping",
    "atlassian_cloud_uat_analytics",
    "custom_field_context_forge",
    "gen-atlassian",
    "jira-dbt",
    "jira_field_toolkit",
    "jsm-modeling",
    "ldap-employee-checker",
    "mcp-atlassian",
    "mcp-atlassian-admin",
    "mcp-org-analytics",
    "mcp-team-analytics",
    "redhat-mcp-server",
    "redhat-user-management",
    "registered-bot-and-service-accounts-assets",
    "uat-broken",
    "uat-compaction",
    "uat-editfix",
    "uat-hallucinate",
    "uat-mixed",
    "uat-test",
    "test-uat",
    "test-verify",
]

# Case-insensitive SUBSTRING patterns matched against a bare file NAME (not a
# directory name — these can't be captured by WORK_TOPIC_EXCLUDE_DIRS since
# they describe loose top-level files under ~/Development, e.g.
# `jira_export_2026.csv` or `atlassian_notes.md`), used by
# `_is_excluded_file_pattern()` / `iter_filtered_files(exclude_patterns=...)`.
# Same user-specific scope/opt-in-only rule as WORK_TOPIC_EXCLUDE_DIRS above.
WORK_TOPIC_EXCLUDE_FILE_PATTERNS = [
    "jira",
    "atlassian",
    "invalid_user",
    "uat_cloud",
    "umb_projects",
    "workflow_projects",
    "redhat_users",
    "jql",
]

# Case-insensitive SUBSTRING signal terms searched for in a file's EXTRACTED
# TEXT (not its path/filename — see WORK_TOPIC_EXCLUDE_DIRS/
# WORK_TOPIC_EXCLUDE_FILE_PATTERNS above for those) by `ingest()`'s
# `exclude_work_content` opt-in check. THIS USER's own confirmed Red
# Hat/Third-Party Service work-content signal list, not a generic default for other
# users of this skill — same user-specific scope/opt-in-only rule as the two
# lists above.
#
# Added after a real ~/Downloads personal-corpus ingestion run (2026-07-21,
# see docs/CHANGELOG.md's dated entry) got contaminated by work files whose
# NAMES gave no signal at all — e.g. `Unknown.pdf`, `Home - JIRA +
# Smartsheet-2.pdf`, `dat-user-fix-errors.csv`,
# `OMEGA-Bot accounts on Next Gen Document Service-160426-161120.pdf`, `Example Organization
# Issue Tracker (1).csv`, `Feb 2026 TCramer Cost Ctrs - TCram_ALL (1).csv` —
# so filename/path-based filtering (WORK_TOPIC_EXCLUDE_DIRS/
# WORK_TOPIC_EXCLUDE_FILE_PATTERNS) could never have caught them. Work
# documents almost always contain distinctive terms in their BODY even when
# the filename is generic, so this list is checked against extracted text
# instead.
WORK_CONTENT_SIGNAL_TERMS = [
    "redhat.com",
    "red hat",
    "atlassian",
    "jira",
    "confluence",
    "jql",
    "smartsheet",
    "cost ctr",
    "cost center",
]

# How many leading characters of a file's extracted text to scan for
# WORK_CONTENT_SIGNAL_TERMS — a simple performance guard so a huge extracted
# document (e.g. a multi-MB CSV/PDF dump) doesn't require a full-text scan
# just to decide whether to skip it. Work-identifying terms in real
# contaminated files (headers, titles, boilerplate, column names) reliably
# show up well within this prefix.
WORK_CONTENT_SCAN_PREFIX_CHARS = 20_000

# Standard Issue Tracker issue-key pattern (e.g. `JBEAP-33417`, `RHEL-180595`,
# `OSPRH-29836`, `ROX-33323`). Issue Tracker CSV/text exports (e.g. a Issue Tracker issue
# tracker export) are reliably full of these even when the export contains
# none of the WORK_CONTENT_SIGNAL_TERMS literal strings — a real gap found
# during verification: `Example Organization Issue Tracker (1).csv` never spells out
# "jira"/"red hat"/"atlassian" anywhere in its extracted text, but is packed
# with issue keys. Matched case-sensitively (Issue Tracker keys are conventionally
# uppercase) against the ORIGINAL-case text, not the lowercased copy used by
# the substring check above. A single match isn't enough signal on its own —
# plenty of legitimate personal content (product SKUs, invoice numbers,
# serial numbers) incidentally matches this shape — so
# WORK_CONTENT_JIRA_KEY_MIN_MATCHES distinct matches are required before this
# is treated as a work-content signal.
_JIRA_ISSUE_KEY_PATTERN = re.compile(r"\b[A-Z]{2,10}-\d{2,6}\b")
WORK_CONTENT_JIRA_KEY_MIN_MATCHES = 3


def _contains_work_content_signal(text: str) -> bool:
    """True if the first WORK_CONTENT_SCAN_PREFIX_CHARS of `text` contain
    EITHER any WORK_CONTENT_SIGNAL_TERMS substring (case-insensitively) OR at
    least WORK_CONTENT_JIRA_KEY_MIN_MATCHES distinct Issue Tracker-style issue-key
    matches (case-sensitively, on the original-case text). Used by
    `EfficientRAG.ingest()`'s opt-in `exclude_work_content` check — see that
    list's comment for why this operates on extracted text rather than
    filenames/paths."""
    prefix = text[:WORK_CONTENT_SCAN_PREFIX_CHARS]
    lowered = prefix.lower()
    if any(term in lowered for term in WORK_CONTENT_SIGNAL_TERMS):
        return True
    if len(_JIRA_ISSUE_KEY_PATTERN.findall(prefix)) >= WORK_CONTENT_JIRA_KEY_MIN_MATCHES:
        return True
    return False


# Payload `source_type` values (set via `extra_metadata` — see `ingest.py`'s
# `--crawl-url`/`--gdrive-query`/`--gdoc-id`/`--gsheet-id`/`--gslide-id`/
# `--confluence-dump-dir` call sites) that should be classified as "public"
# visibility. Only `--crawl-url` content (external, publicly-accessible
# websites fetched via `crawl_site()`/`crawl_confluence_space()` — Third-Party Service
# support/Document Service public docs, Adaptavist, eazyBI, Hermes Agent docs,
# Fivetran REST API docs, etc.) qualifies today. Everything else — local
# files ingested via `--path` (no `source_type` set at all), Google
# Workspace content (`source_type: "google_drive"`), and Document Service TEAM
# space dumps (`source_type: "confluence"`, ingested via
# `--confluence-dump-dir` from private/authenticated Document Service exports, NOT
# the public `crawl_confluence_space()` path) is "private". See
# docs/CHANGELOG.md's dated entry for the rule derivation (verified against
# actual stored payload values via the Qdrant scroll API, not assumed).
PUBLIC_VISIBILITY_SOURCE_TYPES = {"external_docs"}


def _classify_visibility(extra_metadata: Optional[Dict[str, Any]]) -> str:
    """Pure rule-based (no LLM) mapping from a point's `source_type` (as set
    in `extra_metadata` by the various `ingest.py` call sites) to a
    `"public"` / `"private"` `visibility` payload value. `source_type` is
    absent entirely for local-file ingestion (`--path`), which is
    unambiguously private — `.get()` on a possibly-None dict handles that
    case the same as any other non-public `source_type`."""
    source_type = (extra_metadata or {}).get("source_type")
    return "public" if source_type in PUBLIC_VISIBILITY_SOURCE_TYPES else "private"

# Regex matching `.env` / `.env.*` secret files (e.g. `.env.atlassian`,
# `.env.one-atlas-ochw`) — ALWAYS checked by `iter_filtered_files()`
# unconditionally, regardless of whether `exclude_patterns` is passed at all
# (see the fix note below and in `iter_filtered_files`'s main loop). This is
# a standing secrets guard, not something that should be possible to disable
# just by omitting `--exclude-patterns` entirely or passing it with
# unrelated terms and omitting ".env".
#
# FIX 2026-07-21: this used to be checked only inside
# `_is_excluded_file_pattern()`, which itself was only called when
# `exclude_patterns is not None` — meaning a real ingestion run that never
# passes `--exclude-patterns` at all (the common case for e.g. ~/Downloads,
# ~/Documents, ~/Development personal-corpus runs) got ZERO `.env` filtering,
# silently defeating the "always on" guard this comment already claimed to
# provide. `iter_filtered_files()` now checks `_ENV_FILE_PATTERN` directly,
# unconditionally, on every filename, independent of `exclude_patterns`.
_ENV_FILE_PATTERN = re.compile(r"^\.env(\..+)?$", re.IGNORECASE)


def _is_excluded_file_pattern(filename: str, patterns: List[str]) -> bool:
    """True if `filename` (a bare file name, no directory component)
    contains any of `patterns` as a case-insensitive substring.

    Note: this does NOT need to separately check `_ENV_FILE_PATTERN` — that
    guard is applied unconditionally by `iter_filtered_files()` itself before
    this function is ever consulted (see that function's fix note and the
    `_ENV_FILE_PATTERN` comment above), so it's always active even when this
    function isn't called at all (i.e. when `exclude_patterns is None`)."""
    lowered = filename.lower()
    return any(p.lower() in lowered for p in patterns)


def _is_git_repo(dirpath: str) -> bool:
    """True if `dirpath` is the root of a git repository (has a `.git`
    directory directly inside it). Does not attempt to handle a `.git` FILE
    (used by git worktrees/submodules) — real top-level ~/Development clones
    are ordinary repos with a `.git` directory."""
    return os.path.isdir(os.path.join(dirpath, ".git"))


def _has_git_authorship(repo_dir: str, email: str) -> bool:
    """True if `email` has authored at least one commit in the git repo at
    `repo_dir`, checked via `git log --author=<email> --oneline -1`.

    Fails CLOSED (returns False — "no authored commits", i.e. excluded) on
    ANY problem: git binary missing, `repo_dir` not actually a repo, a
    corrupted/shallow repo, a timeout, non-zero exit code, etc. This check
    exists specifically to keep never-contributed-to repos OUT of ingestion
    (see `iter_filtered_files`'s `require_git_authorship_email` param) — a
    version that failed OPEN on error would silently defeat that purpose by
    including exactly the repos this filter is meant to exclude whenever the
    check itself had a problem.
    """
    try:
        result = subprocess.run(
            ["git", "log", f"--author={email}", "--oneline", "-1"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _is_excluded_dir(dirname: str, exclude_dirs: List[str]) -> bool:
    """True if `dirname` (a single path component, not a full path) should be
    pruned from a directory walk — an exact match against `exclude_dirs`, a
    dist-info/egg-info-style metadata directory suffix match (see
    DEFAULT_EXCLUDE_DIR_SUFFIXES), a bare "site-packages" directory (a
    backstop in case a venv sits under a non-excluded parent name), or any
    directory name CONTAINING "venv" case-insensitively.

    The "venv" substring check exists because real-world virtual environment
    directories aren't reliably named exactly ".venv"/"venv" — a dry-run scan
    of ~/Development for this project found a 939MB, 12,082-.py-file
    `.pycaret_venv/` directory that an exact-name-only match would have
    missed entirely (same noise category as `.venv`/`venv`, just named
    differently by whatever tool created it).

    A bare `secrets` directory (any case) is ALWAYS pruned here as a standing
    security guard, independent of `exclude_dirs` entirely — it used to live
    only inside the opt-in `WORK_TOPIC_EXCLUDE_DIRS` list (merged in only via
    `--work-topic-excludes`), which meant a caller who legitimately wants
    --work-topic-excludes OFF (e.g. a personal corpus that should still
    include real Example Organization/Third-Party Service docs) would also silently lose secrets
    protection. Fixed 2026-07-21: checked unconditionally here instead.
    """
    if dirname.lower() == "secrets":
        return True
    if dirname in exclude_dirs:
        return True
    if dirname.lower() == "site-packages":
        return True
    if "venv" in dirname.lower():
        return True
    return any(dirname.endswith(suffix) for suffix in DEFAULT_EXCLUDE_DIR_SUFFIXES)


def iter_filtered_files(
    root: str,
    exclude_dirs: Optional[List[str]] = None,
    include_extensions: Optional[List[str]] = None,
    max_file_size_mb: Optional[float] = DEFAULT_MAX_FILE_SIZE_MB,
    exclude_patterns: Optional[List[str]] = None,
    require_git_authorship_email: Optional[str] = None,
):
    """Walk `root` yielding file paths (as `str`) eligible for `--recursive`
    local ingestion, applying the exclude-dir / include-extension / max-size
    filters shared by both the real ingest path and `--dry-run` reporting in
    scripts/ingest.py.

    Uses `os.walk` (NOT `Path.rglob`, which `scripts/ingest.py` used
    previously) specifically so excluded directories can be PRUNED from
    `dirnames` in place — this avoids ever descending into e.g.
    `node_modules/` or `.venv/` at all, rather than walking the whole
    subtree and filtering results afterward. On a large personal directory
    (the motivating case — `~/Development` can contain many multi-GB
    `node_modules`/`.venv` trees) that distinction is the difference between
    a fast walk and one that touches hundreds of thousands of irrelevant
    files.

    `exclude_dirs` / `include_extensions` default to DEFAULT_EXCLUDE_DIRS /
    DEFAULT_INCLUDE_EXTENSIONS when not provided (None) — NOT when falsy, so
    an explicit `[]` from a caller who really wants "no exclusions" /
    "everything" is honored rather than silently replaced by the default.

    `exclude_patterns`: optional list of case-insensitive substrings; any
    file whose bare name contains one is skipped (see
    `_is_excluded_file_pattern()`). `None` (the default) disables this
    substring-pattern filtering entirely.

    `.env`/`.env.*` files are a SEPARATE, always-on secrets guard applied
    unconditionally in the main loop below via `_ENV_FILE_PATTERN`,
    independent of `exclude_patterns` — it applies even when
    `exclude_patterns` is `None`. (Fixed 2026-07-21: this used to be nested
    inside the `exclude_patterns is not None` branch, which meant callers
    who never pass `--exclude-patterns` at all — e.g. ~/Downloads,
    ~/Documents, ~/Development personal-corpus runs — got no `.env`
    filtering whatsoever, despite the guard being documented as "always on."
    It is now a true unconditional check, matching that documentation.)

    `require_git_authorship_email`: optional email address. When set, for
    each IMMEDIATE (top-level) subdirectory of `root` that is itself a git
    repository (has a `.git` directory), that entire subdirectory is pruned
    from the walk unless `email` has authored at least one commit in it (see
    `_has_git_authorship()`). Immediate subdirectories of `root` that are NOT
    git repositories at all are completely unaffected by this check — it
    only applies to things that ARE git repos. Nested git repos further down
    the tree (rare — e.g. a repo-within-a-repo) are not independently
    re-checked; only the top-level directory identity matters here, matching
    "did I ever clone-and-touch this project" at the granularity a real
    ~/Development directory is organized at. `None` (the default) disables
    this check entirely — no behavior change for callers who don't pass it
    (e.g. ~/Downloads / ~/Documents, which aren't code repos in this sense).
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    if include_extensions is None:
        include_extensions = DEFAULT_INCLUDE_EXTENSIONS
    include_extensions = {e.lower() for e in include_extensions}

    # Precompute which top-level subdirectories of `root` are git repos the
    # given email has never authored a commit in — these are pruned the
    # moment os.walk reaches `root` itself (its first yielded dirpath),
    # before descending into any of them, exactly like the other exclude-dir
    # pruning below.
    top_level_authorship_prune = set()
    if require_git_authorship_email:
        try:
            top_children = os.listdir(root)
        except OSError:
            top_children = []
        for child in top_children:
            child_path = os.path.join(root, child)
            if os.path.isdir(child_path) and _is_git_repo(child_path):
                if not _has_git_authorship(child_path, require_git_authorship_email):
                    top_level_authorship_prune.add(child)

    for dirpath, dirnames, filenames in os.walk(root):
        if require_git_authorship_email and dirpath == root and top_level_authorship_prune:
            dirnames[:] = [d for d in dirnames if d not in top_level_authorship_prune]
        dirnames[:] = [d for d in dirnames if not _is_excluded_dir(d, exclude_dirs)]
        for fname in filenames:
            # Standing secrets guard: `.env`/`.env.*` files are ALWAYS
            # excluded, unconditionally, regardless of whether
            # `exclude_patterns` is passed at all. See `_ENV_FILE_PATTERN`'s
            # 2026-07-21 fix note above for why this must not be nested
            # inside the `exclude_patterns is not None` branch below.
            if _ENV_FILE_PATTERN.match(fname):
                continue
            if exclude_patterns is not None and _is_excluded_file_pattern(fname, exclude_patterns):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in include_extensions:
                continue
            fpath = os.path.join(dirpath, fname)
            if max_file_size_mb is not None:
                try:
                    if os.path.getsize(fpath) > max_file_size_mb * 1024 * 1024:
                        continue
                except OSError:
                    continue
            yield fpath


def load_config(config_path: Optional[str] = None) -> dict:
    """Load RAG configuration.

    Builds a `default_config` dict containing every key the rest of the
    codebase relies on, then — if `config_path` is provided and exists —
    loads the YAML at that path and shallow-merges it on top via
    `default_config.update(user_config)` (top-level merge only, matching
    prior behavior; nested dicts in the user config fully replace the
    corresponding default sub-dict rather than being deep-merged).
    """
    default_config = {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_prefix": "rag_"
        },
        "embedding": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "sparse_model_name": "Qdrant/bm25"
        },
        "indexing": {
            "mode": "light",  # light | balanced | full
            # child/parent chunk sizes and overlap are TOKEN counts (per the
            # embedding model's tokenizer, see EfficientRAG._token_len), not
            # character counts. See config/config.yaml for the full rationale.
            "child_chunk_size": 100,
            "child_chunk_overlap": 20,
            "parent_chunk_size": 400,
            "use_hierarchical": True,
            "use_contextual_summaries": True,
            "quantization": "binary",  # binary | scalar_int8 | none
            "hybrid_search": True,
            "force_binary": False,
            "binary_dim_threshold": 1024
        },
        "retrieval": {
            "top_k_children": 8,
            "oversampling": 3.0,
            "fetch_parents": True,
            "rerank": True,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_candidate_pool": 40,
            "rerank_top_n": 8,
            # CRAG-style ("Corrective RAG", arXiv:2401.15884) decompose-then-
            # recompose context review/distillation pass — see
            # review_and_distill_context() below. OFF by default: this is an
            # experimental addition, not proven to help end-to-end yet, and
            # must not change existing behavior unless explicitly enabled.
            "context_review_enabled": False,
            # Per-sentence relevance threshold (0.0-1.0, scored by a cheap
            # LLM call) below which a sentence is discarded during context
            # review. 0.3 is a deliberately lenient default (bias toward
            # keeping borderline-relevant sentences) since dropping a
            # genuinely relevant sentence is worse for context_recall than
            # keeping a mildly noisy one is for context_precision.
            "context_review_threshold": 0.3,
            # Adaptive-RAG-lite gate — see EfficientRAG._classify_retrieval_need() /
            # EfficientRAG.retrieve() and config/config.yaml's matching comments.
            # OFF by default; v1 scope is "none" vs "light" only.
            "adaptive_retrieval_enabled": False,
            "adaptive_retrieval_model": "glm-5.2:cloud",
            "adaptive_retrieval_provider": "ollama",
            "adaptive_light_top_k": 8,
            "adaptive_light_oversampling": 3.0,
            "adaptive_retrieval_fallback_strategy": "light",
            # Query-adaptive parent-chunk expansion sub-gate — see retrieve()'s
            # fetch_parents block. Only takes effect when adaptive_retrieval_enabled
            # AND fetch_parents are both true; a separate toggle (not folded into
            # adaptive_retrieval_enabled) so it can be isolated independently in a
            # future A/B validation. Default True: the classifier's per-query
            # needs_broad_context verdict decides whether parent expansion runs.
            "adaptive_parent_expansion_enabled": True,
            # Isolation flag for testing adaptive_parent_expansion_enabled WITHOUT the
            # unrelated "none"-strategy skip-retrieval gate confounding the result — see
            # config/config.yaml's matching comment and docs/CHANGELOG.md's 2026-07-20 entry.
            # Default False: retrieve() behaves exactly as before (a "none" classification
            # still short-circuits to []). True: retrieve() never lets strategy == "none"
            # skip retrieval — it's treated as "light" for execution purposes only, while
            # self._last_retrieval_strategy still records the RAW "none" verdict for audit.
            "adaptive_skip_none_strategy": False,
            # HyDE (Hypothetical Document Embeddings, Gao et al. arXiv:2212.10496) — see
            # EfficientRAG._generate_hyde_document() / EfficientRAG.retrieve() and
            # config/config.yaml's matching comments. OFF by default; only the dense
            # search leg is affected when enabled.
            "hyde_enabled": False,
            "hyde_model": "glm-5.2:cloud",
            "hyde_provider": "ollama",
            # MMR (Maximal Marginal Relevance, Carbonell & Goldstein, SIGIR '98) diversity-
            # aware selection pass — see EfficientRAG._apply_mmr() / EfficientRAG.retrieve()
            # and config/config.yaml's matching comments. OFF by default; runs AFTER
            # cross-encoder reranking and BEFORE parent-chunk expansion / CRAG context
            # review, reusing already-computed dense embeddings (no re-embedding).
            "mmr_enabled": False,
            "mmr_lambda": 0.6
        },
        # Google Workspace (Drive/Docs/Sheets) read-only ingestion connector
        # (scripts/google_workspace.py). This is a completely independent
        # integration using the USER'S OWN separate Google Cloud API project's
        # OAuth2 "Desktop app" credentials — never reuses another tool/skill's
        # stored credentials. Paths default to a project-relative
        # `.credentials/` directory (see .gitignore — this directory holds real
        # secrets and must never be committed); override both in config.yaml
        # if you'd rather keep credentials elsewhere (e.g. under
        # ~/.hermes/skills/knowledge/rag-qdrant-efficient/.credentials/).
        "google_workspace": {
            # OAuth2 Desktop-app client secret JSON, downloaded from Google
            # Cloud Console (APIs & Services > Credentials > Create
            # Credentials > OAuth client ID > Desktop app). Must be a real,
            # user-provided file — never fabricated/auto-discovered.
            "client_secret_path": ".credentials/client_secret.json",
            # Cached OAuth token (including refresh token), written after the
            # first successful interactive consent flow so subsequent runs
            # don't need to re-open a browser.
            "token_path": ".credentials/token.json",
        }
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        default_config.update(user_config)

    return default_config


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    chunk_type: str = "child"  # child or parent


class EfficientRAG:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qdrant = QdrantClient(
            host=config.get("qdrant", {}).get("host", "localhost"),
            port=config.get("qdrant", {}).get("port", 6333)
        )
        self.embed_model = SentenceTransformer(
            config.get("embedding", {}).get("model_name", "BAAI/bge-small-en-v1.5")
        )
        self.dim = config.get("embedding", {}).get("dimension", 384)
        self.mode = config.get("indexing", {}).get("mode", "light")
        self.collection_prefix = config.get("qdrant", {}).get("collection_prefix", "rag_")

        # Hybrid (sparse) search support
        self.sparse_model_name = config.get("embedding", {}).get("sparse_model_name", "Qdrant/bm25")
        self.sparse_model = SparseTextEmbedding(self.sparse_model_name)

        # Quantization guard configuration
        self.force_binary = config.get("indexing", {}).get("force_binary", False)
        self.binary_dim_threshold = config.get("indexing", {}).get("binary_dim_threshold", 1024)

        self.hybrid_search_enabled = config.get("indexing", {}).get("hybrid_search", True)

        # Reranking support — lazily instantiate the CrossEncoder only when
        # enabled, to avoid paying model-load cost when reranking is off.
        self.rerank_enabled = config.get("retrieval", {}).get("rerank", True)
        self.reranker = CrossEncoder(
            config.get("retrieval", {}).get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        ) if self.rerank_enabled else None

        # Client-side RRF fusion smoothing constant. Qdrant's server-side
        # FusionQuery(fusion=Fusion.RRF) does not expose a tunable k, so
        # hybrid retrieval fuses the dense/sparse legs itself (see retrieve()).
        self.rrf_k = config.get("retrieval", {}).get("rrf_k", 60)

        # Optional injected summarizer hook. This is internal wiring only
        # (not a YAML-serializable config key) so callers can pass a real
        # LLM-backed summarizer callable via config["_summary_fn"]. When
        # None (the default), `_generate_summary()` is used instead, which
        # is a NAIVE PLACEHOLDER (first few sentences) — NOT an LLM summary.
        self.summary_fn = config.get("_summary_fn")

        # Adaptive-RAG-lite: records the outcome of the last _classify_retrieval_need()
        # call made by retrieve() ("none" | "light"), or None when
        # retrieval.adaptive_retrieval_enabled is False (the default) — i.e. the gate
        # didn't run at all for the most recent retrieve() call. Callers (e.g.
        # retrieve_context() in retrieve.py) read this AFTER calling retrieve() to
        # distinguish "gate is off" from "gate ran and picked light/none", without
        # changing retrieve()'s return type/shape.
        self._last_retrieval_strategy: Optional[str] = None

        # Sibling to _last_retrieval_strategy: records the SAME classification call's
        # verdict on whether the query needs broad/synthesized context (bool), or None
        # when the gate didn't run (adaptive_retrieval_enabled is False). Used by
        # retrieve()'s parent-expansion step to decide, per query, whether to actually
        # expand to parent-chunk text — see retrieve()'s fetch_parents block and
        # docs/RESEARCH_NOTES.md §6 / docs/ARCHITECTURE.md §4.6 for why this is
        # query-adaptive rather than a static toggle.
        self._last_needs_broad_context: Optional[bool] = None

    # Fixed namespace UUID used to derive deterministic Qdrant point IDs for
    # parent chunks from their string parent_id (e.g. "<doc_id>_parent_<idx>").
    # This lets retrieve() fetch a parent chunk directly via
    # `qdrant.retrieve(ids=[...])` instead of a filtered scroll/search.
    _PARENT_ID_NAMESPACE = uuid.UUID("6f6a6f7a-6e61-4d6f-9c4e-6861636b6572")

    def get_collection_name(self, name: str) -> str:
        return f"{self.collection_prefix}{name}"

    def _parent_point_id(self, parent_id: str) -> str:
        """Deterministic Qdrant point ID for a parent chunk, derived from its
        string parent_id. Same parent_id always maps to the same point ID,
        so parent points can be looked up directly with `qdrant.retrieve()`
        without needing a payload filter/scroll."""
        return str(uuid.uuid5(self._PARENT_ID_NAMESPACE, parent_id))

    def _effective_quantization_label(self) -> str:
        """Determine the effective quantization mode per the dimension guard rule:

        binary only if (dim >= binary_dim_threshold AND mode == "light")
                       OR force_binary == True
        else scalar (INT8) if mode in ("light", "balanced")
        else none (mode == "full")
        """
        if self.force_binary or (self.mode == "light" and self.dim >= self.binary_dim_threshold):
            return "binary"
        if self.mode in ("light", "balanced"):
            return "scalar"
        return "none"

    def _resolve_quantization_config(self):
        """Resolve the qdrant quantization_config to use for collection creation,
        applying the binary/scalar/none dimension guard rule and logging a clear
        message stating which quantization was chosen and why.
        """
        label = self._effective_quantization_label()

        if label == "binary":
            if self.dim < self.binary_dim_threshold:
                print(
                    f"WARNING: Quantization: binary (forced via force_binary=true, "
                    f"dim={self.dim} < {self.binary_dim_threshold} — accuracy may degrade)"
                )
            else:
                print(f"Quantization: binary — dim={self.dim} >= threshold={self.binary_dim_threshold}")
            return BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))

        if label == "scalar":
            print(
                f"Quantization: scalar (INT8) — dim={self.dim}, "
                f"threshold={self.binary_dim_threshold}, mode={self.mode}"
            )
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type="int8",
                    quantile=0.99,
                    always_ram=True
                )
            )

        print(f"Quantization: none (mode={self.mode})")
        return None

    def create_collection(self, collection_name: str, force_recreate: bool = False):
        full_name = self.get_collection_name(collection_name)
        exists = self.qdrant.collection_exists(full_name)

        if force_recreate and exists:
            self.qdrant.delete_collection(full_name)
            exists = False

        if exists:
            # Collection already present and not asked to recreate — reuse
            # it as-is. This matters for multi-file directory ingests, which
            # only pass force_recreate=True for the first file; without this
            # check every subsequent file's create_collection() call would
            # either wipe the collection again (if force_recreate stayed
            # True) or hit a 409 Conflict from Qdrant (if it didn't).
            print(f"Collection already exists: {full_name} — reusing")
            return

        vectors_config = {"dense": VectorParams(size=self.dim, distance=Distance.COSINE)}
        sparse_vectors_config = {"sparse": SparseVectorParams()} if self.hybrid_search_enabled else None

        quantization_config = self._resolve_quantization_config()

        create_kwargs = dict(
            collection_name=full_name,
            vectors_config=vectors_config,
            quantization_config=quantization_config,
            on_disk_payload=True  # Helps with storage
        )
        if sparse_vectors_config is not None:
            create_kwargs["sparse_vectors_config"] = sparse_vectors_config

        self.qdrant.create_collection(**create_kwargs)
        print(f"Created collection: {full_name} (mode: {self.mode}, hybrid_search: {self.hybrid_search_enabled})")

    def _load_local_file(self, path: str) -> str:
        """Extract text from local file.

        PDF and DOCX extraction are wrapped in their own try/except: a single
        encrypted/corrupted/malformed file (e.g. pypdf raising
        FileNotDecryptedError deep inside `reader.pages` iteration on an
        encrypted PDF) must never propagate up and kill an entire batch
        ingestion run — see docs/CHANGELOG.md 2026-07-22. On failure we log a
        clear message and return "" so the caller's normal
        `if not raw_text.strip(): return {"status": "error", ...}` path in
        ingest() handles it as an ordinary per-file skip, not a crash.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf" and PdfReader:
            try:
                reader = PdfReader(path)
                return "\n".join([p.extract_text() or "" for p in reader.pages])
            except (PyPdfError, PdfReadError, FileNotDecryptedError) as e:
                print(f"Skipping unreadable/encrypted PDF: {path} ({e!r})")
                return ""
            except Exception as e:
                # Defense in depth: pypdf can also raise plain ValueError/
                # KeyError/struct.error/etc. on sufficiently malformed PDFs
                # that aren't wrapped in its own exception hierarchy.
                print(f"Skipping unreadable PDF (unexpected error): {path} ({e!r})")
                return ""
        elif ext == ".docx" and DocxDocument:
            try:
                doc = DocxDocument(path)
                return "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                print(f"Skipping unreadable/corrupted DOCX: {path} ({e!r})")
                return ""
        elif ext in [".txt", ".md", ".markdown"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            # Fallback: try plain text
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except:
                return ""

    def _extract_readable_text(self, html: str) -> str:
        """Extract clean, readable text from raw HTML.

        Uses trafilatura for extraction (readability-style boilerplate
        removal — benchmarks substantially better than tag-stripping at
        separating article content from nav/related-links/comment noise).
        Falls back to naive BeautifulSoup tag-stripping if trafilatura
        can't extract anything (some pages genuinely fail either way),
        so ingestion never hard-fails on a page the old method could at
        least partially handle.

        Factored out of `_fetch_url()` so `crawl_site()` can reuse the exact
        same extraction logic per-page without duplicating it.
        """
        try:
            import trafilatura
            text = trafilatura.extract(html, include_tables=True, include_formatting=False)
            if text:
                return text[:50000]  # Safety cap
        except ImportError:
            pass
        except Exception as e:
            print(f"trafilatura extraction error (falling back to BeautifulSoup): {e}")

        if not BeautifulSoup:
            return ""
        try:
            soup = BeautifulSoup(html, "lxml")
            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text[:50000]  # Safety cap
        except Exception as e:
            print(f"URL fetch error: {e}")
            return ""

    def _fetch_url(self, url: str) -> str:
        """Fetch a single URL and return its clean extracted text.

        See `_extract_readable_text()` for the extraction strategy
        (trafilatura with a BeautifulSoup tag-stripping fallback).
        """
        if not requests:
            return ""
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Hermes-RAG/1.0"})
        except Exception as e:
            print(f"URL fetch error: {e}")
            return ""
        return self._extract_readable_text(resp.text)

    def crawl_site(self, start_url: str, max_pages: int = 200, same_prefix: bool = True,
                    path_prefix: Optional[str] = None) -> List[Dict[str, str]]:
        """Breadth-first crawl of a documentation site/section starting at `start_url`.

        Discovers linked pages via raw `<a href>` extraction (BeautifulSoup,
        just for link discovery — actual content extraction still goes
        through `_extract_readable_text()`, the same trafilatura/BS4-fallback
        logic `_fetch_url()` uses, so crawled pages are cleaned identically
        to single-page `--url` ingestion).

        Scope control:
        - Only follows links on the SAME DOMAIN (netloc) as `start_url`.
        - When `same_prefix=True` (default), only follows links whose path
          starts with the same "directory prefix" as `start_url`'s path
          (e.g. starting at `/sr4jc/latest` only follows links still under
          `/sr4jc/`), so a hub page for one doc section doesn't wander into
          unrelated parts of a large site.
        - URLs are normalized (fragment/query-free-of-content stripped,
          trailing slash normalized) before dedup so `#section` anchors and
          trailing-slash variants of the same page aren't counted/fetched
          twice.

        `path_prefix` (optional, see docs/CHANGELOG.md 2026-07-22 Fivetran
        entry): overrides the auto-derived "drop the last path segment"
        prefix above with an exact one, for sites where `start_url`'s LAST
        segment is itself the section root (its children live directly
        under it, e.g. `/docs/rest-api` -> children at `/docs/rest-api/...`)
        rather than a hub page one level below the section (the assumption
        the auto-derivation was built for, e.g. a Issue Tracker `/space/resources`
        hub whose sibling articles live under sibling `/space/docs/...`).
        Auto-derivation can't distinguish these two shapes from the URL
        alone; pass `path_prefix="/docs/rest-api/"` explicitly for the
        former case instead of getting the auto-derived `/docs/` (which
        would wander into every unrelated doc section on the domain). Has
        no effect when `same_prefix=False`.

        Stops once `max_pages` unique pages have been fetched or the queue
        is exhausted, whichever comes first.

        Returns a list of {"url", "text", "title"} dicts (title omitted if
        not found) for every page whose extracted text was non-empty.
        """
        if not requests or not BeautifulSoup:
            print("crawl_site: requests/BeautifulSoup not available")
            return []

        from urllib.parse import urljoin, urlparse

        def _normalize(u: str) -> str:
            parsed = urlparse(u)
            path = parsed.path or "/"
            if len(path) > 1 and path.endswith("/"):
                path = path[:-1]
            # Drop fragment always; drop query string too — doc sites rarely
            # encode distinct content in query params, and keeping them
            # tends to multiply "duplicate" pages (e.g. ?sort=, ?utm_*).
            return f"{parsed.scheme}://{parsed.netloc}{path}"

        start_norm = _normalize(start_url)
        start_parsed = urlparse(start_norm)
        start_domain = start_parsed.netloc
        # "Directory prefix" = the doc-section root to stay under, derived by
        # dropping the LAST path segment of start_url (treating it as a
        # specific hub/index page WITHIN a section, e.g.
        # "/jira-software-cloud/resources" -> "/jira-software-cloud/", or
        # "/sr4jc/latest" -> "/sr4jc/") so sibling pages under that section
        # (e.g. "/jira-software-cloud/docs/...") still match — a hub page is
        # rarely itself the root of the URL hierarchy its linked articles
        # live under. If start_url's path has only ONE segment (e.g.
        # "/docs"), there's nothing sensible to drop to without falling all
        # the way to "/" (which would let the crawl wander into an unrelated
        # homepage/other locale) — keep that single segment itself as the
        # prefix (e.g. "/docs/") instead.
        if path_prefix is not None:
            start_prefix = path_prefix if path_prefix.endswith("/") else path_prefix + "/"
        else:
            segments = [s for s in start_parsed.path.split("/") if s]
            if len(segments) >= 2:
                start_prefix = "/" + "/".join(segments[:-1]) + "/"
            elif len(segments) == 1:
                start_prefix = "/" + segments[0] + "/"
            else:
                start_prefix = "/"

        def _in_scope(u: str) -> bool:
            p = urlparse(u)
            if p.netloc != start_domain:
                return False
            if same_prefix and not p.path.startswith(start_prefix):
                return False
            return True

        visited = set()
        queue = [start_norm]
        results = []

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = requests.get(url, timeout=15, headers={"User-Agent": "Hermes-RAG/1.0"})
            except Exception as e:
                print(f"crawl_site: fetch error for {url}: {e}")
                continue

            if resp.status_code >= 400:
                print(f"crawl_site: HTTP {resp.status_code} for {url} — skipped")
                continue

            html = resp.text
            text = self._extract_readable_text(html)

            title = None
            try:
                soup = BeautifulSoup(html, "lxml")
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
            except Exception:
                pass

            if text and text.strip():
                page = {"url": url, "text": text}
                if title:
                    page["title"] = title
                results.append(page)
                print(f"crawl_site: fetched ({len(visited)}/{max_pages}) {url}")
            else:
                print(f"crawl_site: no extractable text for {url} — not ingested, links still followed")

            if len(visited) >= max_pages:
                break

            # Discover links to enqueue.
            try:
                soup = BeautifulSoup(html, "lxml")
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                        continue
                    absolute = urljoin(url, href)
                    norm = _normalize(absolute)
                    if norm in visited:
                        continue
                    if not _in_scope(norm):
                        continue
                    if norm not in queue:
                        queue.append(norm)
            except Exception as e:
                print(f"crawl_site: link discovery error for {url}: {e}")

        return results

    def crawl_confluence_space(self, base_url: str, root_page_id: str, max_pages: int = 150,
                                space_key: Optional[str] = None) -> List[Dict[str, str]]:
        """Enumerate and fetch pages in a Document Service space via the public
        Content REST API, instead of scraping `<a href>` links off the
        space's home page body.

        Background (see docs/CHANGELOG.md 2026-07-22): a plain `crawl_site()`
        BFS starting at a Document Service "space home" page (a `/spaces/{KEY}/
        pages/{id}/...` URL) runs out of links almost immediately, because
        that page's real navigation/page-tree sidebar is NOT present as
        plain `<a href>` tags in its raw HTML.

        THIS METHOD WENT THROUGH TWO WRONG FIXES before landing on the
        current (combined) approach — recorded here so the next person
        doesn't re-try either of them expecting them to be sufficient alone:

        1. CQL `cql=ancestor=<id>` search: matches a page's entire subtree
           in one paginated call. Worked great on the first space tested
           (JIRASOFTWARECLOUD: got the full 150-page cap) but was found,
           during the real production run, to silently return 0-5 results
           for most OTHER spaces (e.g. ConfCloud: 0 results, even though
           that page has real children) — its search index isn't
           consistently populated across all spaces on this instance.
        2. `GET {base_url}/rest/api/content/{id}/child/page`, walked via BFS
           from the space's home/root page: reliable where it applies, and
           fixed ConfCloud (found all 150+ pages) — but some spaces'
           "home" page (e.g. SERVICEDESKCLOUD's "Issue Tracker Cloud Shared Master
           Home") is a shallow stub with only a handful of direct children,
           while the space's real ~690 content pages are NOT nested under
           it in the page tree at all (they're siblings/orphans as far as
           parent/child goes, just members of the same space).

        The fix that actually covers both cases: when `space_key` is given,
        first pull the FULL flat page list for that space via
        `cql=space={key} AND type=page` (also inconsistently indexed — 0
        results on some spaces where BFS instead finds real content, e.g.
        ConfCloud/BITBUCKET — but a strict superset of BFS's results on
        others, e.g. SERVICEDESKCLOUD's 690 vs. BFS's 5), THEN ALSO run the
        `child/page` BFS from `root_page_id`, and take the UNION of both
        (deduped by page ID). Whichever mechanism happens to be populated
        for a given space, the other one covers the gap.

        Each page is fetched at `{base_url}/pages/viewpage.action?
        pageId={id}` (ID-based — unlike a `/display/{SPACE}/{Title}` slug,
        this never depends on title-to-slug encoding matching, which was
        observed to 404 on real pages with renamed titles/unusual
        punctuation) and its text extracted via the same
        `_extract_readable_text()` used by `_fetch_url()`/`crawl_site()`.

        `root_page_id`: the Document Service content ID of the space's home/root
        page (e.g. "764477791" for the Issue Tracker Software Cloud documentation
        home) — found in the URL of a `/spaces/{KEY}/pages/{id}/...` link.

        `space_key`: the Document Service space key (e.g. "JIRASOFTWARECLOUD",
        the `{KEY}` segment of that same URL) — optional, but strongly
        recommended; without it, only the `child/page` BFS runs, which (per
        above) can badly undercount spaces with a shallow/stub home page.

        `base_url`: scheme+netloc of the Document Service instance, e.g.
        "https://confluence.atlassian.com".

        Stops once `max_pages` pages have been fetched, or every known
        candidate page ID is exhausted, whichever comes first. Returns a
        list of {"url", "text", "title"} dicts for every page whose
        extracted text was non-empty; pages that are listed but whose
        content fetch fails or extracts empty text are skipped (logged, not
        fatal — matches `crawl_site()`'s resilience policy).
        """
        if not requests or not BeautifulSoup:
            print("crawl_confluence_space: requests/BeautifulSoup not available")
            return []

        headers = {"User-Agent": "Hermes-RAG/1.0"}

        def _fetch_children(parent_id: str):
            """Return a list of (id, title) for the direct child pages of
            `parent_id`, paginating via start/limit until exhausted."""
            children = []
            start = 0
            limit = 100
            while True:
                api_url = f"{base_url}/rest/api/content/{parent_id}/child/page?limit={limit}&start={start}"
                try:
                    resp = requests.get(api_url, timeout=20, headers=headers)
                except Exception as e:
                    print(f"crawl_confluence_space: child/page fetch error for {parent_id} at start={start}: {e}")
                    break
                if resp.status_code >= 400:
                    print(f"crawl_confluence_space: child/page HTTP {resp.status_code} for {parent_id} at start={start} — stopping enumeration for this node")
                    break
                try:
                    data = resp.json()
                except Exception as e:
                    print(f"crawl_confluence_space: child/page JSON parse error for {parent_id} at start={start}: {e}")
                    break
                results = data.get("results", [])
                for r in results:
                    children.append((r.get("id"), r.get("title")))
                if not (data.get("_links") or {}).get("next"):
                    break
                start += limit
            return children

        def _fetch_space_page_list(key: str, cap: int):
            """Return up to `cap` (id, title) pairs for every page in space
            `key`, via CQL `space=<key> AND type=page`, paginated."""
            found = []
            start = 0
            limit = 100
            while len(found) < cap:
                api_url = (
                    f"{base_url}/rest/api/content/search"
                    f"?cql=space={key}+AND+type=page&limit={limit}&start={start}"
                )
                try:
                    resp = requests.get(api_url, timeout=20, headers=headers)
                except Exception as e:
                    print(f"crawl_confluence_space: space-list fetch error for {key} at start={start}: {e}")
                    break
                if resp.status_code >= 400:
                    print(f"crawl_confluence_space: space-list HTTP {resp.status_code} for {key} at start={start} — stopping")
                    break
                try:
                    data = resp.json()
                except Exception as e:
                    print(f"crawl_confluence_space: space-list JSON parse error for {key} at start={start}: {e}")
                    break
                results = data.get("results", [])
                if not results:
                    break
                for r in results:
                    found.append((r.get("id"), r.get("title")))
                if not (data.get("_links") or {}).get("next"):
                    break
                start += limit
            print(f"crawl_confluence_space: space-list CQL for {key} found {len(found)} page(s)")
            return found

        # Build the candidate queue: space-wide CQL listing first (broadest
        # single source when populated), then BFS from the root — dedup by
        # ID handles the overlap, and each source alone covers the other's
        # gaps (see docstring).
        queue = [(root_page_id, None)]
        seen_candidates = {root_page_id}
        if space_key:
            for pid, title in _fetch_space_page_list(space_key, max_pages * 3):
                if pid and pid not in seen_candidates:
                    seen_candidates.add(pid)
                    queue.append((pid, title))

        results = []
        fetched_ids = set()

        while queue and len(results) < max_pages:
            page_id, title = queue.pop(0)
            if not page_id or page_id in fetched_ids:
                continue
            fetched_ids.add(page_id)
            page_url = f"{base_url}/pages/viewpage.action?pageId={page_id}"

            try:
                resp = requests.get(page_url, timeout=15, headers=headers)
            except Exception as e:
                print(f"crawl_confluence_space: fetch error for {page_url}: {e}")
            else:
                if resp.status_code >= 400:
                    print(f"crawl_confluence_space: HTTP {resp.status_code} for {page_url} — skipped")
                else:
                    text = self._extract_readable_text(resp.text)
                    if text and text.strip():
                        page = {"url": page_url, "text": text}
                        if title:
                            page["title"] = title
                        results.append(page)
                        print(f"crawl_confluence_space: fetched ({len(results)}/{max_pages}) {page_url}")
                    else:
                        print(f"crawl_confluence_space: no extractable text for {page_url} — skipped")

            if len(results) >= max_pages:
                break

            # Enumerate this node's children (BFS supplement) and enqueue any
            # not already discovered via the space-wide listing or a prior
            # BFS step, regardless of whether ITS OWN content fetch above
            # succeeded — a page that 404s or has no extractable body can
            # still have real, ingestible children under it.
            for child_id, child_title in _fetch_children(page_id):
                if child_id and child_id not in seen_candidates:
                    seen_candidates.add(child_id)
                    queue.append((child_id, child_title))

        return results

    def _fetch_api(self, api_url: str, method: str = "GET", headers: Optional[Dict] = None,
                   params: Optional[Dict] = None, data: Optional[Dict] = None) -> str:
        """Fetch API and return text/JSON as string."""
        if not requests:
            return ""
        try:
            resp = requests.request(method, api_url, headers=headers, params=params, json=data, timeout=20)
            if resp.headers.get("content-type", "").startswith("application/json"):
                return json.dumps(resp.json(), indent=2)[:30000]
            return resp.text[:30000]
        except Exception as e:
            print(f"API error: {e}")
            return ""

    def _token_len(self, text: str) -> int:
        """Count tokens the way the embedding model actually sees them, using
        the SentenceTransformer's underlying HuggingFace tokenizer
        (`self.embed_model.tokenizer`, a `BertTokenizer` for
        BAAI/bge-small-en-v1.5). `add_special_tokens=False` so this measures
        pure content length — [CLS]/[SEP] overhead is a fixed +2 the model
        adds at encode time, not something chunk-sizing needs to budget for.
        """
        if not text:
            return 0
        return len(self.embed_model.tokenizer.encode(text, add_special_tokens=False))

    def _hierarchical_chunk(self, text: str, source: str, doc_id: str) -> List[Chunk]:
        """Robust hierarchical parent-child chunking with recursive splitting.

        Sizes (`child_chunk_size`, `parent_chunk_size`, `child_chunk_overlap`)
        are TOKEN counts (per `self.embed_model.tokenizer`), not character
        counts — see config/config.yaml comments. This matters because the
        embedding model has a fixed token budget (512 for bge-small), not a
        character budget, and chars-per-token varies with content density.
        """
        child_size = self.config.get("indexing", {}).get("child_chunk_size", 100)
        parent_size = self.config.get("indexing", {}).get("parent_chunk_size", 400)
        overlap = self.config.get("indexing", {}).get("child_chunk_overlap", 20)

        # NOTE: no empty-string entry here. `"" in t` is True for every string in
        # Python, so an "" sentinel would always match first in the loop below and
        # call `t.split("")`, which Python raises ValueError on ("empty separator")
        # — this was a real, live-hit bug (see docs/CHANGELOG.md, 2026-07-20 Google
        # Workspace ingestion entry). The real last-resort fallback (token-window
        # splitting via the tokenizer, below the for-loop) is only reachable once no
        # separator in this list actually matches — so the list must stop at " ".
        separators = ["\n\n", "\n", ". ", " "]
        tok_len = self._token_len
        tokenizer = self.embed_model.tokenizer

        def recursive_split(t: str, size: int, ov: int, seps: list) -> List[str]:
            if tok_len(t) <= size:
                return [t.strip()] if t.strip() else []
            for sep in seps:
                if sep in t:
                    parts = t.split(sep)
                    # Tokenize each part+sep exactly once up front, then track
                    # a running token-count total while accumulating "current"
                    # — this keeps tokenizer calls O(num_parts) instead of
                    # O(num_parts^2), which matters since re-tokenizing the
                    # whole "current" string on every part (the naive
                    # translation of the old len()-based loop) would be
                    # pathologically slow on a large document.
                    part_tok_counts = [tok_len(part + sep) for part in parts]
                    chunks = []
                    current = ""
                    current_tokens = 0
                    for part, p_tokens in zip(parts, part_tok_counts):
                        if current_tokens + p_tokens > size and current:
                            chunks.append(current.strip())
                            current = part + sep
                            current_tokens = p_tokens
                        else:
                            current += part + sep
                            current_tokens += p_tokens
                    if current.strip():
                        chunks.append(current.strip())
                    # Recurse on large chunks
                    final = []
                    for c in chunks:
                        if tok_len(c) > size:
                            final.extend(recursive_split(c, size, ov, seps[1:]))
                        else:
                            final.append(c)
                    return final
            # Fallback: no separator found at all (e.g. one giant unbroken
            # string) — split directly on token windows via the tokenizer
            # (encode once, slice token ids, decode each window back to text).
            ids = tokenizer.encode(t, add_special_tokens=False)
            step = max(size - ov, 1)
            chunks = []
            for i in range(0, len(ids), step):
                window = ids[i:i + size]
                if window:
                    chunks.append(tokenizer.decode(window).strip())
            return [c for c in chunks if c]

        parent_chunks = recursive_split(text, parent_size, overlap // 2, separators)
        all_chunks = []

        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"{doc_id}_parent_{p_idx}"
            parent_chunk = Chunk(
                text=parent_text,
                metadata={
                    "source": source,
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "chunk_type": "parent",
                    "parent_index": p_idx,
                    "ingested_at": datetime.utcnow().isoformat()
                },
                chunk_type="parent"
            )
            all_chunks.append(parent_chunk)

            child_texts = recursive_split(parent_text, child_size, overlap, separators)
            for c_idx, child_text in enumerate(child_texts):
                child_chunk = Chunk(
                    text=child_text,
                    metadata={
                        "source": source,
                        "doc_id": doc_id,
                        "parent_id": parent_id,
                        "chunk_type": "child",
                        "parent_index": p_idx,
                        "child_index": c_idx,
                        "ingested_at": datetime.utcnow().isoformat()
                    },
                    parent_id=parent_id,
                    chunk_type="child"
                )
                all_chunks.append(child_chunk)

        return all_chunks

    def _generate_summary(self, text: str) -> str:
        """Naive extractive placeholder for a contextual summary.

        NOTE: This is NOT an LLM-generated summary — it simply returns the
        first few sentences of the chunk. To use a real LLM summarizer,
        inject a callable via `config["_summary_fn"]` (see `self.summary_fn`
        in `__init__`); when set, ingestion prefers `self.summary_fn` over
        this placeholder.
        """
        # Placeholder: first 2-3 sentences or key phrases
        sentences = text.split(". ")[:3]
        return ". ".join(sentences) + "."

    def _sparse_vector_for(self, text: str) -> SparseVector:
        """Compute a Qdrant SparseVector for a single piece of text using the
        configured fastembed sparse model."""
        embedding = list(self.sparse_model.embed([text]))[0]
        indices = embedding.indices
        values = embedding.values
        return SparseVector(
            indices=indices.tolist() if hasattr(indices, "tolist") else list(indices),
            values=values.tolist() if hasattr(values, "tolist") else list(values),
        )

    def ingest(self, source: str, collection: str, tags: List[str] = None,
               force_recreate: bool = False, is_url: bool = False, is_api: bool = False,
               api_config: Optional[Dict] = None, raw_text: Optional[str] = None,
               extra_metadata: Optional[Dict[str, Any]] = None,
               exclude_work_content: bool = False) -> Dict[str, Any]:
        """Main ingestion entrypoint. Returns stats.

        `raw_text`: when provided, bypasses local-file/URL/API loading
        entirely and ingests this text directly, with `source` used only as
        the point payload's "source" label / doc_id derivation (e.g. a
        Google Doc's webViewLink — see scripts/google_workspace.py's ingest.py
        wiring). Takes priority over is_url/is_api.

        `extra_metadata`: optional extra key/value pairs merged into every
        resulting point's payload (both parent and child), for source-type
        tagging that doesn't fit the existing fixed payload fields — e.g.
        Google Drive ingestion tags {"source_type": "google_drive",
        "webViewLink": ..., "mime_type": ...}. Existing standard payload
        keys (source/doc_id/parent_id/chunk_type/etc.) always take
        precedence over same-named keys here, so this can't clobber core
        metadata.

        `exclude_work_content`: opt-in (default False — never changes
        behavior for existing callers who don't pass it) content-based work
        filter. When True, the EXTRACTED text (after local-file/URL/API/
        raw_text loading, before chunking/embedding) is scanned for any
        WORK_CONTENT_SIGNAL_TERMS substring (see utils.py's comment on that
        list for why — filename/path-based filtering alone missed real
        contaminated files with generic names like `Unknown.pdf` in a live
        ~/Downloads ingestion run). If a signal term is found, this file is
        skipped entirely: nothing is chunked, embedded, or upserted to
        Qdrant, and a message is logged explaining why, matching how
        `require_git_authorship_email`/`exclude_patterns` exclusions are
        logged in scripts/ingest.py's dry-run reporting.
        """
        self.create_collection(collection, force_recreate=force_recreate)
        full_coll = self.get_collection_name(collection)

        # Load content
        if raw_text is not None:
            loaded_text = raw_text
            doc_id = hashlib.md5(source.encode()).hexdigest()[:12]
        elif is_url:
            loaded_text = self._fetch_url(source)
            doc_id = hashlib.md5(source.encode()).hexdigest()[:12]
        elif is_api:
            loaded_text = self._fetch_api(source, **(api_config or {}))
            doc_id = hashlib.md5(source.encode()).hexdigest()[:12]
        else:
            loaded_text = self._load_local_file(source)
            doc_id = hashlib.md5(source.encode()).hexdigest()[:12]
        raw_text = loaded_text

        if not raw_text.strip():
            return {"status": "error", "message": "No text extracted"}

        if exclude_work_content and _contains_work_content_signal(raw_text):
            print(
                f"Skipping {source}: extracted text contains a "
                "WORK_CONTENT_SIGNAL_TERMS match (--work-topic-excludes "
                "content-based check) — not chunked, embedded, or upserted."
            )
            return {"status": "skipped", "message": "Excluded: work-content signal term found in extracted text"}

        # Hierarchical chunking
        chunks = self._hierarchical_chunk(raw_text, source, doc_id)

        use_summaries = self.config.get("indexing", {}).get("use_contextual_summaries")

        # Rule-based (no LLM) public/private classification — see
        # `_classify_visibility()`'s comment for the mapping. Computed once
        # per `ingest()` call (all chunks of a single source share the same
        # `source_type`/`extra_metadata`, so this can't vary within one
        # ingest) and stamped onto every parent/child point's payload below.
        visibility = _classify_visibility(extra_metadata)

        points = []
        parent_points = 0
        for chunk in chunks:
            if chunk.chunk_type != "child":
                # Parent chunks are never embedded/searched directly — they're
                # stored as payload-only points (a zero "dense" vector, no
                # sparse vector) so the full parent text is durably fetchable
                # by parent_id via a deterministic point ID, for context
                # expansion in retrieve() when retrieval.fetch_parents is set.
                parent_point_id = self._parent_point_id(chunk.metadata.get("parent_id"))
                parent_payload = {
                    "text": chunk.text,
                    "source": chunk.metadata.get("source"),
                    "doc_id": chunk.metadata.get("doc_id"),
                    "parent_id": chunk.metadata.get("parent_id"),
                    "chunk_type": "parent",
                    "tags": tags or [],
                    "visibility": visibility,
                    **(extra_metadata or {}),
                    **{k: v for k, v in chunk.metadata.items() if k not in ["text"]}
                }
                points.append(PointStruct(
                    id=parent_point_id,
                    vector={"dense": [0.0] * self.dim},
                    payload=parent_payload,
                ))
                parent_points += 1
                continue

            dense_vector = self.embed_model.encode(chunk.text, normalize_embeddings=True).tolist()

            if self.hybrid_search_enabled:
                vector = {"dense": dense_vector, "sparse": self._sparse_vector_for(chunk.text)}
            else:
                vector = {"dense": dense_vector}

            summary = None
            if use_summaries:
                summary = self.summary_fn(chunk.text) if self.summary_fn else self._generate_summary(chunk.text)

            payload = {
                "text": chunk.text[:2000],  # Cap payload size
                "source": chunk.metadata.get("source"),
                "doc_id": chunk.metadata.get("doc_id"),
                "parent_id": chunk.parent_id,
                "chunk_type": chunk.chunk_type,
                "tags": tags or [],
                "summary": summary,
                "visibility": visibility,
                **(extra_metadata or {}),
                **{k: v for k, v in chunk.metadata.items() if k not in ["text"]}
            }

            point_id = str(uuid.uuid4())
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if points:
            # Batch upserts instead of one single call for the whole file's
            # points. A single large source document (e.g. a multi-thousand-
            # row CSV/audit-log export) can produce enough child+parent
            # chunks that one unbatched upsert's JSON payload exceeds
            # Qdrant's default 32MB request-body limit, failing with a hard
            # 400 ("JSON payload (...) is larger than allowed") that aborts
            # the whole ingest() call for that file — a real failure hit
            # during a --recursive personal-corpus ingestion run (see
            # docs/CHANGELOG.md's dated entry) on a single ~1,700-point CSV.
            # 256 points/batch is a conservative size that comfortably fits
            # under the limit even for large payload text + dense/sparse
            # vectors combined, without needing to compute exact byte sizes
            # up front.
            upsert_batch_size = 256
            for batch_start in range(0, len(points), upsert_batch_size):
                batch = points[batch_start:batch_start + upsert_batch_size]
                self.qdrant.upsert(collection_name=full_coll, points=batch)

        stats = {
            "status": "success",
            "collection": collection,
            "points_ingested": len(points),
            "child_points_ingested": len(points) - parent_points,
            "parent_points_ingested": parent_points,
            "source": source,
            "mode": self.mode,
            "quantization": self._effective_quantization_label()
        }
        print(stats)
        return stats

    def _classify_retrieval_need(self, query: str) -> Dict[str, Any]:
        """Adaptive-RAG-lite: single cheap LLM call classifying `query` on TWO
        independent dimensions BEFORE the hybrid search / rerank / parent-expansion /
        context-review pipeline runs, gated by config `retrieval.adaptive_retrieval_enabled`
        (OFF by default) — see retrieve()'s call site. Both dimensions are assessed in
        this ONE call (not a second LLM call) to avoid a redundant classifier round-trip
        per query:

        1. "strategy": "none" (no retrieval — answerable from general/common knowledge,
           not specific to any ingested corpus) or "light" (normal retrieval, current
           default settings). Bias toward "light" over "none" when ambiguous — wrongly
           skipping retrieval is worse than wrongly doing a retrieval that turns out
           unnecessary (same lenient-default philosophy as context_review_threshold=0.3).
        2. "needs_broad_context": True for queries that are comparative, multi-part, or
           require synthesizing across multiple facts/sections (the same "deep" criteria
           originally scoped OUT of dimension 1 as a v1 retrieval-depth tier — see below
           — reused here for a DIFFERENT decision: whether retrieve()'s parent-chunk
           expansion step should run for THIS query, not how many chunks to retrieve).
           Bias toward False when ambiguous — the conservative default is to keep the
           precise, un-diluted child-chunk match rather than expand to a noisier parent
           chunk (see docs/RESEARCH_NOTES.md §6 / docs/ARCHITECTURE.md §4.6: parent
           expansion helps multi-hop questions but measurably hurts single-hop factoid
           questions, this project's dominant NFCorpus query style, by diluting a precise
           match with less-relevant surrounding text).

        v1 scope for dimension 1: only "none"/"light" — no third "deep" tier (deferred
        to a future pass). Self-RAG and FLARE were both ruled out as infeasible with
        this project's current infrastructure (no fine-tuning pipeline, no streaming
        generation). Dimension 2 (needs_broad_context) is NOT that deferred "deep" tier
        — it doesn't change top_k/oversampling at all, only whether parent-chunk
        expansion runs; the two are orthogonal, deliberately-separate decisions that
        happen to share one classification call for efficiency.

        Returns: {"strategy": "none" | "light", "needs_broad_context": bool,
                  "raw_response": str}

        Fails open: any call/parse/provider error returns
        {"strategy": config's adaptive_retrieval_fallback_strategy (default "light"),
         "needs_broad_context": False, "raw_response": "<error info>"} — a classifier
        hiccup must degrade to "the gate did nothing this time," never to skipping
        retrieval outright, and never to force-expanding to parent context either.

        Raises:
            ValueError: if retrieval.adaptive_retrieval_provider is not one of
                "ollama" | "neuralwatt" | "claude" — a misconfigured provider name is
                a config error to fail loud on immediately, not something to swallow
                into the fail-open path (which is reserved for call/parse/network
                errors on an otherwise-valid provider).
        """
        retrieval_cfg = self.config.get("retrieval", {})
        provider = retrieval_cfg.get("adaptive_retrieval_provider", "ollama")
        model_name = retrieval_cfg.get("adaptive_retrieval_model", "glm-5.2:cloud")
        fallback_strategy = retrieval_cfg.get("adaptive_retrieval_fallback_strategy", "light")

        if provider not in ("ollama", "neuralwatt", "claude"):
            raise ValueError(
                f"Unknown retrieval.adaptive_retrieval_provider {provider!r} — must be one of "
                "'ollama', 'neuralwatt', or 'claude'."
            )

        prompt = (
            "You are a routing classifier for a retrieval-augmented generation (RAG) system. "
            "Assess the QUERY below on TWO independent dimensions and respond with ONE JSON "
            "object covering both.\n\n"
            "DIMENSION 1 — retrieval strategy (\"strategy\"): decide whether answering the "
            "QUERY requires retrieving information from a specific ingested knowledge base "
            "(\"light\"), or whether it is a general-knowledge question answerable without "
            "looking anything up (\"none\"). If the query is broad common knowledge (e.g. basic "
            "facts, definitions, simple math) with no indication it depends on a specific "
            "document/corpus, choose \"none\". If there is ANY reasonable chance the query "
            "depends on specific, detailed, or corpus-specific information (e.g. it names a "
            "specific study, dataset, document, or asks for a specific/technical detail), "
            "choose \"light\" — when in doubt, prefer \"light\" over \"none\", since skipping "
            "retrieval that was actually needed is worse than doing a retrieval that turns out "
            "unnecessary.\n\n"
            "DIMENSION 2 — broad context need (\"needs_broad_context\", true or false): decide "
            "whether answering the QUERY well requires synthesizing across MULTIPLE facts, "
            "sections, or documents — e.g. it is comparative (\"compare X and Y\"), multi-part "
            "(asks for more than one distinct thing), or requires connecting information "
            "unlikely to sit in one short passage. Set this true for those cases. Set it false "
            "for a single, narrow, single-fact question that one short, precise passage can "
            "answer directly (the common case). When in doubt, prefer false — broadening "
            "context helps genuinely multi-part questions but dilutes precision for the common "
            "single-fact case.\n\n"
            f"QUERY: {query}\n\n"
            "Respond with STRICT JSON only, no markdown fencing, no extra text: "
            '{"strategy": "none"|"light", "needs_broad_context": true|false, '
            '"reasoning": "<one sentence covering both dimensions>"}'
        )

        try:
            raw_response = self._call_llm_provider(provider, model_name, prompt)
            classification = self._parse_retrieval_classification(raw_response)
            return {
                "strategy": classification["strategy"],
                "needs_broad_context": classification["needs_broad_context"],
                "raw_response": raw_response,
            }
        except Exception as exc:
            print(
                f"WARNING: adaptive retrieval classification failed ({exc}) — falling back to "
                f"strategy={fallback_strategy!r}, needs_broad_context=False"
            )
            return {
                "strategy": fallback_strategy,
                "needs_broad_context": False,
                "raw_response": f"<error: {exc}>",
            }

    def _call_llm_provider(self, provider: str, model_name: str, prompt: str) -> str:
        """Dispatch `prompt` to the configured provider (ollama/neuralwatt/claude) and
        return the raw text response. Lazy-imports each provider's client, mirroring the
        cross-module lazy-import pattern review_and_distill_context() already uses to
        avoid a circular import with evaluate_ragas.py (evaluate_ragas.py itself imports
        `load_config` from this module inside its own functions for the same reason).

        Shared by both _classify_retrieval_need() (Adaptive-RAG-lite gate) and
        _generate_hyde_document() (HyDE) — this is purely the "send a prompt to one of
        the three supported providers and get text back" plumbing, with no
        feature-specific logic (prompt content, response parsing, fail-open behavior)
        living here. Formerly named `_call_adaptive_retrieval_llm` before HyDE needed
        the identical dispatch; renamed (call sites updated, no behavior change) rather
        than adding a second near-duplicate copy for HyDE to reuse.
        """
        import sys
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        if provider == "ollama":
            # Mirror _ollama_judge_llm() in evaluate_ragas.py: same base_url env var
            # pattern, direct langchain_ollama.ChatOllama construction.
            from langchain_ollama import ChatOllama
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            llm = ChatOllama(model=model_name, base_url=base_url)
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response))

        if provider == "neuralwatt":
            from evaluate_ragas import _neuralwatt_llm
            wrapper = _neuralwatt_llm(model_name)
            response = wrapper.langchain_llm.invoke(prompt)
            return getattr(response, "content", str(response))

        # provider == "claude" (validated by the caller before this point)
        from evaluate_ragas import _build_anthropic_client
        client, model, mode = _build_anthropic_client()
        if client is None:
            raise RuntimeError(
                "No Anthropic access configured (ANTHROPIC_API_KEY / "
                "ANTHROPIC_VERTEX_PROJECT_ID) for adaptive_retrieval_provider='claude'"
            )
        # Always use a cheap/fast Haiku-class model for this classification, same as
        # review_and_distill_context()'s choice of model for its per-sentence scoring.
        if mode == "direct":
            classify_model = "claude-haiku-4-5"
        else:
            classify_model = os.environ.get("ANTHROPIC_VERTEX_HAIKU_MODEL", "claude-haiku-4-5@20251001")
        response = client.messages.create(
            model=classify_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    @staticmethod
    def _parse_retrieval_classification(raw_response: str) -> Dict[str, Any]:
        """Defensively parse `raw_response` (the classifier LLM's raw text) into
        {"strategy": "none"|"light", "needs_broad_context": bool}.

        Handles: strict JSON, JSON wrapped in a markdown code fence, leading/
        trailing whitespace, and a bare unquoted strategy value (e.g. just `none`
        or `light` with no JSON at all — a tolerance that predates
        needs_broad_context, kept for whatever degenerate model output still
        produces a bare value with no JSON envelope at all).

        `strategy` is mandatory: raises ValueError if nothing recognizable as
        "none"/"light" can be extracted, letting the caller's fail-open handling
        take over — unchanged behavior from before needs_broad_context existed.

        `needs_broad_context` is best-effort and NEVER raises: if it's missing,
        not a boolean (or boolean-like string), or the response isn't valid JSON
        at all, it defaults to False (the conservative "don't expand to parent
        context" choice). The core "none"/"light" gate must keep working exactly
        as before even if this newer, secondary field fails to parse.
        """
        text = (raw_response or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            # Drop a leading language tag like "json\n" left over after stripping fences.
            text = re.sub(r"^\s*json\s*\n", "", text, flags=re.IGNORECASE).strip()

        strategy: Optional[str] = None
        needs_broad_context = False

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                raw_strategy = str(parsed.get("strategy", "")).strip().strip('"').lower()
                if raw_strategy in ("none", "light"):
                    strategy = raw_strategy
                nbc = parsed.get("needs_broad_context")
                if isinstance(nbc, bool):
                    needs_broad_context = nbc
                elif isinstance(nbc, str) and nbc.strip().lower() in ("true", "false"):
                    needs_broad_context = nbc.strip().lower() == "true"
        except (json.JSONDecodeError, TypeError):
            pass

        if strategy is None:
            # Bare unquoted value fallback (e.g. the model just replied `light`).
            bare = text.strip().strip('"').strip("'").lower()
            if bare in ("none", "light"):
                strategy = bare

        if strategy is None:
            # Last resort: look for the literal words anywhere in the response.
            lowered = text.lower()
            if '"strategy"' in lowered or "'strategy'" in lowered:
                match = re.search(r"strategy['\"]?\s*[:=]\s*['\"]?(none|light)", lowered)
                if match:
                    strategy = match.group(1)

        if strategy is None:
            raise ValueError(f"could not parse a 'none'/'light' strategy from response: {raw_response!r}")

        return {"strategy": strategy, "needs_broad_context": needs_broad_context}

    def _generate_hyde_document(self, query: str) -> str:
        """HyDE (Hypothetical Document Embeddings, Gao et al. arXiv:2212.10496): generate
        a short hypothetical passage that would answer `query`, in the style of a factual
        reference document, via a single cheap LLM call — then embed THAT hypothetical
        document for the dense-search leg instead of the literal query (see retrieve()'s
        call site, gated on retrieval.hyde_enabled, OFF by default).

        Why: this project's corpus (NFCorpus) pairs lay-phrased consumer-health queries
        against formal clinical/scientific document language — a vocabulary mismatch that
        dense query-embedding similarity alone often under-serves, since the literal query
        and the document that should match it can use almost entirely different wording
        for the same concept even with hybrid search + cross-encoder reranking already in
        place (see docs/ARCHITECTURE.md §4.5). A generated hypothetical ANSWER passage is
        written in the same register as the target documents, so embedding it (instead of
        the literal question) should land closer to genuinely relevant passages in dense
        vector space.

        Scope — dense leg ONLY: the sparse/BM25 leg in retrieve() deliberately keeps
        embedding the literal `query` (HyDE is specifically a dense-embedding technique;
        literal lexical/BM25 matching should still match the user's actual terms, not a
        generated document's), and the final cross-encoder rerank step scores candidates
        against the literal query text (it scores `(query, child_text)` pairs, i.e. the
        real retrieved passage, not the hypothetical one) — neither is touched here.

        Provider dispatch reuses _call_llm_provider(), the exact same
        ollama/neuralwatt/claude dispatcher _classify_retrieval_need() uses, and the same
        "unknown provider raises ValueError immediately, any call/parse/network error on a
        valid provider fails open" convention. Deliberately kept as a SEPARATE LLM call
        from _classify_retrieval_need() rather than merged into one combined call, even
        though both can run for the same query when both features are enabled: the two
        features are independently toggleable (hyde_enabled / adaptive_retrieval_enabled,
        both off by default) and serve different purposes (a routing decision vs. a query
        rewrite for embedding). Combining them into one call would couple two features
        that each need to keep working correctly when only ONE of them is turned on — a
        prompt/parsing change made for one feature could silently break the other, and the
        combined call would need to run (and pay for) both prompts' worth of reasoning
        even when only one feature is enabled for a given config. If both are ever enabled
        together in practice and the extra latency of two sequential cheap LLM calls turns
        out to matter, revisit combining them as a targeted optimization then — not
        pre-emptively here, before either feature has been validated to help at all.

        Returns: the generated hypothetical document text (or the original `query`
        unchanged on any failure — see below).

        Fails open: any call/parse/provider error returns the ORIGINAL `query` unchanged,
        so the dense-embedding step in retrieve() just behaves as if HyDE were off for
        this one query — this must never fail the whole retrieve() call.

        Raises:
            ValueError: if retrieval.hyde_provider is not one of "ollama" | "neuralwatt" |
                "claude" — a misconfigured provider name is a config error to fail loud on
                immediately, matching _classify_retrieval_need()'s identical convention.
        """
        retrieval_cfg = self.config.get("retrieval", {})
        provider = retrieval_cfg.get("hyde_provider", "ollama")
        model_name = retrieval_cfg.get("hyde_model", "glm-5.2:cloud")

        if provider not in ("ollama", "neuralwatt", "claude"):
            raise ValueError(
                f"Unknown retrieval.hyde_provider {provider!r} — must be one of "
                "'ollama', 'neuralwatt', or 'claude'."
            )

        prompt = (
            "Write a short, plausible passage that would answer this question, in the "
            "style of a factual reference document (e.g. a clinical/scientific reference "
            "passage). Write ONLY the passage itself — 2 to 4 sentences, no preamble, no "
            "mention that this is hypothetical or that you are an AI.\n\n"
            f"QUESTION: {query}"
        )

        try:
            hyde_doc = self._call_llm_provider(provider, model_name, prompt)
            hyde_doc = (hyde_doc or "").strip()
            if not hyde_doc:
                raise ValueError("HyDE call returned an empty document")
            return hyde_doc
        except Exception as exc:
            print(
                f"WARNING: HyDE hypothetical-document generation failed ({exc}) — falling "
                "back to the literal query for the dense-embedding leg."
            )
            return query

    def _apply_mmr(self, candidates: List[Dict[str, Any]], query_embedding: List[float],
                    lambda_param: float, top_n: int) -> List[Dict[str, Any]]:
        """MMR (Maximal Marginal Relevance) diversity-aware selection pass
        (Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering
        Documents and Producing Summaries," SIGIR '98, https://dl.acm.org/doi/10.1145/290941.291025;
        see also the 2025 RAG-specific diversity study, arXiv:2502.09017).

        Positioned in EfficientRAG.retrieve() AFTER cross-encoder reranking (which still
        does the initial relevance-based candidate ordering) and BEFORE parent-chunk
        expansion / CRAG context review — deliberately, so MMR operates on the small,
        still-child-chunk-scale `rerank_top_n` candidate list, not on already-expanded
        parent text. See docs/RESEARCH_NOTES.md §4 and docs/ARCHITECTURE.md §4.7 for the
        full rationale: this project's evaluation corpus is specifically constructed as
        gold documents + randomly-sampled same-topic distractor documents, exactly the
        shape where near-duplicate/overlapping chunks can crowd out unique information
        under a relevance-only reranker.

        Standard iterative MMR: repeatedly picks the remaining candidate maximizing

            lambda_param * relevance_to_query - (1 - lambda_param) * max_similarity_to_selected

        where both similarities are cosine similarity over dense embeddings.
        `relevance_to_query` is each candidate's cosine similarity to `query_embedding`;
        `max_similarity_to_selected` is a candidate's highest cosine similarity to any
        candidate already picked (0.0 for the first pick, since nothing is selected yet).

        Reuses each candidate's existing dense embedding — passed in on the candidate
        dict's "vector" key, itself populated in retrieve() from the same
        qdrant.query_points(..., with_vectors=["dense"]) call already made for the
        hybrid/dense search step (gated on retrieval.mmr_enabled so no extra vector
        payload is pulled when this feature is off). This method deliberately does NOT
        re-embed anything — no new embedding round trip. A candidate with no retained
        vector (e.g. a point ingested before vectors were retained here, or a lookup
        gap) is treated as similarity 0.0 to everything, which degrades that one
        candidate's relative preference slightly rather than raising.

        `lambda_param` closer to 1.0 weights pure relevance — at exactly 1.0 the
        diversity term is always multiplied by 0, so this degenerates to the input's
        original relevance order (a sanity check exercised directly in this feature's
        unit test, see docs/CHANGELOG.md). Closer to 0.0 weights pure diversity. 0.6 is
        this project's chosen default (config key retrieval.mmr_lambda) — a
        relevance-leaning middle of the standard 0.5-0.7 RAG range, pending empirical
        tuning against this project's own NFCorpus harness (UNVALIDATED — see
        docs/CHANGELOG.md's dated MMR entry).

        Args:
            candidates: hit dicts, each expected to carry an "id" key and a "vector" key
                (a dense embedding as a list of floats, or None if unavailable).
            query_embedding: the dense query embedding already computed by retrieve()
                (the literal query's embedding, or the HyDE-substituted one if
                retrieval.hyde_enabled is also on — whichever embedding actually drove
                the dense search leg for this call).
            lambda_param: relevance/diversity trade-off, in [0.0, 1.0].
            top_n: how many candidates to select, in MMR order. Passing
                len(candidates) (retrieve()'s current call pattern) reorders the full
                candidate list for diversity without dropping any; a smaller value would
                additionally truncate.

        Returns:
            A list of up to top_n candidate dicts from `candidates`, in MMR-selected
            order. Returns `candidates` unchanged if it's empty, and `[]` if
            top_n <= 0.
        """
        if not candidates:
            return candidates
        if top_n <= 0:
            return []

        def _cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
            if a is None or b is None:
                return 0.0
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)

        relevance = [_cosine(query_embedding, c.get("vector")) for c in candidates]

        selected_idx: List[int] = []
        remaining_idx = list(range(len(candidates)))

        while remaining_idx and len(selected_idx) < top_n:
            best_i = None
            best_mmr_score = None
            for i in remaining_idx:
                if selected_idx:
                    max_sim_to_selected = max(
                        _cosine(candidates[i].get("vector"), candidates[j].get("vector"))
                        for j in selected_idx
                    )
                else:
                    max_sim_to_selected = 0.0
                mmr_score = lambda_param * relevance[i] - (1 - lambda_param) * max_sim_to_selected
                if best_mmr_score is None or mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_i = i
            selected_idx.append(best_i)
            remaining_idx.remove(best_i)

        return [candidates[i] for i in selected_idx]

    def retrieve(self, query: str, collection: str, top_k: int = 8,
                 filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Retrieve with hybrid (dense + sparse RRF fusion) search, optional
        cross-encoder reranking, and quantization awareness.

        See retrieval.adaptive_skip_none_strategy (config.yaml / load_config() default
        False) for an isolation flag that suppresses the Adaptive-RAG-lite gate's
        "none" -> skip-retrieval short-circuit (without disabling the classification
        itself) — added specifically to let adaptive_parent_expansion_enabled be A/B
        tested independent of that unrelated gate's ~23-25% misfire rate on this
        project's title-style NFCorpus testset. See docs/CHANGELOG.md's 2026-07-20
        entry for the full confound writeup.
        """
        retrieval_cfg = self.config.get("retrieval", {})
        adaptive_enabled = retrieval_cfg.get("adaptive_retrieval_enabled", False)

        if adaptive_enabled:
            classification = self._classify_retrieval_need(query)
            strategy = classification["strategy"]
            # Always record the RAW classifier verdict, regardless of
            # adaptive_skip_none_strategy below — this attribute is the audit trail of
            # "what did the classifier actually say", never rewritten to "light" even
            # when the skip behavior driven by a "none" verdict is being suppressed.
            self._last_retrieval_strategy = strategy
            # Sibling decision from the SAME classification call — see
            # _classify_retrieval_need()'s docstring and the fetch_parents block
            # below for how this is used.
            self._last_needs_broad_context = classification.get("needs_broad_context", False)

            # Isolation flag (config.yaml/load_config() default: False — see those
            # comments and docs/CHANGELOG.md's 2026-07-20 entry) for testing
            # adaptive_parent_expansion_enabled WITHOUT the unrelated "none"-strategy
            # skip-retrieval gate confounding the result. That confound: on this
            # project's title-style NFCorpus testset (bare document titles like
            # "leeks"/"serotonin"/"veal", not natural questions), the classifier judges
            # bare nouns as "general knowledge, no retrieval needed" ~23-25% of the
            # time, which forces RAGAS faithfulness to 0 for those rows (no context ==
            # nothing to be faithful to) — swamping any real parent-expansion effect.
            # The classification itself (strategy AND needs_broad_context) still ALWAYS
            # runs and is still recorded above; this flag ONLY controls whether a
            # "none" verdict is allowed to short-circuit retrieve() to an empty result.
            adaptive_skip_none_strategy = retrieval_cfg.get("adaptive_skip_none_strategy", False)

            if strategy == "none" and not adaptive_skip_none_strategy:
                return []

            # strategy == "light" (or "none" with adaptive_skip_none_strategy=True,
            # treated as "light" for retrieval-EXECUTION purposes only — the raw "none"
            # verdict remains visible via self._last_retrieval_strategy above):
            # override top_k/oversampling for THIS call only, via a shallow-copied
            # config so the instance's persistent config (self.config) is never
            # mutated by a single retrieve() call.
            top_k = retrieval_cfg.get("adaptive_light_top_k", 8)
            light_oversampling = retrieval_cfg.get("adaptive_light_oversampling", 3.0)
            retrieval_cfg = dict(retrieval_cfg)
            retrieval_cfg["oversampling"] = light_oversampling
        else:
            self._last_retrieval_strategy = None
            self._last_needs_broad_context = None

        full_coll = self.get_collection_name(collection)

        # HyDE (Hypothetical Document Embeddings, Gao et al. arXiv:2212.10496) — gated on
        # retrieval.hyde_enabled (OFF by default). When enabled, embed a generated
        # hypothetical answer document instead of the literal query for the DENSE leg
        # only; the sparse/BM25 leg below still embeds the literal `query` (see
        # _generate_hyde_document()'s docstring for why), and reranking further down
        # still scores candidates against the literal `query` text. Fails open to the
        # literal query on any generation error, so this is a no-op when disabled or on
        # failure — never a behavior change to the sparse leg or reranking.
        hyde_enabled = retrieval_cfg.get("hyde_enabled", False)
        dense_query_text = self._generate_hyde_document(query) if hyde_enabled else query
        dense_vec = self.embed_model.encode(dense_query_text, normalize_embeddings=True).tolist()

        # Basic filter support. Always exclude parent-chunk points from
        # normal search — they carry a zero dummy vector and are only meant
        # to be looked up by parent_id for context expansion below, never
        # returned directly as a search hit.
        conditions = [FieldCondition(key="chunk_type", match=MatchValue(value="child"))]
        if filters:
            for k, v in filters.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
        q_filter = Filter(must=conditions)

        oversampling = retrieval_cfg.get("oversampling", 3.0)
        candidate_pool = retrieval_cfg.get("rerank_candidate_pool", 40)
        rerank_active = self.rerank_enabled and self.reranker is not None

        # MMR (retrieval.mmr_enabled, OFF by default — see _apply_mmr()'s docstring)
        # needs each candidate's dense embedding to compute cosine similarity, without
        # re-embedding anything. Only request vectors from Qdrant when MMR is actually
        # enabled, so the default (off) path pulls exactly the same payload over the
        # wire as before this feature existed — no behavior or bandwidth change when
        # mmr_enabled is false.
        mmr_enabled = retrieval_cfg.get("mmr_enabled", False)
        dense_with_vectors = ["dense"] if mmr_enabled else False

        # When hybrid+rerank are both active, fetch a wider candidate pool for
        # the cross-encoder to work with. Otherwise fall back to the simpler
        # oversampled child limit used previously.
        fetch_limit = candidate_pool if rerank_active else int(top_k * oversampling)

        if self.hybrid_search_enabled:
            candidate_limit = fetch_limit
            sparse_vec = self._sparse_vector_for(query)

            dense_hits = self.qdrant.query_points(
                collection_name=full_coll,
                query=dense_vec,
                using="dense",
                limit=candidate_limit,
                query_filter=q_filter,
                with_payload=True,
                with_vectors=dense_with_vectors,
            ).points
            sparse_hits = self.qdrant.query_points(
                collection_name=full_coll,
                query=sparse_vec,
                using="sparse",
                limit=candidate_limit,
                query_filter=q_filter,
                with_payload=True,
                with_vectors=dense_with_vectors,
            ).points

            # Client-side RRF fusion (Qdrant's server-side FusionQuery has no
            # tunable k). For each leg, accumulate 1 / (k + rank) per point id,
            # summing across legs when a point appears in both.
            fused_scores: Dict[Any, float] = {}
            payload_by_id: Dict[Any, Any] = {}
            vector_by_id: Dict[Any, Any] = {}
            for leg in (dense_hits, sparse_hits):
                for rank, point in enumerate(leg, start=1):
                    fused_scores[point.id] = fused_scores.get(point.id, 0.0) + 1.0 / (self.rrf_k + rank)
                    if point.id not in payload_by_id:
                        payload_by_id[point.id] = point.payload
                    if mmr_enabled and point.id not in vector_by_id:
                        pvec = point.vector
                        dense_v = pvec.get("dense") if isinstance(pvec, dict) else pvec
                        if dense_v is not None:
                            vector_by_id[point.id] = dense_v

            ranked_ids = sorted(fused_scores.keys(), key=lambda pid: fused_scores[pid], reverse=True)
            hits = [
                {
                    "id": pid,
                    "score": fused_scores[pid],
                    "payload": payload_by_id.get(pid),
                    "vector": vector_by_id.get(pid),
                }
                for pid in ranked_ids[:candidate_limit]
            ]
        else:
            dense_hits = self.qdrant.query_points(
                collection_name=full_coll,
                query=dense_vec,
                using="dense",
                limit=fetch_limit,
                query_filter=q_filter,
                with_payload=True,
                with_vectors=dense_with_vectors,
            ).points
            hits = []
            for h in dense_hits:
                dense_v = None
                if mmr_enabled:
                    hvec = h.vector
                    dense_v = hvec.get("dense") if isinstance(hvec, dict) else hvec
                hits.append({"id": h.id, "score": h.score, "payload": h.payload, "vector": dense_v})

        # Rerank stage — operates on the fused/dense candidate hits (child
        # chunks), never on expanded parent text, to avoid wasting
        # cross-encoder calls on parent-sized text.
        if rerank_active and hits:
            pairs = [(query, (hit.get("payload") or {}).get("text", "")) for hit in hits]
            scores = self.reranker.predict(pairs)
            scored_hits = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
            rerank_top_n = self.config.get("retrieval", {}).get("rerank_top_n") or top_k
            top_hits = scored_hits[:rerank_top_n]
        else:
            top_hits = [(hit.get("score"), hit) for hit in hits[:top_k]]

        # MMR (Maximal Marginal Relevance, Carbonell & Goldstein 1998) diversity-aware
        # selection pass — gated on retrieval.mmr_enabled (OFF by default, UNVALIDATED).
        # Runs AFTER cross-encoder reranking (relevance ordering above) and BEFORE
        # parent-chunk expansion / CRAG context review below, operating on the small,
        # still-child-chunk-scale rerank_top_n candidates — see _apply_mmr()'s
        # docstring and docs/RESEARCH_NOTES.md §4 for the full rationale. Deliberately
        # kept as a plain on/off toggle independent of the Adaptive-RAG-lite
        # needs_broad_context gate (not coupled to it): the two features address
        # different problems (retrieval breadth vs. redundancy among already-selected
        # candidates), and this project's own convention (see the HyDE/adaptive-parent-
        # expansion entries in docs/CHANGELOG.md) is to ship each new lever as an
        # independently toggleable flag first, only wiring cross-feature interactions
        # later if a validated need for it emerges — not pre-emptively here.
        if mmr_enabled and top_hits:
            mmr_lambda = retrieval_cfg.get("mmr_lambda", 0.6)
            mmr_candidates = [hit for _score, hit in top_hits]
            score_by_id = {hit["id"]: score for score, hit in top_hits}
            reordered = self._apply_mmr(mmr_candidates, dense_vec, mmr_lambda, len(mmr_candidates))
            # Re-pair each reordered candidate with its ORIGINAL relevance/fusion score
            # (MMR's internal lambda-blended score is only used to pick the order/
            # selection; the score surfaced to callers stays the same rerank/fusion
            # score it always was, so this doesn't change the meaning of "score" in the
            # returned hits).
            top_hits = [(score_by_id.get(c["id"]), c) for c in reordered]

        formatted = []
        for score, r in top_hits:
            payload = r.get("payload") or {}
            child_text = payload.get("text", "")
            formatted.append({
                "score": round(float(score), 4) if score is not None else None,
                "text": child_text,
                "child_text": child_text,
                "source": payload.get("source"),
                "parent_id": payload.get("parent_id"),
                "summary": payload.get("summary"),
                "tags": payload.get("tags", []),
                "chunk_type": payload.get("chunk_type")
            })

        # Parent-chunk context expansion. When enabled, replace each hit's
        # "text" with the full parent chunk's text (looked up by its
        # deterministic point ID), keeping the original child chunk text
        # available under "child_text". Falls back gracefully to child-only
        # text on any lookup failure (e.g. points ingested before this
        # feature existed, or a parent point missing/deleted) — never raises.
        fetch_parents = self.config.get("retrieval", {}).get("fetch_parents", False)

        # Query-adaptive parent expansion (docs/RESEARCH_NOTES.md §6 /
        # docs/ARCHITECTURE.md §4.6): when the Adaptive-RAG-lite gate is enabled
        # AND fetch_parents is on, use THIS SAME query's needs_broad_context
        # verdict (from the one classification call already made above — no
        # extra LLM call) to decide whether to actually run the parent-chunk
        # lookup below, instead of always expanding whenever the static
        # fetch_parents config says to. Single-hop factoid queries (this
        # project's dominant NFCorpus style) keep their precise, un-diluted
        # child-chunk match; comparative/multi-part queries still get the
        # broader parent context they genuinely need. If adaptive_retrieval_enabled
        # is False, this whole sub-gate is inert and fetch_parents behaves
        # exactly as before (static, non-query-adaptive) — no behavior change
        # for existing callers who don't opt into the adaptive gate.
        #
        # adaptive_parent_expansion_enabled (new config key, default True) is a
        # SEPARATE toggle from adaptive_retrieval_enabled, deliberately: it lets
        # this specific sub-feature (query-adaptive parent expansion) be turned
        # off independently of the none/light retrieval-depth decision, without
        # touching adaptive_retrieval_enabled itself. This is for a follow-up A/B
        # validation task (see docs/CHANGELOG.md) that needs to isolate which
        # half of the gate — retrieval-depth routing vs. parent-expansion gating
        # — is responsible for any measured effect; a single combined flag would
        # make that isolation impossible without a code change. Defaults to True
        # so the new query-adaptive behavior is what actually ships as soon as
        # adaptive_retrieval_enabled + fetch_parents are both true (that IS the
        # fix being delivered here); setting it False reverts to the old
        # "always expand whenever fetch_parents is true" behavior even with the
        # gate on, for isolation experiments.
        adaptive_parent_expansion_enabled = retrieval_cfg.get("adaptive_parent_expansion_enabled", True)
        if adaptive_enabled and fetch_parents and adaptive_parent_expansion_enabled:
            do_expand_parents = bool(self._last_needs_broad_context)
        else:
            do_expand_parents = fetch_parents

        if do_expand_parents and formatted:
            parent_ids = {f["parent_id"] for f in formatted if f.get("parent_id")}
            if parent_ids:
                try:
                    id_map = {pid: self._parent_point_id(pid) for pid in parent_ids}
                    parent_records = self.qdrant.retrieve(
                        collection_name=full_coll,
                        ids=list(id_map.values()),
                        with_payload=True,
                    )
                    parent_text_by_id = {}
                    for rec in parent_records:
                        rec_payload = rec.payload or {}
                        rec_parent_id = rec_payload.get("parent_id")
                        rec_text = rec_payload.get("text")
                        if rec_parent_id and rec_text:
                            parent_text_by_id[rec_parent_id] = rec_text
                except Exception as exc:
                    print(f"WARNING: parent-chunk lookup failed ({exc}) — falling back to child text only")
                    parent_text_by_id = {}

                for f in formatted:
                    pid = f.get("parent_id")
                    parent_text = parent_text_by_id.get(pid) if pid else None
                    if parent_text:
                        f["text"] = parent_text

        # Optional CRAG-style context review/distillation pass — OFF by
        # default (retrieval.context_review_enabled). Runs LAST, after
        # reranking and parent-chunk expansion, on the final hits that would
        # otherwise be handed straight to generation. See
        # review_and_distill_context()'s docstring for what this does and
        # (importantly) does NOT fix.
        context_review_enabled = self.config.get("retrieval", {}).get("context_review_enabled", False)
        if context_review_enabled and formatted:
            formatted = review_and_distill_context(query, formatted, self.config)

        return formatted

    def get_stats(self, collection: str) -> Dict:
        full_coll = self.get_collection_name(collection)
        try:
            info = self.qdrant.get_collection(full_coll)
            return {
                "collection": collection,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except:
            return {"error": "Collection not found"}


def _split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter used only by review_and_distill_context()'s
    CRAG-style "decompose" step (arXiv:2401.15884). Deliberately a simple
    regex heuristic — not a full NLP sentence tokenizer (spaCy/nltk) — since
    this project already prefers simple text heuristics for boundaries (see
    EfficientRAG._generate_summary()'s `text.split(". ")` placeholder) and no
    new heavy NLP dependency was worth adding just for this. Splits on
    sentence-ending punctuation followed by whitespace, with a short common-
    abbreviation guard (Dr., Mr., e.g., etc., Fig., No., approx., vs.) so
    "Dr. Smith found..." isn't split mid-name. Not perfect (no heuristic
    without a real tokenizer is), but good enough for per-segment relevance
    scoring, where an occasional over/under-split just shifts sentence
    boundaries slightly rather than breaking anything.
    """
    if not text or not text.strip():
        return []

    abbrev_pattern = re.compile(
        r'\b(Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e|Fig|No|approx)\.\s'
    )
    # Temporarily replace the space after a guarded abbreviation with a NUL
    # placeholder so the sentence-boundary regex below doesn't split there,
    # then restore it afterward.
    guarded = abbrev_pattern.sub(lambda m: m.group(0)[:-1] + "\x00", text)
    raw_sentences = re.split(r'(?<=[.!?])\s+', guarded)
    sentences = [s.replace("\x00", " ").strip() for s in raw_sentences]
    return [s for s in sentences if s]


def _score_sentence_relevance(query: str, sentences: List[str], client, model: str) -> List[float]:
    """Score every sentence in `sentences` against `query` in ONE batched LLM
    call (not one call per sentence — see review_and_distill_context()'s
    docstring for why that matters for cost/latency), returning a same-length
    list of floats in [0.0, 1.0].

    On any call/parse failure, fails open: returns all 1.0s (keep everything)
    for this hit rather than raising or silently discarding content, since a
    review-step glitch should degrade to "context review did nothing this
    time," never to "context review silently deleted real content."
    """
    if not sentences:
        return []

    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    prompt = (
        "You are scoring how relevant each numbered sentence below is to a search query, as "
        "part of a retrieval-augmented generation pipeline's context-cleaning step (CRAG-style "
        "decompose-then-recompose, arXiv:2401.15884).\n\n"
        f"QUERY: {query}\n\n"
        f"SENTENCES:\n{numbered}\n\n"
        "For EACH sentence, output a relevance score from 0.0 (irrelevant / off-topic noise) to "
        "1.0 (directly answers or strongly supports answering the query). Respond with ONLY a "
        f"JSON array of exactly {len(sentences)} numbers, one per sentence in order, e.g. "
        "[0.9, 0.1, 0.4]. No other text, no markdown fencing."
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = next((b.text for b in response.content if b.type == "text"), "").strip()
        # Tolerate a model wrapping the array in a ```json fence despite being
        # asked not to.
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("["):raw.rfind("]") + 1]
        scores = json.loads(raw)
        if not isinstance(scores, list) or len(scores) != len(sentences):
            raise ValueError(f"expected a {len(sentences)}-element JSON array, got: {raw[:200]!r}")
        return [max(0.0, min(1.0, float(s))) for s in scores]
    except Exception as exc:
        print(
            f"WARNING: context-review relevance scoring failed ({exc}) — keeping all "
            "sentences for this hit unchanged"
        )
        return [1.0] * len(sentences)


def review_and_distill_context(query: str, hits: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Optional CRAG-style ("Corrective RAG", arXiv:2401.15884) "decompose-
    then-recompose" context review pass, run between EfficientRAG.retrieve()'s
    final reranked/parent-expanded hits and generation (see
    EfficientRAG.retrieve()'s call site, gated on
    retrieval.context_review_enabled — OFF by default).

    For each hit: splits its "text" into sentences (_split_into_sentences()),
    scores every sentence's relevance to `query` in one batched LLM call per
    hit (_score_sentence_relevance() — NOT one call per sentence, to keep
    this practical), discards sentences scoring below
    retrieval.context_review_threshold (default 0.3, see load_config()), and
    recomposes the surviving sentences back into a denser "text" field.
    Every other field on the hit (score, source, parent_id, chunk_type, tags,
    summary, child_text, ...) is passed through unchanged — only "text" is
    ever modified.

    Uses a cheap/fast model (Haiku-class) rather than the project's main
    generation model, since this runs once per retrieved hit on every query
    and needs to be fast/cheap, not high-reasoning: relevance scoring of a
    handful of sentences against a query is a much easier task than answer
    generation. Reuses evaluate_ragas.py's `_build_anthropic_client()` for
    client/credential selection (direct ANTHROPIC_API_KEY, else Claude on
    Vertex AI) rather than duplicating that logic here — imported lazily
    (inside this function) to avoid a circular import at module load time,
    since evaluate_ragas.py itself imports `load_config` from this module
    (utils.py) inside its own functions for the same reason.

    IMPORTANT SCOPE NOTE — read before assuming this fixes faithfulness:
    this step targets context_precision (removing noisy/irrelevant retrieved
    sentences before they reach the prompt), and may help context_recall/
    answer_relevancy indirectly by producing a denser, more focused context.
    It is explicitly NOT expected to move the faithfulness metric, and
    faithfulness is a DIFFERENT, already-diagnosed problem in this project:
    generate_answer()'s attribution-tagged blending policy (see that
    function's docstring) deliberately allows the final generator to draw on
    general knowledge ([G]-tagged) when context is thin, which is what
    suppresses faithfulness — not noise in the retrieved chunks. This was
    checked against a real ablation in a separate paper (Self-Correcting RAG,
    arXiv:2604.10734), which found that context-side cleaning alone left
    faithfulness flat. Do not treat an unchanged faithfulness score after
    enabling this feature as "it didn't work" — check context_precision (and
    secondarily context_recall/answer_relevancy) instead; that is the axis
    this feature actually targets.

    Args:
        query: the user's original query (used for relevance scoring).
        hits: EfficientRAG.retrieve()'s final formatted hit list (each a
            dict with at least a "text" key).
        config: the full RAG config dict (only config["retrieval"]
            ["context_review_threshold"] is read here).

    Returns:
        A new list of hit dicts (same shape/length as `hits`), with "text"
        replaced by the recomposed, distilled sentences for hits where at
        least one sentence survived the threshold. If no Anthropic access is
        configured, or a given hit's text yields zero sentences, that hit is
        passed through completely unchanged (fails open, never raises).
    """
    threshold = config.get("retrieval", {}).get("context_review_threshold", 0.3)

    # Lazy import to avoid a circular import at module load time (see
    # docstring above) — evaluate_ragas.py lives alongside this file in
    # scripts/, so make sure that directory is on sys.path before importing
    # it (mirrors the sys.path.append(...) pattern retrieve.py/
    # evaluate_ragas.py already use for cross-imports within scripts/).
    import sys
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from evaluate_ragas import _build_anthropic_client

    client, model, mode = _build_anthropic_client()
    if client is None:
        print(
            "WARNING: retrieval.context_review_enabled is set but no Anthropic access is "
            "configured (ANTHROPIC_API_KEY / ANTHROPIC_VERTEX_PROJECT_ID) — skipping context "
            "review, returning hits unchanged."
        )
        return hits

    # Always use a cheap/fast (Haiku-class) model for this step regardless of
    # which model _build_anthropic_client() picked for full answer
    # generation — per-sentence relevance scoring runs once per retrieved hit
    # on every query and doesn't need a high-reasoning model.
    if mode == "direct":
        review_model = "claude-haiku-4-5"
    else:
        review_model = os.environ.get("ANTHROPIC_VERTEX_HAIKU_MODEL", "claude-haiku-4-5@20251001")

    reviewed = []
    for hit in hits:
        original_text = hit.get("text", "")
        sentences = _split_into_sentences(original_text)
        if not sentences:
            reviewed.append(hit)
            continue

        scores = _score_sentence_relevance(query, sentences, client, review_model)
        kept = [s for s, sc in zip(sentences, scores) if sc >= threshold]

        new_hit = dict(hit)
        if kept:
            new_hit["text"] = " ".join(kept)
        # If NOTHING survives the threshold, keep the original text rather
        # than emptying the hit: a hit that made it through retrieval +
        # reranking + parent expansion but scores zero sentences above
        # threshold is more likely a miscalibrated threshold or a scoring
        # hiccup than "truly nothing in here is relevant" — silently
        # emptying it would drop a citeable source from the answer entirely.
        reviewed.append(new_hit)

    return reviewed


# Example usage (synthetic for testing)
if __name__ == "__main__":
    config = load_config()
    rag = EfficientRAG(config)
    print("EfficientRAG initialized successfully (light mode with binary quantization + hierarchy).")
