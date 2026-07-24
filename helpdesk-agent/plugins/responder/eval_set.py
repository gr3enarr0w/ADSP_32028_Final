"""Build and load labeled query–document pairs for retrieval fusion tuning.

Production eval sets should be 200–500 labeled pairs (Elastic calibrates fusion
weights on ~300 queries; our threshold work targets 200+). Pairs are auto-labeled
from the responder corpus in SQLite:

  - tickets:     summary or description variant → (tickets, ticket_key)
  - kb_articles: title → (kb_articles, page_id)
  - faq_sources: title → (faq_sources, source_id)
  - atlassian_docs: title → (atlassian_docs, url)

Keyword vs semantic query types are inferred heuristically (title/summary vs
description paraphrase) to preserve coverage in sampled eval sets.

Persist with:  python -m scripts.build_retrieval_eval
Load via:      get_eval_queries() in retrieval.tune_fusion()
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from db import get_db_conn

log = logging.getLogger(__name__)

QueryExample = tuple[str, str, str]
QueryType = Literal["keyword", "semantic"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_PATH = _PROJECT_ROOT / "data" / "retrieval_eval_queries.json"

# Industry rule-of-thumb for fusion weight calibration (see docs/modeling-architecture.md).
PRODUCTION_MIN_QUERIES = 200
PRODUCTION_TARGET_QUERIES = 300
TEST_MIN_QUERIES = 25

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Hand-curated seed pairs — always merged; covers edge cases auto-labeling misses.
SEED_EVAL_QUERIES: list[QueryExample] = [
    ("oauth 2lo", "faq_sources", "FAQ-1"),
    ("2LO authentication", "faq_sources", "FAQ-1"),
    ("<PROJECT_KEY> servicedeskapi", "faq_sources", "FAQ-2"),
    ("OAuth service account", "tickets", "ANTSE-4"),
    (
        "JIRA workflow scheme",
        "atlassian_docs",
        "https://support.atlassian.com/jira-cloud/docs/manage-workflows/",
    ),
    ("password reset", "faq_sources", "FAQ-3"),
    ("can't log in", "tickets", "ANTSE-5"),
    ("need someone else to see my project", "kb_articles", "KB-4"),
    (
        "rest api authentication",
        "atlassian_docs",
        "https://developer.atlassian.com/cloud/jira/platform/rest/v3/authentication/",
    ),
]


@dataclass(frozen=True, slots=True)
class LabeledQuery:
    query: str
    source_type: str
    doc_id: str
    query_type: QueryType = "keyword"
    label_source: str = "manual"

    def as_tuple(self) -> QueryExample:
        return (self.query, self.source_type, self.doc_id)


@dataclass(frozen=True, slots=True)
class EvalSetManifest:
    version: int
    built_at: str
    count: int
    query_type_counts: dict[str, int]
    source_type_counts: dict[str, int]
    queries: tuple[LabeledQuery, ...]

    def to_json(self) -> str:
        payload = {
            "version": self.version,
            "built_at": self.built_at,
            "count": self.count,
            "query_type_counts": self.query_type_counts,
            "source_type_counts": self.source_type_counts,
            "queries": [asdict(q) for q in self.queries],
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "EvalSetManifest":
        data = json.loads(raw)
        queries = tuple(LabeledQuery(**item) for item in data["queries"])
        return cls(
            version=int(data.get("version", 1)),
            built_at=str(data.get("built_at", "")),
            count=int(data.get("count", len(queries))),
            query_type_counts=dict(data.get("query_type_counts", {})),
            source_type_counts=dict(data.get("source_type_counts", {})),
            queries=queries,
        )


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _first_line(text: str, max_len: int = 220) -> str:
    line = (text or "").strip().split("\n")[0].strip()
    return line[:max_len]


def _join_text(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _dedupe_queries(items: list[LabeledQuery]) -> list[LabeledQuery]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[LabeledQuery] = []
    for item in items:
        key = (item.query.strip().lower(), item.source_type, item.doc_id)
        if not item.query.strip() or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _ticket_candidates(conn) -> list[LabeledQuery]:
    rows = conn.execute(
        """
        SELECT ticket_key, summary, description, resolution
        FROM tickets
        WHERE LOWER(COALESCE(status, '')) IN ('resolved', 'closed')
          AND resolution IS NOT NULL
          AND TRIM(resolution) != ''
          AND summary IS NOT NULL
          AND TRIM(summary) != ''
        ORDER BY resolved_at DESC, updated_at DESC, ticket_key DESC
        """
    ).fetchall()

    out: list[LabeledQuery] = []
    for row in rows:
        summary = str(row["summary"]).strip()
        doc_id = str(row["ticket_key"])
        if len(summary) < 8:
            continue

        out.append(
            LabeledQuery(
                query=summary,
                source_type="tickets",
                doc_id=doc_id,
                query_type="keyword",
                label_source="ticket_summary",
            )
        )

        description = _first_line(str(row["description"] or ""))
        if len(description) >= 15 and _jaccard(description, summary) < 0.65:
            out.append(
                LabeledQuery(
                    query=description,
                    source_type="tickets",
                    doc_id=doc_id,
                    query_type="semantic",
                    label_source="ticket_description",
                )
            )

        resolution = _first_line(str(row["resolution"] or ""))
        if len(resolution) >= 20 and _jaccard(resolution, summary) < 0.55:
            out.append(
                LabeledQuery(
                    query=resolution,
                    source_type="tickets",
                    doc_id=doc_id,
                    query_type="semantic",
                    label_source="ticket_resolution_excerpt",
                )
            )

    return out


def _kb_candidates(conn) -> list[LabeledQuery]:
    rows = conn.execute(
        """
        SELECT page_id, title, body_text
        FROM kb_articles
        WHERE title IS NOT NULL AND TRIM(title) != ''
        ORDER BY title
        """
    ).fetchall()
    out: list[LabeledQuery] = []
    for row in rows:
        title = str(row["title"]).strip()
        out.append(
            LabeledQuery(
                query=title,
                source_type="kb_articles",
                doc_id=str(row["page_id"]),
                query_type="keyword",
                label_source="kb_title",
            )
        )
        body = _first_line(str(row["body_text"] or ""))
        if len(body) >= 20 and _jaccard(body, title) < 0.6:
            out.append(
                LabeledQuery(
                    query=body,
                    source_type="kb_articles",
                    doc_id=str(row["page_id"]),
                    query_type="semantic",
                    label_source="kb_body_lead",
                )
            )
    return out


def _faq_candidates(conn) -> list[LabeledQuery]:
    rows = conn.execute(
        """
        SELECT source_id, COALESCE(title, source_id, '') AS title
        FROM faq_sources
        ORDER BY source_type, title
        """
    ).fetchall()
    return [
        LabeledQuery(
            query=str(row["title"]).strip(),
            source_type="faq_sources",
            doc_id=str(row["source_id"]),
            query_type="keyword",
            label_source="faq_title",
        )
        for row in rows
        if str(row["title"]).strip()
    ]


def _atlassian_doc_candidates(conn) -> list[LabeledQuery]:
    rows = conn.execute(
        """
        SELECT url, title, product
        FROM atlassian_docs
        ORDER BY product, title
        """
    ).fetchall()
    out: list[LabeledQuery] = []
    for row in rows:
        title = str(row["title"]).strip()
        url = str(row["url"])
        out.append(
            LabeledQuery(
                query=title,
                source_type="atlassian_docs",
                doc_id=url,
                query_type="keyword",
                label_source="doc_title",
            )
        )
        product = str(row["product"] or "").replace("-", " ")
        if product:
            out.append(
                LabeledQuery(
                    query=_join_text(title, product),
                    source_type="atlassian_docs",
                    doc_id=url,
                    query_type="semantic",
                    label_source="doc_title_product",
                )
            )
    return out


def _source_quota(max_queries: int) -> dict[str, int]:
    """Helpdesk-heavy mix: tickets dominate, docs/KB fill semantic gaps."""
    ratios = {
        "tickets": 0.55,
        "kb_articles": 0.20,
        "faq_sources": 0.10,
        "atlassian_docs": 0.15,
    }
    quotas = {source: int(max_queries * ratio) for source, ratio in ratios.items()}
    remainder = max_queries - sum(quotas.values())
    quotas["tickets"] += remainder
    return quotas


def _sample_stratified(candidates: list[LabeledQuery], max_queries: int) -> list[LabeledQuery]:
    """Sample toward source quotas while preserving keyword/semantic mix."""
    if len(candidates) <= max_queries:
        return candidates

    by_source: dict[str, list[LabeledQuery]] = {}
    for item in candidates:
        by_source.setdefault(item.source_type, []).append(item)

    quotas = _source_quota(max_queries)
    selected: list[LabeledQuery] = []

    for source_type, quota in quotas.items():
        pool = by_source.get(source_type, [])
        if not pool:
            continue
        keywords = [q for q in pool if q.query_type == "keyword"]
        semantics = [q for q in pool if q.query_type == "semantic"]
        target_sem = max(1, quota // 3)
        target_kw = quota - target_sem
        picked = keywords[:target_kw] + semantics[:target_sem]
        if len(picked) < quota:
            remaining = [q for q in pool if q not in picked]
            picked.extend(remaining[: quota - len(picked)])
        selected.extend(picked[:quota])

    if len(selected) < max_queries:
        selected_keys = {(q.query, q.source_type, q.doc_id) for q in selected}
        for item in candidates:
            key = (item.query, item.source_type, item.doc_id)
            if key in selected_keys:
                continue
            selected.append(item)
            if len(selected) >= max_queries:
                break

    return selected[:max_queries]


def build_eval_queries_from_db(
    db=None,
    *,
    max_queries: int = PRODUCTION_TARGET_QUERIES,
    include_seeds: bool = True,
) -> list[LabeledQuery]:
    """Mine auto-labeled pairs from the responder corpus tables."""
    if db is None:
        conn_cm = get_db_conn()
    elif hasattr(db, "execute"):
        conn_cm = nullcontext(db)
    elif hasattr(db, "get_db_conn"):
        conn_cm = db.get_db_conn()
    else:
        raise TypeError("db must be a connection or expose get_db_conn()")

    with conn_cm as conn:
        candidates: list[LabeledQuery] = []
        candidates.extend(_ticket_candidates(conn))
        candidates.extend(_kb_candidates(conn))
        candidates.extend(_faq_candidates(conn))
        candidates.extend(_atlassian_doc_candidates(conn))

    candidates = _dedupe_queries(candidates)

    if include_seeds:
        seed_candidates = [
            LabeledQuery(
                query=query,
                source_type=source_type,
                doc_id=doc_id,
                query_type="keyword",
                label_source="seed",
            )
            for query, source_type, doc_id in SEED_EVAL_QUERIES
        ]
        candidates = seed_candidates + candidates
        candidates = _dedupe_queries(candidates)

    sampled = _sample_stratified(candidates, max_queries)
    log.info(
        "[eval_set] built %d queries from %d candidates (max=%d)",
        len(sampled),
        len(candidates),
        max_queries,
    )
    return sampled


def build_eval_manifest(
    db=None,
    *,
    max_queries: int = PRODUCTION_TARGET_QUERIES,
    include_seeds: bool = True,
) -> EvalSetManifest:
    queries = tuple(build_eval_queries_from_db(db, max_queries=max_queries, include_seeds=include_seeds))
    return EvalSetManifest(
        version=1,
        built_at=datetime.now(timezone.utc).isoformat(),
        count=len(queries),
        query_type_counts=dict(Counter(q.query_type for q in queries)),
        source_type_counts=dict(Counter(q.source_type for q in queries)),
        queries=queries,
    )


def save_eval_manifest(manifest: EvalSetManifest, path: str | Path | None = None) -> Path:
    out_path = Path(path or DEFAULT_EVAL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.to_json())
    log.info("[eval_set] wrote %d queries to %s", manifest.count, out_path)
    return out_path


def load_eval_manifest(path: str | Path | None = None) -> EvalSetManifest | None:
    eval_path = Path(path or DEFAULT_EVAL_PATH)
    if not eval_path.exists():
        return None
    return EvalSetManifest.from_json(eval_path.read_text())


def get_eval_queries(
    *,
    min_queries: int = PRODUCTION_MIN_QUERIES,
    max_queries: int = PRODUCTION_TARGET_QUERIES,
    path: str | Path | None = None,
    db=None,
    prefer_file: bool = True,
) -> list[QueryExample]:
    """Load production eval queries, building from DB when the file is too small."""
    manifest: EvalSetManifest | None = None
    if prefer_file:
        manifest = load_eval_manifest(path)

    if manifest is None or manifest.count < min_queries:
        built = build_eval_queries_from_db(db, max_queries=max_queries, include_seeds=True)
        if manifest is None or len(built) > manifest.count:
            manifest = EvalSetManifest(
                version=1,
                built_at=datetime.now(timezone.utc).isoformat(),
                count=len(built),
                query_type_counts=dict(Counter(q.query_type for q in built)),
                source_type_counts=dict(Counter(q.source_type for q in built)),
                queries=tuple(built),
            )

    if manifest.count < min_queries:
        log.warning(
            "[eval_set] only %d labeled queries available (min=%d); "
            "run `python -m scripts.build_retrieval_eval` after ingesting more corpus data",
            manifest.count,
            min_queries,
        )

    return [q.as_tuple() for q in manifest.queries[:max_queries]]


__all__ = [
    "DEFAULT_EVAL_PATH",
    "PRODUCTION_MIN_QUERIES",
    "PRODUCTION_TARGET_QUERIES",
    "SEED_EVAL_QUERIES",
    "TEST_MIN_QUERIES",
    "EvalSetManifest",
    "LabeledQuery",
    "build_eval_manifest",
    "build_eval_queries_from_db",
    "get_eval_queries",
    "load_eval_manifest",
    "save_eval_manifest",
]
