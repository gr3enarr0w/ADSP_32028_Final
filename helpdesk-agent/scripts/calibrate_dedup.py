"""Dedup threshold calibration using the real production FAQ corpus.

Pulls pairs from the Postgres database, labels them based on article/ticket
metadata (not cosine similarity — that would be circular), runs an F1-optimal
threshold sweep with an 80/20 train/test split, and reports both train and
test F1 so overfitting is visible.

Positive-label strategy (hybrid, three tiers — all sources combined):
  Tier 1 — Explicit Jira duplicate links (highest confidence):
    ticket_links where link_type ILIKE '%duplicate%' OR '%clone%'
  Tier 2 — Duplicate resolution field (high confidence):
    tickets where resolution ILIKE '%duplicate%'
  Tier 3 — High resolution_summary word overlap (medium confidence):
    Same-category tickets whose resolution_summaries share >60% non-stopword
    words — Gemini writes resolutions describing the fix, so near-identical
    fix descriptions mean the same underlying problem was solved.

  If _load_ground_truth_pairs() returns >=100 labeled pairs, those are used
  as the positive examples.  Otherwise the script falls back to the legacy
  category-grouping approach (better than nothing, just noisier).

  --gemini-labels mode (preferred):
    Uses Gemini as an independent judge to label pairs across all overlap
    ranges (0.0-1.0), breaking the circular dependency where word-overlap
    labels train an embedding that tracks word overlap. Gemini-labeled pairs
    are cached to faq/gemini_labeled_pairs.json to avoid repeat API calls.

Negative-label strategy:
  - Cross-category pairs (clearly distinct)
  - Same-category pairs where resolution word overlap < 0.1

Data sources (priority order):
  1. generated_articles  — same article_topic = duplicate; different = distinct
  2. kb_articles         — same space + topic overlap = duplicate; else distinct
  3. Fallback            — ticket_classifications + tickets:
                           same category + same issue_type = duplicate;
                           different categories = distinct

Pair generation rules:
  - Target >=200 pairs, ~40% duplicate / ~60% distinct
  - Cap at 5 pairs per (category, issue_type) group to avoid dominance
  - Stratify distinct pairs so every major category appears
  - No single category exceeds 40% of total pairs
  - Shuffle with --seed (default 42) before 80/20 split

Usage:
    DATABASE_URL=postgresql://... python -m scripts.calibrate_dedup
    DATABASE_URL=postgresql://... python -m scripts.calibrate_dedup --save
    DATABASE_URL=postgresql://... python -m scripts.calibrate_dedup --min-pairs 200 --seed 42 --save
    DATABASE_URL=postgresql://... python -m scripts.calibrate_dedup --gemini-labels --save
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import random
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

CALIBRATION_RESULT_PATH = ROOT / "faq" / "calibration_result.json"
GEMINI_LABELED_PAIRS_PATH = ROOT / "faq" / "gemini_labeled_pairs.json"
MAJOR_CATEGORIES = {"Access", "Configuration", "Permissions", "Integration", "Workflow", "Data"}
MAX_CATEGORY_FRACTION = 0.40
MAX_PAIRS_PER_GROUP = 5


class LabeledPair(NamedTuple):
    text_a: str
    text_b: str
    label: str          # "duplicate" or "distinct"
    category: str
    source: str         # "generated_articles" | "kb_articles" | "ticket"
                        # | "tier1_jira_link" | "tier2_dup_resolution"
                        # | "tier3_word_overlap" | "negative_cross_cat"
                        # | "negative_distinct_resolution" | "gemini_judge"


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

def _strip_html(html_text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", html_text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Hybrid ground-truth pair loading (Tier 1 / 2 / 3)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    'the', 'a', 'an', 'is', 'was', 'to', 'of', 'and', 'or', 'in', 'for',
    'on', 'at', 'by', 'with', 'it', 'this', 'that', 'be', 'are', 'were',
    'has', 'have', 'had', 'not', 'no', 'but', 'so', 'if', 'as', 'from',
    'into', 'their', 'its', 'they', 'we', 'you', 'he', 'she', 'been',
    'will', 'would', 'could', 'can', 'which', 'what', 'how', 'when',
}


def _resolution_similarity(text_a: str, text_b: str, threshold: float = 0.6) -> float:
    """Word-overlap (Jaccard-like) similarity for resolution summaries.

    Uses non-stopword words only.  Returns a float in [0, 1].  Both texts are
    lower-cased and split on whitespace; punctuation attached to tokens is
    included (``fix.`` and ``fix`` are treated as different tokens, which is
    acceptable given the large volumes of shared content between true duplicates).
    """
    words_a = set(text_a.lower().split()) - _STOPWORDS
    words_b = set(text_b.lower().split()) - _STOPWORDS
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def _load_ground_truth_pairs(
    conn,
    max_per_source: int = 100,
) -> list[LabeledPair]:
    """Load labeled duplicate pairs from three tiers of Jira evidence.

    Tier 1 — Explicit Jira duplicate/clone links (highest confidence).
    Tier 2 — Tickets closed with resolution containing "Duplicate".
    Tier 3 — Same-category tickets whose resolution_summary shares >60% of
              non-stopword words (good volume, medium confidence).

    Pairs are deduplicated by their (key_a, key_b) canonical key set so the
    same ticket pair is not included more than once regardless of which tier
    found it first.

    Returns a flat list of LabeledPair (all label="duplicate") plus
    confirmed-distinct negatives from same-category low-overlap pairs.
    The caller is responsible for adding cross-category distinct pairs.

    Logs a breakdown of how many pairs came from each tier.
    """
    pairs: list[LabeledPair] = []
    seen_keys: set[frozenset] = set()  # frozenset({key_a, key_b}) for dedup

    # ------------------------------------------------------------------
    # Tier 1 — Explicit Jira duplicate / clone links
    # ------------------------------------------------------------------
    try:
        tier1_rows = conn.execute(
            """
            SELECT
                tl.ticket_key   AS key_a,
                tl.linked_key   AS key_b,
                t1.summary      AS summary_a,
                t1.resolution_summary AS res_sum_a,
                t1.description  AS desc_a,
                t2.summary      AS summary_b,
                t2.resolution_summary AS res_sum_b,
                t2.description  AS desc_b,
                tc1.category    AS category
            FROM ticket_links tl
            JOIN tickets t1 ON t1.ticket_key = tl.ticket_key
            JOIN tickets t2 ON t2.ticket_key = tl.linked_key
            LEFT JOIN ticket_classifications tc1 ON tc1.ticket_key = tl.ticket_key
            WHERE (tl.link_type ILIKE '%duplicate%' OR tl.link_type ILIKE '%clone%')
              AND t1.summary IS NOT NULL
              AND t2.summary IS NOT NULL
            LIMIT ?
            """,
            (max_per_source,),
        ).fetchall()
    except Exception as exc:
        log.warning("Tier 1 query failed (%s) — skipping", exc)
        tier1_rows = []

    tier1_count = 0
    for row in tier1_rows:
        key_a, key_b = row["key_a"], row["key_b"]
        pair_key = frozenset({key_a, key_b})
        if pair_key in seen_keys:
            continue
        seen_keys.add(pair_key)

        text_a = _pick_text(row["res_sum_a"], row["summary_a"], row["desc_a"])
        text_b = _pick_text(row["res_sum_b"], row["summary_b"], row["desc_b"])
        if not text_a or not text_b:
            continue

        pairs.append(LabeledPair(
            text_a=text_a,
            text_b=text_b,
            label="duplicate",
            category=row["category"] or "unknown",
            source="tier1_jira_link",
        ))
        tier1_count += 1

    log.info("Tier 1 (Jira duplicate/clone links): %d pairs", tier1_count)

    # ------------------------------------------------------------------
    # Tier 2 — resolution field contains "Duplicate"
    # ------------------------------------------------------------------
    try:
        tier2_rows = conn.execute(
            """
            SELECT
                t.ticket_key,
                t.summary,
                t.description,
                t.resolution_summary,
                tc.category
            FROM tickets t
            LEFT JOIN ticket_classifications tc ON tc.ticket_key = t.ticket_key
            WHERE t.resolution ILIKE '%duplicate%'
              AND t.summary IS NOT NULL
            ORDER BY tc.category, t.ticket_key
            LIMIT ?
            """,
            (max_per_source * 2,),
        ).fetchall()
    except Exception as exc:
        log.warning("Tier 2 query failed (%s) — skipping", exc)
        tier2_rows = []

    # Group by category; within each group, pair each ticket with the next
    # (not all combinations — avoids O(n^2) explosion for large categories)
    from collections import defaultdict
    t2_by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in tier2_rows:
        t2_by_cat[row["category"] or "unknown"].append(dict(row))

    tier2_count = 0
    for cat, members in t2_by_cat.items():
        # Consecutive pairs within each category bucket
        for i in range(0, len(members) - 1, 2):
            a, b = members[i], members[i + 1]
            pair_key = frozenset({a["ticket_key"], b["ticket_key"]})
            if pair_key in seen_keys:
                continue
            seen_keys.add(pair_key)
            text_a = _pick_text(a["resolution_summary"], a["summary"], a["description"])
            text_b = _pick_text(b["resolution_summary"], b["summary"], b["description"])
            if not text_a or not text_b:
                continue
            if tier2_count >= max_per_source:
                break
            pairs.append(LabeledPair(
                text_a=text_a,
                text_b=text_b,
                label="duplicate",
                category=cat,
                source="tier2_dup_resolution",
            ))
            tier2_count += 1

    log.info("Tier 2 (duplicate resolution field): %d pairs", tier2_count)

    # ------------------------------------------------------------------
    # Tier 3 — High resolution_summary word overlap within same category
    # ------------------------------------------------------------------
    OVERLAP_THRESHOLD = 0.60
    NEGATIVE_THRESHOLD = 0.10

    try:
        tier3_candidates = conn.execute(
            """
            SELECT
                t.ticket_key,
                t.summary,
                t.description,
                t.resolution_summary,
                tc.category
            FROM tickets t
            JOIN ticket_classifications tc ON tc.ticket_key = t.ticket_key
            WHERE t.resolution_summary IS NOT NULL
              AND TRIM(t.resolution_summary) != ''
              AND LENGTH(t.resolution_summary) > 30
            ORDER BY tc.category, t.ticket_key
            """,
        ).fetchall()
    except Exception as exc:
        log.warning("Tier 3 query failed (%s) — skipping", exc)
        tier3_candidates = []

    # Group by category
    t3_by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in tier3_candidates:
        t3_by_cat[row["category"]].append(dict(row))

    tier3_dup_count = 0
    tier3_neg_count = 0
    negative_pairs: list[LabeledPair] = []

    for cat, members in t3_by_cat.items():
        # Sample at most sqrt(max_per_source)*3 members per category to keep
        # the inner loop fast — avoids O(n^2) over large categories
        sample_size = min(len(members), max(20, max_per_source // 5))
        sampled = members[:sample_size]

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                a, b = sampled[i], sampled[j]
                pair_key = frozenset({a["ticket_key"], b["ticket_key"]})

                res_a = a["resolution_summary"] or ""
                res_b = b["resolution_summary"] or ""
                sim = _resolution_similarity(res_a, res_b)

                if sim >= OVERLAP_THRESHOLD:
                    if pair_key not in seen_keys and tier3_dup_count < max_per_source:
                        seen_keys.add(pair_key)
                        text_a = _pick_text(res_a, a["summary"], a["description"])
                        text_b = _pick_text(res_b, b["summary"], b["description"])
                        if text_a and text_b:
                            pairs.append(LabeledPair(
                                text_a=text_a,
                                text_b=text_b,
                                label="duplicate",
                                category=cat,
                                source="tier3_word_overlap",
                            ))
                            tier3_dup_count += 1

                elif sim < NEGATIVE_THRESHOLD:
                    if pair_key not in seen_keys and tier3_neg_count < max_per_source:
                        seen_keys.add(pair_key)
                        text_a = _pick_text(res_a, a["summary"], a["description"])
                        text_b = _pick_text(res_b, b["summary"], b["description"])
                        if text_a and text_b:
                            negative_pairs.append(LabeledPair(
                                text_a=text_a,
                                text_b=text_b,
                                label="distinct",
                                category=cat,
                                source="negative_distinct_resolution",
                            ))
                            tier3_neg_count += 1

    log.info(
        "Tier 3 (resolution word overlap): %d duplicate pairs, %d confirmed-distinct pairs",
        tier3_dup_count,
        tier3_neg_count,
    )

    # Attach confirmed-distinct negatives
    pairs.extend(negative_pairs)

    total_dup = sum(1 for p in pairs if p.label == "duplicate")
    log.info(
        "Ground-truth pairs total: %d (%d duplicate / %d distinct)",
        len(pairs),
        total_dup,
        len(pairs) - total_dup,
    )
    return pairs


def _pick_text(resolution_summary: str | None, summary: str | None, description: str | None) -> str:
    """Select the richest available text field for a ticket, stripping HTML."""
    if resolution_summary and len(resolution_summary.strip()) > 20:
        return resolution_summary.strip()
    fallback = f"{summary or ''} {_strip_html(description or '')}".strip()
    return fallback if len(fallback) >= 30 else ""


# ---------------------------------------------------------------------------
# Gemini-as-judge labeling (breaks circular word-overlap dependency)
# ---------------------------------------------------------------------------

_GEMINI_JUDGE_PROMPT = """\
You are classifying IT support ticket pairs for a deduplication system.

Ticket A resolution: {text_a}
Ticket B resolution: {text_b}

Are these two tickets describing the SAME underlying IT problem and solution?
Answer with exactly one word: DUPLICATE or DISTINCT.

DUPLICATE: same root cause, same fix, could reasonably be merged
DISTINCT: different problems, different systems, or different users even if similar category\
"""


def _label_pairs_with_gemini(
    pairs_to_label: list[tuple[str, str, str, str]],
    max_pairs: int = 300,
) -> list[LabeledPair]:
    """Use Gemini as judge to label (text_a, text_b, category, source) tuples.

    Breaks circular dependency: labels come from Gemini's semantic understanding,
    not word overlap. Includes hard pairs (medium overlap 0.05-0.55) that
    word-overlap labeling excluded.

    Parameters
    ----------
    pairs_to_label:
        List of (text_a, text_b, category, source) tuples to classify.
        Caller is responsible for systematic sampling across overlap ranges.
    max_pairs:
        Hard cap on API calls. Tuples beyond this index are ignored.

    Returns
    -------
    List of LabeledPair with source="gemini_judge".
    Pairs where Gemini returns an unparseable response are silently dropped.
    """
    from collections import Counter

    from config import GEMINI_MODEL_CLASSIFICATION
    from core.genai import get_genai_client

    client = get_genai_client()
    subset = pairs_to_label[:max_pairs]
    log.info("Gemini judge: sending %d pairs to %s", len(subset), GEMINI_MODEL_CLASSIFICATION)

    results: list[LabeledPair] = []
    word_overlap_labels: list[str] = []  # for agreement logging

    for idx, (text_a, text_b, category, source) in enumerate(subset):
        prompt = _GEMINI_JUDGE_PROMPT.format(text_a=text_a, text_b=text_b)
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_CLASSIFICATION,
                contents=prompt,
            )
            raw = (response.text or "").strip().upper()
        except Exception as exc:
            log.warning("Gemini judge: pair %d failed (%s) -- skipping", idx, exc)
            continue

        if "DUPLICATE" in raw:
            verdict = "duplicate"
        elif "DISTINCT" in raw:
            verdict = "distinct"
        else:
            log.debug("Gemini judge: unparseable response for pair %d: %r", idx, raw)
            continue

        # Track word-overlap label for agreement logging
        overlap = _resolution_word_overlap(text_a, text_b)
        if overlap >= 0.55:
            word_overlap_labels.append("duplicate")
        elif overlap <= 0.05:
            word_overlap_labels.append("distinct")
        else:
            word_overlap_labels.append("ambiguous")

        results.append(LabeledPair(
            text_a=text_a,
            text_b=text_b,
            label=verdict,
            category=category,
            source="gemini_judge",
        ))

    # Log label distribution
    dist = Counter(p.label for p in results)
    log.info(
        "Gemini judge labels: %d duplicate / %d distinct (%.1f%% duplicate)",
        dist["duplicate"],
        dist["distinct"],
        100 * dist["duplicate"] / len(results) if results else 0,
    )

    # Log agreement with word-overlap on non-ambiguous pairs
    agreements = 0
    total_comparable = 0
    for pair, wo_label in zip(results, word_overlap_labels):
        if wo_label == "ambiguous":
            continue
        total_comparable += 1
        if pair.label == wo_label:
            agreements += 1

    if total_comparable > 0:
        agreement_rate = 100 * agreements / total_comparable
        log.info(
            "Agreement with word-overlap labels: %d/%d (%.1f%%) on non-ambiguous pairs",
            agreements,
            total_comparable,
            agreement_rate,
        )
    else:
        log.info("No non-ambiguous word-overlap pairs to compute agreement rate")

    return results


def _build_gemini_candidate_pool(
    conn,
    max_per_bucket: int = 100,
) -> list[tuple[str, str, str, str]]:
    """Load resolution-summary candidates and sample across all overlap ranges.

    Samples up to ``max_per_bucket`` pairs from each of three overlap buckets:
      - Low:    0.0  - 0.20  (clearly distinct by word overlap)
      - Medium: 0.20 - 0.55  (ambiguous -- previously excluded by Tier 3)
      - High:   0.55 - 1.0   (likely duplicate by word overlap)

    Returns a list of (text_a, text_b, category, source) tuples ready for
    ``_label_pairs_with_gemini``.
    """
    from collections import defaultdict

    rows = conn.execute(
        """
        SELECT t.ticket_key, t.resolution_summary, tc.category
        FROM tickets t
        JOIN ticket_classifications tc ON tc.ticket_key = t.ticket_key
        WHERE t.resolution_summary IS NOT NULL
          AND TRIM(t.resolution_summary) != ''
          AND LENGTH(t.resolution_summary) > 30
          AND tc.category IS NOT NULL
        ORDER BY tc.category, t.ticket_key
        """
    ).fetchall()

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rs = (row["resolution_summary"] or "").strip()
        if len(rs) > 20:
            by_category[row["category"]].append({
                "key": row["ticket_key"],
                "text": rs,
                "category": row["category"],
            })

    low_bucket: list[tuple[str, str, str, str]] = []
    mid_bucket: list[tuple[str, str, str, str]] = []
    high_bucket: list[tuple[str, str, str, str]] = []

    SAMPLE_SIZE = 80  # per category to keep combination space manageable
    for cat, members in by_category.items():
        if len(members) < 2:
            continue
        sample = members if len(members) <= SAMPLE_SIZE else random.sample(members, SAMPLE_SIZE)
        pairs_in_cat = list(combinations(sample, 2))
        random.shuffle(pairs_in_cat)

        for a, b in pairs_in_cat:
            if a["text"] == b["text"]:
                continue
            overlap = _resolution_word_overlap(a["text"], b["text"])
            entry: tuple[str, str, str, str] = (a["text"], b["text"], cat, "ticket")
            if overlap < 0.20:
                low_bucket.append(entry)
            elif overlap < 0.55:
                mid_bucket.append(entry)
            else:
                high_bucket.append(entry)

    log.info(
        "Gemini candidate pool: %d low-overlap / %d mid-overlap / %d high-overlap pairs",
        len(low_bucket),
        len(mid_bucket),
        len(high_bucket),
    )

    # Sample up to max_per_bucket from each bucket and shuffle together
    sampled: list[tuple[str, str, str, str]] = []
    for bucket in (low_bucket, mid_bucket, high_bucket):
        random.shuffle(bucket)
        sampled.extend(bucket[:max_per_bucket])

    random.shuffle(sampled)
    return sampled


def _load_or_generate_gemini_pairs(
    conn,
    *,
    max_pairs: int = 300,
    max_per_bucket: int = 100,
    force_refresh: bool = False,
) -> list[LabeledPair]:
    """Load Gemini-labeled pairs from cache, or generate and cache them.

    Cache location: faq/gemini_labeled_pairs.json.
    Pass ``force_refresh=True`` to ignore the cache and re-call the API.
    """
    import json as _json

    if not force_refresh and GEMINI_LABELED_PAIRS_PATH.exists():
        log.info("Loading cached Gemini-labeled pairs from %s", GEMINI_LABELED_PAIRS_PATH)
        raw = _json.loads(GEMINI_LABELED_PAIRS_PATH.read_text())
        pairs = [
            LabeledPair(
                text_a=entry["text_a"],
                text_b=entry["text_b"],
                label=entry["label"],
                category=entry["category"],
                source=entry["source"],
            )
            for entry in raw
        ]
        log.info("Loaded %d cached Gemini-labeled pairs", len(pairs))
        return pairs

    # Build candidate pool from DB
    candidates = _build_gemini_candidate_pool(conn, max_per_bucket=max_per_bucket)
    if not candidates:
        log.warning("Gemini candidate pool is empty -- no pairs to label")
        return []

    pairs = _label_pairs_with_gemini(candidates, max_pairs=max_pairs)

    # Cache to disk
    GEMINI_LABELED_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "text_a": p.text_a,
            "text_b": p.text_b,
            "label": p.label,
            "category": p.category,
            "source": p.source,
        }
        for p in pairs
    ]
    GEMINI_LABELED_PAIRS_PATH.write_text(_json.dumps(payload, indent=2) + "\n")
    log.info("Cached %d Gemini-labeled pairs to %s", len(pairs), GEMINI_LABELED_PAIRS_PATH)
    return pairs


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_generated_articles(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT article_topic, format, body_html
        FROM generated_articles
        WHERE body_html IS NOT NULL AND TRIM(body_html) != ''
        ORDER BY article_topic
        """
    ).fetchall()
    return [
        {
            "group": row["article_topic"],
            "text": _strip_html(row["body_html"]),
            "category": row["format"] or "general",
            "source": "generated_articles",
        }
        for row in rows
        if len(_strip_html(row["body_html"])) >= 50
    ]


def _load_kb_articles(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT space_key, title, body_text, topics_covered
        FROM kb_articles
        WHERE body_text IS NOT NULL AND TRIM(body_text) != ''
        ORDER BY space_key, title
        """
    ).fetchall()
    return [
        {
            "group": f"{row['space_key']}::{row['title']}",
            "text": (row["body_text"] or "").strip(),
            "category": row["space_key"] or "kb",
            "source": "kb_articles",
        }
        for row in rows
        if len((row["body_text"] or "").strip()) >= 50
    ]


def _load_ticket_corpus(conn) -> list[dict]:
    """Load resolved tickets for calibration pair generation.

    Text priority:
      1. ``resolution_summary`` — Gemini-generated from comments + resolution field.
         Preferred because it captures the full arc: what the problem was AND how
         it was resolved, giving the embedding a richer, more discriminative signal.
      2. ``summary + description`` — fallback when ``resolution_summary`` is NULL
         or too short (<=20 chars), e.g. tickets resolved before backfill ran.
    """
    rows = conn.execute(
        """
        SELECT tc.category, tc.issue_type,
               t.summary,
               t.description,
               t.resolution_summary
        FROM tickets t
        LEFT JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
        WHERE tc.category IS NOT NULL
          AND tc.issue_type IS NOT NULL
          AND t.summary IS NOT NULL
        ORDER BY tc.category, tc.issue_type
        """
    ).fetchall()
    items = []
    for row in rows:
        if row['resolution_summary'] and len(row['resolution_summary'].strip()) > 20:
            text = row['resolution_summary'].strip()
        else:
            text = f"{row['summary'] or ''} {row['description'] or ''}".strip()
        if len(text) < 30:
            continue
        items.append({
            "group": f"{row['category']}::{row['issue_type']}",
            "text": text,
            "category": row["category"],
            "source": "ticket",
        })
    return items


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def _make_duplicate_pairs(items: list[dict], max_per_group: int) -> list[LabeledPair]:
    """Within each group, form pairs labelled duplicate."""
    from collections import defaultdict
    by_group: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_group[item["group"]].append(item)

    pairs: list[LabeledPair] = []
    for group, members in by_group.items():
        if len(members) < 2:
            continue
        group_pairs = list(combinations(members, 2))
        random.shuffle(group_pairs)
        for a, b in group_pairs[:max_per_group]:
            pairs.append(LabeledPair(
                text_a=a["text"],
                text_b=b["text"],
                label="duplicate",
                category=a["category"],
                source=a["source"],
            ))
    return pairs


def _make_distinct_pairs(
    items: list[dict],
    target: int,
    existing_pairs: list[LabeledPair],
) -> list[LabeledPair]:
    """Cross-category pairs labelled distinct, stratified by category."""
    from collections import defaultdict
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)

    categories = sorted(by_category.keys())
    pairs: list[LabeledPair] = []

    # Pair each category against a different category round-robin
    cat_list = list(categories)
    random.shuffle(cat_list)
    attempts = 0
    max_attempts = target * 10
    while len(pairs) < target and attempts < max_attempts:
        attempts += 1
        if len(cat_list) < 2:
            break
        cat_a = cat_list[attempts % len(cat_list)]
        cat_b = cat_list[(attempts + 1) % len(cat_list)]
        if cat_a == cat_b:
            continue
        pool_a = by_category[cat_a]
        pool_b = by_category[cat_b]
        if not pool_a or not pool_b:
            continue
        a = random.choice(pool_a)
        b = random.choice(pool_b)
        if a["text"] == b["text"]:
            continue
        pairs.append(LabeledPair(
            text_a=a["text"],
            text_b=b["text"],
            label="distinct",
            category=cat_a,
            source=a["source"],
        ))

    return pairs


def _load_hybrid_duplicate_pairs(conn, max_per_tier: int = 150) -> list[LabeledPair]:
    """Load duplicate/distinct pairs using a two-tier heuristic strategy.

    Tier 2 — Explicit "Duplicate" resolution tickets (26 anchors):
        Each ticket resolved as "Duplicate" is paired with the most
        word-similar ticket in the same category (excluding other
        Duplicate-resolution tickets).  These are confirmed positives.

    Tier 3 — Resolution-summary word overlap within category:
        Within each category, sample up to 200 tickets with a
        resolution_summary populated.  Compute pairwise word overlap using
        ``_resolution_word_overlap()``:
          - overlap >= 0.55  ->  label="duplicate"  (same fix, same problem)
          - overlap <= 0.05  ->  label="distinct"   (clearly different fixes)
          - 0.05 < overlap < 0.55  ->  skipped (ambiguous)
        Cap at ``max_per_tier`` pairs per category, balanced ~40/60 dup/distinct.

    Returns all collected pairs (Tier 2 + Tier 3 combined).
    """
    from collections import defaultdict

    pairs: list[LabeledPair] = []

    # ------------------------------------------------------------------
    # Tier 2: explicit "Duplicate" resolution anchors
    # Each anchor is paired with the most word-similar non-Duplicate ticket
    # in the same category.
    # ------------------------------------------------------------------
    dup_rows = conn.execute(
        """
        SELECT t.ticket_key, t.resolution_summary, tc.category
        FROM tickets t
        LEFT JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
        WHERE t.resolution = 'Duplicate'
          AND t.resolution_summary IS NOT NULL
          AND tc.category IS NOT NULL
        """
    ).fetchall()

    pool_rows = conn.execute(
        """
        SELECT t.ticket_key, t.resolution_summary, tc.category
        FROM tickets t
        LEFT JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
        WHERE t.resolution != 'Duplicate'
          AND t.resolution_summary IS NOT NULL
          AND TRIM(t.resolution_summary) != ''
          AND tc.category IS NOT NULL
        """
    ).fetchall()

    pool_by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in pool_rows:
        rs = (row["resolution_summary"] or "").strip()
        if len(rs) > 20:
            pool_by_cat[row["category"]].append({
                "key": row["ticket_key"],
                "text": rs,
                "category": row["category"],
            })

    tier2_count = 0
    for row in dup_rows:
        anchor_text = (row["resolution_summary"] or "").strip()
        if len(anchor_text) < 20:
            continue
        cat = row["category"]
        candidates = pool_by_cat.get(cat, [])
        if not candidates:
            continue

        best_score = -1.0
        best_candidate = None
        for cand in candidates:
            score = _resolution_word_overlap(anchor_text, cand["text"])
            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_candidate is None:
            continue

        pairs.append(LabeledPair(
            text_a=anchor_text,
            text_b=best_candidate["text"],
            label="duplicate",
            category=cat,
            source="tier2_explicit_duplicate",
        ))
        tier2_count += 1
        if tier2_count >= 26:
            break

    log.info("Hybrid Tier 2: %d pairs from explicit Duplicate-resolution anchors", tier2_count)

    # ------------------------------------------------------------------
    # Tier 3: word-overlap within category (sample up to 200 per category)
    # ------------------------------------------------------------------
    all_pool_rows = conn.execute(
        """
        SELECT t.ticket_key, t.resolution_summary, tc.category
        FROM tickets t
        LEFT JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
        WHERE t.resolution_summary IS NOT NULL
          AND TRIM(t.resolution_summary) != ''
          AND tc.category IS NOT NULL
        ORDER BY tc.category
        """
    ).fetchall()

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in all_pool_rows:
        rs = (row["resolution_summary"] or "").strip()
        if len(rs) > 20:
            by_category[row["category"]].append({
                "key": row["ticket_key"],
                "text": rs,
                "category": row["category"],
            })

    tier3_dup = 0
    tier3_dist = 0
    SAMPLE_SIZE = 200
    DUP_THRESHOLD = 0.55
    DIST_THRESHOLD = 0.05
    TARGET_DUP_FRAC = 0.40

    for cat, members in by_category.items():
        if len(members) < 2:
            continue
        sample = members if len(members) <= SAMPLE_SIZE else random.sample(members, SAMPLE_SIZE)

        cat_dup: list[LabeledPair] = []
        cat_dist: list[LabeledPair] = []

        for a, b in combinations(sample, 2):
            if a["text"] == b["text"]:
                continue
            score = _resolution_word_overlap(a["text"], b["text"])
            if score >= DUP_THRESHOLD:
                cat_dup.append(LabeledPair(
                    text_a=a["text"],
                    text_b=b["text"],
                    label="duplicate",
                    category=cat,
                    source="tier3_word_overlap",
                ))
            elif score <= DIST_THRESHOLD:
                cat_dist.append(LabeledPair(
                    text_a=a["text"],
                    text_b=b["text"],
                    label="distinct",
                    category=cat,
                    source="tier3_word_overlap",
                ))

        target_dup = int(max_per_tier * TARGET_DUP_FRAC)
        target_dist = max_per_tier - target_dup

        random.shuffle(cat_dup)
        random.shuffle(cat_dist)
        selected_dup = cat_dup[:target_dup]
        selected_dist = cat_dist[:target_dist]

        pairs.extend(selected_dup)
        pairs.extend(selected_dist)
        tier3_dup += len(selected_dup)
        tier3_dist += len(selected_dist)

    log.info(
        "Hybrid Tier 3: %d duplicate + %d distinct pairs from resolution_summary word overlap",
        tier3_dup, tier3_dist,
    )
    log.info(
        "Hybrid total: %d pairs (%d dup / %d distinct)",
        len(pairs),
        sum(1 for p in pairs if p.label == "duplicate"),
        sum(1 for p in pairs if p.label == "distinct"),
    )
    return pairs


def _resolution_word_overlap(text_a: str, text_b: str) -> float:
    """Jaccard-like word overlap for resolution summaries (used by hybrid loader).

    Filters a standard stopword list and requires at least 3 content words in
    each text before computing overlap; shorter texts return 0.0.
    """
    _stopwords = {
        'the', 'a', 'an', 'is', 'was', 'to', 'of', 'and', 'or', 'in', 'for',
        'on', 'at', 'by', 'with', 'it', 'this', 'that', 'was', 'has', 'have',
        'been', 'were', 'user', 'issue', 'please', 'could', 'would', 'should',
    }
    words_a = set(text_a.lower().split()) - _stopwords
    words_b = set(text_b.lower().split()) - _stopwords
    if not words_a or not words_b or min(len(words_a), len(words_b)) < 3:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def build_pairs(
    conn,
    *,
    min_pairs: int = 200,
    max_per_group: int = MAX_PAIRS_PER_GROUP,
    seed: int = 42,
    min_gt_pairs: int = 100,
    use_gemini_labels: bool = False,
    gemini_max_pairs: int = 300,
) -> list[LabeledPair]:
    """Generate labelled pairs from the production corpus.

    Strategy selection
    ------------------
    0. If ``use_gemini_labels=True``, load/generate Gemini-judged pairs from
       faq/gemini_labeled_pairs.json (cached) and use them as the sole source.
       Gemini provides semantic labels that break the circular word-overlap
       dependency. The Gemini pairs are supplemented with cross-category
       distinct pairs to reach ``min_pairs`` if needed.

    1. Attempt hybrid duplicate labeling via ``_load_hybrid_duplicate_pairs()``.
       If that yields >= 200 total pairs, those are used as the positive
       examples and the function adds cross-category distinct pairs to reach
       ``min_pairs`` total.

    2. Otherwise fall back to the legacy ``_load_ground_truth_pairs()`` path,
       then to the legacy category-grouping approach using ``generated_articles``,
       ``kb_articles``, and ``ticket_classifications``.
    """
    random.seed(seed)

    # ------------------------------------------------------------------
    # Gemini-judge path (independent semantic labels, no word-overlap circularity)
    # ------------------------------------------------------------------
    if use_gemini_labels:
        gemini_pairs = _load_or_generate_gemini_pairs(
            conn,
            max_pairs=gemini_max_pairs,
            max_per_bucket=gemini_max_pairs // 3,
        )
        if len(gemini_pairs) >= 10:
            log.info(
                "Gemini-label mode: %d pairs loaded (%d dup / %d distinct)",
                len(gemini_pairs),
                sum(1 for p in gemini_pairs if p.label == "duplicate"),
                sum(1 for p in gemini_pairs if p.label == "distinct"),
            )
            gemini_dup = [p for p in gemini_pairs if p.label == "duplicate"]
            gemini_dist = [p for p in gemini_pairs if p.label == "distinct"]

            # Supplement with cross-category distinct pairs if needed
            tickets = _load_ticket_corpus(conn)
            target_cross = max(
                min_pairs - len(gemini_pairs),
                int(len(gemini_dup) * 1.5) - len(gemini_dist),
                0,
            )
            cross_dist: list[LabeledPair] = []
            if target_cross > 0 and tickets:
                cross_dist = [
                    LabeledPair(
                        text_a=p.text_a,
                        text_b=p.text_b,
                        label=p.label,
                        category=p.category,
                        source="negative_cross_cat",
                    )
                    for p in _make_distinct_pairs(tickets, target_cross, gemini_pairs)
                ]
            if cross_dist:
                log.info(
                    "Adding %d cross-category distinct pairs to Gemini-labeled set",
                    len(cross_dist),
                )

            pairs = gemini_pairs + cross_dist
            random.shuffle(pairs)
            pairs = _enforce_category_cap(pairs, MAX_CATEGORY_FRACTION)
            log.info(
                "Final pairs (gemini-labels): %d total (%d dup / %d distinct)",
                len(pairs),
                sum(1 for p in pairs if p.label == "duplicate"),
                sum(1 for p in pairs if p.label == "distinct"),
            )
            return pairs
        else:
            log.warning(
                "Gemini judge returned only %d pairs -- falling back to heuristic labeling",
                len(gemini_pairs),
            )

    # ------------------------------------------------------------------
    # Attempt new hybrid duplicate labeling first (Tier 2 + Tier 3)
    # ------------------------------------------------------------------
    hybrid_pairs = _load_hybrid_duplicate_pairs(conn)

    if len(hybrid_pairs) >= 200:
        log.info(
            "Hybrid labeling produced %d pairs — using as primary positive source",
            len(hybrid_pairs),
        )
        hybrid_dup = [p for p in hybrid_pairs if p.label == "duplicate"]
        hybrid_within_dist = [p for p in hybrid_pairs if p.label == "distinct"]

        # Add cross-category distinct pairs using full ticket corpus
        tickets = _load_ticket_corpus(conn)
        target_cross = max(
            min_pairs - len(hybrid_pairs),
            int(len(hybrid_dup) * 1.5) - len(hybrid_within_dist),
            0,
        )
        cross_dist = []
        if target_cross > 0 and tickets:
            cross_dist = [
                LabeledPair(
                    text_a=p.text_a,
                    text_b=p.text_b,
                    label=p.label,
                    category=p.category,
                    source="negative_cross_cat",
                )
                for p in _make_distinct_pairs(tickets, target_cross, hybrid_pairs)
            ]

        log.info(
            "Adding %d cross-category distinct pairs (target was %d)",
            len(cross_dist), target_cross,
        )

        pairs = hybrid_pairs + cross_dist
        random.shuffle(pairs)
        pairs = _enforce_category_cap(pairs, MAX_CATEGORY_FRACTION)

        log.info(
            "Final pairs (hybrid): %d total (%d dup / %d distinct)",
            len(pairs),
            sum(1 for p in pairs if p.label == "duplicate"),
            sum(1 for p in pairs if p.label == "distinct"),
        )
        return pairs

    # ------------------------------------------------------------------
    # Fallback A: older ground-truth loader (Tier 1 / 2 / 3 with tighter thresholds)
    # ------------------------------------------------------------------
    log.warning(
        "Hybrid labeling returned only %d pairs (< 200) — trying _load_ground_truth_pairs",
        len(hybrid_pairs),
    )

    gt_pairs = _load_ground_truth_pairs(conn, max_per_source=min_gt_pairs)
    gt_dup_count = sum(1 for p in gt_pairs if p.label == "duplicate")

    if gt_dup_count >= min_gt_pairs:
        log.info(
            "Ground-truth fallback: %d duplicate pairs — using as positive examples",
            gt_dup_count,
        )
        tickets = _load_ticket_corpus(conn)
        gt_dist_count = sum(1 for p in gt_pairs if p.label == "distinct")
        target_cross_cat = max(
            min_pairs - len(gt_pairs),
            int(gt_dup_count * 1.5) - gt_dist_count,
            0,
        )
        cross_cat_pairs = [
            LabeledPair(
                text_a=p.text_a,
                text_b=p.text_b,
                label=p.label,
                category=p.category,
                source="negative_cross_cat",
            )
            for p in _make_distinct_pairs(tickets, target_cross_cat, gt_pairs)
        ]
        pairs = gt_pairs + cross_cat_pairs
        random.shuffle(pairs)
        pairs = _enforce_category_cap(pairs, MAX_CATEGORY_FRACTION)
        log.info(
            "Final pairs (gt fallback): %d total (%d dup / %d distinct)",
            len(pairs),
            sum(1 for p in pairs if p.label == "duplicate"),
            sum(1 for p in pairs if p.label == "distinct"),
        )
        return pairs

    # ------------------------------------------------------------------
    # Fallback B: legacy category-grouping approach
    # ------------------------------------------------------------------
    log.warning(
        "Ground-truth pairs insufficient (%d dup < %d threshold) — "
        "falling back to category-grouping approach",
        gt_dup_count,
        min_gt_pairs,
    )

    # Load corpus — try article tables first, fall back to tickets
    generated = _load_generated_articles(conn)
    kb = _load_kb_articles(conn)
    tickets = _load_ticket_corpus(conn)

    log.info(
        "Corpus: %d generated_articles, %d kb_articles, %d resolved tickets",
        len(generated), len(kb), len(tickets),
    )

    # Combine sources; prefer articles over tickets for duplicates
    all_items = generated + kb
    dup_source = "generated_articles+kb_articles"
    # Threshold of 50: with ~1 article per category (10 categories), article
    # tables yield too few pairs to be meaningful; fall back to tickets instead.
    if len(all_items) < 50:
        log.info("Article tables thin — using ticket corpus as primary source")
        all_items = tickets
        dup_source = "ticket_classifications"
    else:
        # Always include tickets as supplemental distinct-pair fodder
        all_items += tickets

    dup_pairs = _make_duplicate_pairs(all_items, max_per_group)
    target_distinct = max(min_pairs - len(dup_pairs), int(len(dup_pairs) * 1.5))
    dist_pairs = _make_distinct_pairs(all_items, target_distinct, dup_pairs)

    pairs = dup_pairs + dist_pairs
    random.shuffle(pairs)

    # Enforce max-category fraction
    pairs = _enforce_category_cap(pairs, MAX_CATEGORY_FRACTION)

    log.info(
        "Generated %d pairs (%d dup / %d distinct) from %s",
        len(pairs), sum(1 for p in pairs if p.label == "duplicate"),
        sum(1 for p in pairs if p.label == "distinct"),
        dup_source,
    )
    return pairs


def _enforce_category_cap(pairs: list[LabeledPair], max_fraction: float) -> list[LabeledPair]:
    """Drop pairs from over-represented categories until no category exceeds cap."""
    from collections import Counter
    while True:
        total = len(pairs)
        if total == 0:
            break
        counts = Counter(p.category for p in pairs)
        over = {cat: cnt for cat, cnt in counts.items() if cnt / total > max_fraction}
        if not over:
            break
        dominated = max(over, key=over.get)
        # Remove one pair from the dominated category
        for i in range(len(pairs) - 1, -1, -1):
            if pairs[i].category == dominated:
                pairs.pop(i)
                break
    return pairs


# ---------------------------------------------------------------------------
# Embedding + scoring
# ---------------------------------------------------------------------------

def _embed_pairs(
    pairs: list[LabeledPair],
    embed_fn,
) -> tuple[list[float], list[int]]:
    """Return (cosine_scores, binary_labels) for all pairs."""
    scores: list[float] = []
    labels: list[int] = []
    for pair in pairs:
        va = np.array(embed_fn(pair.text_a, task_type="SEMANTIC_SIMILARITY"))
        vb = np.array(embed_fn(pair.text_b, task_type="SEMANTIC_SIMILARITY"))
        if va.size == 0 or vb.size == 0:
            log.warning("Skipping pair — empty embedding")
            continue
        scores.append(float(np.dot(va, vb)))
        labels.append(1 if pair.label == "duplicate" else 0)
    return scores, labels


# ---------------------------------------------------------------------------
# Threshold sweep (inline, supports train/test split)
# ---------------------------------------------------------------------------

def _sweep(scores: list[float], labels: list[int]) -> tuple[float, float, float, float]:
    """Return (best_threshold, best_f1, best_precision, best_recall).

    Sweeps thresholds from 0.3 to 0.95 in steps of 0.01.
    """
    best_t, best_f1, best_p, best_r = 0.5, -1.0, 0.0, 0.0
    arr = np.array(scores)
    lab = np.array(labels)
    thresholds = np.arange(0.30, 0.96, 0.01)
    for t in thresholds:
        tp = int(np.sum((arr >= t) & (lab == 1)))
        fp = int(np.sum((arr >= t) & (lab == 0)))
        fn = int(np.sum((arr < t) & (lab == 1)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, float(t), p, r
    return best_t, best_f1, best_p, best_r


def _metrics_at(scores: list[float], labels: list[int], threshold: float) -> dict:
    arr, lab = np.array(scores), np.array(labels)
    tp = int(np.sum((arr >= threshold) & (lab == 1)))
    fp = int(np.sum((arr >= threshold) & (lab == 0)))
    fn = int(np.sum((arr < threshold) & (lab == 1)))
    tn = int(np.sum((arr < threshold) & (lab == 0)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1}


def calibrate_with_split(
    scores: list[float],
    labels: list[int],
    *,
    train_fraction: float = 0.80,
    seed: int = 42,
) -> dict:
    """80/20 train/test split threshold calibration.

    Returns a dict with train_f1, test_f1, threshold, precision, recall,
    score_gap, and a warning flag if test_f1 is significantly below train_f1.
    """
    n = len(scores)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * train_fraction)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_scores = [scores[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_scores = [scores[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    threshold, train_f1, train_p, train_r = _sweep(train_scores, train_labels)

    test_m = _metrics_at(test_scores, test_labels, threshold)
    test_f1 = test_m["f1"]

    # Score gap over full set
    arr, lab = np.array(scores), np.array(labels)
    dup = arr[lab == 1]
    dis = arr[lab == 0]
    score_gap = float(dup.min()) - float(dis.max()) if len(dup) and len(dis) else float("nan")

    overfit_warning = (train_f1 - test_f1) > 0.05
    overlap_warning = not (score_gap != score_gap) and score_gap < 0.05  # nan-safe

    return {
        "threshold": threshold,
        "train_f1": round(train_f1, 4),
        "train_precision": round(train_p, 4),
        "train_recall": round(train_r, 4),
        "test_f1": round(test_f1, 4),
        "test_precision": round(test_m["precision"], 4),
        "test_recall": round(test_m["recall"], 4),
        "test_tp": test_m["tp"],
        "test_fp": test_m["fp"],
        "test_fn": test_m["fn"],
        "test_tn": test_m["tn"],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "score_gap": round(score_gap, 4) if score_gap == score_gap else None,
        "dup_score_mean": round(float(dup.mean()), 4) if len(dup) else None,
        "distinct_score_mean": round(float(dis.mean()), 4) if len(dis) else None,
        "overfit_warning": overfit_warning,
        "overlap_warning": overlap_warning,
    }


# ---------------------------------------------------------------------------
# Repeated-seed k-fold CV for threshold stability
# ---------------------------------------------------------------------------

def repeated_calibrate(
    scores: list[float],
    labels: list[int],
    *,
    n_seeds: int = 5,
    n_folds: int = 5,
) -> dict:
    """Repeated k-fold cross-validation for threshold stability measurement.

    For each of ``n_seeds`` random seeds the data is shuffled and split into
    ``n_folds`` folds.  Each fold is held out once as the test set; the
    remaining folds are used to find the F1-optimal threshold via ``_sweep``,
    which is then evaluated on the held-out fold.

    Parameters
    ----------
    scores:   Cosine similarity scores (one per pair).
    labels:   Binary labels (1 = duplicate, 0 = distinct).
    n_seeds:  Number of independent random seeds to use.
    n_folds:  Number of CV folds per seed.

    Returns
    -------
    dict with keys:
        threshold_mean, threshold_std  — across all seed x fold runs
        f1_mean, f1_std                — across all seed x fold runs
        threshold_min, threshold_max
        f1_min, f1_max
        all_thresholds                 — flat list of every per-fold threshold
        all_f1s                        — flat list of every per-fold test F1
        n_seeds, n_folds
    """
    seed_values = [42, 123, 456, 789, 1011][:n_seeds]

    arr = np.array(scores)
    lab = np.array(labels)
    n = len(arr)

    all_thresholds: list[float] = []
    all_f1s: list[float] = []

    for seed in seed_values:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        fold_boundaries = np.array_split(indices, n_folds)

        for held_out_idx in range(n_folds):
            test_idx = fold_boundaries[held_out_idx]
            train_idx = np.concatenate(
                [fold_boundaries[k] for k in range(n_folds) if k != held_out_idx]
            )

            train_scores = arr[train_idx].tolist()
            train_labels = lab[train_idx].tolist()
            test_scores = arr[test_idx].tolist()
            test_labels = lab[test_idx].tolist()

            if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
                # Skip degenerate folds (all-one-class)
                log.debug("Skipping degenerate fold (seed=%d, fold=%d)", seed, held_out_idx)
                continue

            best_t, _, _, _ = _sweep(train_scores, train_labels)
            fold_m = _metrics_at(test_scores, test_labels, best_t)

            all_thresholds.append(best_t)
            all_f1s.append(fold_m["f1"])

    if not all_thresholds:
        log.warning("repeated_calibrate: no valid folds — returning zeros")
        return {
            "threshold_mean": float("nan"),
            "threshold_std": float("nan"),
            "f1_mean": float("nan"),
            "f1_std": float("nan"),
            "threshold_min": float("nan"),
            "threshold_max": float("nan"),
            "f1_min": float("nan"),
            "f1_max": float("nan"),
            "all_thresholds": [],
            "all_f1s": [],
            "n_seeds": n_seeds,
            "n_folds": n_folds,
        }

    t_arr = np.array(all_thresholds)
    f_arr = np.array(all_f1s)

    return {
        "threshold_mean": round(float(t_arr.mean()), 4),
        "threshold_std": round(float(t_arr.std()), 4),
        "threshold_min": round(float(t_arr.min()), 4),
        "threshold_max": round(float(t_arr.max()), 4),
        "f1_mean": round(float(f_arr.mean()), 4),
        "f1_std": round(float(f_arr.std()), 4),
        "f1_min": round(float(f_arr.min()), 4),
        "f1_max": round(float(f_arr.max()), 4),
        "all_thresholds": [round(v, 4) for v in all_thresholds],
        "all_f1s": [round(v, 4) for v in all_f1s],
        "n_seeds": n_seeds,
        "n_folds": n_folds,
    }


def _stability_verdict(threshold_std: float) -> str:
    """Classify threshold stability based on standard deviation."""
    if threshold_std < 0.02:
        return "STABLE — std < 0.02, safe to deploy"
    if threshold_std <= 0.05:
        return "MARGINAL — collect more pairs before deploying"
    return "UNSTABLE — do not deploy this threshold"


def _print_stability_report(cv: dict) -> None:
    """Print the threshold stability section to stdout."""
    n_seeds = cv["n_seeds"]
    n_folds = cv["n_folds"]
    t_mean = cv["threshold_mean"]
    t_std = cv["threshold_std"]
    t_min = cv["threshold_min"]
    t_max = cv["threshold_max"]
    f_mean = cv["f1_mean"]
    f_std = cv["f1_std"]
    f_min = cv["f1_min"]
    f_max = cv["f1_max"]

    verdict = _stability_verdict(t_std)

    print(f"=== Threshold Stability ({n_seeds}-seed x {n_folds}-fold CV) ===")
    print(f"Threshold: {t_mean:.3f} +/- {t_std:.3f}  (range {t_min:.3f}-{t_max:.3f})")
    print(f"F1:        {f_mean:.3f} +/- {f_std:.3f}  (range {f_min:.3f}-{f_max:.3f})")
    print(f"Verdict: {verdict}")
    print()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(pairs: list[LabeledPair], result: dict) -> None:
    from collections import Counter
    n_dup = sum(1 for p in pairs if p.label == "duplicate")
    n_dist = sum(1 for p in pairs if p.label == "distinct")
    by_source = Counter(p.source for p in pairs)
    by_cat = Counter(p.category for p in pairs)

    # Tier breakdown for duplicate pairs
    tier_dup = Counter(
        p.source for p in pairs if p.label == "duplicate"
    )

    print()
    print("=" * 68)
    print("  DEDUP CALIBRATION — PRODUCTION CORPUS (POSTGRES)")
    print("=" * 68)
    print(f"  Total pairs:       {len(pairs)}  ({n_dup} duplicate / {n_dist} distinct)")
    print(f"  Sources (all):     {dict(by_source)}")
    print(f"  Duplicate sources: {dict(tier_dup)}")
    print(f"  Top categories:    {dict(by_cat.most_common(5))}")
    print(f"  Train / Test:      {result['n_train']} / {result['n_test']} (80/20 split)")
    print()
    print(f"  Threshold (from train):  {result['threshold']:.4f}")
    print()
    print(f"  TRAIN F1:          {result['train_f1']:.4f}  "
          f"(P={result['train_precision']:.4f}  R={result['train_recall']:.4f})")
    print(f"  TEST  F1:          {result['test_f1']:.4f}  "
          f"(P={result['test_precision']:.4f}  R={result['test_recall']:.4f})")
    print(f"  Test confusion:    TP={result['test_tp']} FP={result['test_fp']} "
          f"FN={result['test_fn']} TN={result['test_tn']}")
    print()

    gap = result.get("score_gap")
    print(f"  Score gap:         {gap:.4f}" if gap is not None else "  Score gap: n/a")
    print(f"  Dup score mean:    {result['dup_score_mean']}")
    print(f"  Distinct score mean: {result['distinct_score_mean']}")
    print()

    if result["overfit_warning"]:
        diff = result["train_f1"] - result["test_f1"]
        print(f"  WARNING: train F1 - test F1 = {diff:.4f} (>0.05)")
        print("    Collect more diverse pairs or reduce MAX_PAIRS_PER_GROUP.")
    if result["overlap_warning"]:
        print(f"  WARNING: score gap = {gap:.4f} (<0.05)")
        print("    Duplicate and distinct score distributions overlap — embedding model"
              " may not adequately separate JSM ticket semantics.")

    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# Save calibration result
# ---------------------------------------------------------------------------

def _save_result(
    result: dict,
    threshold: float,
    model_id: str,
    n_pairs: int,
    cv: dict | None = None,
) -> None:
    """Write a CalibrationResult-compatible JSON to faq/calibration_result.json.

    If ``cv`` is supplied (the output of ``repeated_calibrate``), the stability
    fields are included in the saved payload.
    """
    import json
    from datetime import datetime, timezone

    payload = {
        "optimal_threshold": threshold,
        "f1_at_threshold": result["test_f1"],
        "precision_at_threshold": result["test_precision"],
        "recall_at_threshold": result["test_recall"],
        "score_gap": result.get("score_gap"),
        "dup_score_stats": {
            "min": None, "mean": result["dup_score_mean"], "max": None,
        },
        "distinct_score_stats": {
            "min": None, "mean": result["distinct_score_mean"], "max": None,
        },
        "method": "f1_optimal_train_test_split",
        "model_id": model_id,
        "n_pairs": n_pairs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_f1": result["train_f1"],
        "test_f1": result["test_f1"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "platt_slope": None,
        "platt_intercept": None,
        "platt_threshold": None,
    }

    if cv is not None:
        payload["threshold_std"] = cv["threshold_std"]
        payload["f1_cv_mean"] = cv["f1_mean"]
        payload["f1_cv_std"] = cv["f1_std"]
        payload["cv_seeds"] = cv["n_seeds"]
        payload["cv_folds"] = cv["n_folds"]
        payload["stability_verdict"] = _stability_verdict(cv["threshold_std"])

    CALIBRATION_RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("Calibration result saved to %s", CALIBRATION_RESULT_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _probe_tier_counts(conn) -> None:
    """Print data availability counts for all three label tiers and exit.

    Run with ``--probe`` to quickly check how many positive examples are
    available before committing to a full embedding run.
    """
    print()
    print("=" * 68)
    print("  HYBRID LABEL TIER PROBE (no embeddings)")
    print("=" * 68)

    # Tier 1
    try:
        t1 = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM ticket_links
            WHERE link_type ILIKE '%duplicate%' OR link_type ILIKE '%clone%'
            """
        ).fetchone()["cnt"]
        link_types = conn.execute(
            "SELECT DISTINCT link_type FROM ticket_links ORDER BY link_type"
        ).fetchall()
        lt_list = [r["link_type"] for r in link_types] or ["(table empty)"]
    except Exception as exc:
        t1 = f"ERROR: {exc}"
        lt_list = []
    print(f"  Tier 1 — Jira duplicate/clone links:  {t1}")
    print(f"           Link types in DB: {lt_list}")

    # Tier 2
    try:
        t2 = conn.execute(
            "SELECT COUNT(*) AS cnt FROM tickets WHERE resolution ILIKE '%duplicate%'"
        ).fetchone()["cnt"]
        resolutions = conn.execute(
            "SELECT resolution, COUNT(*) AS cnt FROM tickets GROUP BY resolution ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        res_list = [(r["resolution"], r["cnt"]) for r in resolutions]
    except Exception as exc:
        t2 = f"ERROR: {exc}"
        res_list = []
    print(f"  Tier 2 — Duplicate resolution field:  {t2}")
    print(f"           Top resolutions: {res_list}")

    # Tier 3
    try:
        t3 = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM tickets
            WHERE resolution_summary IS NOT NULL
              AND TRIM(resolution_summary) != ''
              AND LENGTH(resolution_summary) > 30
            """
        ).fetchone()["cnt"]
        cat_counts = conn.execute(
            """
            SELECT tc.category, COUNT(*) AS cnt
            FROM tickets t
            JOIN ticket_classifications tc ON tc.ticket_key = t.ticket_key
            WHERE t.resolution_summary IS NOT NULL
              AND TRIM(t.resolution_summary) != ''
              AND LENGTH(t.resolution_summary) > 30
            GROUP BY tc.category
            ORDER BY cnt DESC
            LIMIT 8
            """
        ).fetchall()
        cat_list = [(r["category"], r["cnt"]) for r in cat_counts]
    except Exception as exc:
        t3 = f"ERROR: {exc}"
        cat_list = []
    print(f"  Tier 3 — Tickets with resolution_summary: {t3}")
    print(f"           By category: {cat_list}")

    print("=" * 68)
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", action="store_true",
                        help="Write result to faq/calibration_result.json")
    parser.add_argument("--probe", action="store_true",
                        help="Print data availability counts for all three label tiers and exit")
    parser.add_argument("--min-pairs", type=int, default=200,
                        help="Minimum pair count (default 200)")
    parser.add_argument("--min-gt-pairs", type=int, default=100,
                        help="Min ground-truth duplicate pairs before hybrid mode activates (default 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for pair generation and split (default 42)")
    parser.add_argument("--max-per-group", type=int, default=MAX_PAIRS_PER_GROUP,
                        help=f"Max pairs per group (default {MAX_PAIRS_PER_GROUP})")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of random seeds for repeated CV stability check (default 5)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of folds per seed for CV stability check (default 5)")
    parser.add_argument(
        "--gemini-labels", action="store_true",
        help=(
            "Use Gemini as independent judge to label pairs before calibration. "
            "Samples across all overlap ranges (0.0-1.0), breaking word-overlap "
            "circular dependency. Results cached to faq/gemini_labeled_pairs.json."
        ),
    )
    parser.add_argument(
        "--gemini-max-pairs", type=int, default=300,
        help="Max pairs to send to Gemini judge (default 300, ~100 per overlap bucket)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    from db import get_db_conn

    if args.probe:
        with get_db_conn() as conn:
            _probe_tier_counts(conn)
        return

    from services.embedding import embed_text, EMBEDDING_MODEL

    with get_db_conn() as conn:
        pairs = build_pairs(
            conn,
            min_pairs=args.min_pairs,
            max_per_group=args.max_per_group,
            seed=args.seed,
            min_gt_pairs=args.min_gt_pairs,
            use_gemini_labels=args.gemini_labels,
            gemini_max_pairs=args.gemini_max_pairs,
        )

    if len(pairs) < 10:
        log.error(
            "Only %d pairs generated — corpus too thin. "
            "Ensure generated_articles or ticket resolution_summaries are populated.",
            len(pairs),
        )
        sys.exit(1)

    print(f"\nEmbedding {len(pairs)} pairs with {EMBEDDING_MODEL}...")
    scores, labels = _embed_pairs(pairs, embed_text)

    if len(scores) < 10:
        log.error("Too few scored pairs (%d). Check embedding service.", len(scores))
        sys.exit(1)

    result = calibrate_with_split(scores, labels, seed=args.seed)
    _print_report(pairs, result)

    cv = repeated_calibrate(scores, labels, n_seeds=args.n_seeds, n_folds=args.n_folds)
    _print_stability_report(cv)

    if args.save:
        _save_result(result, result["threshold"], EMBEDDING_MODEL, len(pairs), cv=cv)
        print(f"Saved to {CALIBRATION_RESULT_PATH}")


if __name__ == "__main__":
    main()
