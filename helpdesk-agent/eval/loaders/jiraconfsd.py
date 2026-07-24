"""<PROJECT_KEY> loader — mines real service desk tickets for eval pairs.

Reads from eval/data/jiraconfsd_all.json (pulled by eval/scripts/pull_jiraconfsd.py)
and builds eval pairs at scale.

Two modes:
  "dedup"     — Duplicate ticket detection. Uses exact-duplicate-summary groups
                (285 groups, 8000+ tickets) to test whether the model can find
                semantically identical tickets in a large corpus.
  "retrieval" — Ticket→resolution matching. Summary as query, last comment as doc.

For large-scale eval (--limit 0 = all), builds a retrieval benchmark:
  - All 8000 tickets become the document pool
  - Queries are drawn from duplicate-summary groups
  - Correct answers are all other tickets in the same group
  - Measures Recall@k: can the model find duplicates in the haystack?

Usage:
    python -m eval.loaders.jiraconfsd                  # default 50+25 pairs
    python -m eval.loaders.jiraconfsd --mode dedup     # dedup pairs only
    python -m eval.loaders.jiraconfsd --limit 200      # more pairs
    python -m eval.loaders.jiraconfsd --large-scale    # full 8000-ticket benchmark
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALL_TICKETS = DATA_DIR / "jiraconfsd_all.json"

_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){10,}\b"), "[PHONE]"),
    (re.compile(r"\b(?:accountId|account_id)\s*[:=]\s*\S+", re.IGNORECASE), "[ACCOUNT_ID]"),
]


def _anonymize(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _normalize_summary(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _ticket_text(t: dict, max_len: int = 500) -> str:
    text = _anonymize(t.get("summary", ""))
    desc = t.get("description", "")
    if desc and len(desc) > 30:
        text += ". " + _anonymize(desc[:max_len - len(text) - 2])
    return text[:max_len]


def _load_all_tickets() -> list[dict]:
    if not ALL_TICKETS.exists():
        print(
            f"No ticket data at {ALL_TICKETS}\n"
            f"Run: python -m eval.scripts.pull_jiraconfsd",
            file=sys.stderr,
        )
        return []
    return json.loads(ALL_TICKETS.read_text())


def _build_duplicate_groups(tickets: list[dict]) -> dict[str, list[int]]:
    """Group tickets by normalized summary. Returns {norm_summary: [ticket_indices]}."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tickets):
        if not t.get("summary"):
            continue
        key = _normalize_summary(t["summary"])
        groups[key].append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _build_dedup_pairs(
    tickets: list[dict], pos_limit: int = 50, neg_limit: int = 25, seed: int = 42,
) -> list[dict]:
    """Build duplicate-detection pairs from tickets with similar summaries."""
    rng = random.Random(seed)
    dup_groups = _build_duplicate_groups(tickets)

    pairs = []
    pair_id = 1

    group_keys = sorted(dup_groups.keys(), key=lambda k: -len(dup_groups[k]))
    for gk in group_keys:
        if len(pairs) >= pos_limit:
            break
        indices = dup_groups[gk]
        rng.shuffle(indices)
        a, b = tickets[indices[0]], tickets[indices[1]]

        pairs.append({
            "id": pair_id,
            "query": _ticket_text(a),
            "document": _ticket_text(b),
            "relevance": 5,
            "category": "jira-dedup",
            "source": "jiraconfsd",
        })
        pair_id += 1

    used_summaries = {_normalize_summary(tickets[dup_groups[gk][0]]["summary"])
                      for gk in list(dup_groups.keys())[:pos_limit]}
    non_dup = [
        t for t in tickets
        if t.get("summary") and _normalize_summary(t["summary"]) not in used_summaries
    ]
    rng.shuffle(non_dup)

    neg_count = 0
    for i in range(min(pos_limit, len(pairs))):
        if neg_count >= neg_limit or neg_count >= len(non_dup):
            break
        pairs.append({
            "id": pair_id,
            "query": pairs[i]["query"],
            "document": _ticket_text(non_dup[neg_count]),
            "relevance": 1,
            "category": "jira-dedup-neg",
            "source": "jiraconfsd",
        })
        pair_id += 1
        neg_count += 1

    return pairs


def _build_retrieval_pairs(
    tickets: list[dict], pos_limit: int = 50, neg_limit: int = 25, seed: int = 42,
) -> list[dict]:
    """Build summary->resolution pairs for retrieval eval."""
    rng = random.Random(seed)

    candidates = []
    for t in tickets:
        if t.get("status") not in ("Closed", "Resolved", "Done"):
            continue
        if t.get("resolution") not in ("Done", "Resolved"):
            continue
        if not t.get("summary") or len(t["summary"]) < 10:
            continue
        if not t.get("comments"):
            continue
        last_comment = t["comments"][-1]["body"]
        if len(last_comment) < 40:
            continue
        candidates.append(t)

    rng.shuffle(candidates)

    pairs = []
    pair_id = 1

    for c in candidates[:pos_limit]:
        pairs.append({
            "id": pair_id,
            "query": _anonymize(c["summary"][:500]),
            "document": _anonymize(c["comments"][-1]["body"][:500]),
            "relevance": 5,
            "category": "jira-retrieval",
            "source": "jiraconfsd",
        })
        pair_id += 1

    pool = candidates[pos_limit:pos_limit + neg_limit * 2]
    neg_count = 0
    for i in range(min(pos_limit, len(pairs))):
        if neg_count >= neg_limit or neg_count >= len(pool):
            break
        pairs.append({
            "id": pair_id,
            "query": pairs[i]["query"],
            "document": _anonymize(pool[neg_count]["comments"][-1]["body"][:500]),
            "relevance": 1,
            "category": "jira-retrieval-neg",
            "source": "jiraconfsd",
        })
        pair_id += 1
        neg_count += 1

    return pairs


def load_pairs(
    mode: str = "both",
    pos_limit: int = 50,
    neg_limit: int = 25,
    seed: int = 42,
) -> list[dict]:
    tickets = _load_all_tickets()
    if not tickets:
        return []

    pairs = []
    if mode in ("both", "dedup"):
        dedup = _build_dedup_pairs(tickets, pos_limit, neg_limit, seed)
        pairs.extend(dedup)
    if mode in ("both", "retrieval"):
        retrieval = _build_retrieval_pairs(tickets, pos_limit, neg_limit, seed)
        offset = max((p["id"] for p in pairs), default=0)
        for p in retrieval:
            p["id"] += offset
        pairs.extend(retrieval)

    print(f"<PROJECT_KEY>: loaded {len(pairs)} pairs (mode={mode})")
    return pairs


def load_large_scale(seed: int = 42) -> dict:
    """Build a large-scale retrieval benchmark using all tickets.

    Returns a dict with:
        corpus: list[str]     — all ticket texts (the haystack, 8000+)
        queries: list[str]    — one query per duplicate group
        relevance: list[set]  — for each query, the set of corpus indices that are correct matches
        group_labels: list[str] — normalized summary for each query group
    """
    tickets = _load_all_tickets()
    if not tickets:
        return {}

    rng = random.Random(seed)

    corpus = [_ticket_text(t) for t in tickets]

    dup_groups = _build_duplicate_groups(tickets)
    group_keys = sorted(dup_groups.keys(), key=lambda k: -len(dup_groups[k]))

    queries = []
    relevance_sets = []
    group_labels = []

    for gk in group_keys:
        indices = dup_groups[gk]
        rng.shuffle(indices)
        query_idx = indices[0]
        correct_indices = set(indices[1:])

        queries.append(corpus[query_idx])
        relevance_sets.append(correct_indices)
        group_labels.append(gk)

    print(f"<PROJECT_KEY> large-scale: {len(corpus)} docs, {len(queries)} queries "
          f"({sum(len(r) for r in relevance_sets)} total relevant docs)")

    return {
        "corpus": corpus,
        "queries": queries,
        "relevance": relevance_sets,
        "group_labels": group_labels,
    }


def run_large_scale_eval(model_names: list[str] | None = None, top_k: int = 10):
    """Run retrieval benchmark: each query searches all 8000+ docs."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    try:
        from eval.compare_strategies import MODELS, _get_model
    except ImportError:
        from compare_strategies import MODELS, _get_model

    data = load_large_scale()
    if not data:
        return

    corpus = data["corpus"]
    queries = data["queries"]
    rel_sets = data["relevance"]
    labels = data["group_labels"]

    model_names = model_names or ["minilm", "mpnet", "bge", "e5"]

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"  {model_name} — {len(queries)} queries against {len(corpus)} docs")
        print(f"{'='*80}")

        model = _get_model(model_name)
        info = MODELS[model_name]

        print(f"  Encoding {len(corpus)} corpus docs...", end="", flush=True)
        batch_size = 256
        d_embs = []
        for start in range(0, len(corpus), batch_size):
            batch = [info["d_prefix"] + d for d in corpus[start:start + batch_size]]
            d_embs.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False))
        d_embs = np.vstack(d_embs)
        print(f" done ({d_embs.shape})")

        print(f"  Encoding {len(queries)} queries...", end="", flush=True)
        q_embs = []
        for start in range(0, len(queries), batch_size):
            batch = [info["q_prefix"] + q for q in queries[start:start + batch_size]]
            q_embs.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False))
        q_embs = np.vstack(q_embs)
        print(f" done ({q_embs.shape})")

        print(f"  Computing similarity matrix...", end="", flush=True)
        sim_matrix = q_embs @ d_embs.T
        print(f" done ({sim_matrix.shape})")

        recall_at = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
        mrr_sum = 0
        errors = []

        for qi in range(len(queries)):
            ranked = np.argsort(-sim_matrix[qi])
            correct = rel_sets[qi]

            first_correct_rank = None
            for rank, doc_idx in enumerate(ranked, 1):
                if doc_idx in correct:
                    first_correct_rank = rank
                    break

            if first_correct_rank:
                mrr_sum += 1.0 / first_correct_rank
                for k in recall_at:
                    top_k_set = set(ranked[:k].tolist())
                    if top_k_set & correct:
                        recall_at[k] += 1
            else:
                errors.append(qi)

            if first_correct_rank and first_correct_rank > 20:
                errors.append(qi)

        n_q = len(queries)
        mrr = mrr_sum / n_q

        print(f"\n  Results:")
        print(f"    MRR:        {mrr:.4f}")
        for k in sorted(recall_at):
            r = recall_at[k] / n_q
            print(f"    Recall@{k:<3d}  {r:.4f}  ({recall_at[k]}/{n_q})")

        print(f"\n  Worst misses (rank > 20): {len([e for e in errors if e])} / {n_q}")

        worst = sorted(errors, key=lambda qi: -np.where(
            np.argsort(-sim_matrix[qi]) == list(rel_sets[qi])[0] if rel_sets[qi] else -1
        )[0][0] if rel_sets[qi] else 9999)[:5]

        for qi in worst[:5]:
            correct = rel_sets[qi]
            ranked = np.argsort(-sim_matrix[qi])
            first_rank = None
            for rank, idx in enumerate(ranked, 1):
                if idx in correct:
                    first_rank = rank
                    break
            print(f"    Q: \"{queries[qi][:80]}\"")
            print(f"      Correct at rank {first_rank}, group has {len(correct)} matches")
            print(f"      Label: \"{labels[qi][:60]}\"")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load <PROJECT_KEY> eval pairs")
    parser.add_argument("--mode", choices=["dedup", "retrieval", "both"], default="both")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--neg", type=int, default=25)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--large-scale", action="store_true",
                        help="Run full 8000-ticket retrieval benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to benchmark (default: all 4)")
    args = parser.parse_args()

    if args.large_scale:
        run_large_scale_eval(model_names=args.models)
    else:
        pairs = load_pairs(mode=args.mode, pos_limit=args.limit, neg_limit=args.neg)

        if args.output and pairs:
            out = Path(args.output)
            out.write_text(json.dumps(pairs, indent=2))
            print(f"Saved {len(pairs)} pairs to {out}")
        elif pairs:
            cats = Counter(p["category"] for p in pairs)
            print(f"\nCategory breakdown: {dict(cats)}")
            print(f"\nSample pairs:")
            for p in pairs[:5]:
                print(f"  [{p['id']}] rel={p['relevance']} cat={p['category']}")
                print(f"       Q: {p['query'][:100]}")
                print(f"       D: {p['document'][:100]}")
                print()
