"""CQADupStack loader — extracts duplicate Q&A pairs from the BEIR benchmark.

Downloads corpus, queries, and relevance judgments from HuggingFace for
selected StackExchange forums and converts them into our eval pairs format.

Usage:
    python -m eval.loaders.cqadupstack                # default forums
    python -m eval.loaders.cqadupstack --forums unix   # specific forum
    python -m eval.loaders.cqadupstack --limit 20      # fewer pairs
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FORUMS = ["programmers", "webmasters", "wordpress", "unix", "tex", "stats"]
DEFAULT_DUP_PAIRS = 100
DEFAULT_NEG_PAIRS = 50


def _load_hf_dataset(name: str, config: str | None = None, split: str = "test"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install 'datasets' package: pip install datasets", file=sys.stderr)
        sys.exit(1)
    kwargs = {"cache_dir": str(CACHE_DIR / "hf_cache")}
    if config:
        return load_dataset(name, config, split=split, **kwargs)
    return load_dataset(name, split=split, **kwargs)


def load_pairs(
    forums: list[str] | None = None,
    dup_limit: int = DEFAULT_DUP_PAIRS,
    neg_limit: int = DEFAULT_NEG_PAIRS,
    seed: int = 42,
) -> list[dict]:
    forums = forums or DEFAULT_FORUMS
    rng = random.Random(seed)

    print(f"Loading CQADupStack qrels...")
    qrels_ds = _load_hf_dataset("BeIR/cqadupstack-qrels")
    qrel_map: dict[str, set[str]] = defaultdict(set)
    for row in qrels_ds:
        qrel_map[str(row["query-id"])].add(str(row["corpus-id"]))

    all_queries: dict[str, dict] = {}
    all_corpus: dict[str, dict] = {}

    for forum in forums:
        print(f"Loading forum: {forum}...")
        corpus = _load_hf_dataset("BeIR/cqadupstack", forum, split="corpus")
        queries = _load_hf_dataset("BeIR/cqadupstack", forum, split="queries")

        for row in corpus:
            doc_id = str(row["_id"])
            all_corpus[doc_id] = {
                "text": (row.get("title") or "") + " " + row["text"],
                "forum": forum,
            }
        for row in queries:
            qid = str(row["_id"])
            all_queries[qid] = {
                "text": row["text"],
                "forum": forum,
            }

    dup_pairs = []
    for qid, corpus_ids in qrel_map.items():
        if qid not in all_queries:
            continue
        for cid in corpus_ids:
            if cid not in all_corpus:
                continue
            dup_pairs.append((qid, cid))

    rng.shuffle(dup_pairs)
    dup_pairs = dup_pairs[:dup_limit]

    corpus_ids_in_dups = {cid for _, cid in dup_pairs}
    query_ids_in_dups = {qid for qid, _ in dup_pairs}
    available_corpus = [cid for cid in all_corpus if cid not in corpus_ids_in_dups]

    neg_pairs = []
    for qid, _ in dup_pairs[:neg_limit]:
        neg_cid = rng.choice(available_corpus)
        neg_pairs.append((qid, neg_cid))

    pairs = []
    pair_id = 1

    for qid, cid in dup_pairs:
        q = all_queries[qid]
        d = all_corpus[cid]
        pairs.append({
            "id": pair_id,
            "query": q["text"][:500],
            "document": d["text"][:500],
            "relevance": 5,
            "category": f"CQADup-{d['forum']}",
            "source": "cqadupstack",
        })
        pair_id += 1

    for qid, cid in neg_pairs:
        q = all_queries[qid]
        d = all_corpus[cid]
        pairs.append({
            "id": pair_id,
            "query": q["text"][:500],
            "document": d["text"][:500],
            "relevance": 1,
            "category": f"CQANeg-{d['forum']}",
            "source": "cqadupstack",
        })
        pair_id += 1

    print(f"Loaded {len(dup_pairs)} duplicate + {len(neg_pairs)} negative = {len(pairs)} pairs")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CQADupStack eval pairs")
    parser.add_argument("--forums", nargs="+", default=DEFAULT_FORUMS)
    parser.add_argument("--limit", type=int, default=DEFAULT_DUP_PAIRS,
                        help="Max duplicate pairs to extract")
    parser.add_argument("--neg", type=int, default=DEFAULT_NEG_PAIRS,
                        help="Max negative pairs")
    parser.add_argument("--output", type=str, default=None,
                        help="Save to JSON file")
    args = parser.parse_args()

    pairs = load_pairs(forums=args.forums, dup_limit=args.limit, neg_limit=args.neg)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(pairs, indent=2))
        print(f"Saved {len(pairs)} pairs to {out}")
    else:
        print(f"\nSample pairs:")
        for p in pairs[:3]:
            print(f"  [{p['id']}] rel={p['relevance']} cat={p['category']}")
            print(f"       Q: {p['query'][:80]}...")
            print(f"       D: {p['document'][:80]}...")
