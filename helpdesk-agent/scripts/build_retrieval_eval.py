#!/usr/bin/env python3
"""Build the retrieval fusion eval set from the responder corpus.

Usage:
    python -m scripts.build_retrieval_eval
    python -m scripts.build_retrieval_eval --max 500 --output data/retrieval_eval_queries.json
"""

from __future__ import annotations

import argparse
import logging

from db import init_db
from plugins.responder.eval_set import (
    DEFAULT_EVAL_PATH,
    PRODUCTION_TARGET_QUERIES,
    build_eval_manifest,
    save_eval_manifest,
)

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval fusion eval queries from DB corpus")
    parser.add_argument(
        "--max",
        type=int,
        default=PRODUCTION_TARGET_QUERIES,
        help="Maximum labeled queries to emit (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_EVAL_PATH),
        help="Output JSON path",
    )
    parser.add_argument("--no-seeds", action="store_true", help="Skip hand-curated seed queries")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()

    manifest = build_eval_manifest(max_queries=args.max, include_seeds=not args.no_seeds)
    out = save_eval_manifest(manifest, args.output)

    print(f"Wrote {manifest.count} queries to {out}")
    print(f"  query types: {manifest.query_type_counts}")
    print(f"  sources:     {manifest.source_type_counts}")
    if manifest.count < 200:
        print(
            "\nWARNING: fewer than 200 labeled queries — ingest more resolved tickets, "
            "KB articles, and docs before production tuning."
        )


if __name__ == "__main__":
    main()
