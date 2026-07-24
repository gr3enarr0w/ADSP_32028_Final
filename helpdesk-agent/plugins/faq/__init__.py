"""FAQ plugin — gap analysis, generation, and dedup pipeline.

Re-exports from faq.* modules; new code uses plugins.faq, old code still works.
"""

import logging

from plugins._protocol import BasePlugin

# -- Re-exports from faq.sources --
from faq.sources import gather_all_sources, get_source_status

# -- Re-exports from faq.analyzer --
from faq.analyzer import analyze_faq_gaps

# -- Re-exports from faq.generator --
from faq.generator import generate_faq_entry, generate_all_faq_entries

# -- Re-exports from faq.dedup --
from faq.dedup import (
    compute_embedding,
    compute_fingerprint,
    is_duplicate,
    is_semantic_duplicate,
    is_duplicate_of_sections,
    backfill_fingerprints,
    backfill_embeddings,
    rerank_score,
    rerank_pairs,
)

# -- Re-exports from faq.issue_checker --
from faq.issue_checker import check_linked_issues

# -- Re-exports from faq.threshold_calibration --
from faq.threshold_calibration import (
    CalibrationResult,
    calibrate_threshold,
    save_calibration_result,
    load_calibration_result,
)

log = logging.getLogger(__name__)

__all__ = [
    "gather_all_sources", "get_source_status", "analyze_faq_gaps",
    "generate_faq_entry", "generate_all_faq_entries",
    "compute_embedding", "compute_fingerprint",
    "is_duplicate", "is_semantic_duplicate", "is_duplicate_of_sections",
    "backfill_fingerprints", "backfill_embeddings",
    "rerank_score", "rerank_pairs", "check_linked_issues",
    "CalibrationResult", "calibrate_threshold",
    "save_calibration_result", "load_calibration_result", "plugin",
]


class FaqPlugin(BasePlugin):
    """FAQ gap analysis and entry generation pipeline."""

    name = "faq"

    def on_schedule(self) -> None:
        log.info("[faq] scheduled run — gathering sources + gap analysis + generation")
        sources = gather_all_sources()
        analyze_faq_gaps(sources)
        generate_all_faq_entries(sources)
        log.info("[faq] scheduled run complete")


plugin = FaqPlugin()
