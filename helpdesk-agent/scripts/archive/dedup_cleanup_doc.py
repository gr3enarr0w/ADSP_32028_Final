"""One-time script to remove duplicate articles from the FAQ output Google Doc.

Usage:
    python scripts/dedup_cleanup_doc.py [--dry-run] [--doc-id DOC_ID]

Identifies duplicate articles using cosine similarity (same threshold as the
write-time dedup gate) and rewrites the doc with only unique articles, keeping
the first occurrence of each duplicate group.

Each FAQ article (HEADING_2 topic + all its Steps/Known Limitations subsections)
is treated as one atomic unit for comparison — subsections are NOT compared
independently, which prevents false positives from shared boilerplate text
like "None at this time." appearing in multiple articles' Known Limitations.

Records before/after article counts to stdout for the ANTSE-290 acceptance criteria.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FAQ_OUTPUT_DOC_ID, DEDUP_COSINE_THRESHOLD, mask_id
from faq.dedup import is_duplicate_of_sections
from faq.google_docs import _get_docs_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _extract_article_groups(doc: dict) -> list[dict]:
    """Parse Google Docs JSON into article-level units, splitting only on HEADING_2.

    HEADING_3 paragraphs (Steps, Known Limitations) are treated as content within
    their parent HEADING_2 article, not as separate sections. HEADING_1 (doc title)
    is skipped — it is re-added manually during rewrite.

    Each returned group has:
      - heading:    the HEADING_2 article title (str)
      - full_text:  heading + all subsection text concatenated (for dedup comparison)
      - paragraphs: list of {text, style} for structure-preserving rewrite
    """
    groups: list[dict] = []
    current_heading = ""
    current_texts: list[str] = []
    current_paragraphs: list[dict] = []

    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue

        style = paragraph.get("paragraphStyle", {}).get("namedStyleType", "")
        text_parts: list[str] = []
        for elem in paragraph.get("elements", []):
            text_run = elem.get("textRun")
            if text_run:
                text_parts.append(text_run.get("content", ""))

        line = "".join(text_parts).strip()
        if not line:
            continue

        if style == "HEADING_2":
            if current_heading or current_texts:
                groups.append({
                    "heading": current_heading,
                    "full_text": f"{current_heading}\n" + "\n".join(current_texts),
                    "paragraphs": current_paragraphs,
                })
            current_heading = line
            current_texts = []
            current_paragraphs = [{"text": line, "style": "HEADING_2"}]
        elif style == "HEADING_1":
            # Doc title — flush any pending group, skip the title itself
            if current_heading or current_texts:
                groups.append({
                    "heading": current_heading,
                    "full_text": f"{current_heading}\n" + "\n".join(current_texts),
                    "paragraphs": current_paragraphs,
                })
            current_heading = ""
            current_texts = []
            current_paragraphs = []
        else:
            # HEADING_3, NORMAL_TEXT, etc. — part of the current article
            current_texts.append(line)
            current_paragraphs.append({"text": line, "style": style or "NORMAL_TEXT"})

    if current_heading or current_texts:
        groups.append({
            "heading": current_heading,
            "full_text": f"{current_heading}\n" + "\n".join(current_texts),
            "paragraphs": current_paragraphs,
        })

    return groups


def find_duplicate_articles(groups: list[dict], threshold: float) -> list[int]:
    """Return indices of article groups that are duplicates of an earlier group."""
    kept_texts: list[str] = []
    dup_indices: list[int] = []

    for i, group in enumerate(groups):
        text = group["full_text"]
        if not text.strip():
            dup_indices.append(i)
            continue

        if kept_texts:
            is_dup, match_idx, sim = is_duplicate_of_sections(
                text, kept_texts, threshold=threshold
            )
            if is_dup:
                match_heading = groups[match_idx]["heading"] if match_idx is not None else "?"
                log.info(
                    "  [%d] DUPLICATE of [%d] '%s' (cosine=%.3f): '%s'",
                    i, match_idx, match_heading, sim,
                    group["heading"][:60],
                )
                dup_indices.append(i)
                continue

        kept_texts.append(text)

    return dup_indices


def rewrite_doc(service, doc_id: str, unique_groups: list[dict]) -> None:
    doc = service.documents().get(documentId=doc_id).execute()
    body = doc.get("body", {})
    content = body.get("content", [])
    end_index = content[-1].get("endIndex", 1) if content else 1

    requests: list[dict] = []

    if end_index > 2:
        requests.append({
            "deleteContentRange": {
                "range": {"startIndex": 1, "endIndex": end_index - 1}
            }
        })

    cursor = 1

    # Re-add the doc title as HEADING_1
    title_text = "Atlassian Cloud Migration FAQ\n"
    requests.append({"insertText": {"location": {"index": cursor}, "text": title_text}})
    requests.append({
        "updateParagraphStyle": {
            "range": {"startIndex": cursor, "endIndex": cursor + len(title_text)},
            "paragraphStyle": {"namedStyleType": "HEADING_1"},
            "fields": "namedStyleType",
        }
    })
    cursor += len(title_text)

    for group in unique_groups:
        for para in group["paragraphs"]:
            text = para["text"] + "\n"
            style = para["style"]
            requests.append({"insertText": {"location": {"index": cursor}, "text": text}})
            if style not in ("NORMAL_TEXT", ""):
                requests.append({
                    "updateParagraphStyle": {
                        "range": {"startIndex": cursor, "endIndex": cursor + len(text)},
                        "paragraphStyle": {"namedStyleType": style},
                        "fields": "namedStyleType",
                    }
                })
            cursor += len(text)

        # Blank line between articles
        requests.append({"insertText": {"location": {"index": cursor}, "text": "\n"}})
        cursor += 1

    if requests:
        service.documents().batchUpdate(
            documentId=doc_id, body={"requests": requests}
        ).execute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate FAQ output Google Doc")
    parser.add_argument("--dry-run", action="store_true", help="Report duplicates without modifying the doc")
    parser.add_argument("--doc-id", default=FAQ_OUTPUT_DOC_ID, help="Google Doc ID to clean up")
    parser.add_argument("--threshold", type=float, default=DEDUP_COSINE_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {DEDUP_COSINE_THRESHOLD})")
    args = parser.parse_args()

    doc_id = args.doc_id
    if not doc_id:
        log.error("No doc ID — set FAQ_OUTPUT_DOC_ID or pass --doc-id")
        sys.exit(1)

    log.info("Fetching doc [%s]...", mask_id(doc_id))
    service = _get_docs_service()
    doc = service.documents().get(documentId=doc_id).execute()
    title = doc.get("title", "Untitled")
    groups = _extract_article_groups(doc)

    count_before = len(groups)
    log.info("Doc: '%s'  |  Articles before: %d  |  Threshold: %.2f",
             title, count_before, args.threshold)

    log.info("Scanning for duplicate articles...")
    dup_indices = find_duplicate_articles(groups, threshold=args.threshold)
    unique_groups = [g for i, g in enumerate(groups) if i not in set(dup_indices)]
    count_after = len(unique_groups)
    removed = count_before - count_after

    print(f"\n{'='*60}")
    print(f"Doc:             {title}")
    print(f"Articles before: {count_before}")
    print(f"Duplicates found:{removed}")
    print(f"Articles after:  {count_after}")
    print(f"Threshold used:  {args.threshold}")
    print(f"{'='*60}\n")

    if removed == 0:
        log.info("No duplicate articles found — doc is clean.")
        return

    if args.dry_run:
        log.info("DRY RUN — no changes written.")
        return

    log.info("Rewriting doc with %d unique articles...", count_after)
    rewrite_doc(service, doc_id, unique_groups)
    log.info("Done. Removed %d duplicate articles.", removed)

    # Sampled review output for AC: print 20 evenly-spaced article headings
    print("Sample of remaining articles (for uniqueness review):")
    sample_size = min(20, count_after)
    step = max(1, count_after // sample_size)
    for i in range(0, count_after, step):
        heading = unique_groups[i]["heading"] or "(no heading)"
        print(f"  [{i:3d}] {heading[:80]}")


if __name__ == "__main__":
    main()
