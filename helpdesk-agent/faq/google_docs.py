"""Read from and write to Google Docs for FAQ content."""

import logging
from datetime import datetime, timezone

from config import FAQ_SOURCE_DOC_IDS, FAQ_OUTPUT_DOC_ID
from config import validate_google_id, mask_id, apply_cloud_terminology
from faq.dedup import COSINE_THRESHOLD, is_duplicate_of_sections
from utils.google_api import get_google_service
from utils.tracking import upsert_faq_source

log = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/documents"]


def _get_docs_service():
    """Build authenticated Google Docs API service."""
    return get_google_service("docs", "v1", _SCOPES)


def _extract_text(doc: dict) -> list[dict]:
    """Parse Google Docs JSON into heading -> content sections.

    Returns list of {heading: str, content: str}.
    """
    sections: list[dict] = []
    current_heading = ""
    current_content: list[str] = []

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

        if style.startswith("HEADING"):
            if current_heading or current_content:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content),
                })
            current_heading = line
            current_content = []
        else:
            current_content.append(line)

    if current_heading or current_content:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_content),
        })

    return sections


def read_doc(doc_id: str) -> str:
    """Read a single Google Doc and return its text content.

    Also updates the faq_sources table with a content hash for change detection.
    """
    validate_google_id(doc_id)

    service = _get_docs_service()
    doc = service.documents().get(documentId=doc_id).execute()
    title = doc.get("title", "Untitled")

    sections = _extract_text(doc)
    full_text = "\n\n".join(
        f"## {s['heading']}\n{s['content']}" if s["heading"] else s["content"]
        for s in sections
    )

    upsert_faq_source("google_doc", doc_id, title, full_text)

    log.info("Read Google Doc: %s [%s] (%d sections, %d chars)",
             title, mask_id(doc_id), len(sections), len(full_text))
    return full_text


def read_all_source_docs() -> str:
    """Read all configured source Google Docs and return combined text."""
    if not FAQ_SOURCE_DOC_IDS:
        log.warning("No source Google Doc IDs configured (FAQ_SOURCE_DOC_IDS)")
        return ""

    all_text: list[str] = []
    for doc_id in FAQ_SOURCE_DOC_IDS:
        try:
            text = read_doc(doc_id)
            if text:
                all_text.append(text)
        except Exception as e:
            log.warning("Failed to read Google Doc %s: %s", mask_id(doc_id), e)

    combined = "\n\n---\n\n".join(all_text)
    log.info("Read %d source docs (%d total chars)", len(all_text), len(combined))
    return combined


def write_faq_entries(entries: list[dict], doc_id: str | None = None) -> bool:
    """Write FAQ entries to the output Google Doc (replaces content).

    Each entry: {topic, question, answer, steps: [], known_limitations}
    """
    doc_id = doc_id or FAQ_OUTPUT_DOC_ID
    if not doc_id:
        log.error("No output Google Doc ID configured (FAQ_OUTPUT_DOC_ID)")
        return False

    validate_google_id(doc_id)
    service = _get_docs_service()

    # Get current doc to find content length for clearing
    doc = service.documents().get(documentId=doc_id).execute()
    body = doc.get("body", {})
    content = body.get("content", [])
    end_index = content[-1].get("endIndex", 1) if content else 1

    # ── Output-layer dedup gate ──
    # Compare each incoming entry against sections already in the doc.
    # Skip any entry whose content is a near-duplicate of an existing section.
    existing_sections = _extract_text(doc)
    section_texts = [
        f"{s['heading']}\n{s['content']}" if s["heading"] else s["content"]
        for s in existing_sections
    ]

    if section_texts:
        filtered: list[dict] = []
        for entry in entries:
            # Build a comparable plain-text representation of this entry
            parts = []
            if entry.get("topic"):
                parts.append(entry["topic"])
            if entry.get("question"):
                parts.append(entry["question"])
            if entry.get("answer"):
                parts.append(entry["answer"])
            for step in entry.get("steps", []):
                parts.append(step)
            if entry.get("known_limitations"):
                parts.append(entry["known_limitations"])
            article_text = "\n".join(parts)

            is_dup, match_idx, sim = is_duplicate_of_sections(
                article_text, section_texts, threshold=COSINE_THRESHOLD,
            )
            if is_dup:
                match_heading = existing_sections[match_idx]["heading"] if match_idx is not None else "?"
                log.warning(
                    "Skipping duplicate FAQ entry '%s' — matches existing doc "
                    "section '%s' (cosine=%.3f)",
                    entry.get("topic", "?"), match_heading, sim,
                )
                continue
            filtered.append(entry)

        skipped = len(entries) - len(filtered)
        if skipped:
            log.info(
                "Output-layer dedup: %d of %d entries skipped as duplicates",
                skipped, len(entries),
            )
        entries = filtered

    if not entries:
        log.info("No new entries to write after dedup filtering")
        return True

    requests: list[dict] = []

    # Clear existing content (leave the first newline)
    if end_index > 2:
        requests.append({
            "deleteContentRange": {
                "range": {"startIndex": 1, "endIndex": end_index - 1}
            }
        })

    # Build content from entries
    insert_parts: list[tuple[str, str]] = []

    insert_parts.append(("Atlassian Cloud Migration FAQ\n", "HEADING_1"))
    insert_parts.append((
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n",
        "NORMAL_TEXT",
    ))

    for entry in entries:
        topic = apply_cloud_terminology(entry.get("topic", "Untitled"))
        question = apply_cloud_terminology(entry.get("question", ""))
        answer = apply_cloud_terminology(entry.get("answer", ""))
        steps = [apply_cloud_terminology(s) for s in entry.get("steps", [])]
        limitations = apply_cloud_terminology(entry.get("known_limitations", ""))

        insert_parts.append((f"{topic}\n", "HEADING_2"))

        if question:
            insert_parts.append((f"Q: {question}\n\n", "NORMAL_TEXT"))
        if answer:
            insert_parts.append((f"{answer}\n\n", "NORMAL_TEXT"))
        if steps:
            insert_parts.append(("Steps:\n", "HEADING_3"))
            for i, step in enumerate(steps, 1):
                insert_parts.append((f"{i}. {step}\n", "NORMAL_TEXT"))
            insert_parts.append(("\n", "NORMAL_TEXT"))
        if limitations:
            insert_parts.append(("Known Limitations\n", "HEADING_3"))
            insert_parts.append((f"{limitations}\n\n", "NORMAL_TEXT"))

    # Build insertText + updateParagraphStyle requests
    cursor = 1
    for text, style in insert_parts:
        requests.append({
            "insertText": {"location": {"index": cursor}, "text": text}
        })
        if style != "NORMAL_TEXT":
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": cursor, "endIndex": cursor + len(text)},
                    "paragraphStyle": {"namedStyleType": style},
                    "fields": "namedStyleType",
                }
            })
        cursor += len(text)

    if requests:
        service.documents().batchUpdate(
            documentId=doc_id, body={"requests": requests}
        ).execute()

    log.info("Wrote %d FAQ entries to Google Doc [%s]", len(entries), mask_id(doc_id))
    return True
