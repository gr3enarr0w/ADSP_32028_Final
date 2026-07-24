"""Read content from Google Slides for FAQ source material."""

import logging

from config import FAQ_SLIDES_ID
from config import validate_google_id, mask_id
from utils.google_api import get_google_service
from utils.tracking import upsert_faq_source

log = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/presentations.readonly"]


def _get_slides_service():
    """Build authenticated Google Slides API service."""
    return get_google_service("slides", "v1", _SCOPES)


def read_slides(presentation_id: str | None = None) -> str:
    """Read a Google Slides presentation and return extracted text.

    Returns formatted text with slide numbers and titles.
    """
    presentation_id = presentation_id or FAQ_SLIDES_ID
    if not presentation_id:
        log.warning("No Google Slides ID configured (FAQ_SLIDES_ID)")
        return ""

    validate_google_id(presentation_id)

    service = _get_slides_service()
    presentation = service.presentations().get(presentationId=presentation_id).execute()
    title = presentation.get("title", "Untitled Presentation")

    slides_text: list[str] = []
    for i, slide in enumerate(presentation.get("slides", []), 1):
        slide_title = ""
        body_parts: list[str] = []

        for element in slide.get("pageElements", []):
            shape = element.get("shape")
            if not shape:
                continue

            text_content = shape.get("text", {})
            placeholder_type = shape.get("placeholder", {}).get("type", "")

            text_runs: list[str] = []
            for text_elem in text_content.get("textElements", []):
                text_run = text_elem.get("textRun")
                if text_run:
                    text_runs.append(text_run.get("content", "").strip())

            combined = " ".join(t for t in text_runs if t)
            if not combined:
                continue

            if placeholder_type in ("TITLE", "CENTERED_TITLE"):
                slide_title = combined
            else:
                body_parts.append(combined)

        slide_body = "\n".join(body_parts)
        if slide_title or slide_body:
            slides_text.append(f"--- Slide {i}: {slide_title} ---\n{slide_body}")

    full_text = "\n\n".join(slides_text)

    upsert_faq_source("google_slides", presentation_id, title, full_text)

    log.info("Read Google Slides: %s [%s] (%d slides, %d chars)",
             title, mask_id(presentation_id), len(slides_text), len(full_text))
    return full_text
