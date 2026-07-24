"""Read from Google Sheets for FAQ documentation source material.

Reads all worksheets from configured source spreadsheets and returns
the content as text for gap analysis comparison.
"""

import logging
import os

import gspread

from config import FAQ_SOURCE_SHEET_IDS, GOOGLE_SERVICE_ACCOUNT_JSON, mask_id
from utils.tracking import upsert_faq_source

log = logging.getLogger(__name__)


def _get_sheets_client():
    """Build authenticated gspread client."""
    sa_path = GOOGLE_SERVICE_ACCOUNT_JSON
    if not os.path.isabs(sa_path):
        sa_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sa_path)
    return gspread.service_account(filename=sa_path)


def read_sheet(sheet_id: str) -> str:
    """Read all worksheets from a Google Sheet and return as text.

    Each worksheet is formatted with its tab name as a heading,
    followed by rows as pipe-delimited text (header row bolded).
    """
    gc = _get_sheets_client()
    spreadsheet = gc.open_by_key(sheet_id)
    title = spreadsheet.title

    all_text = []
    for worksheet in spreadsheet.worksheets():
        rows = worksheet.get_all_values()
        if not rows:
            continue

        tab_text = [f"## {worksheet.title}"]

        # First row as headers
        headers = rows[0]
        tab_text.append(" | ".join(headers))
        tab_text.append(" | ".join(["---"] * len(headers)))

        for row in rows[1:]:
            # Skip completely empty rows
            if not any(cell.strip() for cell in row):
                continue
            tab_text.append(" | ".join(row))

        all_text.append("\n".join(tab_text))

    full_text = "\n\n".join(all_text)

    upsert_faq_source("google_sheet", sheet_id, title, full_text)

    log.info("Read Google Sheet: %s [%s] (%d tabs, %d chars)",
             title, mask_id(sheet_id), len(spreadsheet.worksheets()), len(full_text))
    return full_text


def read_source_sheets() -> str:
    """Read all configured source Google Sheets and return combined text."""
    if not FAQ_SOURCE_SHEET_IDS:
        log.warning("No source Google Sheet IDs configured (FAQ_SOURCE_SHEET_IDS)")
        return ""

    all_text = []
    for sheet_id in FAQ_SOURCE_SHEET_IDS:
        try:
            text = read_sheet(sheet_id)
            if text:
                all_text.append(text)
        except Exception as e:
            log.warning("Failed to read Google Sheet %s: %s", mask_id(sheet_id), e)

    combined = "\n\n---\n\n".join(all_text)
    log.info("Read %d source sheets (%d total chars)", len(all_text), len(combined))
    return combined
