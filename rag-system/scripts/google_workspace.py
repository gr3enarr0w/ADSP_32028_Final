"""
Google Workspace (Drive / Docs / Sheets) READ-ONLY ingestion connector for the
Hermes RAG skill.

This is a completely independent Google integration: it does NOT reuse any
other tool/skill's stored OAuth credentials. It expects the user's own,
separate Google Cloud API project with an OAuth2 "Desktop app" client, and
authenticates using the standard `google-auth-oauthlib` Desktop-app flow —
the same pattern documented in Google's own quickstarts, e.g.:
  https://developers.google.com/workspace/drive/api/quickstart/python
  https://developers.google.com/workspace/docs/api/quickstart/python
  https://developers.google.com/workspace/sheets/api/quickstart/python

Flow, matching Google's documented pattern exactly:
  1. If a cached token file exists, load it via
     `google.oauth2.credentials.Credentials.from_authorized_user_file()`.
  2. If the loaded credentials are expired but carry a refresh token, refresh
     them silently via `google.auth.transport.requests.Request()`.
  3. Otherwise, run the interactive Desktop-app consent flow via
     `google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file()`
     + `.run_local_server(port=0)` — this opens the user's local browser to
     Google's consent screen and spins up a temporary local HTTP server to
     receive the OAuth redirect. This step can ONLY be done by the user
     interactively; it cannot be scripted/faked.
  4. Persist the resulting credentials (including refresh token) to
     `token_path` (`creds.to_json()`) so future runs skip step 3.

Scopes are READ-ONLY on purpose (least privilege for an ingestion-only
feature — this affects what the OAuth consent screen shows the user):
  - https://www.googleapis.com/auth/drive.readonly
  - https://www.googleapis.com/auth/documents.readonly
  - https://www.googleapis.com/auth/spreadsheets.readonly
  - https://www.googleapis.com/auth/presentations.readonly

NOTE: the presentations.readonly scope was added after documents/spreadsheets
(see docs/CHANGELOG.md 2026-07-21 for Slides support). Any previously cached
token.json was authorized WITHOUT this scope, so it will no longer satisfy
`Credentials.from_authorized_user_file(...)` against the updated SCOPES list
above — the next real run must go through the interactive consent flow again
(step 3 in the Flow description above) to mint a token covering all four
scopes. This is expected, one-time, and cannot be avoided/scripted around.

SECURITY NOTE: `client_secret.json` and the cached `token.json` are real
secrets (an OAuth client secret and a long-lived refresh token,
respectively). Nothing in this module ever logs/prints their contents. Keep
both out of version control (see the project's .gitignore).
"""

import os
from typing import Any, Dict, List, Optional

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    _GOOGLE_LIBS_AVAILABLE = True
except ImportError:
    Request = Credentials = InstalledAppFlow = build = None  # type: ignore
    _GOOGLE_LIBS_AVAILABLE = False

# Read-only scopes only — this is an ingestion (read) feature, never a write
# feature, and requesting narrower scopes means a less alarming OAuth
# consent screen for the user.
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/presentations.readonly",
]

# MIME types this connector knows how to turn into ingestible plain text.
GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"
GOOGLE_SLIDES_MIME_TYPE = "application/vnd.google-apps.presentation"


def _require_google_libs():
    if not _GOOGLE_LIBS_AVAILABLE:
        raise ImportError(
            "Google Workspace ingestion requires the google-api-python-client, "
            "google-auth-httplib2, and google-auth-oauthlib packages. Install with:\n"
            "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )


def _escape_drive_query_literal(value: str) -> str:
    """Escape a literal string for embedding in a Drive API `q` search query
    (Drive's query language uses single-quoted string literals; backslash and
    single-quote inside the literal must be backslash-escaped)."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


class GoogleWorkspaceClient:
    """Thin, read-only wrapper around the Drive/Docs/Sheets APIs, authenticated
    via a user-provided OAuth2 Desktop-app `client_secret.json` from the
    user's OWN Google Cloud project. Never reuses credentials belonging to
    any other tool/skill.
    """

    def __init__(self, client_secret_path: str, token_path: str):
        """
        Args:
            client_secret_path: path to the OAuth2 Desktop-app client secret
                JSON downloaded from Google Cloud Console
                (APIs & Services > Credentials > Create Credentials > OAuth
                client ID > Desktop app). Must point to a REAL file — this
                class deliberately does not fabricate or search for one.
            token_path: path where the cached OAuth token (including refresh
                token) will be read from / written to after the first
                successful interactive consent flow.
        """
        _require_google_libs()
        self.client_secret_path = os.path.expanduser(client_secret_path) if client_secret_path else client_secret_path
        self.token_path = os.path.expanduser(token_path) if token_path else token_path
        self._creds = None
        self._drive_service = None
        self._docs_service = None
        self._sheets_service = None
        self._slides_service = None

    def _load_credentials(self):
        """Load cached credentials, refreshing or running the interactive
        Desktop-app consent flow as needed. Mirrors Google's own quickstart
        pattern (see module docstring for links) essentially verbatim."""
        creds = None

        if self.token_path and os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.client_secret_path or not os.path.exists(self.client_secret_path):
                    raise FileNotFoundError(
                        f"Google OAuth client secret not found at "
                        f"{self.client_secret_path!r}. This feature requires YOUR OWN "
                        "Google Cloud API project's OAuth2 'Desktop app' client secret "
                        "(Google Cloud Console > APIs & Services > Credentials > Create "
                        "Credentials > OAuth client ID > Desktop app > Download JSON). "
                        "Set google_workspace.client_secret_path in config.yaml (or pass "
                        "--google-client-secret) to point at the downloaded file. This "
                        "is a brand-new, separate credential — it is never fabricated "
                        "or auto-discovered by this code."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_path, SCOPES)
                # Opens the user's local browser to Google's consent screen and runs
                # a temporary local HTTP server to receive the OAuth redirect. This
                # is the ONE step that requires the user to interact manually.
                creds = flow.run_local_server(port=0)

            if self.token_path:
                token_dir = os.path.dirname(self.token_path)
                if token_dir:
                    os.makedirs(token_dir, exist_ok=True)
                with open(self.token_path, "w") as token_file:
                    token_file.write(creds.to_json())

        self._creds = creds
        return creds

    def _get_service(self, api_name: str, api_version: str):
        creds = self._creds or self._load_credentials()
        return build(api_name, api_version, credentials=creds)

    @property
    def drive(self):
        if self._drive_service is None:
            self._drive_service = self._get_service("drive", "v3")
        return self._drive_service

    @property
    def docs(self):
        if self._docs_service is None:
            self._docs_service = self._get_service("docs", "v1")
        return self._docs_service

    @property
    def sheets(self):
        if self._sheets_service is None:
            self._sheets_service = self._get_service("sheets", "v4")
        return self._sheets_service

    @property
    def slides(self):
        if self._slides_service is None:
            self._slides_service = self._get_service("slides", "v1")
        return self._slides_service

    def search_drive(
        self,
        query: str,
        max_results: int = 10,
        involvement: str = "owner_or_writer",
    ) -> List[Dict[str, Any]]:
        """Search Drive for files matching `query` (free-text, matched against
        file content and metadata via Drive's `fullText contains` operator),
        excluding trashed files, optionally filtered to files the current
        user is actually INVOLVED with (not just able to see — some orgs
        share broadly by default, so visibility != engagement).

        Args:
            query: free-text search term (matched via `fullText contains`).
            max_results: max files to return (Drive API `pageSize`). Note
                that in `"owner_writer_or_commented"` mode the actual number
                of files inspected/returned can exceed this, since a second,
                independent query is also run (see below).
            involvement: one of:
                - "owner_only": query-level filter, `'me' in owners`.
                - "owner_or_writer" (default): query-level filter,
                  `'me' in owners or 'me' in writers`.
                - "owner_writer_or_commented": same query-level owner/writer
                  filter, PLUS a supplementary pass that re-runs the search
                  WITHOUT any involvement filter and, for every additional
                  file not already matched by owner/writer, calls the Drive
                  API's `comments.list` endpoint to check whether the
                  current user has left a comment on it (there is no
                  query-level operator for "has commented" — see
                  `_has_user_commented` for why this must be a post-filter).
                  This is a heavier operation (one extra Drive API call per
                  extra candidate file), so it is opt-in only.
                - "any": no involvement filter at all — the original,
                  visibility-only behavior (backward compatible default
                  prior to this filter's introduction).

        Both `'me' in owners` and `'me' in writers` are real, documented
        Drive API v3 query terms/operators (see the `owners`/`writers`
        entries and the `in` operator at
        https://developers.google.com/workspace/drive/api/guides/ref-search-terms
        and the general query-string guide at
        https://developers.google.com/workspace/drive/api/guides/search-files
        , which shows the `in` operator used with `owners`/`writers` against
        an explicit user identifier). The literal special value `'me'` for
        these terms — resolving to the currently authenticated user — is
        Google's own documented shorthand, e.g. used verbatim as
        `'me' in owners` in Google's Apps Script "Importing and Exporting
        Projects" guide:
        https://developers.google.com/apps-script/guides/import-export

        Returns a list of dicts, each with: id, name, mimeType, modifiedTime,
        webViewLink (fields requested straight from the Drive API — no local
        renaming/mutation).
        """
        valid_modes = ("owner_only", "owner_or_writer", "owner_writer_or_commented", "any")
        if involvement not in valid_modes:
            raise ValueError(f"involvement must be one of {valid_modes}, got {involvement!r}")

        if involvement == "owner_writer_or_commented":
            owner_writer_files = self._search_drive_raw(query, max_results, "owner_or_writer")
            matched_ids = {f["id"] for f in owner_writer_files}

            # Supplementary pass: re-run with NO involvement filter (the
            # full visible/searchable result set) and comment-check every
            # file not already matched by the owner/writer query above, to
            # also catch "commented on someone else's doc" cases.
            all_visible_files = self._search_drive_raw(query, max_results, "any")
            commented_files = [
                f for f in all_visible_files
                if f["id"] not in matched_ids and self._has_user_commented(f["id"])
            ]
            return owner_writer_files + commented_files

        return self._search_drive_raw(query, max_results, involvement)

    def _search_drive_raw(self, query: str, max_results: int, involvement: str) -> List[Dict[str, Any]]:
        """Run a single Drive `files.list` query with the given query-level
        `involvement` filter applied (no comment post-filtering — see
        `search_drive` for that)."""
        drive_query = f"fullText contains '{_escape_drive_query_literal(query)}' and trashed = false"
        if involvement == "owner_only":
            drive_query += " and 'me' in owners"
        elif involvement in ("owner_or_writer", "owner_writer_or_commented"):
            drive_query += " and ('me' in owners or 'me' in writers)"
        elif involvement != "any":
            raise ValueError(f"unknown involvement mode: {involvement!r}")

        response = self.drive.files().list(
            q=drive_query,
            pageSize=max_results,
            fields="files(id, name, mimeType, modifiedTime, webViewLink)",
        ).execute()
        return response.get("files", [])

    def _has_user_commented(self, file_id: str) -> bool:
        """Return True if the current user has left at least one (non-deleted)
        comment on `file_id`.

        There is no Drive API v3 `files.list` query-level operator for "has
        the current user commented on this file" (the documented file query
        terms — https://developers.google.com/workspace/drive/api/guides/ref-search-terms
        — cover owners/writers/readers/sharedWithMe/etc. but nothing
        comment-related), so this is implemented as a genuine post-filter
        API call: `comments.list` —
        https://developers.google.com/workspace/drive/api/reference/rest/v3/comments/list
        (`GET .../files/{fileId}/comments`) — for each candidate file,
        checking each returned Comment's `author` (a User resource —
        https://developers.google.com/workspace/drive/api/reference/rest/v3/User)
        for `author.me == True`, per that resource's documented `me` field
        ("Whether this user is the requesting user.").

        Paginates through all comment pages (bounded to a generous cap to
        avoid pathological runaway on a file with an enormous comment
        thread). Deleted comments are excluded (Drive's default
        `includeDeleted=false`) since a deleted comment's content/author
        association is no longer meaningful engagement signal. Errors (e.g.
        comments not supported/permitted for a given file) are swallowed and
        treated as "no comment found" rather than aborting the whole search.
        """
        try:
            page_token = None
            for _ in range(10):  # cap: up to 10 pages (<=1000 comments) per file
                request_kwargs = {
                    "fileId": file_id,
                    "fields": "nextPageToken, comments(author(me))",
                    "pageSize": 100,
                }
                if page_token:
                    request_kwargs["pageToken"] = page_token
                response = self.drive.comments().list(**request_kwargs).execute()
                for comment in response.get("comments", []) or []:
                    author = comment.get("author", {}) or {}
                    if author.get("me"):
                        return True
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
            return False
        except Exception:
            # Don't let one file's comment lookup failure (e.g. unsupported
            # file type, transient API error) abort the whole search.
            return False

    def get_doc_text(self, doc_id: str) -> str:
        """Fetch a Google Doc via the Docs API and flatten its structured JSON
        body into plain text (see `flatten_google_doc`)."""
        document = self.docs.documents().get(documentId=doc_id).execute()
        return flatten_google_doc(document)

    def get_spreadsheet_sheet_titles(self, spreadsheet_id: str) -> List[str]:
        """Return the titles of all tabs/sheets in a spreadsheet, in the
        order the Sheets API reports them, via `spreadsheets.get()`
        (https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/get)
        requesting only `sheets.properties.title`.

        Do NOT assume a spreadsheet's first tab is named literally "Sheet1"
        — that's merely Sheets' default name for a brand-new blank
        spreadsheet's first tab, not a guarantee. Any spreadsheet whose
        first (or only) tab has been renamed, or that was created
        programmatically with a different name (e.g. a date-stamped title
        like "2026-04-16T183930Z_audit_log"), will 400 with "Unable to
        parse range: Sheet1" if a caller hardcodes that literal. Calling
        this first and using an actual returned title avoids that.
        """
        response = self.sheets.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields="sheets.properties.title",
        ).execute()
        return [
            sheet["properties"]["title"]
            for sheet in response.get("sheets", []) or []
            if sheet.get("properties", {}).get("title")
        ]

    def get_slides_text(self, presentation_id: str) -> str:
        """Fetch a Google Slides presentation via the Slides API and flatten
        its structured JSON body into plain text (see `flatten_google_slides`)."""
        presentation = self.slides.presentations().get(
            presentationId=presentation_id
        ).execute()
        return flatten_google_slides(presentation)

    def get_sheet_values(self, sheet_id: str, range_: str) -> List[List[str]]:
        """Fetch a value range from a Google Sheet via the Sheets API.
        `range_` follows the Sheets A1 notation (e.g. "Sheet1", "Sheet1!A1:D50").
        Returns the raw `values` list-of-lists as given by the API (missing
        trailing cells in a row are simply absent, per the Sheets API's own
        behavior — not padded here)."""
        result = self.sheets.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=range_
        ).execute()
        return result.get("values", [])


def _flatten_paragraph(paragraph: Dict[str, Any]) -> str:
    """Flatten a single Docs API `paragraph` structural element's `elements`
    array into plain text, concatenating each `textRun.content` in order.
    Non-text elements (inline images, footnote references, page/column
    breaks, horizontal rules) are skipped except where they should render as
    a line break in plain text."""
    text = ""
    for element in paragraph.get("elements", []) or []:
        text_run = element.get("textRun")
        if text_run is not None:
            text += text_run.get("content", "")
        elif "horizontalRule" in element:
            text += "\n"
        elif "pageBreak" in element:
            text += "\n"
        # inlineObjectElement / footnoteReference / columnBreak / autoText /
        # richLink etc. carry no directly extractable plain text — skipped.
    return text


def _flatten_table(table: Dict[str, Any]) -> str:
    """Flatten a Docs API `table` structural element into plain text: each
    row's cells are flattened recursively (a cell's `content` is itself a
    list of structural elements, same shape as the document body) and joined
    with " | ", rows joined with newlines."""
    row_lines = []
    for row in table.get("tableRows", []) or []:
        cell_texts = []
        for cell in row.get("tableCells", []) or []:
            cell_content = cell.get("content", []) or []
            cell_texts.append(_flatten_structural_elements(cell_content).strip())
        row_lines.append(" | ".join(cell_texts))
    return "\n".join(row_lines) + ("\n" if row_lines else "")


def _flatten_structural_elements(elements: List[Dict[str, Any]]) -> str:
    """Flatten a list of Docs API structural elements (as found in
    `body.content`, or recursively inside a table cell's `content`, or a
    table-of-contents' `content`) into plain text."""
    parts = []
    for element in elements or []:
        if "paragraph" in element:
            parts.append(_flatten_paragraph(element["paragraph"]))
        elif "table" in element:
            parts.append(_flatten_table(element["table"]))
        elif "tableOfContents" in element:
            toc_content = (element["tableOfContents"] or {}).get("content", []) or []
            parts.append(_flatten_structural_elements(toc_content))
        # "sectionBreak" elements carry no text content — nothing to add.
    return "".join(parts)


def flatten_google_doc(document: Dict[str, Any]) -> str:
    """Flatten a Google Docs API `documents.get` response (the full document
    resource) into plain text.

    Walks the well-documented Docs API structure:
      document.body.content[]  (structural elements, in order)
        .paragraph.elements[].textRun.content   -> plain text runs
        .table.tableRows[].tableCells[].content[]  -> recursively flattened
        .tableOfContents.content[]                 -> recursively flattened
        .sectionBreak                               -> no text

    This is a real structural walk of the nested JSON (not a raw str() of
    the response) so ingested Google Docs content is coherent, readable text
    rather than a JSON dump.

    Returns the flattened, whitespace-trimmed plain text. An empty/absent
    body yields an empty string rather than raising.
    """
    body = document.get("body", {}) or {}
    content = body.get("content", []) or []
    return _flatten_structural_elements(content).strip()


def _flatten_slide_page_element(page_element: Dict[str, Any]) -> str:
    """Flatten a single Slides API `pageElement`'s visible text into plain
    text. Only `shape.text.textElements[].textRun.content` is extracted
    (standard Slides API structure for text boxes, titles, body placeholders,
    etc.) — non-text elements (images, videos, lines, tables are handled
    separately by the caller, sheet charts) contribute nothing here."""
    shape = page_element.get("shape") or {}
    text_content = shape.get("text") or {}
    parts = []
    for text_element in text_content.get("textElements", []) or []:
        text_run = text_element.get("textRun")
        if text_run is not None:
            parts.append(text_run.get("content", ""))
    return "".join(parts)


def _flatten_slide_table(page_element: Dict[str, Any]) -> str:
    """Flatten a Slides API `table` pageElement into plain text: each row's
    cells are flattened (a cell's `text.textElements[]`, same shape as a
    shape's text) and joined with " | ", rows joined with newlines."""
    table = page_element.get("table") or {}
    row_lines = []
    for row in table.get("tableRows", []) or []:
        cell_texts = []
        for cell in row.get("tableCells", []) or []:
            cell_text_content = cell.get("text") or {}
            cell_parts = []
            for text_element in cell_text_content.get("textElements", []) or []:
                text_run = text_element.get("textRun")
                if text_run is not None:
                    cell_parts.append(text_run.get("content", ""))
            cell_texts.append("".join(cell_parts).strip())
        row_lines.append(" | ".join(cell_texts))
    return "\n".join(row_lines)


def flatten_google_slides(presentation: Dict[str, Any]) -> str:
    """Flatten a Google Slides API `presentations.get` response (the full
    presentation resource) into plain text, one section per slide separated
    by a clear `--- Slide N ---` boundary marker so retrieved chunks retain
    some structure/citability (rather than one undifferentiated text blob).

    Walks the documented Slides API structure:
      presentation.slides[]  (in slide order)
        .pageElements[].shape.text.textElements[].textRun.content -> text runs
        .pageElements[].table.tableRows[].tableCells[].text.textElements[]
            -> recursively flattened as " | "-joined rows

    Non-text page elements (images, videos, lines, word art, etc.) are
    skipped — they carry no directly extractable plain text.

    Returns the flattened, whitespace-trimmed plain text. An
    empty/absent slide list yields an empty string rather than raising.
    Slides with no extractable text still get their boundary marker (so
    slide numbering stays accurate against the real deck) but contribute no
    body text beneath it.
    """
    slides = presentation.get("slides", []) or []
    sections = []
    for idx, slide in enumerate(slides, start=1):
        slide_parts = []
        for page_element in slide.get("pageElements", []) or []:
            if "shape" in page_element:
                text = _flatten_slide_page_element(page_element)
                if text.strip():
                    slide_parts.append(text.strip())
            elif "table" in page_element:
                text = _flatten_slide_table(page_element)
                if text.strip():
                    slide_parts.append(text.strip())
            # image/video/line/wordArt/sheetsChart page elements carry no
            # directly extractable plain text — skipped.
        body = "\n".join(slide_parts)
        sections.append(f"--- Slide {idx} ---\n{body}".rstrip())
    return "\n\n".join(sections).strip()


def sheet_values_to_text(values: List[List[str]]) -> str:
    """Convert a Sheets API `values.get()` result (list of rows, each a list
    of cell strings) into a single ingestible plain-text blob: one line per
    row, cells joined with " | ". Ragged rows (Sheets omits trailing empty
    cells) are joined as-is, no padding."""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in values)
