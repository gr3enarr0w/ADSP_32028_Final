"""FastAPI router for FAQ service endpoints."""

import hmac

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from config import FAQ_API_TOKEN, validate_google_id
from db import get_db_conn
from faq.sources import get_source_status
from faq.lookup import lookup, format_not_found
from analysis.gaps import get_gap_summary
from utils.html_parser import parse_faq_html as _parse_faq_html

faq_router = APIRouter()
_security = HTTPBearer(auto_error=False)


# ── Auth dependency ──

def _verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> str:
    """Validate Bearer token if FAQ_API_TOKEN is configured."""
    if not FAQ_API_TOKEN:
        return "no-auth"  # Auth disabled when token not set
    if not credentials or not hmac.compare_digest(credentials.credentials, FAQ_API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ── Request/Response models (Pydantic) ──

class LookupRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Ticket key (e.g. <PROJECT_KEY>-1234) or topic keywords")


class LookupResponse(BaseModel):
    found: bool
    query: str
    faq_matches: list[dict]
    kb_matches: list[dict]
    ticket_matches: list[dict]
    response_draft: str | None


class GenerateRequest(BaseModel):
    theme: str | None = Field(None, max_length=500, description="Specific theme to generate for")


class GenerateResponse(BaseModel):
    generated: int
    errors: int


class ExportResponse(BaseModel):
    exported: int


class FAQEntry(BaseModel):
    topic: str
    question: str
    answer: str
    steps: list[str]
    known_limitations: str


# ── Endpoints ──

@faq_router.get("/sources")
def list_sources(_token: str = Depends(_verify_token)):
    """List configured FAQ sources and their freshness."""
    return {
        "sources": get_source_status(),
    }


@faq_router.get("/gaps")
def faq_gaps(_token: str = Depends(_verify_token)):
    """FAQ-specific gap analysis results."""
    summary = get_gap_summary()
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No gap analysis data. Run 'faq_service.py analyze' first.",
        )
    return summary


@faq_router.get("/entries")
def list_faq_entries(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _token: str = Depends(_verify_token),
):
    """List generated FAQ entries."""
    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT id, article_topic, title, format, status, generated_at
            FROM generated_articles
            WHERE format = 'faq'
            ORDER BY generated_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
    return {"entries": [dict(r) for r in rows]}


@faq_router.get("/entries/{entry_id}")
def get_faq_entry(entry_id: int, _token: str = Depends(_verify_token)):
    """Get a single FAQ entry with full content."""
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM generated_articles WHERE id = ? AND format = 'faq'",
            (entry_id,),
        ).fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="FAQ entry not found",
        )
    return dict(row)


@faq_router.post("/lookup", response_model=LookupResponse)
def agent_lookup(
    request: LookupRequest,
    _token: str = Depends(_verify_token),
):
    """Look up existing answers for a ticket or topic.

    Searches FAQ entries, KB articles, and resolved tickets. Returns a
    Slack-formatted response draft with links if matches are found,
    or a 'not found' message if no matches exist.
    """
    result = lookup(request.query)
    if not result["found"]:
        result["response_draft"] = format_not_found(request.query)
    return LookupResponse(**result)


@faq_router.post("/generate", response_model=GenerateResponse,
                  status_code=status.HTTP_201_CREATED)
def trigger_generate(
    request: GenerateRequest,
    _token: str = Depends(_verify_token),
):
    """Trigger FAQ generation for a specific gap (or all gaps)."""
    # Validate theme exists if provided
    if request.theme:
        with get_db_conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM kb_coverage WHERE theme = ?", (request.theme,)
            ).fetchone()
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Theme not found in gap analysis: {request.theme}",
            )

    from faq.sources import gather_all_sources
    from faq.generator import generate_all_faq_entries

    sources = gather_all_sources()
    generated, errors = generate_all_faq_entries(sources, theme_filter=request.theme)
    return GenerateResponse(generated=generated, errors=errors)


@faq_router.post("/export", response_model=ExportResponse)
def trigger_export(_token: str = Depends(_verify_token)):
    """Export FAQ entries to output Google Doc."""
    from faq.google_docs import write_faq_entries

    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT article_topic, title, body_html FROM generated_articles
            WHERE format = 'faq' AND status = 'draft'
            ORDER BY generated_at
        """).fetchall()

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No FAQ entries to export",
        )

    entries = []
    for r in rows:
        r = dict(r)
        parsed = _parse_faq_html(r["body_html"])
        entries.append({
            "topic": r["title"],
            **parsed,
        })

    success = write_faq_entries(entries)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to write to Google Doc",
        )
    return ExportResponse(exported=len(entries))
