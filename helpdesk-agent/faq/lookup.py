# DEPRECATED: use plugins/responder/lookup.py — ANTSE-309

"""Agent lookup service — search FAQ entries, KB articles, and resolved
tickets to craft a potential response for a support query."""

import logging
import requests

from config import CLOUD_URL, FAQ_CONFLUENCE_SPACES, CLOUD_CUTOVER_DATE
from db import get_db_conn
from faq.atlassian_docs import search_atlassian_docs
from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url, clear_cache as _clear_oauth_cache

log = logging.getLogger(__name__)


def _jira_browse_url(ticket_key):
    """Return the Jira Cloud browse URL for a ticket."""
    return f"{CLOUD_URL}/browse/{ticket_key}"


def lookup(query: str) -> dict:
    """Search all knowledge sources for a query (ticket key or topic).

    Returns a dict with:
        found: bool — whether any matches were found
        faq_matches: list of FAQ entries
        kb_matches: list of KB articles
        ticket_matches: list of resolved tickets with similar classifications
        response_draft: str — crafted response text if matches found
    """
    with get_db_conn() as conn:
        is_ticket_key = query.upper().startswith(("<PROJECT_KEY>-", "HATMOS-", "ANTSE-", "RH1-"))

        faq_matches = []
        kb_matches = []
        ticket_matches = []
        query_words = []  # for fallback search

        if is_ticket_key:
            ticket_key = query.upper()
            classification = conn.execute("""
                SELECT category, issue_type, keywords, resolution_summary
                FROM ticket_classifications WHERE ticket_key = ?
            """, (ticket_key,)).fetchone()

            if classification:
                cl = dict(classification)
                faq_matches = _search_faq(conn, cl["category"], cl["issue_type"])
                kb_matches = _search_kb(conn, cl["keywords"])
                ticket_matches = _search_resolved(conn, cl["category"], cl["issue_type"], exclude_key=ticket_key)
                query_words = [w for w in (cl["keywords"] or "").split() if len(w) > 3][:5]
            else:
                ticket = conn.execute(
                    "SELECT summary FROM tickets WHERE ticket_key = ?", (ticket_key,)
                ).fetchone()
                if ticket:
                    faq_matches = _search_faq_text(conn, ticket["summary"])
                    kb_matches = _search_kb(conn, ticket["summary"])
                    ticket_matches = _search_resolved_text(conn, ticket["summary"], exclude_key=ticket_key)
                    query_words = [w for w in ticket["summary"].split() if len(w) > 3][:5]
        else:
            faq_matches = _search_faq_text(conn, query)
            kb_matches = _search_kb(conn, query)
            ticket_matches = _search_resolved_text(conn, query)
            query_words = [w for w in query.split() if len(w) > 3][:5]

        # Supplement with Atlassian docs — boost limit when KB has no results
        doc_limit = 5 if not kb_matches else 3
        atlassian_matches = search_atlassian_docs(conn, query_words, limit=doc_limit) if query_words else []

    found = bool(faq_matches or kb_matches or ticket_matches or atlassian_matches)
    response_draft = _craft_response(query, faq_matches, kb_matches, ticket_matches, atlassian_matches) if found else None

    return {
        "found": found,
        "query": query,
        "faq_matches": faq_matches,
        "kb_matches": kb_matches,
        "ticket_matches": ticket_matches,
        "atlassian_matches": atlassian_matches,
        "response_draft": response_draft,
    }


def _search_faq(conn, category, issue_type):
    """Find FAQ entries matching a category or issue type."""
    rows = conn.execute("""
        SELECT id, article_topic, title, body_html, confluence_url
        FROM generated_articles
        WHERE format = 'faq'
          AND (article_topic LIKE ? OR article_topic LIKE ?)
        ORDER BY generated_at DESC
        LIMIT 3
    """, (f"%{category}%", f"%{issue_type}%")).fetchall()
    return [dict(r) for r in rows]


def _search_faq_text(conn, text):
    """Full-text search across FAQ topics and titles."""
    words = [w for w in text.split() if len(w) > 3][:5]
    if not words:
        return []
    conditions = " OR ".join(["article_topic LIKE ? OR title LIKE ?"] * len(words))
    params = []
    for w in words:
        params.extend([f"%{w}%", f"%{w}%"])
    rows = conn.execute(f"""
        SELECT id, article_topic, title, body_html, confluence_url
        FROM generated_articles
        WHERE format = 'faq' AND ({conditions})
        ORDER BY generated_at DESC
        LIMIT 3
    """, params).fetchall()
    return [dict(r) for r in rows]


def _search_kb(conn, keywords_or_text):
    """Search Confluence KB via live CQL API query.

    Searches configured spaces (FAQ_CONFLUENCE_SPACES) using Confluence's
    built-in text search. Post-cutover content is auto-included. Pre-cutover
    content is scored for Cloud relevance via Gemini Flash and only included
    if confidence >= 70%.
    """
    if not keywords_or_text:
        return []
    words = [w for w in keywords_or_text.split() if len(w) > 3][:5]
    if not words:
        return []

    search_text = " ".join(words).replace('"', '\\"')
    spaces = ",".join(FAQ_CONFLUENCE_SPACES)
    base_cql = f'space IN ({spaces}) AND type=page AND text ~ "{search_text}"'

    # Primary: Cloud-era content (auto-include, no scoring needed)
    cloud_cql = f'{base_cql} AND lastModified >= "{CLOUD_CUTOVER_DATE}"'
    matches = _run_confluence_cql(cloud_cql, limit=3)

    # If room remains, check pre-cutover content with Cloud confidence scoring
    remaining = 3 - len(matches)
    if remaining > 0:
        pre_cutover = _run_confluence_cql(base_cql, limit=remaining + 2)
        # Remove any already in cloud matches
        cloud_ids = {m["page_id"] for m in matches}
        pre_cutover = [p for p in pre_cutover if p["page_id"] not in cloud_ids]
        if pre_cutover:
            scored = _score_cloud_relevance(pre_cutover[:remaining + 2])
            matches.extend(scored[:remaining])

    return matches


def _run_confluence_cql(cql: str, limit: int = 3) -> list[dict]:
    """Execute a CQL query against Confluence and return page matches."""
    try:
        headers = get_cloud_auth("confluence_search")
        base = get_cloud_base_url("confluence_search")
        resp = requests.get(
            f"{base}/wiki/rest/api/content/search",
            headers=headers,
            params={"cql": cql, "limit": limit},
            timeout=15,
        )
        if resp.status_code == 401:
            log.warning("Confluence CQL search got 401 — clearing token cache and retrying")
            _clear_oauth_cache()
            headers = get_cloud_auth("confluence_search")
            resp = requests.get(
                f"{base}/wiki/rest/api/content/search",
                headers=headers,
                params={"cql": cql, "limit": limit},
                timeout=15,
            )
        if resp.status_code != 200:
            log.debug("Confluence CQL search failed: %d", resp.status_code)
            return []

        data = resp.json()
        results = data.get("results", [])
        matches = []
        for page in results:
            page_url = page.get("_links", {}).get("webui", "")
            if page_url and not page_url.startswith("http"):
                base_url = data.get("_links", {}).get("base", CLOUD_URL)
                page_url = f"{base_url}{page_url}"
            matches.append({
                "page_id": str(page.get("id", "")),
                "title": page.get("title", ""),
                "url": page_url,
                "space_key": page.get("space", {}).get("key", "") if "space" in page else "",
            })
        return matches
    except Exception as e:
        log.debug("Confluence CQL search error: %s", e)
        return []


_CLOUD_SCORE_THRESHOLD = 70


def _score_cloud_relevance(pages: list[dict]) -> list[dict]:
    """Score pre-cutover KB pages for Cloud relevance using Gemini Flash.

    Checks the kb_cloud_scores cache first. Uncached pages get scored
    via a single batched Gemini Flash call. Returns only pages with
    confidence >= threshold, sorted by confidence descending.
    """
    if not pages:
        return []

    scored = []
    to_score = []

    # Check cache
    with get_db_conn() as conn:
        for page in pages:
            cached = conn.execute(
                "SELECT cloud_applicable, confidence FROM kb_cloud_scores WHERE page_id = ?",
                (page["page_id"],),
            ).fetchone()
            if cached:
                if cached["cloud_applicable"] and cached["confidence"] >= _CLOUD_SCORE_THRESHOLD:
                    page["confidence"] = cached["confidence"]
                    scored.append(page)
            else:
                to_score.append(page)

    if not to_score:
        return sorted(scored, key=lambda p: p.get("confidence", 0), reverse=True)

    # Score uncached pages with Gemini Flash
    try:
        from faq.auto_responder import _get_genai_client
        from config import GEMINI_MODEL_CLASSIFICATION

        client = _get_genai_client()

        items = "\n".join(
            f'- Title: "{p["title"]}", URL: {p["url"]}, Page ID: {p["page_id"]}'
            for p in to_score
        )
        prompt = f"""You are evaluating Confluence KB articles to determine if they are applicable to Atlassian Cloud (not Jira Data Center or Jira Server).

Our organization is migrating from Jira Data Center to Atlassian Cloud. Articles about Cloud-specific features, Cloud administration, Cloud APIs, or general concepts that apply to both are Cloud-applicable. Articles about Server/DC-only features (e.g., DC clustering, Server plugins, on-premise installation) are not.

For each article below, score its Cloud applicability:

{items}

Return valid JSON only (no markdown fencing) — an array of objects:
[{{"page_id": "...", "cloud_applicable": true/false, "confidence": 0-100}}]"""

        response = client.models.generate_content(
            model=GEMINI_MODEL_CLASSIFICATION,
            contents=prompt,
        )

        from utils.gemini import parse_json_response
        results = parse_json_response(response.text)
        if isinstance(results, dict):
            results = [results]

        scores_by_id = {str(r["page_id"]): r for r in results if isinstance(r, dict)}

        # Cache and filter
        with get_db_conn() as conn:
            for page in to_score:
                score_data = scores_by_id.get(page["page_id"], {})
                applicable = score_data.get("cloud_applicable", False)
                confidence = score_data.get("confidence", 0)

                conn.execute(
                    """INSERT INTO kb_cloud_scores
                       (page_id, title, url, cloud_applicable, confidence)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT (page_id) DO UPDATE SET
                           title=EXCLUDED.title, url=EXCLUDED.url,
                           cloud_applicable=EXCLUDED.cloud_applicable,
                           confidence=EXCLUDED.confidence""",
                    (page["page_id"], page["title"], page["url"], applicable, confidence),
                )

                if applicable and confidence >= _CLOUD_SCORE_THRESHOLD:
                    page["confidence"] = confidence
                    scored.append(page)

    except Exception as e:
        log.debug("Cloud relevance scoring failed: %s", e)

    return sorted(scored, key=lambda p: p.get("confidence", 0), reverse=True)


def _search_resolved(conn, category, issue_type, exclude_key=None):
    """Find resolved tickets with matching classification.

    Tries exact category+issue_type first, falls back to category-only
    if no exact matches found.
    """
    exclude_clause = "AND t.ticket_key != ?" if exclude_key else ""

    # Try exact match first
    params = [category, issue_type]
    if exclude_key:
        params.append(exclude_key)
    rows = conn.execute(f"""
        SELECT t.ticket_key, t.summary, t.status, t.resolution,
               c.resolution_summary, c.category, c.issue_type
        FROM tickets t
        JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
        WHERE c.category = ? AND c.issue_type = ?
          AND t.resolution IS NOT NULL
          AND c.resolution_summary IS NOT NULL
          AND c.resolution_summary != ''
          {exclude_clause}
        ORDER BY t.resolved_at DESC
        LIMIT 5
    """, params).fetchall()

    if rows:
        return [dict(r) for r in rows]

    # Fallback: category-only match
    params = [category]
    if exclude_key:
        params.append(exclude_key)
    rows = conn.execute(f"""
        SELECT t.ticket_key, t.summary, t.status, t.resolution,
               c.resolution_summary, c.category, c.issue_type
        FROM tickets t
        JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
        WHERE c.category = ?
          AND t.resolution IS NOT NULL
          AND c.resolution_summary IS NOT NULL
          AND c.resolution_summary != ''
          {exclude_clause}
        ORDER BY t.resolved_at DESC
        LIMIT 5
    """, params).fetchall()
    return [dict(r) for r in rows]


def _search_resolved_text(conn, text, exclude_key=None):
    """Find resolved tickets with matching text in summary or keywords."""
    words = [w for w in text.split() if len(w) > 3][:4]
    if not words:
        return []
    conditions = " OR ".join(["t.summary LIKE ? OR c.keywords LIKE ?"] * len(words))
    params = []
    for w in words:
        params.extend([f"%{w}%", f"%{w}%"])
    exclude_clause = ""
    if exclude_key:
        exclude_clause = "AND t.ticket_key != ?"
        params.append(exclude_key)
    rows = conn.execute(f"""
        SELECT t.ticket_key, t.summary, t.status, t.resolution,
               c.resolution_summary, c.category, c.issue_type
        FROM tickets t
        JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
        WHERE ({conditions})
          AND t.resolution IS NOT NULL
          AND c.resolution_summary IS NOT NULL
          AND c.resolution_summary != ''
          {exclude_clause}
        ORDER BY t.resolved_at DESC
        LIMIT 5
    """, params).fetchall()
    return [dict(r) for r in rows]


def _craft_response(query, faq_matches, kb_matches, ticket_matches, atlassian_matches=None):
    """Craft a Slack-formatted response with links and references."""
    parts = []

    if faq_matches:
        parts.append("*FAQ Entries:*")
        for faq in faq_matches:
            link = faq.get("confluence_url") or ""
            title = faq["title"]
            if link:
                parts.append(f"  - <{link}|{title}>")
            else:
                parts.append(f"  - {title} (FAQ #{faq['id']})")

    if kb_matches:
        parts.append("\n*Knowledge Base Articles:*")
        for kb in kb_matches:
            url = kb.get("url") or ""
            if url:
                parts.append(f"  - <{url}|{kb['title']}>")
            else:
                parts.append(f"  - {kb['title']} ({kb['space_key']})")

    if ticket_matches:
        parts.append("\n*Similar Resolved Tickets:*")
        for t in ticket_matches[:3]:
            ticket_url = _jira_browse_url(t["ticket_key"])
            resolution = t.get("resolution_summary", "")
            if len(resolution) > 200:
                resolution = resolution[:200] + "..."
            parts.append(f"  - <{ticket_url}|{t['ticket_key']}> — {t['summary']}")
            if resolution:
                parts.append(f"    _Resolution: {resolution}_")

    if atlassian_matches:
        parts.append("\n*[Atlassian Docs] Official Documentation:*")
        for doc in atlassian_matches:
            product = doc.get("product", "").replace("-", " ").title()
            parts.append(f"  - <{doc['url']}|{doc['title']}> ({product})")

    return "\n".join(parts)


def format_not_found(query):
    """Return a Slack-formatted 'not found' message."""
    return (
        f"No existing answers found for *{query}*.\n"
        f"No matching FAQ entries, KB articles, or resolved tickets. "
        f"This may require a new response."
    )
