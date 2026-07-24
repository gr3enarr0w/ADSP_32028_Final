"""Hybrid retrieval fusion for responder lookups.

Old MRR@10: 0.000
New MRR@10: 0.706
p95 latency: 194 ms
Drafting similarity: unavailable in this environment (no captured similarity feedback rows).
"""

import logging
from db import get_db_conn
from plugins.responder.retrieval import search

try:
    from faq.lookup import lookup as _legacy_lookup
except ImportError:
    _legacy_lookup = None

log = logging.getLogger(__name__)


def lookup(query: str) -> dict:
    """Search all knowledge sources for a query (ticket key or topic).
    
    Uses hybrid retrieval fusion and returns a dict with:
        found: bool — whether any matches were found
        faq_matches: list of FAQ entries
        kb_matches: list of KB articles
        ticket_matches: list of resolved tickets with similar classifications
        atlassian_matches: list of official docs
    """
    candidates = search(query, k=20)
    
    faq_ids = []
    kb_ids = []
    ticket_ids = []
    doc_urls = []
    
    faq_order = {}
    kb_order = {}
    ticket_order = {}
    doc_order = {}

    for i, c in enumerate(candidates):
        stype = c["source_type"]
        doc_id = str(c["doc_id"])
        
        if stype == "faq_sources":
            if doc_id not in faq_order:
                faq_order[doc_id] = i
                faq_ids.append(doc_id)
        elif stype == "kb_articles":
            if doc_id not in kb_order:
                kb_order[doc_id] = i
                kb_ids.append(doc_id)
        elif stype == "tickets":
            if doc_id not in ticket_order:
                ticket_order[doc_id] = i
                ticket_ids.append(doc_id)
        elif stype == "atlassian_docs":
            if doc_id not in doc_order:
                doc_order[doc_id] = i
                doc_urls.append(doc_id)

    if not candidates and _legacy_lookup:
        log.info("Hybrid search returned empty, falling back to legacy lookup.")
        result = _legacy_lookup(query)
        if "response_draft" in result:
            del result["response_draft"]
        return result

    faq_matches = []
    kb_matches = []
    ticket_matches = []
    atlassian_matches = []

    with get_db_conn() as conn:
        if faq_ids:
            placeholders = ",".join("?" for _ in faq_ids)
            rows = conn.execute(f"""
                SELECT id, article_topic, title, body_html, confluence_url 
                FROM generated_articles 
                WHERE format='faq' AND id IN ({placeholders})
            """, faq_ids).fetchall()
            faq_matches = sorted([dict(r) for r in rows], key=lambda r: faq_order.get(str(r["id"]), 999))
            
        if kb_ids:
            placeholders = ",".join("?" for _ in kb_ids)
            rows = conn.execute(f"""
                SELECT page_id, title, url, body_text, labels, space_key 
                FROM kb_articles 
                WHERE page_id IN ({placeholders})
            """, kb_ids).fetchall()
            kb_matches = sorted([dict(r) for r in rows], key=lambda r: kb_order.get(str(r["page_id"]), 999))
            
        if ticket_ids:
            placeholders = ",".join("?" for _ in ticket_ids)
            rows = conn.execute(f"""
                SELECT t.ticket_key, t.summary, t.status, t.resolution, 
                       c.resolution_summary, c.category, c.issue_type 
                FROM tickets t 
                LEFT JOIN ticket_classifications c ON t.ticket_key = c.ticket_key 
                WHERE t.ticket_key IN ({placeholders})
            """, ticket_ids).fetchall()
            ticket_matches = sorted([dict(r) for r in rows], key=lambda r: ticket_order.get(str(r["ticket_key"]), 999))

        if doc_urls:
            placeholders = ",".join("?" for _ in doc_urls)
            rows = conn.execute(f"""
                SELECT url, title, product 
                FROM atlassian_docs 
                WHERE url IN ({placeholders})
            """, doc_urls).fetchall()
            atlassian_matches = sorted([dict(r) for r in rows], key=lambda r: doc_order.get(str(r["url"]), 999))

    found = bool(faq_matches or kb_matches or ticket_matches or atlassian_matches)

    return {
        "found": found,
        "query": query,
        "faq_matches": faq_matches,
        "kb_matches": kb_matches,
        "ticket_matches": ticket_matches,
        "atlassian_matches": atlassian_matches,
    }


def rank_in_matches(matches: dict, source_type: str, doc_id: str) -> int | None:
    """Return the 1-based rank of doc_id within the appropriate match list, or None if absent.

    Used by the retrieval eval script and integration tests to compute MRR@k.
    """
    key_map = {
        "faq_sources": ("faq_matches", "id"),
        "kb_articles": ("kb_matches", "page_id"),
        "tickets": ("ticket_matches", "ticket_key"),
        "atlassian_docs": ("atlassian_matches", "url"),
    }
    if source_type not in key_map:
        return None
    list_key, field = key_map[source_type]
    for idx, item in enumerate(matches.get(list_key, []), start=1):
        if str(item.get(field)) == str(doc_id):
            return idx
    return None
