"""PII scrubbing — 15 regex categories for data sanitization plus
consistent identity anonymization for modeling-safe pseudonyms.

Assignee identity is no longer stored in the DB (ANTSE-209). Only reporter
and comment author identities are anonymized here.
"""

import re
import logging

from db import get_db

log = logging.getLogger(__name__)

# Consistent pseudonym mapping — same real ID always maps to same number.
# Built per scrub run from all unique identities in the database.
_identity_maps = {}


def _build_identity_map(conn):
    """Build role-separated pseudonym maps for identity fields.

    Each role gets its own independent numbering (reporter_0001,
    author_0001). Uses sorted order so pseudonyms are stable across runs.
    """
    global _identity_maps

    reporters = sorted({
        r[0] for r in conn.execute(
            "SELECT DISTINCT reporter_id FROM tickets WHERE reporter_id != '' AND reporter_id NOT LIKE 'reporter_%'"
        ).fetchall()
    })
    authors = sorted({
        r[0] for r in conn.execute(
            "SELECT DISTINCT author_id FROM ticket_comments WHERE author_id != '' AND author_id NOT LIKE 'author_%'"
        ).fetchall()
    })

    _identity_maps = {
        "reporter": {name: f"reporter_{i+1:04d}" for i, name in enumerate(reporters)},
        "author": {name: f"author_{i+1:04d}" for i, name in enumerate(authors)},
    }

    return {
        "reporters": len(reporters),
        "authors": len(authors),
    }

# Ordered: specific patterns first, generic (email) last to avoid false matches
PII_PATTERNS = {
    "url_creds": re.compile(r'https?://[^:]+:[^@]+@'),
    "kerberos": re.compile(r'\b\w+@REDHAT\.COM\b'),
    "jira_mention": re.compile(r'\[~[\w:.]+\]'),
    "bearer_token": re.compile(r'Bearer\s+[A-Za-z0-9_.-]{20,}'),
    "api_key_header": re.compile(r'(?:api[_-]?key|token|secret)\s*[:=]\s*["\']?[A-Za-z0-9_.-]{16,}', re.IGNORECASE),
    "private_key": re.compile(r'-----BEGIN (RSA |EC )?PRIVATE KEY-----'),
    "aws_key": re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    "ldap_dn": re.compile(r'uid=\w+,'),
    "email": re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b'),
    "handle_mention": re.compile(r'(?<!\w)@\w+'),
    "ipv4": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    "ipv6": re.compile(r'\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
    "phone": re.compile(r'\b\+?\d{1,3}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
}

PII_REPLACEMENTS = {
    "email": "[EMAIL]",
    "jira_mention": "[MENTION]",
    "handle_mention": "[MENTION]",
    "ipv4": "[IP]",
    "ipv6": "[IP]",
    "ssn": "[SSN]",
    "credit_card": "[CC]",
    "phone": "[PHONE]",
    "url_creds": "[CREDS_URL]",
    "aws_key": "[AWS_KEY]",
    "private_key": "[PRIVATE_KEY]",
    "kerberos": "[KERBEROS]",
    "ldap_dn": "[LDAP_DN]",
    "bearer_token": "[BEARER_TOKEN]",
    "api_key_header": "[API_KEY]",
}


def scrub_pii(text):
    """Replace PII patterns in text with redaction markers."""
    if not text:
        return text
    for name, pattern in PII_PATTERNS.items():
        text = pattern.sub(PII_REPLACEMENTS[name], text)
    return text


def audit_pii(text):
    """Count PII matches in text. Returns dict of {category: count}."""
    if not text:
        return {}
    found = {}
    for name, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[name] = len(matches)
    return found


def scrub_database(dry_run=False):
    """Scrub PII from all stored ticket data. Returns summary dict."""
    conn = get_db()

    if dry_run:
        log.info("DRY RUN — scanning for PII without modifying data")

    tickets = conn.execute("SELECT ticket_key, summary, description FROM tickets").fetchall()
    total_found = {}
    scrubbed_tickets = 0

    for ticket in tickets:
        for field in ["summary", "description"]:
            text = ticket[field] or ""
            pii = audit_pii(text)
            if pii:
                for cat, count in pii.items():
                    total_found[cat] = total_found.get(cat, 0) + count
                if not dry_run:
                    conn.execute(
                        f"UPDATE tickets SET {field} = ? WHERE ticket_key = ?",
                        (scrub_pii(text), ticket["ticket_key"])
                    )
                    scrubbed_tickets += 1

    comments = conn.execute("SELECT ticket_key, comment_id, body FROM ticket_comments").fetchall()
    scrubbed_comments = 0

    for comment in comments:
        text = comment["body"] or ""
        pii = audit_pii(text)
        if pii:
            for cat, count in pii.items():
                total_found[cat] = total_found.get(cat, 0) + count
            if not dry_run:
                conn.execute(
                    "UPDATE ticket_comments SET body = ? WHERE ticket_key = ? AND comment_id = ?",
                    (scrub_pii(text), comment["ticket_key"], comment["comment_id"])
                )
                scrubbed_comments += 1

    # --- Identity anonymization ---
    identity_counts = _build_identity_map(conn)
    anon_reporters = 0
    anon_authors = 0

    if not dry_run:
        # Use executemany for bulk updates to avoid N+1 query performance issues
        if _identity_maps["reporter"]:
            conn.executemany(
                "UPDATE tickets SET reporter_id = ? WHERE reporter_id = ?",
                [(pseudo, real) for real, pseudo in _identity_maps["reporter"].items()]
            )
            anon_reporters += len(_identity_maps["reporter"])

        if _identity_maps["author"]:
            conn.executemany(
                "UPDATE ticket_comments SET author_id = ?, author_name = ? WHERE author_id = ?",
                [(pseudo, pseudo, real) for real, pseudo in _identity_maps["author"].items()]
            )
            anon_authors += len(_identity_maps["author"])

        conn.execute("UPDATE tickets SET reporter_email = '' WHERE reporter_email != ''")
        conn.commit()

    conn.close()

    return {
        "total_found": total_found,
        "tickets_scanned": len(tickets),
        "comments_scanned": len(comments),
        "scrubbed_tickets": scrubbed_tickets,
        "scrubbed_comments": scrubbed_comments,
        "anon_reporters": anon_reporters,
        "anon_authors": anon_authors,
        "identity_counts": identity_counts,
        "dry_run": dry_run,
    }
