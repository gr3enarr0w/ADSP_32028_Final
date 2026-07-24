"""Tests for PII scrubbing — all 15 regex categories plus identity anonymization."""

import os
import tempfile
from unittest.mock import patch

import pytest

from db import get_db_conn, init_db
from ingest.scrubber import audit_pii, scrub_database, scrub_pii


class TestScrubPii:
    def test_empty_input(self):
        assert scrub_pii("") == ""
        assert scrub_pii(None) is None

    def test_no_pii(self):
        text = "Normal ticket about Jira configuration"
        assert scrub_pii(text) == text

    def test_email(self):
        assert "[EMAIL]" in scrub_pii("Contact user@example.com for help")

    def test_jira_mention(self):
        assert "[MENTION]" in scrub_pii("Assigned to [~john.doe]")
        assert "[MENTION]" in scrub_pii("See [~accountId:abc123]")

    def test_handle_mention(self):
        assert "[MENTION]" in scrub_pii("Hey @username check this")

    def test_ipv4(self):
        assert "[IP]" in scrub_pii("Server at 192.168.1.100 is down")

    def test_ipv6(self):
        assert "[IP]" in scrub_pii("IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334")

    def test_ssn(self):
        assert "[SSN]" in scrub_pii("SSN is 123-45-6789")

    def test_credit_card(self):
        assert "[CC]" in scrub_pii("Card 4111 1111 1111 1111")
        assert "[CC]" in scrub_pii("Card 4111-1111-1111-1111")

    def test_phone(self):
        assert "[PHONE]" in scrub_pii("Call +1 (555) 123-4567")

    def test_url_creds(self):
        assert "[CREDS_URL]" in scrub_pii("URL: https://user:pass@host.com/path")

    def test_aws_key(self):
        assert "[AWS_KEY]" in scrub_pii("Key: AKIAIOSFODNN7EXAMPLE")

    def test_private_key(self):
        assert "[PRIVATE_KEY]" in scrub_pii("-----BEGIN PRIVATE KEY-----")
        assert "[PRIVATE_KEY]" in scrub_pii("-----BEGIN RSA PRIVATE KEY-----")

    def test_kerberos(self):
        assert "[KERBEROS]" in scrub_pii("Principal: jsmith@REDHAT.COM")

    def test_ldap_dn(self):
        assert "[LDAP_DN]" in scrub_pii("LDAP: uid=jsmith,ou=People,dc=redhat")

    def test_bearer_token(self):
        assert "[BEARER_TOKEN]" in scrub_pii("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc123")

    def test_api_key(self):
        assert "[API_KEY]" in scrub_pii("api_key = 'abcdef1234567890xx'")
        assert "[API_KEY]" in scrub_pii("token: ABCDEFGHIJKLMNOP1234")

    def test_multiple_patterns(self):
        text = "Email user@test.com from 192.168.1.1, SSN 123-45-6789"
        result = scrub_pii(text)
        assert "[EMAIL]" in result
        assert "[IP]" in result
        assert "[SSN]" in result
        assert "user@test.com" not in result


class TestAuditPii:
    def test_empty(self):
        assert audit_pii("") == {}
        assert audit_pii(None) == {}

    def test_counts(self):
        text = "user@a.com and admin@b.com from 10.0.0.1"
        result = audit_pii(text)
        assert result["email"] == 2
        assert result["ipv4"] == 1

    def test_no_pii(self):
        assert audit_pii("Just a normal string") == {}


class TestIdentityAnonymization:
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        with patch("db.DB_PATH", db_path):
            init_db()
            yield db_path
        os.unlink(db_path)

    def test_scrub_database_anonymizes_identities(self, temp_db):
        with patch("db.DB_PATH", temp_db):
            with get_db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO tickets (
                        ticket_key, summary, status, reporter_id, reporter_email, assignee_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("T-1", "Help", "Open", "acct-reporter-a", "a@example.com", "acct-assignee-1"),
                )
                conn.execute(
                    """
                    INSERT INTO tickets (
                        ticket_key, summary, status, reporter_id, reporter_email
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    ("T-2", "Other", "Open", "acct-reporter-b", "b@example.com"),
                )
                conn.execute(
                    """
                    INSERT INTO ticket_comments (
                        comment_id, ticket_key, author_id, author_name, body, is_public
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("c1", "T-1", "acct-author-a", "Alice Agent", "Reply one", 1),
                )
                conn.execute(
                    """
                    INSERT INTO ticket_comments (
                        comment_id, ticket_key, author_id, author_name, body, is_public
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("c2", "T-1", "acct-author-a", "Alice Agent", "Reply two", 1),
                )

            result = scrub_database(dry_run=False)

            with get_db_conn() as conn:
                reporters = sorted(
                    r[0]
                    for r in conn.execute(
                        "SELECT DISTINCT reporter_id FROM tickets ORDER BY reporter_id"
                    ).fetchall()
                )
                authors = conn.execute(
                    "SELECT author_id, author_name FROM ticket_comments ORDER BY comment_id"
                ).fetchall()
                emails = conn.execute(
                    "SELECT reporter_email FROM tickets ORDER BY ticket_key"
                ).fetchall()
                assignee = conn.execute(
                    "SELECT assignee_id FROM tickets WHERE ticket_key = 'T-1'"
                ).fetchone()

        assert reporters == ["reporter_0001", "reporter_0002"]
        assert all(row["author_id"] == "author_0001" for row in authors)
        assert all(row["author_name"] == "author_0001" for row in authors)
        assert all(row["reporter_email"] == "" for row in emails)
        assert assignee["assignee_id"] == "acct-assignee-1"
        assert result["anon_reporters"] == 2
        assert result["anon_authors"] == 1
        assert result["identity_counts"] == {"reporters": 2, "authors": 1}

    def test_scrub_database_dry_run_leaves_identities(self, temp_db):
        with patch("db.DB_PATH", temp_db):
            with get_db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO tickets (
                        ticket_key, summary, status, reporter_id, reporter_email
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    ("T-1", "Help", "Open", "acct-reporter-a", "a@example.com"),
                )

            result = scrub_database(dry_run=True)

            with get_db_conn() as conn:
                row = conn.execute(
                    "SELECT reporter_id, reporter_email FROM tickets WHERE ticket_key = 'T-1'"
                ).fetchone()

        assert row["reporter_id"] == "acct-reporter-a"
        assert row["reporter_email"] == "a@example.com"
        assert result["identity_counts"] == {"reporters": 1, "authors": 0}
        assert result["anon_reporters"] == 0
        assert result["anon_authors"] == 0
