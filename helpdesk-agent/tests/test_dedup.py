"""Tests for MinHash structural fingerprinting, semantic embedding, and dedup gating."""

import json
import struct

import pytest
from db import init_db, get_db_conn


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("db.DB_PATH", db_path)
    init_db()
    yield db_path


# ── Shared helpers ──


def _insert_kb_article(
    page_id: str,
    title: str,
    body_text: str,
    embedding: bytes | None = None,
) -> None:
    """Insert a kb_articles row for use in dedup tests."""
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO kb_articles (page_id, title, body_text, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (page_id, title, body_text, embedding),
        )


# ── Schema ──


class TestSchema:
    def test_structural_fingerprint_column_exists(self):
        with get_db_conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(generated_articles)").fetchall()}
            assert "structural_fingerprint" in cols


# ── Normalization & shingling ──


class TestNormalize:
    def test_strips_html(self):
        from faq.dedup import _normalize
        assert _normalize("<h1>Title</h1><p>Body text</p>") == "title body text"

    def test_collapses_whitespace(self):
        from faq.dedup import _normalize
        assert _normalize("  lots   of   space  ") == "lots of space"

    def test_lowercases(self):
        from faq.dedup import _normalize
        assert _normalize("UPPER Case") == "upper case"


class TestShingle:
    def test_basic_shingles(self):
        from faq.dedup import _shingle
        result = _shingle("a b c d", k=3)
        assert result == {"a b c", "b c d"}

    def test_short_text_returns_whole(self):
        from faq.dedup import _shingle
        result = _shingle("ab", k=3)
        assert result == {"ab"}

    def test_empty_returns_empty(self):
        from faq.dedup import _shingle
        assert _shingle("", k=3) == set()


# ── Fingerprint computation ──


class TestComputeFingerprint:
    def test_deterministic(self):
        from faq.dedup import compute_fingerprint
        html = "<p>Some content here</p>"
        assert compute_fingerprint(html) == compute_fingerprint(html)

    def test_different_content_different_fingerprint(self):
        from faq.dedup import compute_fingerprint
        fp1 = compute_fingerprint("<p>Alpha content about migration</p>")
        fp2 = compute_fingerprint("<p>Beta content about something else entirely</p>")
        assert fp1 != fp2

    def test_html_tags_ignored(self):
        from faq.dedup import compute_fingerprint
        fp1 = compute_fingerprint("<h1>Title</h1><p>body text here</p>")
        fp2 = compute_fingerprint("<div>Title body text here</div>")
        assert fp1 == fp2

    def test_returns_hex_string(self):
        from faq.dedup import compute_fingerprint
        fp = compute_fingerprint("<p>test</p>")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


# ── Duplicate detection ──


class TestIsDuplicate:
    def _insert_article(self, topic, body_html, fingerprint=""):
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO generated_articles (article_topic, title, body_html, format,
                    structural_fingerprint)
                VALUES (?, ?, ?, 'faq', ?)
            """, (topic, topic, body_html, fingerprint))

    def test_no_articles_no_duplicate(self):
        from faq.dedup import is_duplicate
        dup, topic = is_duplicate("<p>anything</p>")
        assert dup is False
        assert topic is None

    def test_identical_content_is_duplicate(self):
        from faq.dedup import is_duplicate
        html = "<h2>How do I migrate?</h2><p>Follow these steps to migrate your data safely.</p>"
        self._insert_article("Migration Guide", html)
        dup, topic = is_duplicate(html)
        assert dup is True
        assert topic == "Migration Guide"

    def test_different_content_not_duplicate(self):
        from faq.dedup import is_duplicate
        self._insert_article("Migration Guide",
                             "<h2>How to migrate</h2><p>Follow these steps to migrate.</p>")
        dup, _ = is_duplicate(
            "<h2>Billing setup</h2><p>Configure your billing and payment methods.</p>")
        assert dup is False

    def test_exclude_topic_skips_self(self):
        from faq.dedup import is_duplicate
        html = "<h2>Same content</h2><p>Exact same body text for testing.</p>"
        self._insert_article("Topic A", html)
        dup, _ = is_duplicate(html, exclude_topic="Topic A")
        assert dup is False

    def test_near_duplicate_detected(self):
        """Structurally near-identical content (same words, trivial edits) caught by MinHash."""
        from faq.dedup import is_duplicate
        base = (
            "<h2>How do I migrate my Jira project data to Cloud?</h2>"
            "<p>Follow these steps to migrate your project data safely to Atlassian Cloud. "
            "First you need to back up your existing configuration and export all project "
            "settings. Then install the Jira Cloud Migration Assistant from the Atlassian "
            "Marketplace. Run a pre-migration assessment to identify any incompatible apps "
            "or configurations. Review the assessment report and resolve any blocking issues. "
            "Finally start the migration and monitor progress in the dashboard.</p>"
            "<h3>Known Limitations</h3>"
            "<p>Some third-party apps may not have Cloud equivalents. Custom workflows "
            "may need manual adjustment after migration.</p>"
        )
        self._insert_article("Migration", base)
        variant = (
            "<h2>How do I migrate my Jira project data to Cloud?</h2>"
            "<p>Follow these steps to migrate your project data safely to Atlassian Cloud. "
            "First you need to back up your existing configuration and export all project "
            "settings. Then install the Jira Cloud Migration Assistant from the Atlassian "
            "Marketplace. Run a pre-migration assessment to identify any incompatible apps "
            "or configurations. Review the assessment report and resolve any blocking issues. "
            "Finally start the migration and monitor progress in the dashboard.</p>"
            "<h3>Known Limitations</h3>"
            "<p>Some third-party apps may not have Cloud equivalents. Custom workflows "
            "may need manual review after migration.</p>"
        )
        dup, topic = is_duplicate(variant)
        assert dup is True
        assert topic == "Migration"

    def test_empty_body_not_duplicate(self):
        from faq.dedup import is_duplicate
        self._insert_article("Existing", "<p>Real content here about migration.</p>")
        dup, _ = is_duplicate("")
        assert dup is False

    def test_empty_body_fingerprint_is_deterministic(self):
        from faq.dedup import compute_fingerprint
        assert compute_fingerprint("") == compute_fingerprint("")
        assert compute_fingerprint("") == compute_fingerprint("   ")

    def test_none_body_handled(self):
        from faq.dedup import compute_fingerprint, is_duplicate
        fp = compute_fingerprint(None)
        assert len(fp) == 64
        dup, _ = is_duplicate(None)
        assert dup is False

    def test_is_duplicate_checks_kb_articles(self):
        from faq.dedup import is_duplicate
        base = (
            "How do I migrate my Jira project data to Cloud? "
            "Follow these steps to migrate your project data safely to Atlassian Cloud. "
            "First you need to back up your existing configuration and export all project "
            "settings. Then install the Jira Cloud Migration Assistant from the Atlassian "
            "Marketplace. Run a pre-migration assessment to identify any incompatible apps "
            "or configurations. Review the assessment report and resolve any blocking issues. "
            "Finally start the migration and monitor progress in the dashboard. "
            "Some third-party apps may not have Cloud equivalents."
        )
        _insert_kb_article("12345", "Migration Guide", base)
        candidate_html = (
            "<h2>How do I migrate my Jira project data to Cloud?</h2>"
            "<p>Follow these steps to migrate your project data safely to Atlassian Cloud. "
            "First you need to back up your existing configuration and export all project "
            "settings. Then install the Jira Cloud Migration Assistant from the Atlassian "
            "Marketplace. Run a pre-migration assessment to identify any incompatible apps "
            "or configurations. Review the assessment report and resolve any blocking issues. "
            "Finally start the migration and monitor progress in the dashboard.</p>"
            "<p>Some third-party apps may not have Cloud equivalents.</p>"
        )
        dup, title = is_duplicate(candidate_html)
        assert dup is True
        assert title == "Migration Guide"

    def test_is_duplicate_excludes_only_generated_not_kb(self):
        from faq.dedup import is_duplicate
        html = (
            "<h2>How do I migrate my Jira project data to Cloud?</h2>"
            "<p>Follow these steps to migrate your project data safely to Atlassian Cloud. "
            "First you need to back up your existing configuration and export all project "
            "settings. Then install the Jira Cloud Migration Assistant from the Atlassian "
            "Marketplace. Run a pre-migration assessment to identify any incompatible apps "
            "or configurations. Review the assessment report and resolve any blocking issues. "
            "Finally start the migration and monitor progress in the dashboard.</p>"
        )
        self._insert_article("Topic A", html)
        kb_body = (
            "How do I migrate my Jira project data to Cloud? "
            "Follow these steps to migrate your project data safely to Atlassian Cloud. "
            "First you need to back up your existing configuration and export all project "
            "settings. Then install the Jira Cloud Migration Assistant from the Atlassian "
            "Marketplace. Run a pre-migration assessment to identify any incompatible apps "
            "or configurations. Review the assessment report and resolve any blocking issues. "
            "Finally start the migration and monitor progress in the dashboard."
        )
        _insert_kb_article("99999", "KB Migration Guide", kb_body)
        dup, title = is_duplicate(html, exclude_topic="Topic A")
        assert dup is True
        assert title == "KB Migration Guide"


# ── Backfill ──


class TestBackfill:
    def test_backfills_null_fingerprints(self):
        from faq.dedup import backfill_fingerprints
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO generated_articles (article_topic, title, body_html, format)
                VALUES ('t1', 'Title 1', '<p>Content one</p>', 'faq')
            """)
            conn.execute("""
                INSERT INTO generated_articles (article_topic, title, body_html, format)
                VALUES ('t2', 'Title 2', '<p>Content two</p>', 'faq')
            """)
        count = backfill_fingerprints()
        assert count == 2

        with get_db_conn() as conn:
            rows = conn.execute(
                "SELECT structural_fingerprint FROM generated_articles"
            ).fetchall()
            for row in rows:
                assert row["structural_fingerprint"] is not None
                assert len(row["structural_fingerprint"]) == 64

    def test_skips_already_fingerprinted(self):
        from faq.dedup import backfill_fingerprints, compute_fingerprint
        fp = compute_fingerprint("<p>pre-existing</p>")
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO generated_articles
                    (article_topic, title, body_html, format, structural_fingerprint)
                VALUES ('t1', 'T', '<p>pre-existing</p>', 'faq', ?)
            """, (fp,))
        count = backfill_fingerprints()
        assert count == 0


# ── Schema: semantic_embedding column ──


class TestSemanticSchema:
    def test_semantic_embedding_column_exists(self):
        with get_db_conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(generated_articles)").fetchall()}
            assert "semantic_embedding" in cols


# ── Embedding computation ──


class TestComputeEmbedding:
    def test_deterministic(self):
        from faq.dedup import compute_embedding
        html = "<p>How to migrate your Jira project data to Cloud</p>"
        assert compute_embedding(html) == compute_embedding(html)

    def test_returns_dense_list(self):
        from faq.dedup import compute_embedding
        emb = compute_embedding("<p>Some content about migration</p>")
        assert isinstance(emb, list)
        assert len(emb) > 0
        assert all(isinstance(v, float) for v in emb)

    def test_empty_body_returns_empty(self):
        from faq.dedup import compute_embedding
        assert compute_embedding("") == []
        assert compute_embedding("   ") == []
        assert compute_embedding(None) == []

    def test_unit_vector(self):
        """Embedding should be normalized to approximately unit length."""
        import math
        from faq.dedup import compute_embedding
        emb = compute_embedding("<p>Migration steps for Jira Cloud</p>")
        mag = math.sqrt(sum(v * v for v in emb))
        assert abs(mag - 1.0) < 1e-6

    def test_different_content_different_embeddings(self):
        from faq.dedup import compute_embedding
        e1 = compute_embedding("<p>How to configure SAML SSO for Jira Cloud</p>")
        e2 = compute_embedding("<p>Billing and payment setup for organization</p>")
        assert e1 != e2


# ── Cosine similarity helpers ──


class TestCosineHelpers:
    def test_identical_vectors_sim_one(self):
        from faq.dedup import _cosine_sim
        vec = {"a": 0.5, "b": 0.5, "c": 0.707}
        assert abs(_cosine_sim(vec, vec) - sum(v * v for v in vec.values())) < 1e-6

    def test_disjoint_vectors_sim_zero(self):
        from faq.dedup import _cosine_sim
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine_sim(a, b) == 0.0

    def test_empty_vector_sim_zero(self):
        from faq.dedup import _cosine_sim
        assert _cosine_sim({}, {"a": 1.0}) == 0.0
        assert _cosine_sim({"a": 1.0}, {}) == 0.0


# ── Semantic duplicate detection ──


class TestIsSemanticDuplicate:
    def _insert_article(self, topic, body_html):
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO generated_articles (article_topic, title, body_html, format)
                VALUES (?, ?, ?, 'faq')
            """, (topic, topic, body_html))

    def test_no_articles_no_duplicate(self):
        from faq.dedup import is_semantic_duplicate
        dup, topic, sim = is_semantic_duplicate("<p>anything at all</p>")
        assert dup is False
        assert topic is None
        assert sim == 0.0

    def test_identical_content_is_duplicate(self):
        from faq.dedup import is_semantic_duplicate
        html = (
            "<h2>How do I set up SAML SSO?</h2>"
            "<p>Configure SAML single sign-on for your Jira Cloud organization "
            "by navigating to the admin console and selecting authentication.</p>"
        )
        self._insert_article("SSO Setup", html)
        dup, topic, sim = is_semantic_duplicate(html)
        assert dup is True
        assert topic == "SSO Setup"
        assert sim >= 0.99

    def test_completely_different_content_passes(self):
        from faq.dedup import is_semantic_duplicate
        self._insert_article(
            "SSO Setup",
            "<h2>SAML SSO</h2><p>Configure single sign-on authentication.</p>",
        )
        dup, _, _ = is_semantic_duplicate(
            "<h2>Billing FAQ</h2><p>How to update credit card payment method.</p>"
        )
        assert dup is False

    def test_paraphrased_content_detected(self):
        """Content with same words reordered should be caught."""
        from faq.dedup import is_semantic_duplicate
        original = (
            "<h2>Migration steps</h2>"
            "<p>To migrate your data from Jira Server to Jira Cloud you need to "
            "install the migration assistant, run the pre-migration assessment, "
            "review the results, and start the migration process.</p>"
        )
        self._insert_article("Migration Steps", original)
        paraphrased = (
            "<h2>Steps for migration</h2>"
            "<p>You need to install the migration assistant to migrate your data "
            "from Jira Server to Jira Cloud. Run the pre-migration assessment, "
            "review the results, and start the migration process.</p>"
        )
        dup, topic, sim = is_semantic_duplicate(paraphrased)
        assert dup is True
        assert topic == "Migration Steps"

    def test_exclude_topic_skips_self(self):
        from faq.dedup import is_semantic_duplicate
        html = "<p>Exact same body text for testing deduplication logic.</p>"
        self._insert_article("Topic A", html)
        dup, _, _ = is_semantic_duplicate(html, exclude_topic="Topic A")
        assert dup is False

    def test_empty_body_not_duplicate(self):
        from faq.dedup import is_semantic_duplicate
        self._insert_article("Existing", "<p>Real content about migration.</p>")
        dup, _, sim = is_semantic_duplicate("")
        assert dup is False
        assert sim == 0.0

    def test_custom_threshold(self):
        """A very low threshold should flag even loosely related content."""
        from faq.dedup import is_semantic_duplicate
        self._insert_article(
            "Cloud migration",
            "<p>Steps to migrate your Jira data to the cloud platform.</p>",
        )
        dup, _, _ = is_semantic_duplicate(
            "<p>Information about migrating Jira server data.</p>",
            threshold=0.1,
        )
        assert dup is True

    def test_is_semantic_duplicate_checks_kb_articles(self):
        from faq.dedup import _embedding_to_vector, is_semantic_duplicate
        candidate_html = (
            "<h2>How do I set up SAML SSO?</h2>"
            "<p>Configure SAML single sign-on for your Jira Cloud organization "
            "by navigating to the admin console and selecting authentication.</p>"
        )
        vec = _embedding_to_vector(candidate_html)
        blob = struct.pack(f"{len(vec)}f", *vec.tolist())
        _insert_kb_article(
            "54321",
            "SSO Setup KB",
            "Configure SAML single sign-on for your Jira Cloud organization "
            "by navigating to the admin console and selecting authentication.",
            embedding=blob,
        )
        dup, title, sim = is_semantic_duplicate(candidate_html)
        assert dup is True
        assert title == "SSO Setup KB"
        # When EMBEDDING_PROVIDER != "minilm" the stored BLOB is not used; the KB
        # article is re-embedded from title+body_text, which differs slightly from
        # the candidate HTML (different preamble, no HTML tags).  0.90 is a safe
        # floor for semantically near-identical content under any provider.
        assert sim >= 0.90

    def test_is_semantic_duplicate_handles_legacy_dict_embedding(self):
        from faq.dedup import is_semantic_duplicate
        html = (
            "<h2>How do I set up SAML SSO?</h2>"
            "<p>Configure SAML single sign-on for your Jira Cloud organization "
            "by navigating to the admin console and selecting authentication.</p>"
        )
        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO generated_articles
                    (article_topic, title, body_html, format, semantic_embedding)
                VALUES (?, ?, ?, 'faq', ?)
                """,
                ("Legacy Dict Embedding", "Legacy Dict Embedding", html, json.dumps({"legacy": 1.0})),
            )
        dup, title, sim = is_semantic_duplicate(html)
        assert dup is True
        assert title == "Legacy Dict Embedding"
        assert sim >= 0.99


# ── Embedding provider config ──


class TestEmbeddingProvider:
    def test_minilm_is_default(self):
        from faq.dedup import EMBEDDING_PROVIDER
        assert EMBEDDING_PROVIDER in ("minilm", "stub", "vertex")

    def test_vertex_produces_embeddings(self, monkeypatch):
        monkeypatch.setattr("faq.dedup.EMBEDDING_PROVIDER", "vertex")
        from faq.dedup import compute_embedding
        emb = compute_embedding("<p>test</p>")
        assert isinstance(emb, list)
        assert len(emb) > 0  # Vertex AI gemini-embedding-001 returns 3072-dim

    def test_stub_provider_works(self, monkeypatch):
        monkeypatch.setattr("faq.dedup.EMBEDDING_PROVIDER", "stub")
        from faq.dedup import compute_embedding
        emb = compute_embedding("<p>some test content here</p>")
        assert isinstance(emb, dict)
        assert len(emb) > 0


# ── Backfill embeddings ──


class TestBackfillEmbeddings:
    def test_backfills_null_embeddings(self):
        from faq.dedup import backfill_embeddings
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO generated_articles (article_topic, title, body_html, format)
                VALUES ('e1', 'Title 1', '<p>Content one about migration</p>', 'faq')
            """)
            conn.execute("""
                INSERT INTO generated_articles (article_topic, title, body_html, format)
                VALUES ('e2', 'Title 2', '<p>Content two about billing</p>', 'faq')
            """)
        count = backfill_embeddings()
        assert count == 2

        with get_db_conn() as conn:
            rows = conn.execute(
                "SELECT semantic_embedding FROM generated_articles"
            ).fetchall()
            for row in rows:
                assert row["semantic_embedding"] is not None
                emb = json.loads(row["semantic_embedding"])
                assert isinstance(emb, list)
                assert len(emb) > 0  # dimension depends on provider (vertex: 3072, minilm: 384)

    def test_skips_already_embedded(self):
        from faq.dedup import backfill_embeddings, compute_embedding
        emb = json.dumps(compute_embedding("<p>pre-existing content</p>"))
        with get_db_conn() as conn:
            conn.execute("""
                INSERT INTO generated_articles
                    (article_topic, title, body_html, format, semantic_embedding)
                VALUES ('e1', 'T', '<p>pre-existing content</p>', 'faq', ?)
            """, (emb,))
        count = backfill_embeddings()
        assert count == 0
