"""MinHash structural fingerprinting and semantic embedding dedup for FAQ/KB articles."""

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter

import numpy as np
from datasketch import MinHash

from config import DEDUP_JACCARD_THRESHOLD, DEDUP_COSINE_THRESHOLD
from db import get_db, get_db_conn
from services.embedding import embed_text as _shared_embed_text

log = logging.getLogger(__name__)

NUM_PERM = 128
JACCARD_THRESHOLD = DEDUP_JACCARD_THRESHOLD
COSINE_THRESHOLD = DEDUP_COSINE_THRESHOLD

# Embedding provider: "vertex" (default — Vertex AI gemini-embedding-001, 3072-dim),
# "minilm" (legacy self-hosted sentence-transformers, 384-dim), or
# "stub" (legacy TF-IDF sparse vectors, no external deps, low quality).
# MiniLM achieved F1=0.54 with threshold stuck at floor 0.30 on JSM tickets.
# Switch back to "minilm" by setting EMBEDDING_PROVIDER=minilm (or via env).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "vertex")

# ── MiniLM embedding (delegated to shared services.embedding) ──


_DOMAIN_EXPANSIONS = {
    r"\bjsm\b": "jira service management",
    r"\bdc\b": "data center",
    r"\bjcma\b": "jira cloud migration assistant",
    r"\bscim\b": "system for cross-domain identity management",
    r"\bsla\b": "service level agreement",
    r"\bsso\b": "single sign-on",
    r"\bsaml\b": "security assertion markup language",
    r"\b2lo\b": "two-legged oauth",
    r"\b3lo\b": "three-legged oauth",
    r"\bjql\b": "jira query language",
    r"\bcql\b": "confluence query language",
    r"\bjsd\b": "jira service desk",
    r"\brbac\b": "role-based access control",
}

_JIRA_MARKUP = re.compile(
    r"\{(?:code|noformat|color|quote|panel|anchor|jira)[^}]*\}|"
    r"\[~[^\]]+\]|"
    r"\![\w/.-]+\!|"
    r"\{[a-z]+\}"
)


def _normalize(html: str) -> str:
    """Strip HTML tags, Jira markup, collapse whitespace, lowercase."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = _JIRA_MARKUP.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _normalize_for_embedding(html: str) -> str:
    """Normalize and expand domain acronyms for better embedding quality."""
    text = _normalize(html)
    for pattern, expansion in _DOMAIN_EXPANSIONS.items():
        text = re.sub(pattern, lambda m: f"{m.group(0)} ({expansion})", text)
    return text


def _shingle(text: str, k: int = 4) -> set[str]:
    """Produce word-level k-shingles from text."""
    words = text.split()
    if len(words) < k:
        return {text} if text else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def compute_fingerprint(body_html: str) -> str:
    """Return a hex digest MinHash fingerprint for article HTML content.

    The fingerprint is a SHA-256 of the MinHash digest — useful as a stored
    identifier but NOT for similarity comparison (use compute_minhash + Jaccard).
    """
    text = _normalize(body_html or "")
    if not text:
        return hashlib.sha256(b"__empty__").hexdigest()
    shingles = _shingle(text)
    mh = MinHash(num_perm=NUM_PERM)
    for s in shingles:
        mh.update(s.encode("utf-8"))
    return hashlib.sha256(mh.digest().tobytes()).hexdigest()


def compute_minhash(body_html: str) -> MinHash | None:
    """Return a MinHash object for similarity comparison, or None for empty content."""
    text = _normalize(body_html or "")
    if not text:
        return None
    shingles = _shingle(text)
    mh = MinHash(num_perm=NUM_PERM)
    for s in shingles:
        mh.update(s.encode("utf-8"))
    return mh


def _kb_minhash(title: str, body_text: str) -> MinHash | None:
    """Compute MinHash from a KB article's title and plain-text body."""
    combined = f"{title or ''} {body_text or ''}".strip()
    if not combined:
        return None
    return compute_minhash(combined)


# Full-table scans: both queries fetch all rows on every dedup call.
# Acceptable at current KB scale (hundreds of pages); if kb_articles grows to
# thousands, replace with an ANN index (e.g. pgvector or a faiss sidecar) so
# only the top-k candidates are loaded for similarity comparison.
_KB_ARTICLES_QUERY = (
    "SELECT page_id, title, body_text FROM kb_articles "
    "WHERE body_text IS NOT NULL AND TRIM(body_text) != ''"
)


def is_duplicate(body_html: str, exclude_topic: str | None = None) -> tuple[bool, str | None]:
    """Check whether body_html is a near-duplicate of any existing article.

    Compares against generated_articles and kb_articles (Confluence KB pages)
    using MinHash Jaccard similarity.  The *exclude_topic* filter applies only
    to generated_articles — all kb_articles are always checked.

    Returns (is_dup, matching_title) where matching_title is:
    - the article_topic from generated_articles, or
    - the Confluence page title from kb_articles.
    Callers must treat this string as an opaque label for logging; do NOT use
    it as a generated_articles lookup key, since KB hits return a page title,
    not an article_topic.
    """
    candidate = compute_minhash(body_html)
    if candidate is None:
        return False, None
    conn = get_db()
    try:
        query = "SELECT article_topic, body_html FROM generated_articles"
        params: tuple = ()
        if exclude_topic:
            query += " WHERE article_topic != ?"
            params = (exclude_topic,)
        gen_rows = conn.execute(query, params).fetchall()
        kb_rows = conn.execute(_KB_ARTICLES_QUERY).fetchall()
    finally:
        conn.close()

    for row in gen_rows:
        existing_mh = compute_minhash(row["body_html"])
        if existing_mh is None:
            continue
        sim = candidate.jaccard(existing_mh)
        if sim >= JACCARD_THRESHOLD:
            log.warning(
                "Near-duplicate detected (sim=%.2f): candidate vs '%s'",
                sim, row["article_topic"],
            )
            return True, row["article_topic"]

    for row in kb_rows:
        existing_mh = _kb_minhash(row["title"], row["body_text"])
        if existing_mh is None:
            continue
        sim = candidate.jaccard(existing_mh)
        if sim >= JACCARD_THRESHOLD:
            log.warning(
                "Near-duplicate detected (sim=%.2f): candidate vs KB '%s'",
                sim, row["title"],
            )
            return True, row["title"]
    return False, None


def backfill_fingerprints() -> int:
    """Compute fingerprints for all articles that lack one. Returns count updated."""
    updated = 0
    with get_db_conn() as conn:
        rows = conn.execute(
            "SELECT id, body_html FROM generated_articles WHERE structural_fingerprint IS NULL"
        ).fetchall()
        for row in rows:
            fp = compute_fingerprint(row["body_html"])
            conn.execute(
                "UPDATE generated_articles SET structural_fingerprint = ? WHERE id = ?",
                (fp, row["id"]),
            )
            updated += 1
    log.info("Backfilled fingerprints for %d articles", updated)
    return updated


# ── Semantic embedding dedup ──


def _tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens, dropping very short words."""
    return [w for w in text.split() if len(w) > 1]


def _build_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency for a corpus of token lists."""
    n = len(corpus_tokens)
    if n == 0:
        return {}
    df: Counter = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    return {word: math.log((n + 1) / (count + 1)) + 1 for word, count in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute a TF-IDF weighted word-frequency vector."""
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {w: (c / total) * idf.get(w, 1.0) for w, c in tf.items()}


def _dict_to_unit_vector(vec: dict[str, float]) -> dict[str, float]:
    """Normalize a sparse vector dict to unit length."""
    mag = math.sqrt(sum(v * v for v in vec.values()))
    if mag == 0:
        return vec
    return {k: v / mag for k, v in vec.items()}


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse unit vectors."""
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    return sum(a[k] * b[k] for k in keys)


def compute_embedding(body_html: str) -> list[float] | dict[str, float]:
    """Compute an embedding for article HTML content.

    Uses a pluggable provider controlled by EMBEDDING_PROVIDER env var:
    - "vertex" (default): Vertex AI gemini-embedding-001, 3072-dim dense vectors.
      Requires GOOGLE_SERVICE_ACCOUNT_JSON, GEMINI_PROJECT, GEMINI_LOCATION.
    - "minilm": all-MiniLM-L6-v2 dense 384-dim vectors (self-hosted CPU).
      Low domain accuracy on JSM tickets (F1=0.54, threshold stuck at floor 0.30).
    - "stub": legacy word-frequency unit vector (no external deps, low quality).

    Returns a dense list[float] for vertex/minilm, or sparse dict for stub.
    """
    if EMBEDDING_PROVIDER in ("vertex", "minilm"):
        normalized = _normalize_for_embedding(body_html or "")
        if not normalized:
            return []
        return _dense_embed(normalized)

    # stub fallback — no acronym expansion needed
    normalized = _normalize(body_html or "")
    if not normalized:
        return {}
    tokens = _tokenize(normalized)
    if not tokens:
        return {}
    idf = {w: 1.0 for w in set(tokens)}
    vec = _tfidf_vector(tokens, idf)
    return _dict_to_unit_vector(vec)


def _dense_embed(text: str) -> list[float]:
    """Compute a dense embedding via the shared embedding service.

    Delegates to services.embedding.embed_text, which routes to Vertex AI
    (gemini-embedding-001) or SentenceTransformer (all-MiniLM-L6-v2) based on
    the EMBEDDING_MODEL environment variable.
    """
    return _shared_embed_text(text)


def _minilm_embed(text: str) -> list[float]:
    """Compute a 384-dim dense embedding via the shared embedding service.

    Kept for backward compatibility — callers should prefer _dense_embed().
    """
    return _shared_embed_text(text)


def _embedding_to_vector(text: str) -> np.ndarray | dict[str, float]:
    """Return the embedding for article or KB text, used for pairwise comparison.

    Accepts HTML (generated articles) or plain text (KB articles) — both pass
    through the same normalizer before embedding.

    For vertex/minilm: returns np.ndarray (3072-dim or 384-dim, unit-normalized).
    For stub: returns sparse dict (legacy path).
    """
    if EMBEDDING_PROVIDER in ("vertex", "minilm"):
        normalized = _normalize_for_embedding(text or "")
        if not normalized:
            return np.array([])
        return np.array(_shared_embed_text(normalized))

    # stub fallback
    normalized = _normalize(text or "")
    if not normalized:
        return {}
    tokens = _tokenize(normalized)
    if not tokens:
        return {}
    idf = {w: 1.0 for w in set(tokens)}
    vec = _tfidf_vector(tokens, idf)
    return _dict_to_unit_vector(vec)


def _embedding_to_sparse(body_html: str) -> dict[str, float]:
    """Legacy stub-only sparse embedding. Kept for backward compat with tests."""
    text = _normalize(body_html or "")
    if not text:
        return {}
    tokens = _tokenize(text)
    if not tokens:
        return {}
    idf = {w: 1.0 for w in set(tokens)}
    vec = _tfidf_vector(tokens, idf)
    return _dict_to_unit_vector(vec)


def _compute_similarity(vec_a, vec_b) -> float:
    """Cosine similarity that works with both dense (ndarray) and sparse (dict) vectors."""
    if isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b))
    if isinstance(vec_a, dict) and isinstance(vec_b, dict):
        return _cosine_sim(vec_a, vec_b)
    return 0.0


def _embedding_from_stored(
    body_html: str,
    semantic_embedding: str | None,
) -> np.ndarray | dict[str, float]:
    """Deserialize a stored semantic_embedding or recompute from body_html."""
    if semantic_embedding:
        try:
            emb = json.loads(semantic_embedding)
            if EMBEDDING_PROVIDER in ("vertex", "minilm") and isinstance(emb, list):
                arr = np.array(emb, dtype=np.float32)
                if arr.size > 0:
                    return arr
            elif EMBEDDING_PROVIDER == "stub" and isinstance(emb, dict) and emb:
                return _dict_to_unit_vector({str(k): float(v) for k, v in emb.items()})
        except (json.JSONDecodeError, TypeError, ValueError):
            log.debug("Failed to deserialize semantic_embedding, recomputing")
    return _embedding_to_vector(body_html)


def _kb_embedding_from_stored(
    title: str,
    body_text: str,
    embedding_blob: bytes | None,
) -> np.ndarray | dict[str, float] | None:
    """Deserialize a kb_articles embedding BLOB or embed title+body on the fly.

    The BLOB is packed float32 written by services/vector_store._pack.
    Falls back to on-the-fly embedding when the BLOB is absent or corrupt.
    Note: stored BLOBs from MiniLM (384-dim) are incompatible with Vertex AI
    (3072-dim); when EMBEDDING_PROVIDER=vertex, on-the-fly embedding is used.
    """
    if EMBEDDING_PROVIDER == "minilm" and embedding_blob:
        try:
            vec = np.frombuffer(embedding_blob, dtype=np.float32).copy()
            if vec.size > 0:
                mag = float(np.linalg.norm(vec))
                if mag > 0:
                    vec = vec / mag
                return vec
        except (ValueError, TypeError):
            log.debug("Failed to deserialize kb embedding, recomputing")
    combined = f"{title or ''} {body_text or ''}".strip()
    if not combined:
        return None
    return _embedding_to_vector(combined)


# See note on _KB_ARTICLES_QUERY above regarding the full-table scan.
_KB_ARTICLES_EMBEDDING_QUERY = (
    "SELECT page_id, title, body_text, embedding FROM kb_articles "
    "WHERE body_text IS NOT NULL AND TRIM(body_text) != ''"
)


def is_semantic_duplicate(
    body_html: str,
    exclude_topic: str | None = None,
    threshold: float = COSINE_THRESHOLD,
) -> tuple[bool, str | None, float]:
    """Check whether body_html is a semantic duplicate of any existing article.

    Compares against generated_articles and kb_articles (Confluence KB pages)
    using cosine similarity on embedding vectors.  Stored embeddings are reused
    when available (semantic_embedding for generated_articles, embedding BLOB
    for kb_articles).  The *exclude_topic* filter applies only to
    generated_articles — all kb_articles are always checked.

    Returns (is_dup, matching_title, similarity) where matching_title is:
    - the article_topic from generated_articles, or
    - the Confluence page title from kb_articles.
    Callers must treat this string as an opaque label for logging; do NOT use
    it as a generated_articles lookup key, since KB hits return a page title,
    not an article_topic.
    """
    candidate = _embedding_to_vector(body_html)
    if isinstance(candidate, np.ndarray) and candidate.size == 0:
        return False, None, 0.0
    if isinstance(candidate, dict) and not candidate:
        return False, None, 0.0

    conn = get_db()
    try:
        query = (
            "SELECT article_topic, body_html, semantic_embedding "
            "FROM generated_articles"
        )
        params: tuple = ()
        if exclude_topic:
            query += " WHERE article_topic != ?"
            params = (exclude_topic,)
        gen_rows = conn.execute(query, params).fetchall()
        kb_rows = conn.execute(_KB_ARTICLES_EMBEDDING_QUERY).fetchall()
    finally:
        conn.close()

    for row in gen_rows:
        existing = _embedding_from_stored(row["body_html"], row["semantic_embedding"])
        if isinstance(existing, np.ndarray) and existing.size == 0:
            continue
        if isinstance(existing, dict) and not existing:
            continue
        sim = _compute_similarity(candidate, existing)
        if sim >= threshold:
            log.warning(
                "Semantic duplicate detected (cosine=%.3f): candidate vs '%s'",
                sim, row["article_topic"],
            )
            return True, row["article_topic"], sim

    for row in kb_rows:
        existing = _kb_embedding_from_stored(
            row["title"], row["body_text"], row["embedding"],
        )
        if existing is None:
            continue
        if isinstance(existing, np.ndarray) and existing.size == 0:
            continue
        if isinstance(existing, dict) and not existing:
            continue
        sim = _compute_similarity(candidate, existing)
        if sim >= threshold:
            log.warning(
                "Semantic duplicate detected (cosine=%.3f): candidate vs KB '%s'",
                sim, row["title"],
            )
            return True, row["title"], sim
    return False, None, 0.0


def is_duplicate_of_sections(
    article_text: str,
    existing_sections: list[str],
    threshold: float = COSINE_THRESHOLD,
) -> tuple[bool, int | None, float]:
    """Check whether article_text is a semantic duplicate of any existing section.

    Compares the embedding of *article_text* against each string in
    *existing_sections*.  This is a pure-function variant that does NOT
    query the database — it works entirely on the provided lists, making it
    suitable for output-layer dedup (e.g. Google Docs).

    Returns:
        (is_dup, matching_section_index, similarity)
        - is_dup: True if any section exceeds *threshold*
        - matching_section_index: index into *existing_sections* of the match,
          or None if no match
        - similarity: the highest cosine similarity found
    """
    candidate = _embedding_to_vector(article_text)
    if isinstance(candidate, np.ndarray) and candidate.size == 0:
        return False, None, 0.0
    if isinstance(candidate, dict) and not candidate:
        return False, None, 0.0

    best_sim = 0.0
    best_idx: int | None = None

    for idx, section in enumerate(existing_sections):
        existing = _embedding_to_vector(section)
        if isinstance(existing, np.ndarray) and existing.size == 0:
            continue
        if isinstance(existing, dict) and not existing:
            continue
        sim = _compute_similarity(candidate, existing)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    if best_sim >= threshold and best_idx is not None:
        log.warning(
            "Output-layer duplicate detected (cosine=%.3f) vs section %d",
            best_sim, best_idx,
        )
        return True, best_idx, best_sim

    return False, None, best_sim


# ── Cross-encoder re-ranking ──

_cross_encoder_model = None
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder(CROSS_ENCODER_NAME)
        log.info("Loaded cross-encoder: %s", CROSS_ENCODER_NAME)
    return _cross_encoder_model


def rerank_score(query: str, document: str) -> float:
    """Score a query-document pair using the cross-encoder (higher = more relevant)."""
    model = _get_cross_encoder()
    text_q = _normalize_for_embedding(query)
    text_d = _normalize_for_embedding(document)
    return float(model.predict([(text_q, text_d)])[0])


def rerank_pairs(query: str, documents: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
    """Re-rank documents by cross-encoder relevance to query.

    Returns list of (original_index, score) sorted by score descending.
    If top_k is set, only the top-k results are returned.
    """
    model = _get_cross_encoder()
    text_q = _normalize_for_embedding(query)
    pairs = [(text_q, _normalize_for_embedding(d)) for d in documents]
    scores = model.predict(pairs)
    ranked = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
    if top_k:
        ranked = ranked[:top_k]
    return [(idx, float(s)) for idx, s in ranked]


def backfill_embeddings() -> int:
    """Compute embeddings for all articles that lack one. Returns count updated."""
    updated = 0
    with get_db_conn() as conn:
        rows = conn.execute(
            "SELECT id, body_html FROM generated_articles WHERE semantic_embedding IS NULL"
        ).fetchall()
        for row in rows:
            emb = compute_embedding(row["body_html"])
            serialized = json.dumps(emb if isinstance(emb, (list, dict)) else emb)
            conn.execute(
                "UPDATE generated_articles SET semantic_embedding = ? WHERE id = ?",
                (serialized, row["id"]),
            )
            updated += 1
    log.info(
        "Backfilled embeddings for %d articles (provider=%s, model=%s)",
        updated, EMBEDDING_PROVIDER,
        os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
    )
    return updated
