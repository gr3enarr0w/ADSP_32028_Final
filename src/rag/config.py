"""
config.py — single source of truth for all tunables, driven by environment
variables so the whole pipeline is *swappable via .env* (grading requirement:
Model-Agnostic / config-driven).

Nothing here imports heavy libraries, so it is safe to import anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Optionally load a .env file if python-dotenv is installed (no hard dependency).
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # ---- data paths ----
    raw_csv: str = field(default_factory=lambda: _env(
        "RAW_CSV", str(REPO_ROOT / "data" / "raw" / "SAMPLE_amazon_household_cleaning.csv")))
    processed_dir: str = field(default_factory=lambda: _env(
        "PROCESSED_DIR", str(REPO_ROOT / "data" / "processed")))
    index_dir: str = field(default_factory=lambda: _env(
        "INDEX_DIR", str(REPO_ROOT / "data" / "index")))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "household_cleaning"))

    # ---- catalog slice ----
    # Category keywords used to filter the full Kaggle catalog down to the slice.
    slice_keywords: tuple = ("clean", "household", "cleaning", "deterg", "dish",
                             "laundry", "polish", "wipe", "disinfect", "bath")

    # ---- embeddings (Model-Agnostic) ----
    # provider: "local" (sentence-transformers, no key), "openai", or "hash" (offline test).
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "local"))
    embedding_model_local: str = field(default_factory=lambda: _env(
        "EMBEDDING_MODEL_LOCAL", "sentence-transformers/all-MiniLM-L6-v2"))
    embedding_model_openai: str = field(default_factory=lambda: _env(
        "EMBEDDING_MODEL_OPENAI", "text-embedding-3-small"))
    hash_dim: int = field(default_factory=lambda: _env_int("HASH_DIM", 384))

    # ---- LLM (used by Answerer/Critic + prompt disclosure; teammates' nodes) ----
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-sonnet-5"))

    # ---- vector store ----
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "qdrant"))  # qdrant|chroma
    # Qdrant: leave QDRANT_URL blank for embedded local mode (no server needed);
    # set it (e.g. http://localhost:6333 or a Qdrant Cloud URL) to use a server.
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY", ""))

    # ---- retrieval / hybrid fusion ----
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 25))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    # weight on vector vs bm25 rank in reciprocal-rank fusion (0..1, 1 = vector only)
    hybrid_alpha: float = field(default_factory=lambda: _env_float("HYBRID_ALPHA", 0.5))

    # ---- reranker (optional cross-encoder) ----
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", True))
    reranker_model: str = field(default_factory=lambda: _env(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # ---- speech-to-text (voice input; Final deliverable) ------------------
    # faster-whisper model size (see src/rag/asr.py). "small.en" is what
    # notebooks/02_whisper_asr.ipynb prototyped with; "tiny.en"/"base.en" are
    # faster for a live demo, "medium.en" more accurate.
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "small.en"))
    # "auto" -> cuda when a GPU is visible, else cpu. Force with "cpu"/"cuda".
    asr_device: str = field(default_factory=lambda: _env("ASR_DEVICE", "auto"))
    # "auto" -> float16 on cuda, int8 on cpu.
    asr_compute_type: str = field(default_factory=lambda: _env("ASR_COMPUTE_TYPE", "auto"))
    asr_beam_size: int = field(default_factory=lambda: _env_int("ASR_BEAM_SIZE", 5))

    # ---- text-to-speech (Answerer's spoken summary; Final deliverable) ----
    # provider: "pyttsx3" (offline, no key, default so notebooks run with zero
    # setup), "openai" (needs OPENAI_API_KEY), or "elevenlabs" (needs
    # ELEVENLABS_API_KEY). See src/rag/tts.py for the dispatch + fallback logic.
    tts_provider: str = field(default_factory=lambda: _env("TTS_PROVIDER", "pyttsx3"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", ""))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "tts-1"))

    def embedding_signature(self) -> str:
        """Identifies the embedding space so we never mix vectors from two models."""
        if self.embedding_provider == "local":
            return f"local:{self.embedding_model_local}"
        if self.embedding_provider == "openai":
            return f"openai:{self.embedding_model_openai}"
        return f"hash:{self.hash_dim}"


_CONFIG: Config | None = None


def get_config(refresh: bool = False) -> Config:
    """Process-wide singleton config."""
    global _CONFIG
    if _CONFIG is None or refresh:
        _CONFIG = Config()
    return _CONFIG
