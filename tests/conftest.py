"""Test fixtures — force the offline hash embedder so the suite runs anywhere."""
import os
import pathlib
import sys

# Must be set BEFORE importing rag.config (env-driven singleton).
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("USE_RERANKER", "false")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import pytest  # noqa: E402
from rag.config import get_config  # noqa: E402
from rag import ingest, index  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def built_index():
    """Build sample data + index once for the whole test session."""
    cfg = get_config(refresh=True)
    if not os.path.exists(cfg.raw_csv):
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_data.py")], check=True)
    ingest.run(cfg=cfg)
    index.build_index(cfg)
    yield
