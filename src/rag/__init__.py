"""
rag — Shane's Agentic RAG package for the Voice-to-Voice Product Assistant.

Public surface used by teammates:
    from rag.config import get_config
    from rag.ingest import run as run_ingest
    from rag.index import build_index, load_collection
    from rag.retrieval import HybridRetriever
    from rag.rag_search import rag_search        # <- the rag.search MCP tool body
"""
from .config import get_config  # noqa: F401

__all__ = ["get_config"]
__version__ = "0.1.0"
