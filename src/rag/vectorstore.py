"""
vectorstore.py — pluggable vector-store backend behind `VECTOR_STORE`.

Backends:
  * "chroma" (default) — persistent, file-based, zero setup.
  * "qdrant"           — Qdrant. Embedded/local by default (no server needed);
                         set QDRANT_URL to point at a Qdrant server / Qdrant Cloud.

Both backends expose the same tiny interface so `index.py` and `retrieval.py`
never import a specific DB:

    store = get_store(cfg)
    store.rebuild(ids, embeddings, documents, metadatas, signature)  # build
    store.load()                                                     # open + verify
    ids, docs, metas = store.fetch_all()                             # BM25 corpus
    ranked_ids = store.query(embedding, n, filters)                  # dense search

`filters` is the generic planner dict ({price_max, price_min, min_rating, brand,
material}); each backend translates it to its own filter language. The embedding
space is stamped on build and verified on load so you can't query an index built
with a different model.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .config import Config, get_config


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

def _chroma_where(filters: Optional[dict]) -> Optional[dict]:
    if not filters:
        return None
    clauses = []
    if filters.get("price_max") is not None:
        clauses.append({"price": {"$gte": 0}})
        clauses.append({"price": {"$lte": float(filters["price_max"])}})
    if filters.get("price_min") is not None:
        clauses.append({"price": {"$gte": float(filters["price_min"])}})
    if filters.get("min_rating") is not None:
        clauses.append({"rating": {"$gte": float(filters["min_rating"])}})
    if filters.get("brand"):
        clauses.append({"brand": {"$eq": str(filters["brand"])}})
    if filters.get("material"):
        clauses.append({"material": {"$eq": str(filters["material"]).lower()}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.col = None

    def _client(self):
        import chromadb
        Path(self.cfg.index_dir).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=self.cfg.index_dir)

    def rebuild(self, ids, embeddings, documents, metadatas, signature) -> dict:
        client = self._client()
        try:
            client.delete_collection(self.cfg.collection)
        except Exception:  # noqa: BLE001
            pass
        col = client.create_collection(
            name=self.cfg.collection,
            metadata={"embedding_signature": signature, "hnsw:space": "cosine"},
        )
        B = 256
        for i in range(0, len(ids), B):
            col.add(
                ids=ids[i:i + B],
                embeddings=[e.tolist() for e in embeddings[i:i + B]],
                documents=documents[i:i + B],
                metadatas=metadatas[i:i + B],
            )
        return {"n_vectors": len(ids), "dim": int(len(embeddings[0])) if len(ids) else 0}

    def load(self):
        client = self._client()
        self.col = client.get_collection(self.cfg.collection)
        stored = (self.col.metadata or {}).get("embedding_signature")
        want = self.cfg.embedding_signature()
        if stored and stored != want:
            raise RuntimeError(
                f"Embedding mismatch: index built with {stored!r} but config wants "
                f"{want!r}. Rebuild the index or fix EMBEDDING_PROVIDER.")
        return self

    def fetch_all(self):
        d = self.col.get(include=["documents", "metadatas"])
        return d["ids"], d["documents"], d["metadatas"]

    def query(self, embedding, n, filters):
        emb = embedding.tolist() if hasattr(embedding, "tolist") else [float(x) for x in embedding]
        res = self.col.query(
            query_embeddings=[emb], n_results=n,
            where=_chroma_where(filters), include=["distances"],
        )
        return list(res.get("ids", [[]])[0])


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def _qdrant_filter(filters: Optional[dict]):
    if not filters:
        return None
    from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

    must = []
    if filters.get("price_max") is not None:
        must.append(FieldCondition(key="price", range=Range(gte=0, lte=float(filters["price_max"]))))
    if filters.get("price_min") is not None:
        must.append(FieldCondition(key="price", range=Range(gte=float(filters["price_min"]))))
    if filters.get("min_rating") is not None:
        must.append(FieldCondition(key="rating", range=Range(gte=float(filters["min_rating"]))))
    if filters.get("brand"):
        must.append(FieldCondition(key="brand", match=MatchValue(value=str(filters["brand"]))))
    if filters.get("material"):
        must.append(FieldCondition(key="material", match=MatchValue(value=str(filters["material"]).lower())))
    return Filter(must=must) if must else None


class QdrantStore:
    """Qdrant backend. Embedded local mode unless QDRANT_URL is set.

    Point ids are integers (Qdrant requires int/UUID); the real doc id and the
    embed-text document are carried in the payload under `_id` / `_document`.
    """

    _SIG_FILE = "qdrant_signature.txt"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = None
        self.name = cfg.collection

    def _connect(self):
        from qdrant_client import QdrantClient
        if self.cfg.qdrant_url:
            return QdrantClient(url=self.cfg.qdrant_url,
                                api_key=self.cfg.qdrant_api_key or None)
        path = os.path.join(self.cfg.index_dir, "qdrant")
        Path(path).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=path)  # embedded, no server

    def rebuild(self, ids, embeddings, documents, metadatas, signature) -> dict:
        from qdrant_client.models import Distance, VectorParams, PointStruct

        client = self._connect()
        dim = int(len(embeddings[0])) if len(ids) else 384
        try:
            client.delete_collection(self.name)
        except Exception:  # noqa: BLE001
            pass
        client.create_collection(
            self.name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

        points = []
        for idx, (did, emb, doc, meta) in enumerate(zip(ids, embeddings, documents, metadatas)):
            payload = dict(meta)
            payload["_id"] = did
            payload["_document"] = doc
            vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            points.append(PointStruct(id=idx, vector=vec, payload=payload))
        B = 256
        for i in range(0, len(points), B):
            client.upsert(self.name, points=points[i:i + B])

        Path(self.cfg.index_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.cfg.index_dir, self._SIG_FILE), "w") as f:
            f.write(signature)
        # release the embedded lock so a reader can open the path afterwards
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        return {"n_vectors": len(points), "dim": dim}

    def load(self):
        sig_path = os.path.join(self.cfg.index_dir, self._SIG_FILE)
        if os.path.exists(sig_path):
            stored = open(sig_path).read().strip()
            want = self.cfg.embedding_signature()
            if stored and stored != want:
                raise RuntimeError(
                    f"Embedding mismatch: index built with {stored!r} but config wants "
                    f"{want!r}. Rebuild the index or fix EMBEDDING_PROVIDER.")
        self.client = self._connect()
        return self

    def fetch_all(self):
        ids, docs, metas = [], [], []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.name, with_payload=True, with_vectors=False, limit=256, offset=offset)
            for p in points:
                pl = p.payload or {}
                ids.append(pl.get("_id", str(p.id)))
                docs.append(pl.get("_document", ""))
                metas.append({k: v for k, v in pl.items() if k not in ("_id", "_document")})
            if offset is None:
                break
        return ids, docs, metas

    def query(self, embedding, n, filters):
        vec = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        qfilter = _qdrant_filter(filters)
        # query_points is the modern API; fall back to search on older clients
        try:
            resp = self.client.query_points(
                self.name, query=vec, query_filter=qfilter, limit=n, with_payload=True)
            pts = resp.points
        except AttributeError:  # pragma: no cover - old qdrant-client
            pts = self.client.search(
                self.name, query_vector=vec, query_filter=qfilter, limit=n, with_payload=True)
        return [(p.payload or {}).get("_id", str(p.id)) for p in pts]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_store(cfg: Optional[Config] = None):
    cfg = cfg or get_config()
    vs = cfg.vector_store.lower()
    if vs == "chroma":
        return ChromaStore(cfg)
    if vs == "qdrant":
        return QdrantStore(cfg)
    raise ValueError(
        f"Unknown VECTOR_STORE={cfg.vector_store!r} (supported: chroma, qdrant)")
