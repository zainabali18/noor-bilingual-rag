"""Thin wrapper over a persistent ChromaDB collection for dense retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .chunker import Chunk
from .config import config
from .embeddings import Embedder


@dataclass
class Hit:
    chunk_id: str
    text: str
    topic: str
    lang: str
    source: str
    score: float  # similarity in [0, 1] for dense; rank-based elsewhere
    doc_id: str = ""


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self.embedder = embedder or Embedder()
        self.collection = self.client.get_or_create_collection(
            name=config.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(config.collection)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=config.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: List[Chunk]) -> None:
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"topic": c.topic, "lang": c.lang, "source": c.source, "doc_id": c.doc_id}
                for c in chunks
            ],
        )

    def search(self, query: str, k: int) -> List[Hit]:
        q_vec = self.embedder.embed_one(query)
        res = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Hit] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                Hit(
                    chunk_id=cid,
                    text=doc,
                    topic=meta.get("topic", ""),
                    lang=meta.get("lang", ""),
                    source=meta.get("source", ""),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                    doc_id=meta.get("doc_id", ""),
                )
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
