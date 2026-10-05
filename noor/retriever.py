"""Hybrid retrieval: dense + lexical, fused with RRF, then cross-encoder rerank.

Pipeline:
    1. Dense search (semantic, cross-lingual) -> candidate list A
    2. Lexical BM25 search (exact terms)      -> candidate list B
    3. Reciprocal Rank Fusion merges A and B into one ranked shortlist
    4. A multilingual cross-encoder reranks the shortlist (optional)
    5. Return the top_k passages for grounding

Reciprocal Rank Fusion (Cormack et al., 2009) scores each document by
    sum over lists of 1 / (rrf_k + rank)
It needs only the rank of a document in each list, not comparable scores, so it
fuses the cosine similarities and BM25 scores cleanly without normalisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import config
from .lexical import LexicalIndex
from .reranker import Reranker
from .vectorstore import Hit, VectorStore


@dataclass
class RetrievedPassage:
    chunk_id: str
    text: str
    topic: str
    lang: str
    source: str
    score: float
    doc_id: str = ""
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    # Per-stage scores, kept for transparency (shown in the UI's retrieval details)
    dense_score: Optional[float] = None    # cosine similarity from vector search
    lexical_score: Optional[float] = None  # raw BM25 score
    rrf_score: Optional[float] = None      # fused Reciprocal Rank Fusion score
    rerank_score: Optional[float] = None   # cross-encoder relevance score


def reciprocal_rank_fusion(
    dense: List[Hit], lexical: List[Hit], rrf_k: int
) -> List[RetrievedPassage]:
    table: Dict[str, RetrievedPassage] = {}
    fused: Dict[str, float] = {}

    for rank, hit in enumerate(dense, start=1):
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        p = table.setdefault(hit.chunk_id, _to_passage(hit))
        p.dense_rank = rank
        p.dense_score = hit.score

    for rank, hit in enumerate(lexical, start=1):
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        p = table.setdefault(hit.chunk_id, _to_passage(hit))
        p.lexical_rank = rank
        p.lexical_score = hit.score

    for cid, score in fused.items():
        table[cid].score = score
        table[cid].rrf_score = score

    return sorted(table.values(), key=lambda p: p.score, reverse=True)


def _to_passage(hit: Hit) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=hit.chunk_id, text=hit.text, topic=hit.topic,
        lang=hit.lang, source=hit.source, score=hit.score, doc_id=hit.doc_id,
    )


class HybridRetriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        lexical_index: Optional[LexicalIndex] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.vs = vector_store or VectorStore()
        self.lex = lexical_index or LexicalIndex.from_disk()
        self.reranker = reranker
        if config.use_reranker and self.reranker is None:
            self.reranker = Reranker()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedPassage]:
        top_k = top_k or config.top_k
        dense = self.vs.search(query, config.dense_k)
        lexical = self.lex.search(query, config.lexical_k)
        fused = reciprocal_rank_fusion(dense, lexical, config.rrf_k)

        if self.reranker is not None and fused:
            shortlist = fused[: max(config.dense_k, config.lexical_k)]
            hits = [
                Hit(p.chunk_id, p.text, p.topic, p.lang, p.source, p.score, p.doc_id)
                for p in shortlist
            ]
            reranked = self.reranker.rerank(query, hits, top_k)
            order = {h.chunk_id: i for i, h in enumerate(reranked)}
            chosen = [p for p in shortlist if p.chunk_id in order]
            chosen.sort(key=lambda p: order[p.chunk_id])
            for p in chosen:
                p.score = next(h.score for h in reranked if h.chunk_id == p.chunk_id)
                p.rerank_score = p.score
            return chosen
        return fused[:top_k]
