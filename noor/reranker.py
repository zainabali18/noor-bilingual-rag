"""Multilingual cross-encoder reranker.

Bi-encoders (used for the dense search) embed query and passage separately,
which is fast but approximate. A cross-encoder reads the query and passage
*together* and scores their relevance directly, which is far more accurate.
We use it only on the fused shortlist, getting the accuracy of a cross-encoder
at the cost of ranking a handful of candidates.
"""
from __future__ import annotations

from typing import List

from .config import config
from .model_loading import MODEL_LOCK, get_model
from .vectorstore import Hit


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.reranker_model

    @property
    def model(self):
        from sentence_transformers import CrossEncoder

        return get_model("reranker", CrossEncoder, self.model_name, config.model_device)

    def rerank(self, query: str, hits: List[Hit], top_k: int) -> List[Hit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        with MODEL_LOCK:
            scores = self.model.predict(pairs, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
