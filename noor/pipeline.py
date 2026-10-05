"""End-to-end RAG: retrieve -> (rerank) -> generate a grounded, cited answer."""
from __future__ import annotations

from typing import Optional

from .generator import Answer, generate
from .retriever import HybridRetriever


class RAGPipeline:
    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or HybridRetriever()

    def answer(self, question: str, top_k: Optional[int] = None) -> Answer:
        passages = self.retriever.retrieve(question, top_k=top_k)
        return generate(question, passages)
