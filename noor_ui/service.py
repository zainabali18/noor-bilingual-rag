"""Thin service layer between the UI and the RAG pipeline.

Reuses the existing backend (HybridRetriever + generate) rather than
re-implementing it. Retrieval and generation are called separately so that,
if generation fails (no key, network error, quota), Noor can still show the
best retrieved evidence instead of an error page.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import streamlit as st

from noor.config import config
from noor.generator import generate
from noor.pipeline import RAGPipeline
from noor.retriever import RetrievedPassage
from noor.vectorstore import VectorStore

from . import content


@dataclass
class Turn:
    """One question and Noor's response, stored in session state."""

    question: str
    answer_text: str = ""
    passages: List[RetrievedPassage] = field(default_factory=list)
    provider: str = "none"            # "openai" when a model wrote the answer
    notice: Optional[str] = None      # soft message shown under the answer
    error: Optional[str] = None       # set when the turn failed entirely
    elapsed: float = 0.0              # seconds


@st.cache_data(ttl=30, show_spinner=False)
def index_size() -> int:
    """Number of chunks in the vector store (0 means ingest hasn't run)."""
    try:
        return VectorStore().count()
    except Exception:
        return 0


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    """Load models and indexes once per server process."""
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def warm_up() -> bool:
    """Load both models once, while the visitor is still reading the home page.

    The embedder and reranker load lazily on first use, so without this the
    first question pays the full model-loading cost.
    """
    load_pipeline().retriever.retrieve("warm up", top_k=1)
    return True


def _normalise(question: str) -> str:
    return " ".join(question.split())


# Answers are cached per question so repeated questions (like the suggestions)
# return instantly. Exceptions are never cached, so a failed OpenAI call is
# retried next time rather than remembered.
@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def _retrieve(question: str, top_k: int) -> List[RetrievedPassage]:
    return load_pipeline().retriever.retrieve(question, top_k=top_k)


@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def _generate(question: str, top_k: int, provider: str, model: str) -> tuple:
    # provider and model are part of the cache key, so changing them in .env
    # never serves an answer written by a different setup
    answer = generate(question, _retrieve(question, top_k))
    return answer.text, answer.provider


def ask(question: str, top_k: int) -> Turn:
    """Retrieve evidence, then try to write an answer from it."""
    started = time.perf_counter()
    key = _normalise(question)
    passages = _retrieve(key, top_k)
    turn = Turn(question=question, passages=passages)

    if not passages:
        turn.answer_text = content.NO_RESULTS
        turn.elapsed = time.perf_counter() - started
        return turn

    try:
        turn.answer_text, turn.provider = _generate(
            key, top_k, config.llm_provider, config.openai_model
        )
    except Exception as exc:  # generation is optional; never lose the evidence
        turn.provider = "none"
        turn.notice = (
            f"Noor couldn't write an answer this time ({type(exc).__name__}), "
            "so it's showing the most relevant passage instead."
        )

    if turn.provider == "none":
        # Present the top passage as the answer, cited like a written one.
        turn.answer_text = f"{passages[0].text} [1]"
        turn.notice = turn.notice or content.RETRIEVAL_ONLY_NOTE

    turn.elapsed = time.perf_counter() - started
    return turn


def pipeline_summary() -> List[tuple]:
    """Technical configuration, shown only in About and retrieval details."""
    generator = (
        f"OpenAI · {config.openai_model}" if config.llm_provider == "openai"
        else "Off (retrieval only)"
    )
    return [
        ("Embeddings", config.embed_model.split("/")[-1]),
        ("Keyword search", "BM25 with Arabic normalisation"),
        ("Fusion", f"Reciprocal Rank Fusion (k={config.rrf_k})"),
        ("Reranker", config.reranker_model.split("/")[-1] if config.use_reranker else "Off"),
        ("Answer model", generator),
        ("Candidates", f"{config.dense_k} semantic + {config.lexical_k} keyword"),
    ]
