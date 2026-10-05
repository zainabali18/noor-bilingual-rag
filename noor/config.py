"""Central configuration, driven by environment variables with sensible defaults.

Nothing here requires a paid API. The dense embedder and the reranker are
open-source multilingual models downloaded from the Hugging Face Hub on first
run. Generation is optional: without a key, Noor returns an extractive answer
built from the retrieved passages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # Data / storage
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    chroma_dir: Path = field(default_factory=lambda: ROOT / ".chroma")
    collection: str = "noor"

    # Models (multilingual, open-source)
    embed_model: str = _get(
        "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    reranker_model: str = _get(
        "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    use_reranker: bool = _get("USE_RERANKER", "true").lower() == "true"
    # "cpu" is the safe default: these models are small, and Apple's GPU backend
    # (mps) can abort when a web server runs them from several threads.
    model_device: str = _get("MODEL_DEVICE", "cpu")

    # Chunking
    chunk_size: int = _get_int("CHUNK_SIZE", 700)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 120)

    # Retrieval
    dense_k: int = _get_int("DENSE_K", 10)      # candidates from vector search
    lexical_k: int = _get_int("LEXICAL_K", 10)  # candidates from BM25
    rrf_k: int = _get_int("RRF_K", 60)          # RRF smoothing constant
    top_k: int = _get_int("TOP_K", 4)           # passages passed to the generator

    # Generation (optional)
    llm_provider: str = _get("LLM_PROVIDER", "none")  # openai | none
    openai_model: str = _get("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = _get_float("TEMPERATURE", 0.1)


config = Config()
