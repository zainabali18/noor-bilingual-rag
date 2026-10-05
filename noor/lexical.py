"""Lexical (BM25) retrieval with light Arabic normalisation.

Dense retrieval is strong on meaning but can miss exact terms, names, and
numbers. BM25 covers that gap. Fusing both (see retriever.py) is more robust
than either alone, which is the whole point of hybrid search.

The chunk corpus is persisted to JSON at ingest time so the BM25 index can be
rebuilt in memory on startup and stays in sync with the vector store.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import List

from .chunker import Chunk
from .config import config
from .vectorstore import Hit

_WORD = re.compile(r"\w+", re.UNICODE)

# Arabic normalisation: unify alef forms, ya/alef-maqsura, ta-marbuta, and
# strip tashkeel (diacritics) and tatweel so lexical matching is forgiving.
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا",  # أ إ آ -> ا
    "ى": "ي",  # ى -> ي
    "ة": "ه",  # ة -> ه
})


def normalise(text: str) -> str:
    text = _TASHKEEL.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ARABIC_MAP)
    return text.lower()


def tokenize(text: str) -> List[str]:
    return _WORD.findall(normalise(text))


class LexicalIndex:
    def __init__(self, chunks: List[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self._tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, k: int) -> List[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits: List[Hit] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            hits.append(
                Hit(chunk_id=c.chunk_id, text=c.text, topic=c.topic,
                    lang=c.lang, source=c.source, score=float(scores[i]),
                    doc_id=c.doc_id)
            )
        return hits

    # ---- persistence -------------------------------------------------------
    @staticmethod
    def corpus_path() -> Path:
        return config.chroma_dir / "chunks.json"

    @classmethod
    def save_corpus(cls, chunks: List[Chunk]) -> None:
        path = cls.corpus_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(c) for c in chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_disk(cls) -> "LexicalIndex":
        path = cls.corpus_path()
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m noor.ingest` first."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**d) for d in data]
        return cls(chunks)
