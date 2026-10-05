"""Sentence-aware chunking that works for both English and Arabic text.

Text is split into sentences first (respecting Arabic punctuation such as the
Arabic question mark and comma), then sentences are packed into overlapping
windows of roughly `chunk_size` characters. Chunking on sentence boundaries
keeps passages coherent, which improves both retrieval and grounding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .loader import Document

# End-of-sentence markers for Latin and Arabic scripts.
#   .  !  ?   ।   Arabic: ؟ (U+061F) ۔ (U+06D4)   plus newline breaks
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?؟۔])\s+|\n{2,}")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    topic: str
    lang: str
    text: str
    source: str


def split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Pack sentences into overlapping character windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    length = 0

    for sentence in sentences:
        # +1 accounts for the joining space
        if current and length + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # build the overlap tail from the end of the current window
            tail: List[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) + 1 > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current = tail
            length = tail_len
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(docs: List[Document], chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{doc.lang}-{i}",
                    doc_id=doc.doc_id,
                    topic=doc.topic,
                    lang=doc.lang,
                    text=piece,
                    source=doc.source,
                )
            )
    return chunks
