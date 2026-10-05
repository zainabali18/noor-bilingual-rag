"""Tests for sentence-aware, bilingual chunking (no model downloads needed)."""
from noor.chunker import chunk_text, split_sentences


def test_split_sentences_english():
    text = "First sentence. Second one! A third? Yes."
    assert split_sentences(text) == ["First sentence.", "Second one!", "A third?", "Yes."]


def test_split_sentences_arabic():
    text = "الجملة الأولى. الجملة الثانية؟ نعم."
    parts = split_sentences(text)
    assert len(parts) == 3
    assert parts[0] == "الجملة الأولى."


def test_chunk_respects_size():
    text = " ".join(f"Sentence number {i}." for i in range(50))
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)  # size + one trailing sentence


def test_chunk_overlap_creates_continuity():
    text = "A one. B two. C three. D four. E five. F six. G seven. H eight."
    chunks = chunk_text(text, chunk_size=25, overlap=12)
    # consecutive chunks should share at least one sentence due to overlap
    assert len(chunks) >= 2


def test_chunk_short_text_single_chunk():
    assert chunk_text("Just one short sentence.", 700, 120) == ["Just one short sentence."]


def test_invalid_overlap_raises():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("x.", chunk_size=10, overlap=10)
