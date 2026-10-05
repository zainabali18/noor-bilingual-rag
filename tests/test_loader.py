"""Tests for the document loader against the shipped bilingual dataset."""
from noor.config import config
from noor.loader import load_documents


def test_loads_parallel_bilingual_corpus():
    docs = load_documents(config.data_dir)
    langs = {d.lang for d in docs}
    assert langs == {"en", "ar"}

    en_ids = {d.doc_id for d in docs if d.lang == "en"}
    ar_ids = {d.doc_id for d in docs if d.lang == "ar"}
    # every English topic must have an Arabic counterpart and vice versa
    assert en_ids == ar_ids
    assert len(en_ids) >= 5


def test_documents_have_text_and_topic():
    for d in load_documents(config.data_dir):
        assert d.text.strip()
        assert d.topic.strip()
