"""Tests for Arabic normalisation and BM25 tokenisation (no model downloads)."""
from noor.lexical import normalise, tokenize


def test_normalise_strips_diacritics():
    assert normalise("مَرْحَبًا") == "مرحبا"


def test_normalise_unifies_alef():
    # أ إ آ should all fold to ا
    assert normalise("أحمد") == normalise("احمد")
    assert normalise("إيمان") == normalise("ايمان")


def test_normalise_taa_marbuta_and_alef_maqsura():
    assert normalise("مدرسة") == normalise("مدرسه")
    assert normalise("مصطفى") == normalise("مصطفي")


def test_tokenize_mixed_script():
    toks = tokenize("Visa تأشيرة 2025")
    assert "visa" in toks
    assert "2025" in toks
    assert any("؀" <= ch <= "ۿ" for tok in toks for ch in tok)


def test_tokenize_lowercases_english():
    assert tokenize("Golden VISA") == ["golden", "visa"]
