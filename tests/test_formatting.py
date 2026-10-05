"""Tests for the UI's pure-Python formatting helpers (no Streamlit needed)."""
from noor_ui.formatting import (
    answer_to_html,
    fmt_rank,
    fmt_score,
    link_citations,
    source_anchor,
    text_direction,
)


def test_direction_english_and_arabic():
    assert text_direction("How long is the Golden Visa valid?") == "ltr"
    assert text_direction("ما مدة صلاحية التأشيرة الذهبية؟") == "rtl"


def test_direction_mixed_uses_majority_script():
    assert text_direction("هل يمكن كفالة أفراد الأسرة (Family sponsorship)؟") == "rtl"
    assert text_direction("The Arabic word نور means light") == "ltr"


def test_citations_link_to_matching_source_cards():
    out = link_citations("Valid for ten years [1][2].", turn_index=0, n_sources=2)
    assert f'href="#{source_anchor(0, 1)}"' in out
    assert f'href="#{source_anchor(0, 2)}"' in out
    assert "[1]" not in out


def test_grouped_and_arabic_comma_citations():
    out = link_citations("نعم [1، 2]", turn_index=3, n_sources=2)
    assert out.count('class="noor-cite"') == 2


def test_out_of_range_citation_is_not_linked():
    out = link_citations("Claim [5]", turn_index=0, n_sources=2)
    assert "is-orphan" in out and "href" not in out


def test_answer_html_escapes_model_output():
    html = answer_to_html("<script>alert(1)</script> **bold** [1]", 0, 1)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<strong>bold</strong>" in html


def test_answer_html_lists_and_paragraphs():
    text = "Intro line.\n\n- first [1]\n- second\n\n1. step one\n2. step two\n\nOutro."
    html = answer_to_html(text, 0, 1)
    assert html.startswith("<p>Intro line.</p>")
    assert "<ul><li>first" in html and "<li>second</li></ul>" in html
    assert "<ol><li>step one</li><li>step two</li></ol>" in html
    assert html.endswith("<p>Outro.</p>")
    assert "\n\n" not in html  # no blank lines that could break HTML rendering


def test_score_and_rank_formatting():
    assert fmt_score(None) == "—"
    assert fmt_score(0.71234) == "0.712"
    assert fmt_rank(None) == "—"
    assert fmt_rank(2) == "#2"
