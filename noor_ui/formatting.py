"""Pure-Python text helpers for the UI (no Streamlit import, so it is testable).

* `text_direction` decides whether text should render right-to-left.
* `answer_to_html` turns a model answer (light Markdown with [n] citations)
  into safe HTML with clickable citation chips that link to source cards.
"""
from __future__ import annotations

import html
import re
from typing import List, Optional

# Arabic script blocks: Arabic, Arabic Supplement, Arabic Extended-A,
# Arabic Presentation Forms-A and -B.
_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")

# [1]  [1, 2]  [1،2]  (also matches Arabic-Indic digits, which int() accepts)
_CITATION = re.compile(r"\[(\d+(?:\s*[,،]\s*\d+)*)\]")
_CITATION_SPLIT = re.compile(r"\s*[,،]\s*")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_BULLET = re.compile(r"^[-*•]\s+(.*)$")
_NUMBERED = re.compile(r"^\d+[.)]\s+(.*)$")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def text_direction(text: str) -> str:
    """Return "rtl" when Arabic letters outnumber Latin letters, else "ltr"."""
    arabic = len(_ARABIC.findall(text or ""))
    latin = len(_LATIN.findall(text or ""))
    return "rtl" if arabic > latin else "ltr"


def is_arabic(text: str) -> bool:
    return text_direction(text) == "rtl"


def esc(text: str) -> str:
    """HTML-escape text from documents or the model before rendering."""
    return html.escape(text or "", quote=True)


def source_anchor(turn_index: int, source_number: int) -> str:
    return f"noor-src-{turn_index}-{source_number}"


def link_citations(escaped: str, turn_index: int, n_sources: int) -> str:
    """Replace [n] markers in already-escaped text with citation chips."""

    def _chip(match: re.Match) -> str:
        chips = []
        for raw in _CITATION_SPLIT.split(match.group(1)):
            n = int(raw)
            if 1 <= n <= n_sources:
                chips.append(
                    f'<a class="noor-cite" href="#{source_anchor(turn_index, n)}" '
                    f'title="Go to source {n}">{n}</a>'
                )
            else:  # a number the model invented; show it, but don't link it
                chips.append(f'<span class="noor-cite is-orphan">{n}</span>')
        return "".join(chips)

    return _CITATION.sub(_chip, escaped)


def _inline(text: str, turn_index: int, n_sources: int) -> str:
    out = esc(text)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _CODE.sub(r"<code>\1</code>", out)
    return link_citations(out, turn_index, n_sources)


def answer_to_html(text: str, turn_index: int, n_sources: int) -> str:
    """Render a light Markdown subset as HTML.

    Supports paragraphs, line breaks, bullet and numbered lists, headings,
    **bold**, `code` and [n] citations. Everything else is escaped, so model
    output can never inject markup into the page.
    """
    blocks: List[str] = []
    paragraph: List[str] = []
    items: List[str] = []
    list_tag: Optional[str] = None

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append("<p>" + "<br>".join(paragraph) + "</p>")
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if items:
            body = "".join(f"<li>{item}</li>" for item in items)
            blocks.append(f"<{list_tag}>{body}</{list_tag}>")
            items.clear()
        list_tag = None

    for raw in (text or "").strip().splitlines():
        line = raw.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue

        heading = _HEADING.match(line)
        bullet = _BULLET.match(line)
        numbered = _NUMBERED.match(line)

        if heading:
            flush_paragraph()
            flush_list()
            blocks.append(f"<h4>{_inline(heading.group(1), turn_index, n_sources)}</h4>")
        elif bullet or numbered:
            flush_paragraph()
            tag = "ul" if bullet else "ol"
            if list_tag != tag:
                flush_list()
                list_tag = tag
            content = (bullet or numbered).group(1)
            items.append(_inline(content, turn_index, n_sources))
        else:
            flush_list()
            paragraph.append(_inline(line, turn_index, n_sources))

    flush_paragraph()
    flush_list()
    return "".join(blocks)


def fmt_score(value: Optional[float], digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def fmt_rank(value: Optional[int]) -> str:
    return "—" if value is None else f"#{value}"
