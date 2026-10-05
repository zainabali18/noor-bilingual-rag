"""Reusable rendering functions for Noor's interface.

Static markup is built as HTML strings (rendered with st.html) and styled by
assets/noor.css. Interactive pieces use native Streamlit widgets inside keyed
containers, so the CSS can target them through stable `st-key-*` classes
rather than fragile auto-generated class names.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from typing import Callable, List

import streamlit as st

from noor.retriever import RetrievedPassage

from . import content
from .formatting import (
    answer_to_html,
    esc,
    fmt_rank,
    fmt_score,
    source_anchor,
    text_direction,
)
from .service import Turn, pipeline_summary


# --------------------------------------------------------------------------- #
#  Brand + icons
# --------------------------------------------------------------------------- #
def mark_svg(uid: str = "mark") -> str:
    """Noor's mark as raw SVG: light glowing inside a Gulf-style arched window."""
    arch = "M7 37V17.5C7 10.3 12.8 4.5 20 4.5S33 10.3 33 17.5V37Z"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
        "<defs>"
        f'<radialGradient id="g-{uid}" cx="50%" cy="50%" r="50%">'
        '<stop offset="0" stop-color="#FFE9B8"/>'
        '<stop offset=".38" stop-color="#E2AD4A" stop-opacity=".95"/>'
        '<stop offset="1" stop-color="#E2AD4A" stop-opacity="0"/>'
        "</radialGradient>"
        f'<clipPath id="c-{uid}"><path d="{arch}"/></clipPath>'
        "</defs>"
        f'<path d="{arch}" fill="#13213D"/>'
        f'<circle cx="20" cy="22.5" r="11" fill="url(#g-{uid})" clip-path="url(#c-{uid})"/>'
        '<circle cx="20" cy="22.5" r="3.4" fill="#FFF4DA"/>'
        "</svg>"
    )


@lru_cache(maxsize=1)
def _mark_src() -> str:
    encoded = base64.b64encode(mark_svg().encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def mark(css_class: str = "noor-mark", label: str = "") -> str:
    """The mark as an <img> (st.html sanitises inline SVG away)."""
    alt = esc(label)
    hidden = "" if label else ' aria-hidden="true"'
    return f'<img class="{css_class}" src="{_mark_src()}" alt="{alt}"{hidden}>'


def _html(markup: str) -> None:
    st.html(markup)


# --------------------------------------------------------------------------- #
#  Navigation, settings, about
# --------------------------------------------------------------------------- #
def nav(on_new_chat: Callable[[], None]) -> None:
    with st.container(key="noor_nav", horizontal=True, vertical_alignment="center", gap="small"):
        st.html(
            f'<div class="noor-brand">{mark()}'
            f'<span class="noor-wordmark">Noor</span>'
            f'<span class="noor-wordmark-ar">نور</span></div>',
            width="stretch",
        )
        st.button("New chat", key="nav_new_chat", type="tertiary", on_click=on_new_chat)
        if st.button("About", key="nav_about", type="tertiary"):
            about_dialog()
        with st.popover("Preferences", icon=":material/tune:", key="nav_prefs"):
            preferences()


def preferences() -> None:
    _html(
        '<p class="noor-pop-title">Preferences</p>'
        '<p class="noor-pop-sub">How many passages Noor reads before answering. '
        "More can help broad questions; fewer keep answers focused.</p>"
    )
    st.slider("Sources per answer", min_value=1, max_value=8, key="top_k")
    st.toggle("Open retrieval details by default", key="show_details")


@st.dialog("About Noor", width="medium")
def about_dialog() -> None:
    rows = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in pipeline_summary())
    _html(
        '<div class="noor-about">'
        "<p>Noor is a bilingual assistant for everyday questions about living and "
        "working in the UAE. Ask in English or Arabic: Noor looks for the answer in "
        "both languages and shows you exactly where it came from.</p>"
        "<h4>How Noor answers</h4>"
        "<ol>"
        "<li><b>Understands your question</b> in either language, by meaning rather than exact wording.</li>"
        "<li><b>Searches both languages</b> with meaning-based and keyword search together.</li>"
        "<li><b>Checks the strongest evidence</b> with a second model that re-reads the best candidates.</li>"
        "<li><b>Answers with sources</b> you can open and read for yourself.</li>"
        "</ol>"
        "<h4>Under the hood</h4>"
        f'<dl class="noor-kv">{rows}</dl>'
        '<p class="small">Retrieval quality is measured on a bilingual evaluation set '
        "(Hit@1, Recall@3, MRR, nDCG@5). The results and code are in the project repository.</p>"
        "<h4>About the information</h4>"
        '<p class="small">The knowledge base is original sample content written for this '
        "project. It is general information, not legal, financial or official advice. "
        "Always confirm important details with the relevant UAE authority.</p>"
        '<p class="small">Built by Zainab Ali.</p>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
#  Home
# --------------------------------------------------------------------------- #
def hero() -> None:
    _html(
        '<section class="noor-hero">'
        f'<div class="noor-hero-mark">{mark()}</div>'
        f"<h1>{esc(content.HEADLINE)}</h1>"
        f'<p class="noor-hero-ar" lang="ar">{esc(content.HEADLINE_AR)}</p>'
        f'<p class="noor-hero-intro">{esc(content.INTRO)}</p>'
        "</section>"
    )


def home_ask(on_submit: Callable[[str], None]) -> None:
    with st.container(key="noor_home_ask"):
        st.chat_input(
            content.INPUT_PLACEHOLDER,
            key="home_input",
            on_submit=on_submit,
            args=("home_input",),
        )
    _html(
        '<p class="noor-ask-hint">English · <span class="ar" lang="ar">العربية</span>'
        " · Answers grounded in Noor's knowledge base</p>"
    )


def suggestions(on_pick: Callable[[str], None]) -> None:
    _html('<p class="noor-section-label">Or start with one of these</p>')
    with st.container(key="noor_suggestions"):
        items = list(enumerate(content.SUGGESTIONS))
        for row in (items[:3], items[3:]):
            for col, (i, (question, arabic)) in zip(st.columns(3, gap="small"), row):
                with col:
                    st.button(
                        question,
                        key=f"sugg_{'ar' if arabic else 'en'}_{i}",
                        on_click=on_pick,
                        args=(question,),
                        width="stretch",
                        wrap=True,
                    )


def flow_and_coverage() -> None:
    steps = "".join(f"<li>{esc(s)}</li>" for s in content.FLOW_STEPS)
    _html(
        f'<ol class="noor-flow" aria-label="How Noor answers">{steps}</ol>'
        f'<p class="noor-coverage">{esc(content.COVERAGE)}</p>'
        f'<p class="noor-footer">{esc(content.FOOTER)}</p>'
    )


def empty_index_notice() -> None:
    with st.container(key="noor_home_notice"):
        _html(
            '<div class="noor-card">'
            f"<h3>{esc(content.EMPTY_INDEX_TITLE)}</h3>"
            f"<p>{content.EMPTY_INDEX_BODY}</p>"
            "</div>"
        )


# --------------------------------------------------------------------------- #
#  Conversation
# --------------------------------------------------------------------------- #
def question(text: str, index: int) -> None:
    _html(
        f'<h2 class="noor-question" id="noor-turn-{index}" dir="{text_direction(text)}">'
        f"{esc(text)}</h2>"
    )


def thinking_html() -> str:
    return (
        f'<div class="noor-thinking" role="status">{mark()}'
        f'<span>{esc(content.THINKING)}<span class="dots"></span></span></div>'
    )


def response(turn: Turn, index: int, show_details: bool) -> None:
    if turn.error:
        _error(turn.error)
        return

    _answer(turn, index)
    if turn.passages:
        _sources(turn.passages, index)
        _retrieval_details(turn, index, show_details)


def _answer(turn: Turn, index: int) -> None:
    n = len(turn.passages)
    meta = f"{n} source{'s' if n != 1 else ''} · {turn.elapsed:.1f}s" if n else ""
    body = answer_to_html(turn.answer_text, index, n)
    notice = (
        f'<div class="noor-notice"><span>{esc(turn.notice)}</span></div>'
        if turn.notice else ""
    )
    _html(
        f'<div class="noor-byline">{mark()}'
        f'<span class="name">Noor</span><span class="meta">{esc(meta)}</span></div>'
        f'<div class="noor-answer" dir="{text_direction(turn.answer_text)}">{body}</div>'
        f"{notice}"
    )


def _source_card(p: RetrievedPassage, number: int, index: int) -> str:
    direction = "rtl" if p.lang == "ar" else text_direction(p.text)
    lang_class = "lang is-ar" if p.lang == "ar" else "lang"
    label = content.LANG_LABEL.get(p.lang, p.lang)
    return (
        f'<details class="noor-src" id="{source_anchor(index, number)}" dir="{direction}">'
        "<summary>"
        f'<span class="num">{number}</span>'
        f'<span class="title">{esc(p.topic)}</span>'
        f'<span class="{lang_class}">{esc(label)}</span>'
        f'<span class="passage">{esc(p.text)}</span>'
        '<span class="toggle"></span>'
        "</summary>"
        "</details>"
    )


def _sources(passages: List[RetrievedPassage], index: int) -> None:
    cards = "".join(_source_card(p, i, index) for i, p in enumerate(passages, start=1))
    _html(
        f'<div class="noor-sources-head">Sources <span>{len(passages)}</span></div>'
        f'<div class="noor-sources">{cards}</div>'
        f'<div class="noor-disclaimer"><span>{esc(content.DISCLAIMER)}</span></div>'
    )


def _retrieval_details(turn: Turn, index: int, expanded: bool) -> None:
    with st.expander("View retrieval details", expanded=expanded):
        rows = []
        for i, p in enumerate(turn.passages, start=1):
            rows.append(
                "<tr>"
                f"<td>{i}</td>"
                f'<td class="topic">{esc(p.topic)}<div class="file">{esc(p.source)}</div></td>'
                f"<td>{esc(content.LANG_LABEL.get(p.lang, p.lang))}</td>"
                # getattr: tolerate passages created before these fields existed
                f"<td>{fmt_score(getattr(p, 'dense_score', None))}</td>"
                f"<td>{fmt_score(getattr(p, 'lexical_score', None), 2)}</td>"
                f"<td>{fmt_score(getattr(p, 'rrf_score', None), 4)}</td>"
                f"<td>{fmt_score(getattr(p, 'rerank_score', None))}</td>"
                f"<td>{fmt_rank(p.dense_rank)} / {fmt_rank(p.lexical_rank)}</td>"
                "</tr>"
            )
        config_rows = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in pipeline_summary())
        _html(
            '<div class="noor-table-wrap"><table class="noor-table">'
            "<thead><tr><th>Order</th><th>Passage</th><th>Language</th>"
            "<th>Semantic</th><th>BM25</th><th>Fusion</th><th>Reranker</th>"
            "<th>Initial rank (sem. / kw.)</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            f'<dl class="noor-kv">{config_rows}</dl>'
            '<p class="noor-detail-note">Semantic is cosine similarity from the multilingual '
            "embeddings. BM25 is the keyword score, blank when a passage wasn't in the keyword "
            "results (common across languages). Fusion combines both rankings with Reciprocal "
            "Rank Fusion, and the reranker score sets the final order.</p>"
        )


def _error(message: str) -> None:
    _html(
        '<div class="noor-card is-error">'
        "<h3>Noor couldn't answer that</h3>"
        f"<p>{esc(message)}</p>"
        "</div>"
    )


def scroll_to_turn(index: int) -> None:
    """Bring a freshly answered turn into view (the page doesn't auto-scroll)."""
    # Wrapped in a function scope: these scripts share the page's global scope,
    # so a top-level `const` would collide on the second answer.
    st.html(
        "<script>(() => {"
        f"const go = () => document.getElementById('noor-turn-{index}')"
        "?.scrollIntoView({behavior: 'smooth', block: 'start'});"
        # wait for Streamlit to finish laying out the rest of the turn
        "setTimeout(go, 250);"
        "})();</script>",
        unsafe_allow_javascript=True,
    )
