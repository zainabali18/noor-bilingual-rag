"""Noor - bilingual Arabic/English RAG assistant for life in the UAE.

Run:     python -m streamlit run app.py
Ingest:  python -m noor.ingest   (once, before the first run)

This file only wires state and layout together. Rendering lives in
`noor_ui.components`, styling in `noor_ui/assets/noor.css`, and all RAG logic
stays in the `noor` backend package.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from noor.config import config
from noor_ui import components as ui
from noor_ui import content, service, theme
from noor_ui.service import Turn

st.set_page_config(
    page_title=content.PAGE_TITLE,
    page_icon=str(Path(__file__).parent / "noor_ui" / "assets" / "noor-mark.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
#  State + callbacks
# --------------------------------------------------------------------------- #
def init_state() -> None:
    state = st.session_state
    state.setdefault("turns", [])
    state.setdefault("top_k", config.top_k)
    state.setdefault("show_details", False)


def queue_question(text: str) -> None:
    text = (text or "").strip()
    if text:
        st.session_state["pending"] = text


def queue_from_input(key: str) -> None:
    queue_question(st.session_state.get(key))


def new_chat() -> None:
    st.session_state["turns"] = []
    st.session_state.pop("pending", None)


# --------------------------------------------------------------------------- #
#  Views
# --------------------------------------------------------------------------- #
def home_view() -> None:
    theme.inject_home_css()
    ui.hero()
    ui.home_ask(on_submit=queue_from_input)
    if service.index_size() == 0:
        ui.empty_index_notice()
    else:
        ui.suggestions(on_pick=queue_question)
    ui.flow_and_coverage()

    # Runs after the page is drawn, so models load while the visitor reads.
    if service.index_size() > 0:
        try:
            service.warm_up()
        except Exception:
            pass  # any real problem resurfaces, with a clear message, on the first question


def answer(question: str) -> Turn:
    if service.index_size() == 0:
        return Turn(
            question=question,
            error="The knowledge base hasn't been built yet. Run python -m noor.ingest, then try again.",
        )
    try:
        return service.ask(question, top_k=st.session_state["top_k"])
    except Exception as exc:  # surface the problem instead of a stack trace
        return Turn(question=question, error=f"Something went wrong while searching: {exc}")


def conversation_view(pending: str | None) -> None:
    turns = st.session_state["turns"]
    show_details = st.session_state["show_details"]

    for index, turn in enumerate(turns):
        with st.container(key=f"noor_turn_{index}"):
            ui.question(turn.question, index)
            ui.response(turn, index, show_details)

    if pending:
        index = len(turns)
        with st.container(key=f"noor_turn_{index}"):
            ui.question(pending, index)
            status = st.empty()
            status.html(ui.thinking_html())
            turn = answer(pending)
            status.empty()
            turns.append(turn)
            ui.response(turn, index, show_details)
        ui.scroll_to_turn(index)

    st.chat_input(
        content.THREAD_PLACEHOLDER,
        key="thread_input",
        on_submit=queue_from_input,
        args=("thread_input",),
    )


def main() -> None:
    init_state()
    theme.inject_base_css()
    ui.nav(on_new_chat=new_chat)

    pending = st.session_state.pop("pending", None)
    if not st.session_state["turns"] and not pending:
        home_view()
    else:
        conversation_view(pending)


main()
