"""Loads Noor's CSS design system into the Streamlit page."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).parent / "assets" / "noor.css"

# A faint eight-fold lattice, the kind found in Gulf screens and windows.
_LATTICE = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='64' height='64' "
    "viewBox='0 0 64 64'%3E%3Cpath d='M14 14H50V50H14Z M32 6.5L57.5 32L32 57.5L6.5 32Z "
    "M32 0V6.5 M32 57.5V64 M0 32H6.5 M57.5 32H64 M0 0L14 14 M64 0L50 14 M0 64L14 50 "
    "M64 64L50 50' fill='none' stroke='%2313213D' stroke-width='0.7'/%3E%3C/svg%3E"
)

# Home only: dawn light behind the hero, with the lattice fading out from it.
_HOME_CSS = f"""
<style>
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(52% 420px at 50% 120px, rgba(226,173,74,.30), rgba(226,173,74,0) 70%),
    radial-gradient(90% 640px at 50% 0, rgba(255,252,246,.95), rgba(245,239,228,0) 75%),
    var(--sand);
}}
[data-testid="stAppViewContainer"]::before {{
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 780px;
  background-image: url("{_LATTICE}");
  background-size: 64px 64px;
  opacity: .07;
  -webkit-mask-image: radial-gradient(60% 520px at 50% 160px, #000 0%, transparent 75%);
          mask-image: radial-gradient(60% 520px at 50% 160px, #000 0%, transparent 75%);
  pointer-events: none;
  z-index: 0;
}}
</style>
"""


@lru_cache(maxsize=1)
def _base_css() -> str:
    return f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>"


def inject_base_css() -> None:
    st.html(_base_css())


def inject_home_css() -> None:
    st.html(_HOME_CSS)
