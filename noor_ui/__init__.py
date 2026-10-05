"""Noor's Streamlit frontend.

Kept separate from the `noor` backend package so the RAG pipeline stays
UI-agnostic. Modules:

    content     - user-facing copy, suggestions and topics
    formatting  - pure-Python text helpers (direction detection, answer HTML)
    service     - thin layer over the RAG pipeline (caching, fallbacks)
    theme       - loads the CSS design system
    components  - reusable rendering functions
"""
