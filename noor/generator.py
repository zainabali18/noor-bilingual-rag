"""Grounded answer generation with inline citations.

Two modes are supported:
  * openai     - calls the OpenAI Chat Completions API
  * none       - no API key needed; returns an extractive answer assembled from
                 the top retrieved passages, still with citations

Whatever the provider, the model is instructed to answer *only* from the
retrieved context and to cite sources as [1], [2], ... so answers stay grounded
and auditable. The context can mix English and Arabic passages; the model is
asked to answer in the language of the question.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import config
from .retriever import RetrievedPassage

SYSTEM_PROMPT = (
    "You are Noor, a careful bilingual (Arabic/English) assistant. "
    "Answer the user's question using ONLY the numbered context passages. "
    "The passages may be in Arabic or English; use them regardless of language. "
    "Reply in the SAME language as the question. "
    "Cite every claim with the passage number in square brackets, e.g. [1]. "
    "If the context does not contain the answer, say so plainly instead of guessing."
)


@dataclass
class Answer:
    text: str
    passages: List[RetrievedPassage]
    provider: str


def _format_context(passages: List[RetrievedPassage]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(f"[{i}] (topic: {p.topic}, lang: {p.lang})\n{p.text}")
    return "\n\n".join(blocks)


def _extractive_answer(question: str, passages: List[RetrievedPassage]) -> str:
    """No-LLM fallback: return the most relevant passage(s) with citations."""
    if not passages:
        return "I could not find anything relevant in the knowledge base."
    lead = passages[0]
    out = [f"Top match for your question (source [1], topic: {lead.topic}):", "", lead.text, "", "[1] " + lead.source]
    if len(passages) > 1:
        out.append("[2] " + passages[1].source)
    return "\n".join(out)


def _openai_answer(question: str, context: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=config.openai_model,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def generate(question: str, passages: List[RetrievedPassage]) -> Answer:
    provider = config.llm_provider.lower()
    if provider == "none" or not passages:
        return Answer(_extractive_answer(question, passages), passages, "none")

    context = _format_context(passages)
    if provider == "openai":
        text = _openai_answer(question, context)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    return Answer(text, passages, provider)
