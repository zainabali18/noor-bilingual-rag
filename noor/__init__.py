"""Noor: a bilingual (Arabic/English) Retrieval-Augmented Generation system.

Noor combines dense semantic search with lexical (BM25) search, fuses the two
with Reciprocal Rank Fusion, reranks with a multilingual cross-encoder, and
generates grounded, cited answers. It is built to retrieve across languages:
a question in English can be answered from Arabic sources and vice versa.
"""

__version__ = "0.1.0"
