"""Build the indexes: load -> chunk -> embed into Chroma + persist BM25 corpus.

Run once (and again whenever the knowledge base changes):

    python -m noor.ingest
"""
from __future__ import annotations

from .chunker import chunk_documents
from .config import config
from .lexical import LexicalIndex
from .loader import load_documents
from .vectorstore import VectorStore


def main() -> None:
    print(f"Loading documents from {config.data_dir} ...")
    docs = load_documents(config.data_dir)
    langs = {}
    for d in docs:
        langs[d.lang] = langs.get(d.lang, 0) + 1
    print(f"  {len(docs)} documents: " + ", ".join(f"{v} {k}" for k, v in langs.items()))

    print(f"Chunking (size={config.chunk_size}, overlap={config.chunk_overlap}) ...")
    chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)
    print(f"  {len(chunks)} chunks")

    print(f"Embedding with {config.embed_model} and writing to Chroma ...")
    vs = VectorStore()
    vs.reset()
    vs.add(chunks)
    print(f"  vector store now holds {vs.count()} chunks")

    print("Persisting BM25 corpus ...")
    LexicalIndex.save_corpus(chunks)
    print(f"  wrote {LexicalIndex.corpus_path()}")

    print("Done. Try:  streamlit run app.py   or   python -m eval.evaluate")


if __name__ == "__main__":
    main()
