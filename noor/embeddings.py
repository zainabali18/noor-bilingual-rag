"""Dense embeddings with a multilingual sentence-transformer.

The default model, paraphrase-multilingual-MiniLM-L12-v2, maps text from 50+
languages (English and Arabic included) into one shared vector space. That
shared space is what makes cross-lingual retrieval possible: an English query
and an Arabic passage about the same topic land close together.

The model is loaded lazily, once, and shared safely between threads (see
model_loading.py), so importing this module is cheap.
"""
from __future__ import annotations

from typing import List

from .config import config
from .model_loading import MODEL_LOCK, get_model


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.embed_model

    @property
    def model(self):
        # Imported here so the dependency is only needed when embeddings are used.
        from sentence_transformers import SentenceTransformer

        return get_model("embedder", SentenceTransformer, self.model_name, config.model_device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        with MODEL_LOCK:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,   # cosine similarity via dot product
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vectors.tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
