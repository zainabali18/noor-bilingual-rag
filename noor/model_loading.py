"""Load and share the Hugging Face models safely.

Two concerns live here:

1. Local cache first. By default sentence-transformers contacts the Hugging
   Face Hub on every load to check for a newer revision, even when the model is
   on disk. On a slow network that check can hang until it times out (over a
   minute on macOS). Trying `local_files_only=True` first skips the network when
   the model is cached, and only downloads when it genuinely isn't.

2. Thread safety. Web apps such as Streamlit serve requests from several
   threads. Two threads loading the same model at once would load it twice, and
   two threads running a model at once can crash GPU backends (Apple's Metal
   aborts the whole process). Every load and every forward pass therefore goes
   through one shared re-entrant lock. The models are small, so serialising
   them costs almost nothing.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Tuple, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)

# Held while loading a model and while running one.
MODEL_LOCK = threading.RLock()

_models: Dict[Tuple[str, str, str], Any] = {}


def load_local_first(factory: Callable[..., T], name: str, **kwargs: Any) -> T:
    try:
        return factory(name, local_files_only=True, **kwargs)
    except Exception:  # not cached yet (first run): download it once
        log.info("Model %s not in local cache, downloading.", name)
        return factory(name, **kwargs)


def get_model(kind: str, factory: Callable[..., T], name: str, device: str) -> T:
    """Return the shared instance of a model, loading it exactly once."""
    key = (kind, name, device)
    with MODEL_LOCK:
        if key not in _models:
            _models[key] = load_local_first(factory, name, device=device)
        return _models[key]
