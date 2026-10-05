"""Load the bilingual knowledge base from disk.

Each document is a Markdown file with a small YAML-style front matter block:

    ---
    id: golden-visa
    topic: UAE Golden Visa
    lang: en
    ---
    # Title
    body...
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class Document:
    doc_id: str
    topic: str
    lang: str          # "en" or "ar"
    text: str
    source: str        # file path, for citations


def _parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(raw)
    if not match:
        raise ValueError(f"{path} is missing a front-matter block")

    meta_block, body = match.groups()
    meta = {}
    for line in meta_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    # drop the leading "# Title" line from the body, keep the prose
    body_lines = [ln for ln in body.strip().splitlines() if not ln.startswith("# ")]
    text = "\n".join(body_lines).strip()

    return Document(
        doc_id=meta.get("id", path.stem),
        topic=meta.get("topic", path.stem),
        lang=meta.get("lang", "en"),
        text=text,
        source=str(path.relative_to(path.parents[2])),
    )


def _has_front_matter(path: Path) -> bool:
    with path.open(encoding="utf-8") as f:
        return f.read(3) == "---"


def load_documents(data_dir: Path) -> List[Document]:
    """Load every knowledge-base .md file under data_dir (recursively).

    Markdown files without a front-matter block (e.g. a README) are skipped so
    documentation can live alongside the data.
    """
    paths = sorted(p for p in Path(data_dir).rglob("*.md") if _has_front_matter(p))
    docs = [_parse(p) for p in paths]
    if not docs:
        raise FileNotFoundError(f"No knowledge-base documents found under {data_dir}")
    return docs
