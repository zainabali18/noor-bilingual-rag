# Sample knowledge base

These documents are **original, illustrative sample text** written for this
project so the repository is self-contained. Each topic exists in both English
(`en/`) and Arabic (`ar/`) so the system can be tested on **cross-lingual
retrieval**.

The content is general information for demonstration only. It is **not** legal,
financial, or official advice, and it is not scraped from any source.

## Use your own data

Drop `.md` files into `en/` and `ar/` (or any sub-folders) with this header:

```markdown
---
id: my-topic
topic: My Topic Title
lang: en
---
# My Topic Title

Body text...
```

Then rebuild the indexes:

```bash
rm -rf .chroma
python -m noor.ingest
```
