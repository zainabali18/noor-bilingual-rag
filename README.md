# 🔦 Noor — Bilingual Arabic/English RAG

A retrieval-augmented generation system that answers questions in **Arabic or
English** and retrieves **across languages**: ask in English and it can ground
its answer in Arabic sources, and vice versa.

Noor is built to show how a RAG system is engineered for quality rather than as
a demo chatbot. It combines **dense semantic search** with **lexical BM25
search**, fuses them with **Reciprocal Rank Fusion**, reranks the shortlist with
a **multilingual cross-encoder**, and ships with an **evaluation harness** that
measures retrieval quality and compares configurations.

Runs **fully offline with no paid API**. An OpenAI key is optional and only
used for the final answer synthesis step.

---

## Why this is more than a chatbot

| Concern | How Noor handles it |
|---|---|
| Meaning vs exact terms | **Hybrid retrieval**: dense embeddings for meaning + BM25 for exact words, names and numbers |
| Combining two rankings | **Reciprocal Rank Fusion** merges them without needing comparable scores |
| Bi-encoder is approximate | **Cross-encoder reranker** reads query + passage together for a precise final order |
| Arabic text quirks | Normalisation of alef/ya/ta-marbuta + diacritic stripping before lexical matching |
| Cross-language questions | A **shared multilingual embedding space** places EN and AR text on the same topic close together |
| "Is it actually good?" | An **evaluation harness** reporting Hit@1, Recall@3, MRR, nDCG@5 and a cross-lingual breakdown |
| Hallucination | Answers are grounded in retrieved context and **cite their sources**; the model is told to refuse when the context lacks the answer |

---

## Architecture

```
                         INGEST (offline, once)
  data/en/*.md ┐
  data/ar/*.md ┘─▶ load ─▶ sentence-aware chunk ─▶ ┬─▶ multilingual embeddings ─▶ ChromaDB
                                                    └─▶ BM25 corpus (JSON)

                         QUERY (per question)
  question ─┬─▶ dense search (ChromaDB)   ─┐
            └─▶ lexical search (BM25)      ─┴─▶ Reciprocal Rank Fusion
                                                        │
                                          cross-encoder reranker (top shortlist)
                                                        │
                                               top-k grounded passages
                                                        │
                                      LLM (OpenAI / none) ─▶ cited answer
```

The two retrievers are deliberately complementary: dense search understands
*meaning* and works across languages, while BM25 nails *exact* tokens. RRF needs
only each document's rank in each list, so it fuses cosine similarities and BM25
scores without any fragile score normalisation. The cross-encoder then does the
expensive, accurate relevance judgement on just the shortlist.

---

## The interface

Noor is built to feel like a product rather than a notebook demo:

- **Bilingual by default.** Questions, answers and sources each take the
  direction of their own text, so Arabic reads right-to-left and English
  left-to-right, even when the two are mixed in one answer.
- **Cited answers.** Inline citations like [1] jump to the matching source
  card, which shows the passage, its document and its language.
- **Retrieval details on demand.** Each answer has a "View retrieval details"
  panel with the semantic, BM25, fusion and reranker scores and the initial
  ranks, so the pipeline stays inspectable without cluttering the answer.
- **Graceful fallback.** If answer generation is unavailable (no key, network
  error), Noor still shows the strongest retrieved evidence instead of failing.

The frontend lives in `noor_ui/` and only calls the backend's public functions,
so the RAG pipeline stays independent of the UI.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the indexes (downloads the multilingual models on first run)
python -m noor.ingest

# 3a. Launch the app
python -m streamlit run app.py

# 3b. ...or evaluate retrieval quality
python -m eval.evaluate
```

Optional generated answers: copy `.env.example` to `.env`, set
`LLM_PROVIDER=openai` and add your OpenAI key. Without a key Noor still
retrieves, reranks, and returns the grounded source passages.

Ask things like:
- "How long is the UAE Golden Visa valid and who can apply?"
- "هل التأمين الصحي إلزامي في الإمارات؟"  *(Is health insurance mandatory in the UAE?)*
- "What is the difference between a mainland and a free-zone company?"

---

## Evaluation

Retrieval quality is measured against a labelled set (`eval/eval_set.yaml`) of
16 questions, 8 English and 8 Arabic, each tagged with the topic that answers
it. Because every topic exists in both languages, the harness also measures
**cross-lingual** retrieval: how often a relevant passage in the *other*
language is surfaced.

```bash
python -m eval.evaluate
```

reports, for three configurations, the metrics below (results on the shipped
knowledge base):

```
configuration          Hit@1  Recall@3     MRR   nDCG@5
--------------------------------------------------------
dense                  1.000     0.562   1.000    0.711
hybrid                 1.000     0.318   1.000    0.538
hybrid+rerank          1.000     0.594   1.000    0.729

Cross-lingual retrieval (top-5):
configuration       cross-hit rate   other-lang share
-----------------------------------------------------
dense                        1.000             0.463
hybrid                       0.438             0.087
hybrid+rerank                1.000             0.487
```

Metrics are implemented from scratch in `eval/metrics.py`:
- **Hit@1** — was the top result relevant?
- **Recall@3** — fraction of relevant chunks found in the top 3
- **MRR** — mean reciprocal rank of the first relevant result
- **nDCG@5** — rank-weighted relevance, discounted by position

### What the results show

The top result is always correct (Hit@1 = MRR = 1.000), so the interesting
differences are in how well each configuration ranks the *rest* of the relevant
passages, and how it handles cross-language retrieval.

- **Dense retrieval is already strong**, including cross-lingually (every
  question surfaces a relevant passage in the other language). That is the
  multilingual embedding space doing its job.
- **Naive hybrid actually hurts here.** BM25 matches surface tokens, and an
  English query shares no tokens with an Arabic document, so fusing BM25 in
  pulls same-language passages up and pushes cross-language ones down
  (cross-hit rate drops to 0.438). A useful reminder that lexical search is
  inherently monolingual.
- **Reranking recovers and wins overall.** The multilingual cross-encoder
  re-scores the fused shortlist on meaning rather than tokens, restoring full
  cross-lingual coverage and giving the best Recall@3 and nDCG@5. This is why
  the shortlist is reranked rather than trusted as-is.

*(Run `make eval` to reproduce on your machine; the models download on first
run.)*

An optional **LLM-as-judge** step (`eval/answer_eval.py`) scores the *generated*
answers for faithfulness and relevancy when an OpenAI key is set — a
lightweight take on the idea behind RAGAS.

---

## Project structure

```
noor-bilingual-rag/
├── app.py                 # Streamlit entry point: state + layout wiring
├── noor_ui/               # frontend (kept separate from the RAG backend)
│   ├── components.py      # reusable rendering functions
│   ├── formatting.py      # RTL detection, safe answer HTML, citation links
│   ├── service.py         # pipeline caching + graceful generation fallback
│   ├── content.py         # all user-facing copy in one place
│   ├── theme.py           # loads the design system
│   └── assets/noor.css    # design tokens + component styles
├── .streamlit/config.toml # theme for native widgets
├── noor/
│   ├── config.py          # env-driven configuration
│   ├── loader.py          # read bilingual Markdown + front matter
│   ├── chunker.py         # sentence-aware chunking (EN + AR)
│   ├── embeddings.py      # multilingual sentence-transformer
│   ├── vectorstore.py     # ChromaDB dense retrieval
│   ├── lexical.py         # BM25 + Arabic normalisation
│   ├── reranker.py        # multilingual cross-encoder
│   ├── retriever.py       # hybrid retrieval + RRF + rerank
│   ├── generator.py       # grounded, cited generation (+ no-key fallback)
│   ├── pipeline.py        # end-to-end RAG
│   └── ingest.py          # build the indexes
├── eval/
│   ├── eval_set.yaml      # labelled EN + AR questions
│   ├── metrics.py         # Hit@k, Recall@k, MRR, nDCG (from scratch)
│   ├── evaluate.py        # retrieval evaluation + ablation
│   └── answer_eval.py     # optional LLM-as-judge answer scoring
├── data/{en,ar}/*.md      # bilingual sample knowledge base
├── tests/                 # chunking, Arabic norm, RRF, metrics, loader, UI formatting
└── .github/workflows/ci.yml
```

---

## Key design decisions

- **Multilingual bi-encoder** (`paraphrase-multilingual-MiniLM-L12-v2`) gives one
  shared vector space for 50+ languages, which is what enables cross-lingual
  retrieval at all.
- **Hybrid over dense-only** because pure vector search misses exact tokens
  (visa numbers, scheme names); BM25 recovers them.
- **RRF over score-weighting** because cosine similarity and BM25 scores are not
  on the same scale; rank fusion sidesteps that entirely.
- **Rerank only the shortlist** to get cross-encoder accuracy at bi-encoder cost.
- **Graceful no-key mode** so the system is runnable and reviewable by anyone,
  with generation as an optional enhancement.
- **Fast cold starts.** Models load from the local cache first and only touch
  the network when they aren't downloaded yet, the app warms them up while the
  home page is open, and repeated questions are served from an answer cache.

## Possible extensions

- Agentic layer (LangGraph) that decides when to retrieve vs ask a follow-up
- Query translation to boost the weaker retrieval direction
- Metadata filters (e.g. restrict to a language or topic)
- Larger, domain-specific knowledge base with human-labelled relevance judgements

## Notes

The knowledge base is original sample content for demonstration (see
`data/README.md`); it is not official or legal advice. This is a personal
project built to demonstrate bilingual RAG engineering end to end.

## License

MIT — see [LICENSE](LICENSE).
