"""Evaluate retrieval quality and compare three configurations.

    1. dense        - semantic vector search only
    2. hybrid       - dense + BM25 fused with Reciprocal Rank Fusion
    3. hybrid+rerank- hybrid shortlist reordered by a cross-encoder

For each configuration we report, over the labelled eval set:
    Hit@1, Recall@3, MRR, nDCG@5

and a cross-lingual breakdown showing that questions are answered even when the
only same-topic sources are in the *other* language.

Run:  python -m eval.evaluate
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from noor.config import config
from noor.lexical import LexicalIndex
from noor.reranker import Reranker
from noor.retriever import reciprocal_rank_fusion
from noor.vectorstore import VectorStore

from .metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

EVAL_PATH = Path(__file__).parent / "eval_set.yaml"
POOL = 10  # retrieve this many candidates per query for metric computation


def load_eval():
    data = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    return data["questions"]


def relevance_flags(passages, relevant_ids) -> List[int]:
    return [1 if p.doc_id in relevant_ids else 0 for p in passages]


def count_relevant_chunks(vs_meta: Dict[str, int], relevant_ids) -> int:
    return sum(vs_meta.get(rid, 0) for rid in relevant_ids)


def rank_dense(vs, query):
    hits = vs.search(query, POOL)
    from noor.retriever import _to_passage
    return [_to_passage(h) for h in hits]


def rank_hybrid(vs, lex, query):
    dense = vs.search(query, config.dense_k)
    lexical = lex.search(query, config.lexical_k)
    return reciprocal_rank_fusion(dense, lexical, config.rrf_k)[:POOL]


def rank_hybrid_rerank(vs, lex, reranker, query):
    fused = rank_hybrid(vs, lex, query)
    if not fused:
        return []
    from noor.vectorstore import Hit
    hits = [Hit(p.chunk_id, p.text, p.topic, p.lang, p.source, p.score, p.doc_id) for p in fused]
    reranked = reranker.rerank(query, hits, POOL)
    by_id = {p.chunk_id: p for p in fused}
    return [by_id[h.chunk_id] for h in reranked if h.chunk_id in by_id]


def evaluate_config(name, rank_fn, questions, chunks_per_doc):
    agg = {"hit@1": 0.0, "recall@3": 0.0, "mrr": 0.0, "ndcg@5": 0.0}
    cross = {"total": 0, "cross_hit": 0, "other_lang_passages": 0, "all_passages": 0}

    for item in questions:
        passages = rank_fn(item["q"])
        rel = relevance_flags(passages, set(item["relevant"]))
        total_rel = count_relevant_chunks(chunks_per_doc, item["relevant"])

        agg["hit@1"] += hit_at_k(rel, 1)
        agg["recall@3"] += recall_at_k(rel, total_rel, 3)
        agg["mrr"] += reciprocal_rank(rel)
        agg["ndcg@5"] += ndcg_at_k(rel, total_rel, 5)

        # cross-lingual: did we retrieve a relevant passage in the OTHER language?
        cross["total"] += 1
        top5 = passages[:5]
        other = [p for p in top5 if p.lang != item["lang"]]
        cross["all_passages"] += len(top5)
        cross["other_lang_passages"] += len(other)
        if any(p.doc_id in set(item["relevant"]) and p.lang != item["lang"] for p in top5):
            cross["cross_hit"] += 1

    n = len(questions)
    for k in agg:
        agg[k] /= n
    return name, agg, cross


def print_table(results):
    cols = ["Hit@1", "Recall@3", "MRR", "nDCG@5"]
    keys = ["hit@1", "recall@3", "mrr", "ndcg@5"]
    header = f"{'configuration':<18}" + "".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, agg, _ in results:
        row = f"{name:<18}" + "".join(f"{agg[k]:>10.3f}" for k in keys)
        print(row)


def print_cross(results):
    print("\nCross-lingual retrieval (top-5):")
    print(f"{'configuration':<18}{'cross-hit rate':>16}{'other-lang share':>18}")
    print("-" * 52)
    for name, _, cross in results:
        rate = cross["cross_hit"] / cross["total"] if cross["total"] else 0
        share = cross["other_lang_passages"] / cross["all_passages"] if cross["all_passages"] else 0
        print(f"{name:<18}{rate:>16.3f}{share:>18.3f}")
    print("\ncross-hit rate = fraction of questions where a relevant passage in the")
    print("OTHER language appeared in the top 5 (proof of cross-lingual retrieval).")


def main():
    questions = load_eval()
    vs = VectorStore()
    if vs.count() == 0:
        raise SystemExit("Index is empty. Run `python -m noor.ingest` first.")
    lex = LexicalIndex.from_disk()

    # how many chunks exist per doc_id (for recall denominators)
    chunks_per_doc: Dict[str, int] = {}
    for c in lex.chunks:
        chunks_per_doc[c.doc_id] = chunks_per_doc.get(c.doc_id, 0) + 1

    reranker = Reranker()

    results = [
        evaluate_config("dense", lambda q: rank_dense(vs, q), questions, chunks_per_doc),
        evaluate_config("hybrid", lambda q: rank_hybrid(vs, lex, q), questions, chunks_per_doc),
        evaluate_config("hybrid+rerank", lambda q: rank_hybrid_rerank(vs, lex, reranker, q), questions, chunks_per_doc),
    ]

    print(f"\nEvaluated {len(questions)} questions "
          f"({sum(1 for x in questions if x['lang']=='en')} EN, "
          f"{sum(1 for x in questions if x['lang']=='ar')} AR)\n")
    print_table(results)
    print_cross(results)


if __name__ == "__main__":
    main()
