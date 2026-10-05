"""Standard information-retrieval metrics, implemented from scratch.

All functions take `relevance`, a list of 0/1 flags in rank order (index 0 is
the top-ranked result), where 1 means the item at that rank is relevant.
"""
from __future__ import annotations

import math
from typing import List


def hit_at_k(relevance: List[int], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def recall_at_k(relevance: List[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


def reciprocal_rank(relevance: List[int]) -> float:
    for i, r in enumerate(relevance, start=1):
        if r:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevance: List[int], k: int) -> float:
    return sum(r / math.log2(i + 1) for i, r in enumerate(relevance[:k], start=1))


def ndcg_at_k(relevance: List[int], total_relevant: int, k: int) -> float:
    ideal = [1] * min(total_relevant, k)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevance, k) / idcg
