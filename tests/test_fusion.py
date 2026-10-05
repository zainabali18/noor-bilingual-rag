"""Tests for Reciprocal Rank Fusion and the IR metrics (no model downloads)."""
from eval.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from noor.retriever import reciprocal_rank_fusion
from noor.vectorstore import Hit


def _hit(cid, lang="en"):
    return Hit(chunk_id=cid, text=cid, topic=cid, lang=lang, source=cid, score=1.0, doc_id=cid)


def test_rrf_rewards_agreement():
    dense = [_hit("a"), _hit("b"), _hit("c")]
    lexical = [_hit("b"), _hit("a"), _hit("d")]
    fused = reciprocal_rank_fusion(dense, lexical, rrf_k=60)
    ids = [p.chunk_id for p in fused]
    # a and b appear in both lists, so they should rank above c and d
    assert set(ids[:2]) == {"a", "b"}


def test_rrf_records_both_ranks():
    dense = [_hit("a"), _hit("b")]
    lexical = [_hit("b")]
    fused = reciprocal_rank_fusion(dense, lexical, rrf_k=60)
    b = next(p for p in fused if p.chunk_id == "b")
    assert b.dense_rank == 2
    assert b.lexical_rank == 1


def test_metrics_basic():
    rel = [0, 1, 0, 1]
    assert hit_at_k(rel, 1) == 0.0
    assert hit_at_k(rel, 2) == 1.0
    assert reciprocal_rank(rel) == 0.5
    assert recall_at_k(rel, total_relevant=2, k=4) == 1.0


def test_ndcg_perfect_is_one():
    rel = [1, 1, 0, 0]
    assert ndcg_at_k(rel, total_relevant=2, k=4) == 1.0


def test_ndcg_rewards_higher_rank():
    good = ndcg_at_k([1, 0, 0], total_relevant=1, k=3)
    bad = ndcg_at_k([0, 0, 1], total_relevant=1, k=3)
    assert good > bad


def test_rrf_keeps_per_stage_scores():
    dense = [_hit("a"), _hit("b")]
    lexical = [Hit(chunk_id="b", text="b", topic="b", lang="en", source="b", score=7.5, doc_id="b")]
    fused = {p.chunk_id: p for p in reciprocal_rank_fusion(dense, lexical, rrf_k=60)}
    assert fused["b"].dense_score == 1.0
    assert fused["b"].lexical_score == 7.5
    assert fused["a"].lexical_score is None          # not found by keyword search
    assert fused["b"].rrf_score == fused["b"].score  # fusion score recorded
