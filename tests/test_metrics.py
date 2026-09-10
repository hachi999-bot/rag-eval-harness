import pytest

from harness.metrics import (
    aggregate,
    evaluate_query,
    hit_at_k,
    mean_ignoring_none,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# --- normal cases ---------------------------------------------------------


def test_hit_at_k_true_when_relevant_in_top_k():
    assert hit_at_k(["a", "b", "c"], ["c"], k=3) == 1.0


def test_hit_at_k_false_when_relevant_outside_top_k():
    assert hit_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)


def test_recall_at_k_partial():
    # 1 of 2 relevant docs found in top 3
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == pytest.approx(0.5)


def test_precision_at_k_partial():
    assert precision_at_k(["a", "x", "y"], ["a", "b"], k=3) == pytest.approx(1 / 3)


# --- boundary cases --------------------------------------------------------


def test_hit_at_k_exact_boundary_position():
    # relevant doc is exactly the k-th item
    assert hit_at_k(["x", "y", "a"], ["a"], k=3) == 1.0
    assert hit_at_k(["x", "y", "a"], ["a"], k=2) == 0.0


def test_recall_at_k_all_found():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0


def test_precision_at_k_k_larger_than_ranked_list():
    # denominator stays k even if fewer than k documents were retrieved
    assert precision_at_k(["a"], ["a"], k=5) == pytest.approx(1 / 5)


def test_recall_and_hit_with_empty_relevant_ids():
    # nothing to find -> hit is always 0, recall is undefined (None)
    assert hit_at_k(["a", "b"], [], k=3) == 0.0
    assert recall_at_k(["a", "b"], [], k=3) is None


def test_precision_at_k_with_empty_relevant_ids_is_zero():
    assert precision_at_k(["a", "b"], [], k=2) == 0.0


def test_precision_at_k_with_empty_ranked_ids_is_zero():
    assert precision_at_k([], ["a"], k=3) == 0.0


def test_reciprocal_rank_no_hit_returns_zero():
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_reciprocal_rank_empty_relevant_ids_returns_zero():
    assert reciprocal_rank(["a", "b"], []) == 0.0


# --- abnormal cases ---------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, -5])
def test_hit_at_k_rejects_non_positive_k(k):
    with pytest.raises(ValueError):
        hit_at_k(["a"], ["a"], k=k)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_non_positive_k(k):
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], k=k)


@pytest.mark.parametrize("k", [0, -1])
def test_precision_at_k_rejects_non_positive_k(k):
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], k=k)


def test_evaluate_query_requires_k_values():
    with pytest.raises(ValueError):
        evaluate_query("q1", ["a"], ["a"], k_values=[])


def test_aggregate_requires_non_empty_input():
    with pytest.raises(ValueError):
        aggregate([], k_values=[1])


# --- mean_ignoring_none -----------------------------------------------------


def test_mean_ignoring_none_mixed():
    assert mean_ignoring_none([1.0, None, 3.0]) == 2.0


def test_mean_ignoring_none_all_none_returns_none():
    assert mean_ignoring_none([None, None]) is None


# --- evaluate_query / aggregate integration --------------------------------


def test_evaluate_query_and_aggregate_end_to_end():
    metrics_a = evaluate_query("q1", ["a", "b", "c"], ["a"], k_values=[1, 3])
    metrics_b = evaluate_query("q2", ["x", "y", "z"], ["y"], k_values=[1, 3])

    assert metrics_a.hit_at_k[1] == 1.0
    assert metrics_b.hit_at_k[1] == 0.0
    assert metrics_b.hit_at_k[3] == 1.0

    summary = aggregate([metrics_a, metrics_b], k_values=[1, 3])
    assert summary["n_queries"] == 2
    assert summary["hit_at_1"] == pytest.approx(0.5)
    assert summary["hit_at_3"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx((1.0 + 1 / 2) / 2)


def test_aggregate_excludes_none_recall_from_mean():
    # q1 has no relevant docs at all -> its recall is None and must not drag
    # the aggregate mean down as if it scored 0.
    metrics_none = evaluate_query("q1", ["a", "b"], [], k_values=[2])
    metrics_full = evaluate_query("q2", ["a", "b"], ["a", "b"], k_values=[2])
    summary = aggregate([metrics_none, metrics_full], k_values=[2])
    assert summary["recall_at_2"] == pytest.approx(1.0)
