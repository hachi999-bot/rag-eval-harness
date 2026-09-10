import pytest

from harness.metrics import evaluate_query
from harness.report import compare_metric, format_markdown
from harness.runner import (
    NaiveOverlapRetriever,
    TfidfRetriever,
    load_corpus,
    load_eval_set,
    run_eval,
)


@pytest.fixture
def before_after_runs(tiny_corpus_dir, tiny_eval_set_path):
    corpus = load_corpus(tiny_corpus_dir)
    eval_set = load_eval_set(tiny_eval_set_path)
    before, _ = run_eval(corpus, eval_set, NaiveOverlapRetriever(corpus), k_values=[1, 3])
    after, _ = run_eval(corpus, eval_set, TfidfRetriever(corpus), k_values=[1, 3])
    return before, after


# --- normal cases ---------------------------------------------------------


def test_compare_metric_mrr_reports_means(before_after_runs):
    before, after = before_after_runs
    result = compare_metric(before, after, "mrr", seed=1)
    assert result["metric"] == "mrr"
    assert result["before_mean"] >= 0.0
    assert result["after_mean"] >= 0.0
    assert result["delta"] == pytest.approx(result["after_mean"] - result["before_mean"])


def test_compare_metric_hit_at_k_requires_k(before_after_runs):
    before, after = before_after_runs
    with pytest.raises(ValueError):
        compare_metric(before, after, "hit_at_k", k=None)


def test_compare_metric_per_k_excludes_none_recall(before_after_runs):
    before, after = before_after_runs
    # t4 has no relevant docs -> recall is None on both sides -> excluded, not
    # treated as a comparable zero.
    result = compare_metric(before, after, "recall_at_k", k=1, seed=1)
    assert result["n"] == 3


def test_format_markdown_contains_metric_rows(before_after_runs):
    before, after = before_after_runs
    comparisons = [compare_metric(before, after, "mrr", seed=1)]
    text = format_markdown(comparisons, "naive", "tfidf")
    assert "naive" in text
    assert "tfidf" in text
    assert "mrr" in text


# --- boundary cases --------------------------------------------------------


def test_compare_metric_identical_runs_has_zero_delta():
    # Two queries, since a single shared query has no variance to size an
    # effect from (see test_compare_metric_single_shared_query_has_no_stats).
    metrics = [
        evaluate_query("q1", ["a", "b"], ["a"], k_values=[1]),
        evaluate_query("q2", ["x", "y"], ["y"], k_values=[1]),
    ]
    result = compare_metric(metrics, metrics, "hit_at_k", k=1, seed=1)
    assert result["delta"] == 0.0
    assert result["effect_size_cohens_d"] == 0.0


def test_compare_metric_single_shared_query_has_no_stats(before_after_runs):
    before, after = before_after_runs
    single_before = [before[0]]
    single_after = [after[0]]
    result = compare_metric(single_before, single_after, "mrr", seed=1)
    assert result["n"] == 1
    assert result["p_value"] is None
    assert result["effect_size_cohens_d"] is None


# --- abnormal cases ---------------------------------------------------------


def test_compare_metric_unknown_metric_raises(before_after_runs):
    before, after = before_after_runs
    with pytest.raises(ValueError):
        compare_metric(before, after, "not_a_metric")


def test_compare_metric_no_overlapping_queries_raises():
    before = [evaluate_query("q1", ["a"], ["a"], k_values=[1])]
    after = [evaluate_query("q2", ["a"], ["a"], k_values=[1])]
    with pytest.raises(ValueError):
        compare_metric(before, after, "mrr")
