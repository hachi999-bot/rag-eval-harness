"""Before/After comparison report for two evaluation runs over the same
fixed eval set (e.g. a "naive" baseline retriever vs. a "tfidf" retriever).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from harness.metrics import QueryMetrics
from harness.runner import (
    DEFAULT_K_VALUES,
    build_retriever,
    load_corpus,
    load_eval_set,
    run_eval,
    validate_eval_set_against_corpus,
)
from harness.stats import bootstrap_ci, cohens_d_paired, paired_permutation_test

SCALAR_METRICS = ("mrr",)
PER_K_METRICS = ("hit_at_k", "recall_at_k", "precision_at_k")


def _extract_scalar(per_query: Sequence[QueryMetrics], metric: str) -> dict[str, float]:
    if metric != "mrr":
        raise ValueError(f"unknown scalar metric: {metric!r}")
    return {qm.query_id: qm.reciprocal_rank for qm in per_query}


def _extract_per_k(per_query: Sequence[QueryMetrics], metric: str, k: int) -> dict[str, float]:
    attr = {"hit_at_k": "hit_at_k", "recall_at_k": "recall_at_k", "precision_at_k": "precision_at_k"}[
        metric
    ]
    values = {}
    for qm in per_query:
        value = getattr(qm, attr)[k]
        if value is not None:  # recall_at_k is None for queries with no relevant docs
            values[qm.query_id] = value
    return values


def compare_metric(
    before: Sequence[QueryMetrics],
    after: Sequence[QueryMetrics],
    metric_name: str,
    k: int | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Compare one metric between two runs over the queries present in both."""
    if metric_name in SCALAR_METRICS:
        before_map = _extract_scalar(before, metric_name)
        after_map = _extract_scalar(after, metric_name)
        label = metric_name
    elif metric_name in PER_K_METRICS:
        if k is None:
            raise ValueError(f"metric {metric_name!r} requires a k value")
        before_map = _extract_per_k(before, metric_name, k)
        after_map = _extract_per_k(after, metric_name, k)
        label = f"{metric_name.replace('_k', '')}{k}"
    else:
        raise ValueError(f"unknown metric: {metric_name!r}")

    shared_ids = sorted(set(before_map) & set(after_map))
    if not shared_ids:
        raise ValueError(f"no comparable queries for metric {metric_name!r} (k={k})")

    before_values = [before_map[qid] for qid in shared_ids]
    after_values = [after_map[qid] for qid in shared_ids]

    before_mean = sum(before_values) / len(before_values)
    after_mean = sum(after_values) / len(after_values)

    result: dict = {
        "metric": label,
        "n": len(shared_ids),
        "before_mean": before_mean,
        "after_mean": after_mean,
        "delta": after_mean - before_mean,
        "before_ci": bootstrap_ci(before_values, confidence, n_resamples, seed),
        "after_ci": bootstrap_ci(after_values, confidence, n_resamples, seed),
    }

    if len(shared_ids) >= 2:
        result["p_value"] = paired_permutation_test(
            before_values, after_values, n_permutations, seed
        )
        result["effect_size_cohens_d"] = cohens_d_paired(before_values, after_values)
    else:
        # A single comparable query has no variance to test or size an effect from.
        result["p_value"] = None
        result["effect_size_cohens_d"] = None

    return result


def format_markdown(comparisons: Sequence[dict], before_label: str, after_label: str) -> str:
    lines = [
        f"# Before/After Comparison: `{before_label}` -> `{after_label}`",
        "",
        "| Metric | n | Before | After | Delta | p-value | Cohen's d |",
        "|---|---|---|---|---|---|---|",
    ]
    for comp in comparisons:
        p_value = "n/a" if comp["p_value"] is None else f"{comp['p_value']:.4f}"
        effect = "n/a" if comp["effect_size_cohens_d"] is None else f"{comp['effect_size_cohens_d']:.3f}"
        lines.append(
            f"| {comp['metric']} | {comp['n']} | {comp['before_mean']:.3f} | "
            f"{comp['after_mean']:.3f} | {comp['delta']:+.3f} | {p_value} | {effect} |"
        )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two retriever runs over the fixed eval set.")
    parser.add_argument("--eval", default="data/eval_set.json")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--before", choices=["naive", "tfidf", "openai"], default="naive")
    parser.add_argument("--after", choices=["naive", "tfidf", "openai"], default="tfidf")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    corpus = load_corpus(args.corpus)
    eval_set = load_eval_set(args.eval)
    validate_eval_set_against_corpus(eval_set, corpus)

    before_retriever = build_retriever(args.before, corpus)
    after_retriever = build_retriever(args.after, corpus)

    before_per_query, _ = run_eval(corpus, eval_set, before_retriever, args.k)
    after_per_query, _ = run_eval(corpus, eval_set, after_retriever, args.k)

    comparisons = [compare_metric(before_per_query, after_per_query, "mrr", seed=args.seed)]
    for k in args.k:
        for metric in PER_K_METRICS:
            comparisons.append(
                compare_metric(before_per_query, after_per_query, metric, k, seed=args.seed)
            )

    if args.format == "json":
        print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    else:
        print(format_markdown(comparisons, args.before, args.after))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
