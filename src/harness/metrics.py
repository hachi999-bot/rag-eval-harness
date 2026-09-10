"""Retrieval quality metrics: Hit@K, MRR, Recall@K, Precision@K.

All functions operate on plain sequences of document IDs so they have no
dependency on how retrieval was performed (mock or real).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """1.0 if any relevant document appears in the top-k, else 0.0.

    An empty relevant_ids set has nothing to hit, so this always returns 0.0
    in that case (there is no "correct" retrieval to reward).
    """
    _validate_k(k)
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """1 / (rank of first relevant document), 0.0 if none is retrieved."""
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float | None:
    """Fraction of relevant documents found in the top-k.

    Returns None when relevant_ids is empty: recall is undefined (there is
    nothing to recall), and callers should exclude these queries from the
    aggregate mean rather than silently treating them as a perfect score.
    """
    _validate_k(k)
    if not relevant_ids:
        return None
    top_k = set(ranked_ids[:k])
    found = top_k & set(relevant_ids)
    return len(found) / len(set(relevant_ids))


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of the top-k retrieved documents that are relevant.

    Defined even when relevant_ids is empty (any retrieval is then a false
    positive, so precision is 0.0), and even when fewer than k documents
    were retrieved (the denominator is always k, per the standard definition).
    """
    _validate_k(k)
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    found = len(set(top_k) & set(relevant_ids))
    return found / k


def mean_ignoring_none(values: Sequence[float | None]) -> float | None:
    """Arithmetic mean of the non-None values, or None if all are None."""
    usable = [v for v in values if v is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


@dataclass
class QueryMetrics:
    query_id: str
    k_values: list[int]
    hit_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float | None] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "reciprocal_rank": self.reciprocal_rank,
            "hit_at_k": dict(self.hit_at_k),
            "recall_at_k": dict(self.recall_at_k),
            "precision_at_k": dict(self.precision_at_k),
        }


def evaluate_query(
    query_id: str,
    ranked_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k_values: Sequence[int],
) -> QueryMetrics:
    """Compute every metric for a single query across all requested k values."""
    if not k_values:
        raise ValueError("k_values must be non-empty")
    metrics = QueryMetrics(query_id=query_id, k_values=list(k_values))
    metrics.reciprocal_rank = reciprocal_rank(ranked_ids, relevant_ids)
    for k in k_values:
        metrics.hit_at_k[k] = hit_at_k(ranked_ids, relevant_ids, k)
        metrics.recall_at_k[k] = recall_at_k(ranked_ids, relevant_ids, k)
        metrics.precision_at_k[k] = precision_at_k(ranked_ids, relevant_ids, k)
    return metrics


def aggregate(query_metrics: Sequence[QueryMetrics], k_values: Sequence[int]) -> dict:
    """Aggregate per-query metrics into corpus-level means (MRR + Hit/Recall/Precision@K)."""
    if not query_metrics:
        raise ValueError("query_metrics must be non-empty")
    result: dict = {
        "n_queries": len(query_metrics),
        "mrr": mean_ignoring_none([qm.reciprocal_rank for qm in query_metrics]),
    }
    for k in k_values:
        result[f"hit_at_{k}"] = mean_ignoring_none([qm.hit_at_k[k] for qm in query_metrics])
        result[f"recall_at_{k}"] = mean_ignoring_none([qm.recall_at_k[k] for qm in query_metrics])
        result[f"precision_at_{k}"] = mean_ignoring_none(
            [qm.precision_at_k[k] for qm in query_metrics]
        )
    return result
