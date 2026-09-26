"""Evaluation helpers for instrument retrieval."""

from collections import defaultdict


def precision_at_k(results: list[dict], expected_instrument: str, k: int = 5) -> float:
    """Calculate precision among the first k search results."""
    selected = results[:k]
    if not selected:
        return 0.0
    return sum(item["instrument"] == expected_instrument for item in selected) / len(selected)


def mean_reciprocal_rank(ranked_results: list[list[dict]], expected_labels: list[str]) -> float:
    """Calculate MRR for several queries with known instrument labels."""
    reciprocal_ranks = []
    for results, expected in zip(ranked_results, expected_labels):
        rank = next((index for index, item in enumerate(results, start=1) if item["instrument"] == expected), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
