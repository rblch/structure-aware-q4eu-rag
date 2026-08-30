"""IR metrics over the complete answer-unit ranking derived from top-k chunks."""

from __future__ import annotations

import math


def compute_ir_metrics(
    *,
    ranked_answer_units: list[str],
    gold_unit_ids: list[str],
) -> dict[str, float]:
    gold = set(gold_unit_ids)
    if not gold:
        raise ValueError("gold_unit_ids must not be empty")

    relevant_flags = [unit_id in gold for unit_id in ranked_answer_units]
    relevant_retrieved = sum(relevant_flags)
    retrieved_count = len(ranked_answer_units)
    gold_count = len(gold)
    precision = relevant_retrieved / retrieved_count if retrieved_count else 0.0
    recall = relevant_retrieved / gold_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # R is the query's gold-set size.
    r_precision = sum(relevant_flags[:gold_count]) / gold_count
    mrr = reciprocal_rank(relevant_flags)
    dcg = discounted_cumulative_gain(relevant_flags)
    ideal_count = min(gold_count, retrieved_count)
    ideal_dcg = discounted_cumulative_gain([True] * ideal_count)
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r_precision": r_precision,
        "mrr": mrr,
        "ndcg": ndcg,
        "retrieved_answer_unit_count": retrieved_count,
        "relevant_retrieved_count": relevant_retrieved,
    }


def reciprocal_rank(relevant_flags: list[bool]) -> float:
    for index, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            return 1 / index
    return 0.0


def discounted_cumulative_gain(relevant_flags: list[bool]) -> float:
    return sum(
        (1.0 if is_relevant else 0.0) / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevant_flags, start=1)
    )


def mean_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {
        key: sum(metrics[key] for metrics in metric_dicts) / len(metric_dicts)
        for key in keys
    }
