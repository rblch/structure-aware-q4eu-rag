"""Shared statistical helpers."""

from __future__ import annotations

import math
import random

from scipy import stats


def percentile(values: list[float | int], percentile_value: float) -> float:
    """Linear-interpolation percentile (same convention as numpy 'linear')."""
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def scipy_wilcoxon_method(
    non_zero_differences: list[float],
) -> tuple[str | stats.PermutationMethod, str]:
    n = len(non_zero_differences)
    tied_abs_values = len({abs(value) for value in non_zero_differences}) < n
    if tied_abs_values and n <= 20:
        return (
            stats.PermutationMethod(n_resamples=math.inf, random_state=42),
            "scipy.stats.wilcoxon_permutation_exact_for_ties",
        )
    if tied_abs_values:
        return "approx", "scipy.stats.wilcoxon_approx_tie_corrected"
    if n <= 50:
        return "exact", "scipy.stats.wilcoxon_exact"
    return "approx", "scipy.stats.wilcoxon_approx"


def wilcoxon_signed_rank(differences: list[float]) -> dict[str, object]:
    non_zero = [
        round(difference, 12)
        for difference in differences
        if round(difference, 12) != 0
    ]
    n = len(non_zero)
    if n == 0:
        return {
            "test": "scipy_wilcoxon_signed_rank",
            "method": "all_zero_differences",
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
            "effective_n": 0,
        }
    abs_values = [abs(difference) for difference in non_zero]
    ranks = average_ranks(abs_values)
    w_positive = sum(
        rank for rank, difference in zip(ranks, non_zero) if difference > 0
    )
    w_negative = sum(
        rank for rank, difference in zip(ranks, non_zero) if difference < 0
    )
    scipy_method, method_label = scipy_wilcoxon_method(non_zero)
    result = stats.wilcoxon(
        non_zero,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method=scipy_method,
    )
    total_rank = n * (n + 1) / 2
    return {
        "test": "scipy_wilcoxon_signed_rank",
        "method": method_label,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "rank_biserial": (w_positive - w_negative) / total_rank if total_rank else 0.0,
        "effective_n": n,
    }


def holm_adjusted_p_values(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (index, p_value) in enumerate(indexed):
        value = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


# A shared fixed seed keeps independent contrast CIs reproducible.
def bootstrap_ci(
    values: list[float],
    *,
    iterations: int = 10000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    generator = random.Random(random_seed)
    estimates = []
    for _ in range(iterations):
        sample = [values[generator.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    alpha = 1 - confidence
    lower_index = int((alpha / 2) * (iterations - 1))
    upper_index = int((1 - alpha / 2) * (iterations - 1))
    return estimates[lower_index], estimates[upper_index]
