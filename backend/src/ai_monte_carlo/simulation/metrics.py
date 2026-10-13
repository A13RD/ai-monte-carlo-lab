"""Statistical estimation and metrics calculation module.

Provides independent, pure functions for computing:
- Sample Mean (bar{x})
- Sample Standard Deviation (s, with Bessel's correction ddof=1)
- Sample Percentiles (P5, P50, P95)
- Empirical Probability of Exceeding a Budget / Threshold (P(X > Threshold))
- Comprehensive statistical summary dataclass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SimulationMetricsSummary:
    """Summary of statistical estimators for a Monte Carlo sample.

    Attributes:
        mean: Sample mean (bar{x}).
        sample_std: Sample standard deviation (s) with ddof=1.
        p5: 5th percentile (P5).
        p50: 50th percentile (Median, P50).
        p95: 95th percentile (P95).
        prob_exceed_budget: Empirical probability P(X > threshold), or None if threshold not provided.
        sample_size: Number of observations (N).
        min_value: Minimum sample value.
        max_value: Maximum sample value.
    """

    mean: float
    sample_std: float
    p5: float
    p50: float
    p95: float
    prob_exceed_budget: float | None
    sample_size: int
    min_value: float
    max_value: float


def _validate_numeric_array(
    samples: np.ndarray | Sequence[float], min_elements: int = 1
) -> np.ndarray:
    """Validates and converts input into a clean 1D float NumPy array."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.ravel()

    if len(arr) < min_elements:
        raise ValueError(
            f"Insufficient observations: expected at least {min_elements}, got {len(arr)}"
        )

    if np.any(np.isnan(arr)):
        raise ValueError("Samples contain NaN values, which cannot produce valid estimators.")
    if np.any(np.isinf(arr)):
        raise ValueError("Samples contain Infinite values.")

    return arr


def calculate_mean(samples: np.ndarray | Sequence[float]) -> float:
    """Calculates the sample mean.

    Formula:
        bar{x} = (1 / n) * sum_{i=1}^n x_i

    Args:
        samples: Array or sequence of numeric observations.

    Returns:
        Sample mean as float.
    """
    arr = _validate_numeric_array(samples, min_elements=1)
    return float(np.mean(arr))


def calculate_sample_std(
    samples: np.ndarray | Sequence[float], ddof: int = 1
) -> float:
    """Calculates the sample standard deviation with Bessel's correction.

    Formula:
        s = sqrt( (1 / (n - ddof)) * sum_{i=1}^n (x_i - bar{x})^2 )

    Args:
        samples: Array or sequence of numeric observations.
        ddof: Delta Degrees of Freedom (default: 1 for sample standard deviation).

    Returns:
        Sample standard deviation as float.
    """
    if ddof < 0:
        raise ValueError(f"ddof must be non-negative, got {ddof}")

    arr = _validate_numeric_array(samples, min_elements=ddof + 1)
    return float(np.std(arr, ddof=ddof))


def calculate_percentiles(
    samples: np.ndarray | Sequence[float],
    percentiles: Sequence[float] = (5.0, 50.0, 95.0),
) -> dict[str, float]:
    """Calculates arbitrary percentiles from the empirical sample distribution.

    Args:
        samples: Array or sequence of numeric observations.
        percentiles: Sequence of percentile levels in [0.0, 100.0].

    Returns:
        Dictionary mapping percentile names (e.g. 'p5', 'p50', 'p95') to values.
    """
    arr = _validate_numeric_array(samples, min_elements=1)

    for p in percentiles:
        if not (0.0 <= p <= 100.0):
            raise ValueError(f"Percentile must be between 0 and 100, got {p}")

    computed = np.percentile(arr, percentiles, method="linear")

    result: dict[str, float] = {}
    for p, val in zip(percentiles, computed):
        # Format key as p5, p50, p95 or p_xx_x for decimals
        if p.is_integer():
            key = f"p{int(p)}"
        else:
            key = f"p{str(p).replace('.', '_')}"
        result[key] = float(val)

    return result


def calculate_probability_exceeding(
    samples: np.ndarray | Sequence[float], threshold: float
) -> float:
    """Calculates the empirical probability of exceeding a given budget or threshold.

    Formula:
        P(X > Threshold) = (1 / n) * sum_{i=1}^n I(x_i > Threshold)

    Args:
        samples: Array or sequence of numeric observations.
        threshold: Real value threshold (e.g., budget limit in USD).

    Returns:
        Empirical probability in [0.0, 1.0].
    """
    arr = _validate_numeric_array(samples, min_elements=1)

    if np.isnan(threshold) or np.isinf(threshold):
        raise ValueError(f"Threshold must be a finite real number, got {threshold}")

    indicator = arr > threshold
    return float(np.mean(indicator))


def summarize_metrics(
    samples: np.ndarray | Sequence[float],
    threshold: float | None = None,
) -> SimulationMetricsSummary:
    """Computes a complete statistical metrics summary for a sample distribution.

    Requires at least 2 observations to compute sample standard deviation (ddof=1).

    Args:
        samples: Array or sequence of numeric observations (n >= 2).
        threshold: Optional threshold for probability of exceedance.

    Returns:
        SimulationMetricsSummary object.
    """
    arr = _validate_numeric_array(samples, min_elements=2)

    mean_val = calculate_mean(arr)
    std_val = calculate_sample_std(arr, ddof=1)
    perc = calculate_percentiles(arr, percentiles=(5.0, 50.0, 95.0))
    p_exceed = (
        calculate_probability_exceeding(arr, threshold)
        if threshold is not None
        else None
    )

    return SimulationMetricsSummary(
        mean=round(mean_val, 6),
        sample_std=round(std_val, 6),
        p5=round(perc["p5"], 6),
        p50=round(perc["p50"], 6),
        p95=round(perc["p95"], 6),
        prob_exceed_budget=round(p_exceed, 6) if p_exceed is not None else None,
        sample_size=len(arr),
        min_value=round(float(np.min(arr)), 6),
        max_value=round(float(np.max(arr)), 6),
    )
