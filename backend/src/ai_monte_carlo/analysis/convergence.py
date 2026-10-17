"""Monte Carlo Convergence Analysis Module.

Provides tools for evaluating estimator stability, Monte Carlo standard error,
cumulative statistics across adaptive sample checkpoints, and confidence intervals
(Student-t / Normal for sample mean, Wilson score for binomial budget exceedance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ai_monte_carlo.simulation.metrics import (
    _validate_numeric_array,
    calculate_mean,
    calculate_percentiles,
    calculate_probability_exceeding,
    calculate_sample_std,
)
from ai_monte_carlo.simulation.monte_carlo import MonteCarloResult


@dataclass(frozen=True)
class ConvergenceCheckpoint:
    """Statistical estimators and numerical uncertainty bounds at a specific sample size.

    Attributes:
        checkpoint_size: Subsample size n evaluated (2 <= n <= M).
        mean: Cumulative sample mean at n observations.
        sample_std: Sample standard deviation (s_n, ddof=1).
        se_mean: Estimated Monte Carlo standard error of the mean (s_n / sqrt(n)).
        ci_mean_lower: Lower bound of confidence interval for the mean.
        ci_mean_upper: Upper bound of confidence interval for the mean.
        p95: 95th percentile estimated from the first n observations.
        prob_exceed_budget: Empirical probability of exceeding budget limit, or None if no threshold.
        se_prob_exceed: Standard error of the binomial proportion sqrt(p*(1-p)/n), or None.
        ci_prob_lower: Lower bound of Wilson score confidence interval for exceedance prob.
        ci_prob_upper: Upper bound of Wilson score confidence interval for exceedance prob.
    """

    checkpoint_size: int
    mean: float
    sample_std: float
    se_mean: float
    ci_mean_lower: float
    ci_mean_upper: float
    p95: float
    prob_exceed_budget: float | None
    se_prob_exceed: float | None
    ci_prob_lower: float | None
    ci_prob_upper: float | None


@dataclass(frozen=True)
class ConvergenceAnalysisResult:
    """Full convergence trajectory across multiple sample size checkpoints.

    Attributes:
        checkpoints: List of ConvergenceCheckpoint objects ordered by sample size.
        total_samples: Total number of observations M available.
        budget_limit: Budget threshold used for exceedance analysis, or None.
        confidence_level: Nominal confidence level (e.g. 0.95 for 95% CI).
        method_ci: Method used for mean CI ('t' for Student-t, 'normal' for Gaussian).
    """

    checkpoints: list[ConvergenceCheckpoint]
    total_samples: int
    budget_limit: float | None
    confidence_level: float
    method_ci: str

    def to_dataframe(self) -> pd.DataFrame:
        """Converts checkpoints trajectory into a pandas DataFrame ready for plotting."""
        records = []
        for cp in self.checkpoints:
            records.append(
                {
                    "checkpoint_size": cp.checkpoint_size,
                    "mean": cp.mean,
                    "sample_std": cp.sample_std,
                    "se_mean": cp.se_mean,
                    "ci_mean_lower": cp.ci_mean_lower,
                    "ci_mean_upper": cp.ci_mean_upper,
                    "ci_mean_width": cp.ci_mean_upper - cp.ci_mean_lower,
                    "p95": cp.p95,
                    "prob_exceed_budget": cp.prob_exceed_budget,
                    "se_prob_exceed": cp.se_prob_exceed,
                    "ci_prob_lower": cp.ci_prob_lower,
                    "ci_prob_upper": cp.ci_prob_upper,
                }
            )
        return pd.DataFrame(records)

    def summary_dict(self) -> dict[str, Any]:
        """Provides a structured dictionary summary of convergence results."""
        return {
            "total_samples": self.total_samples,
            "budget_limit": self.budget_limit,
            "confidence_level": self.confidence_level,
            "method_ci": self.method_ci,
            "n_checkpoints": len(self.checkpoints),
            "final_estimate": {
                "checkpoint_size": self.checkpoints[-1].checkpoint_size,
                "mean": self.checkpoints[-1].mean,
                "se_mean": self.checkpoints[-1].se_mean,
                "ci_mean_95": [self.checkpoints[-1].ci_mean_lower, self.checkpoints[-1].ci_mean_upper],
                "p95": self.checkpoints[-1].p95,
                "prob_exceed_budget": self.checkpoints[-1].prob_exceed_budget,
                "se_prob_exceed": self.checkpoints[-1].se_prob_exceed,
            },
        }

    def relative_se_final(self) -> float:
        """Calculates relative standard error of the mean for the final checkpoint (SE / |Mean|)."""
        final_cp = self.checkpoints[-1]
        if final_cp.mean == 0.0:
            return 0.0
        return abs(final_cp.se_mean / final_cp.mean)


def calculate_wilson_score_interval(
    count_success: int,
    n_trials: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculates the Wilson score confidence interval for a binomial proportion.

    Formula:
        center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
        margin = (z / (1 + z^2 / n)) * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4n^2))
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

    Justification:
        Unlike the Wald interval (p_hat +/- z * sqrt(p*(1-p)/n)), the Wilson score
        interval does not degenerate to zero width when p_hat is 0 or 1, and strictly
        respects the [0, 1] probability support.

    Args:
        count_success: Number of successes (non-negative integer <= n_trials).
        n_trials: Total number of trials (n >= 1).
        confidence_level: Confidence level in (0, 1).

    Returns:
        tuple (lower_bound, upper_bound) in [0.0, 1.0].
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if not (0 <= count_success <= n_trials):
        raise ValueError(
            f"count_success must be between 0 and {n_trials}, got {count_success}"
        )
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))

    p_hat = count_success / n_trials
    denominator = 1.0 + (z**2) / n_trials
    center = (p_hat + (z**2) / (2.0 * n_trials)) / denominator
    spread = (z / denominator) * math.sqrt(
        (p_hat * (1.0 - p_hat) / n_trials) + ((z**2) / (4.0 * (n_trials**2)))
    )

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return round(lower, 6), round(upper, 6)


def generate_adaptive_checkpoints(
    total_size: int,
    preferred_checkpoints: Sequence[int] | None = None,
) -> list[int]:
    """Generates an ordered, deduplicated list of valid checkpoint sample sizes.

    Args:
        total_size: Total available sample size M (must be >= 2).
        preferred_checkpoints: Optional user-provided checkpoint sequence.
            Default nominal progression: [100, 500, 1_000, 5_000, 10_000].

    Returns:
        Sorted list of unique integers n where 2 <= n <= total_size, always ending with total_size.
    """
    if total_size < 2:
        raise ValueError(f"total_size must be at least 2, got {total_size}")

    if preferred_checkpoints is not None:
        cps = [cp for cp in preferred_checkpoints if 2 <= cp <= total_size]
    else:
        nominal = [100, 500, 1_000, 5_000, 10_000]
        cps = [cp for cp in nominal if cp < total_size]

        # For small sample sizes (M < 100), generate reasonable intermediate steps
        if not cps:
            steps = [int(round(total_size * frac)) for frac in (0.2, 0.4, 0.6, 0.8)]
            cps = [s for s in steps if 2 <= s < total_size]

    # Always include total_size as final checkpoint
    cps.append(total_size)

    # Sort and remove duplicates
    unique_cps = sorted(set(cps))
    return unique_cps


def analyze_convergence(
    data: MonteCarloResult | np.ndarray | Sequence[float],
    budget_limit: float | None = None,
    checkpoints: Sequence[int] | None = None,
    confidence_level: float = 0.95,
    method_ci: str = "t",
) -> ConvergenceAnalysisResult:
    """Analyzes Monte Carlo estimator convergence across increasing sample checkpoints.

    Computes:
    - Cumulative sample mean: bar{x}_n = (1/n) sum x_i
    - Sample standard deviation: s_n = sqrt(sum (x_i - bar{x}_n)^2 / (n-1))
    - Estimated Monte Carlo standard error: SE = s_n / sqrt(n)
    - 95% Confidence interval for the mean:
        * Student-t: bar{x}_n +/- t_{1-alpha/2, n-1} * SE
        * Normal: bar{x}_n +/- z_{1-alpha/2} * SE
    - Cumulative P95 percentile
    - Cumulative empirical exceedance probability: p_hat_n = (1/n) sum I(x_i > budget)
    - Binomial standard error: SE_p = sqrt(p_hat * (1 - p_hat) / n)
    - Wilson score confidence interval for the exceedance probability

    Args:
        data: Either a MonteCarloResult object or an array of sample values.
        budget_limit: Budget limit in USD. If None and data is MonteCarloResult,
            inherits data.config.budget_limit.
        checkpoints: Custom checkpoints sequence. If None, uses adaptive checkpoints.
        confidence_level: Nominal confidence level (default: 0.95).
        method_ci: Confidence interval distribution for the mean: 't' (default) or 'normal'.

    Returns:
        ConvergenceAnalysisResult containing all computed checkpoints.
    """
    if isinstance(data, MonteCarloResult):
        samples = data.cost_samples
        threshold = budget_limit if budget_limit is not None else data.config.budget_limit
    else:
        samples = data
        threshold = budget_limit

    arr = _validate_numeric_array(samples, min_elements=2)
    total_size = len(arr)

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if method_ci not in {"t", "normal"}:
        raise ValueError(f"method_ci must be 't' or 'normal', got '{method_ci}'")

    valid_checkpoints = generate_adaptive_checkpoints(total_size, preferred_checkpoints=checkpoints)
    alpha = 1.0 - confidence_level

    checkpoint_objects: list[ConvergenceCheckpoint] = []

    for n in valid_checkpoints:
        subsample = arr[:n]

        # 1. Mean and Sample Std
        m = calculate_mean(subsample)
        s = calculate_sample_std(subsample, ddof=1)

        # 2. Monte Carlo Standard Error
        se_mean = s / math.sqrt(n)

        # 3. Confidence Interval for the Mean
        if method_ci == "t":
            # Student-t with df = n - 1
            crit_val = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        else:
            # Standard Normal
            crit_val = float(stats.norm.ppf(1.0 - alpha / 2.0))

        ci_mean_low = m - crit_val * se_mean
        ci_mean_upp = m + crit_val * se_mean

        # 4. Cumulative P95 Percentile
        p95_val = float(np.percentile(subsample, 95.0, method="linear"))

        # 5. Budget Exceedance Probability and Wilson Score Interval
        if threshold is not None:
            count_exceed = int(np.sum(subsample > threshold))
            prob_exceed = count_exceed / n
            se_prob = math.sqrt(prob_exceed * (1.0 - prob_exceed) / n)
            ci_p_low, ci_p_upp = calculate_wilson_score_interval(
                count_success=count_exceed,
                n_trials=n,
                confidence_level=confidence_level,
            )
        else:
            prob_exceed = None
            se_prob = None
            ci_p_low = None
            ci_p_upp = None

        checkpoint_objects.append(
            ConvergenceCheckpoint(
                checkpoint_size=n,
                mean=round(m, 6),
                sample_std=round(s, 6),
                se_mean=round(se_mean, 6),
                ci_mean_lower=round(ci_mean_low, 6),
                ci_mean_upper=round(ci_mean_upp, 6),
                p95=round(p95_val, 6),
                prob_exceed_budget=round(prob_exceed, 6) if prob_exceed is not None else None,
                se_prob_exceed=round(se_prob, 6) if se_prob is not None else None,
                ci_prob_lower=ci_p_low,
                ci_prob_upper=ci_p_upp,
            )
        )

    return ConvergenceAnalysisResult(
        checkpoints=checkpoint_objects,
        total_samples=total_size,
        budget_limit=threshold,
        confidence_level=confidence_level,
        method_ci=method_ci,
    )

