"""Monte Carlo Simulation Engine for AI LLM Consumption and Costs.

Executes stochastic monthly simulations to estimate:
- Aggregated monthly input and output token consumption
- Total monthly monetary expense in USD
- Sample statistics: Mean, Sample Standard Deviation, P5, P50 (median), P95
- Empirical Probability of Exceeding a Budget Limit P(Cost > B)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai_monte_carlo.config import MonteCarloConfig
from ai_monte_carlo.simulation.generators import (
    sample_requests_count,
    sample_tokens,
)
from ai_monte_carlo.simulation.metrics import (
    SimulationMetricsSummary,
    summarize_metrics,
)

MAX_TOKEN_DRAWS_PER_CHUNK = 250_000


@dataclass(frozen=True)
class MonteCarloResult:
    """Encapsulates the raw trajectory samples and summary statistics from a simulation run.

    Attributes:
        cost_samples: 1D array of simulated monthly costs in USD (length M).
        input_token_samples: 1D array of simulated monthly input tokens (length M).
        output_token_samples: 1D array of simulated monthly output tokens (length M).
        total_token_samples: 1D array of simulated monthly total tokens (length M).
        request_count_samples: 1D array of simulated monthly request counts (length M).
        cost_metrics: Statistical summary of monthly costs (includes P(Cost > Budget)).
        input_token_metrics: Statistical summary of monthly input tokens.
        output_token_metrics: Statistical summary of monthly output tokens.
        total_token_metrics: Statistical summary of monthly total tokens.
        request_count_metrics: Statistical summary of monthly request counts.
        config: Simulation configuration used to generate this result.
    """

    cost_samples: np.ndarray
    input_token_samples: np.ndarray
    output_token_samples: np.ndarray
    total_token_samples: np.ndarray
    request_count_samples: np.ndarray
    cost_metrics: SimulationMetricsSummary
    input_token_metrics: SimulationMetricsSummary
    output_token_metrics: SimulationMetricsSummary
    total_token_metrics: SimulationMetricsSummary
    request_count_metrics: SimulationMetricsSummary
    config: MonteCarloConfig

    def to_dataframe(self) -> pd.DataFrame:
        """Converts raw iteration samples into a pandas DataFrame."""
        return pd.DataFrame(
            {
                "iteration": np.arange(1, len(self.cost_samples) + 1),
                "request_count": self.request_count_samples,
                "input_tokens": self.input_token_samples,
                "output_tokens": self.output_token_samples,
                "total_tokens": self.total_token_samples,
                "monthly_cost_usd": self.cost_samples,
            }
        )

    def summary_dict(self) -> dict[str, Any]:
        """Returns a structured dictionary with high-level simulation statistics."""
        return {
            "iterations": self.config.n_iterations,
            "budget_limit_usd": self.config.budget_limit,
            "probability_exceeding_budget": self.cost_metrics.prob_exceed_budget,
            "cost_usd": {
                "mean": self.cost_metrics.mean,
                "sample_std": self.cost_metrics.sample_std,
                "p5": self.cost_metrics.p5,
                "p50_median": self.cost_metrics.p50,
                "p95": self.cost_metrics.p95,
                "min": self.cost_metrics.min_value,
                "max": self.cost_metrics.max_value,
            },
            "tokens": {
                "input_mean": self.input_token_metrics.mean,
                "output_mean": self.output_token_metrics.mean,
                "total_mean": self.total_token_metrics.mean,
                "total_p95": self.total_token_metrics.p95,
            },
            "requests": {
                "mean": self.request_count_metrics.mean,
                "p50_median": self.request_count_metrics.p50,
                "p95": self.request_count_metrics.p95,
            },
        }


def _aggregate_compound_tokens(
    rng: np.random.Generator,
    request_counts: np.ndarray,
    token_sampler_fn,
    chunk_size: int = 2_000,
) -> np.ndarray:
    """Aggregates compound random sums of tokens across replications using chunked vectorization.

    For each replication m with N_m requests, sums N_m independent token realizations:
        S_m = sum_{i=1}^{N_m} T_{m, i}

    Args:
        rng: NumPy random generator.
        request_counts: 1D array of request counts per replication (length M).
        token_sampler_fn: Callable taking integer `size` and returning sampled token array.
        chunk_size: Replications per chunk to keep memory footprint bounded.

    Returns:
        1D array of summed tokens per replication (length M).
    """
    n_iterations = len(request_counts)
    aggregated_tokens = np.zeros(n_iterations, dtype=np.int64)

    for start_idx in range(0, n_iterations, chunk_size):
        end_idx = min(start_idx + chunk_size, n_iterations)
        chunk_reqs = request_counts[start_idx:end_idx]

        active_indices = np.flatnonzero(chunk_reqs > 0)
        if len(active_indices) == 0:
            continue

        active_counts = chunk_reqs[active_indices]
        total_tokens_to_draw = int(np.sum(active_counts))

        sampled_tokens = token_sampler_fn(total_tokens_to_draw)

        # Compute partition prefix offsets
        offsets = np.zeros(len(active_counts), dtype=np.int64)
        offsets[1:] = np.cumsum(active_counts)[:-1]

        # Vectorized sum per replication using NumPy reduceat
        replication_sums = np.add.reduceat(sampled_tokens, offsets)
        aggregated_tokens[start_idx + active_indices] = replication_sums

    return aggregated_tokens


def run_monte_carlo(config: MonteCarloConfig) -> MonteCarloResult:
    """Executes the full Monte Carlo simulation workflow.

    Workflow:
        1. Initializes reproducible RNG from config.seed.
        2. Simulates monthly request counts N^{(m)} for m = 1...M.
        3. Simulates compound monthly input tokens sum T_in^{(m)}.
        4. Simulates compound monthly output tokens sum T_out^{(m)}.
        5. Computes total tokens T_total^{(m)} = T_in^{(m)} + T_out^{(m)}.
        6. Computes monthly monetary costs in USD:
           Cost^{(m)} = (T_in^{(m)} / 1e6) * R_in + (T_out^{(m)} / 1e6) * R_out + C_fixed
        7. Computes statistical metrics:
           - Sample mean
           - Sample standard deviation (ddof=1)
           - Percentiles: P5, P50 (median), P95
           - Empirical probability of exceeding budget P(Cost > budget_limit)

    Args:
        config: MonteCarloConfig containing all distribution, pricing, and iteration settings.

    Returns:
        MonteCarloResult containing all raw samples and computed metrics.
    """
    rng = np.random.default_rng(config.seed)
    m = config.n_iterations

    # 1. Simulate monthly request counts
    monthly_expected_requests = config.workload.expected_monthly_requests
    chunk_size = max(
        1,
        min(
            2_000,
            int(MAX_TOKEN_DRAWS_PER_CHUNK / max(monthly_expected_requests, 1.0)),
        ),
    )
    request_counts = sample_requests_count(
        rng=rng,
        family=config.workload.family,
        mean_requests=monthly_expected_requests,
        size=m,
        dispersion=config.workload.dispersion_parameter,
    )

    # 2. Simulate compound input and output tokens
    input_tokens = _aggregate_compound_tokens(
        rng=rng,
        request_counts=request_counts,
        token_sampler_fn=lambda n: sample_tokens(rng, config.input_tokens, size=n),
        chunk_size=chunk_size,
    )

    output_tokens = _aggregate_compound_tokens(
        rng=rng,
        request_counts=request_counts,
        token_sampler_fn=lambda n: sample_tokens(rng, config.output_tokens, size=n),
        chunk_size=chunk_size,
    )

    total_tokens = input_tokens + output_tokens

    # 3. Compute monthly monetary costs
    cost_in = (input_tokens / 1_000_000.0) * config.pricing.cost_per_million_input
    cost_out = (output_tokens / 1_000_000.0) * config.pricing.cost_per_million_output
    monthly_costs = cost_in + cost_out + config.pricing.fixed_monthly_cost

    # 4. Compute statistical summaries
    cost_metrics = summarize_metrics(monthly_costs, threshold=config.budget_limit)
    in_token_metrics = summarize_metrics(input_tokens.astype(np.float64))
    out_token_metrics = summarize_metrics(output_tokens.astype(np.float64))
    total_token_metrics = summarize_metrics(total_tokens.astype(np.float64))
    req_metrics = summarize_metrics(request_counts.astype(np.float64))

    return MonteCarloResult(
        cost_samples=monthly_costs,
        input_token_samples=input_tokens,
        output_token_samples=output_tokens,
        total_token_samples=total_tokens,
        request_count_samples=request_counts,
        cost_metrics=cost_metrics,
        input_token_metrics=in_token_metrics,
        output_token_metrics=out_token_metrics,
        total_token_metrics=total_token_metrics,
        request_count_metrics=req_metrics,
        config=config,
    )
