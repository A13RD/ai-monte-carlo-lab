"""AI Monte Carlo Lab - Simulation Engine.

Phase 1 Core Library for LLM Workload and Cost Monte Carlo Simulation.
"""

from ai_monte_carlo.config import (
    DistributionType,
    LatencyConfig,
    MonteCarloConfig,
    PricingConfig,
    TokenDistributionConfig,
    WorkloadConfig,
    WorkloadDistributionType,
)
from ai_monte_carlo.data.synthetic import (
    generate_synthetic_requests,
    save_synthetic_requests,
)
from ai_monte_carlo.simulation.generators import (
    sample_latency,
    sample_requests_count,
    sample_tokens,
    sample_tokens_gamma,
    sample_tokens_lognormal,
    sample_tokens_normal,
)
from ai_monte_carlo.simulation.metrics import (
    SimulationMetricsSummary,
    calculate_mean,
    calculate_percentiles,
    calculate_probability_exceeding,
    calculate_sample_std,
    summarize_metrics,
)
from ai_monte_carlo.simulation.monte_carlo import (
    MonteCarloResult,
    run_monte_carlo,
)

__all__ = [
    # Config
    "DistributionType",
    "WorkloadDistributionType",
    "PricingConfig",
    "TokenDistributionConfig",
    "LatencyConfig",
    "WorkloadConfig",
    "MonteCarloConfig",
    # Data
    "generate_synthetic_requests",
    "save_synthetic_requests",
    # Generators
    "sample_tokens",
    "sample_tokens_lognormal",
    "sample_tokens_gamma",
    "sample_tokens_normal",
    "sample_requests_count",
    "sample_latency",
    # Metrics
    "calculate_mean",
    "calculate_sample_std",
    "calculate_percentiles",
    "calculate_probability_exceeding",
    "summarize_metrics",
    "SimulationMetricsSummary",
    # Monte Carlo Engine
    "run_monte_carlo",
    "MonteCarloResult",
]
