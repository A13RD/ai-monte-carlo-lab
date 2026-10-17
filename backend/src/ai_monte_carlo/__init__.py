"""AI Monte Carlo Lab - Simulation Engine.

Phase 1 & Phase 2 Core Library for LLM Workload, Predictive Demand Modeling,
and Cost Monte Carlo Simulation.
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
from ai_monte_carlo.data.demand import (
    DailyDemandConfig,
    generate_daily_demand,
)
from ai_monte_carlo.data.synthetic import (
    generate_synthetic_requests,
    save_synthetic_requests,
)
from ai_monte_carlo.models.demand import (
    BaselineDemandModel,
    DemandMetrics,
    HorizonForecast,
    ModelComparison,
    PoissonDemandModel,
    calculate_demand_metrics,
    extract_demand_features,
    forecast_demand_horizon,
    temporal_train_test_split,
)
from ai_monte_carlo.services.experiment import (
    DemandExperimentResult,
    ExperimentConfig,
    run_demand_experiment,
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
    # Data - Phase 1
    "generate_synthetic_requests",
    "save_synthetic_requests",
    # Data - Phase 2
    "DailyDemandConfig",
    "generate_daily_demand",
    # Models - Phase 2
    "DemandMetrics",
    "ModelComparison",
    "HorizonForecast",
    "temporal_train_test_split",
    "extract_demand_features",
    "calculate_demand_metrics",
    "BaselineDemandModel",
    "PoissonDemandModel",
    "forecast_demand_horizon",
    # Services - Phase 2
    "ExperimentConfig",
    "DemandExperimentResult",
    "run_demand_experiment",
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
