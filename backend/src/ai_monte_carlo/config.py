"""Configuration models and parameters for AI Monte Carlo Lab.

Provides strongly-typed dataclasses, enum definitions, and input validations
for pricing models, token distributions, latency models, and simulation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


MAX_EXPECTED_SIMULATED_REQUESTS = 30_000_000


class DistributionType(str, Enum):
    """Supported parametric probability distributions for token generation."""

    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    NORMAL = "normal"
    UNIFORM = "uniform"


class WorkloadDistributionType(str, Enum):
    """Supported parametric probability distributions for request counts."""

    POISSON = "poisson"
    NEGATIVE_BINOMIAL = "negative_binomial"
    NORMAL = "normal"
    FIXED = "fixed"


@dataclass(frozen=True)
class PricingConfig:
    """Pricing configuration based on rates per million tokens.

    Attributes:
        cost_per_million_input: Cost in USD per 1,000,000 input (prompt) tokens.
        cost_per_million_output: Cost in USD per 1,000,000 output (completion) tokens.
        fixed_monthly_cost: Fixed recurring platform or hosting cost in USD.
    """

    cost_per_million_input: float = 2.50
    cost_per_million_output: float = 10.00
    fixed_monthly_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_per_million_input < 0:
            raise ValueError(
                f"cost_per_million_input must be non-negative, got {self.cost_per_million_input}"
            )
        if self.cost_per_million_output < 0:
            raise ValueError(
                f"cost_per_million_output must be non-negative, got {self.cost_per_million_output}"
            )
        if self.fixed_monthly_cost < 0:
            raise ValueError(
                f"fixed_monthly_cost must be non-negative, got {self.fixed_monthly_cost}"
            )


@dataclass(frozen=True)
class TokenDistributionConfig:
    """Statistical distribution parameters for token count modeling.

    Attributes:
        family: Probability distribution family (e.g. LOGNORMAL, GAMMA).
        mean: Expected token count in natural scale. Must be > 0.
        std: Standard deviation of token count in natural scale. Must be >= 0.
        min_tokens: Minimum allowable token count (clipping lower bound >= 1).
        max_tokens: Maximum allowable token count (clipping upper bound, optional).
    """

    family: DistributionType = DistributionType.LOGNORMAL
    mean: float = 500.0
    std: float = 250.0
    min_tokens: int = 1
    max_tokens: int | None = 128_000

    def __post_init__(self) -> None:
        if self.mean <= 0:
            raise ValueError(f"mean tokens must be strictly positive, got {self.mean}")
        if self.std < 0:
            raise ValueError(f"std tokens must be non-negative, got {self.std}")
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be at least 1, got {self.min_tokens}")
        if self.max_tokens is not None and self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) cannot be less than min_tokens ({self.min_tokens})"
            )


@dataclass(frozen=True)
class LatencyConfig:
    """Latency generation model parameters.

    Latency is modeled as:
        Latency = Base + (alpha_in * input_tokens) + (alpha_out * output_tokens) + epsilon
    where epsilon ~ Normal(0, jitter_std_ms^2), truncated at min_latency_ms.

    Attributes:
        base_latency_ms: Fixed Time-to-First-Token overhead in milliseconds.
        per_input_token_ms: Marginal latency per prompt token (ms).
        per_output_token_ms: Marginal latency per generated token (ms).
        jitter_std_ms: Stochastic network and queue jitter standard deviation (ms).
        min_latency_ms: Absolute minimum physical latency bound (ms).
    """

    base_latency_ms: float = 180.0
    per_input_token_ms: float = 0.05
    per_output_token_ms: float = 0.80
    jitter_std_ms: float = 40.0
    min_latency_ms: float = 10.0

    def __post_init__(self) -> None:
        if self.base_latency_ms < 0:
            raise ValueError(f"base_latency_ms must be non-negative, got {self.base_latency_ms}")
        if self.per_input_token_ms < 0:
            raise ValueError(
                f"per_input_token_ms must be non-negative, got {self.per_input_token_ms}"
            )
        if self.per_output_token_ms < 0:
            raise ValueError(
                f"per_output_token_ms must be non-negative, got {self.per_output_token_ms}"
            )
        if self.jitter_std_ms < 0:
            raise ValueError(f"jitter_std_ms must be non-negative, got {self.jitter_std_ms}")
        if self.min_latency_ms < 0:
            raise ValueError(f"min_latency_ms must be non-negative, got {self.min_latency_ms}")


@dataclass(frozen=True)
class WorkloadConfig:
    """Configuration for demand/traffic generation over a monthly horizon.

    Attributes:
        family: Distribution family for request volume.
        mean_daily_requests: Expected number of requests per day.
        dispersion_parameter: Overdispersion parameter alpha for Negative Binomial
            where Var = mu + alpha * mu^2, or relative dispersion factor.
        days_per_month: Operational days in a billing month (default: 30).
    """

    family: WorkloadDistributionType = WorkloadDistributionType.POISSON
    mean_daily_requests: float = 1_000.0
    dispersion_parameter: float = 0.1
    days_per_month: int = 30

    def __post_init__(self) -> None:
        if self.mean_daily_requests < 0:
            raise ValueError(
                f"mean_daily_requests must be non-negative, got {self.mean_daily_requests}"
            )
        if self.dispersion_parameter <= 0:
            raise ValueError(
                f"dispersion_parameter must be strictly positive, got {self.dispersion_parameter}"
            )
        if self.days_per_month < 1:
            raise ValueError(f"days_per_month must be at least 1, got {self.days_per_month}")

    @property
    def expected_monthly_requests(self) -> float:
        """Calculates expected monthly request count."""
        return self.mean_daily_requests * self.days_per_month


@dataclass(frozen=True)
class MonteCarloConfig:
    """Master configuration for monthly Monte Carlo simulation runs.

    Attributes:
        n_iterations: Number of Monte Carlo monthly iterations (M >= 2).
        budget_limit: Monthly budget threshold in USD to evaluate P(Cost > Budget).
        pricing: Pricing parameters for input/output tokens.
        input_tokens: Token distribution config for input/prompt length.
        output_tokens: Token distribution config for output/completion length.
        workload: Request demand volume distribution config.
        seed: Random seed for deterministic reproducibility.
    """

    n_iterations: int = 10_000
    budget_limit: float = 500.0
    pricing: PricingConfig = field(default_factory=PricingConfig)
    input_tokens: TokenDistributionConfig = field(
        default_factory=lambda: TokenDistributionConfig(
            family=DistributionType.LOGNORMAL, mean=600.0, std=300.0
        )
    )
    output_tokens: TokenDistributionConfig = field(
        default_factory=lambda: TokenDistributionConfig(
            family=DistributionType.LOGNORMAL, mean=350.0, std=180.0
        )
    )
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.n_iterations < 2:
            raise ValueError(
                f"n_iterations must be at least 2 to compute sample statistics, got {self.n_iterations}"
            )
        if self.budget_limit < 0:
            raise ValueError(f"budget_limit must be non-negative, got {self.budget_limit}")
