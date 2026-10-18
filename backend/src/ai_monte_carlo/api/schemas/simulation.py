"""Pydantic schemas and serialization models for Monte Carlo simulation."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, model_validator

from ai_monte_carlo.config import (
    DistributionType,
    LatencyConfig,
    MonteCarloConfig,
    PricingConfig,
    TokenDistributionConfig,
    WorkloadConfig,
    WorkloadDistributionType,
)


class PricingSchema(BaseModel):
    """Token pricing and fixed monthly cost parameters."""

    cost_per_million_input: float = Field(
        default=2.50, ge=0.0, description="Cost in USD per 1,000,000 input tokens"
    )
    cost_per_million_output: float = Field(
        default=10.00, ge=0.0, description="Cost in USD per 1,000,000 output tokens"
    )
    fixed_monthly_cost: float = Field(
        default=0.0, ge=0.0, description="Fixed monthly operational / hosting cost in USD"
    )

    def to_domain(self) -> PricingConfig:
        return PricingConfig(
            cost_per_million_input=self.cost_per_million_input,
            cost_per_million_output=self.cost_per_million_output,
            fixed_monthly_cost=self.fixed_monthly_cost,
        )


class TokenDistributionSchema(BaseModel):
    """Statistical distribution specification for token lengths."""

    family: DistributionType = Field(
        default=DistributionType.LOGNORMAL,
        description="Probability distribution family (lognormal, gamma, normal, uniform)",
    )
    mean: float = Field(
        default=500.0, gt=0.0, description="Expected token length in natural scale (mean > 0)"
    )
    std: float = Field(
        default=250.0, ge=0.0, description="Standard deviation of token length in natural scale (std >= 0)"
    )
    min_tokens: int = Field(
        default=1, ge=1, description="Minimum allowable token count (clipping lower bound >= 1)"
    )
    max_tokens: int | None = Field(
        default=128_000, description="Maximum allowable token count (clipping upper bound)"
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> TokenDistributionSchema:
        if self.max_tokens is not None and self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) cannot be less than min_tokens ({self.min_tokens})"
            )
        return self

    def to_domain(self) -> TokenDistributionConfig:
        return TokenDistributionConfig(
            family=self.family,
            mean=self.mean,
            std=self.std,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
        )


class WorkloadSchema(BaseModel):
    """Traffic and request volume distribution specification."""

    family: WorkloadDistributionType = Field(
        default=WorkloadDistributionType.POISSON,
        description="Demand distribution family (poisson, negative_binomial, normal, fixed)",
    )
    mean_daily_requests: float = Field(
        default=1_000.0, ge=0.0, description="Expected average requests per operational day"
    )
    dispersion_parameter: float = Field(
        default=0.1, gt=0.0, description="Overdispersion parameter alpha (must be strictly positive)"
    )
    days_per_month: int = Field(
        default=30, ge=1, description="Number of operational days in billing cycle"
    )

    def to_domain(self) -> WorkloadConfig:
        return WorkloadConfig(
            family=self.family,
            mean_daily_requests=self.mean_daily_requests,
            dispersion_parameter=self.dispersion_parameter,
            days_per_month=self.days_per_month,
        )


class SimulationRequest(BaseModel):
    """Request payload to execute a monthly Monte Carlo simulation run."""

    n_iterations: int = Field(
        default=10_000,
        ge=2,
        le=500_000,
        description="Number of Monte Carlo iterations (M >= 2)",
    )
    budget_limit: float = Field(
        default=500.0,
        ge=0.0,
        description="Monthly monetary budget threshold in USD to evaluate P(Cost > Budget)",
    )
    pricing: PricingSchema = Field(
        default_factory=PricingSchema,
        description="Token pricing configuration",
    )
    input_tokens: TokenDistributionSchema = Field(
        default_factory=lambda: TokenDistributionSchema(
            family=DistributionType.LOGNORMAL, mean=600.0, std=300.0, min_tokens=10
        ),
        description="Prompt token distribution parameters",
    )
    output_tokens: TokenDistributionSchema = Field(
        default_factory=lambda: TokenDistributionSchema(
            family=DistributionType.LOGNORMAL, mean=350.0, std=180.0, min_tokens=10
        ),
        description="Completion token distribution parameters",
    )
    workload: WorkloadSchema = Field(
        default_factory=WorkloadSchema,
        description="Monthly request volume configuration",
    )
    seed: int | None = Field(
        default=42,
        description="Pseudorandom generator seed for deterministic reproducibility (optional)",
    )

    def to_domain(self) -> MonteCarloConfig:
        return MonteCarloConfig(
            n_iterations=self.n_iterations,
            budget_limit=self.budget_limit,
            pricing=self.pricing.to_domain(),
            input_tokens=self.input_tokens.to_domain(),
            output_tokens=self.output_tokens.to_domain(),
            workload=self.workload.to_domain(),
            seed=self.seed,
        )


class MetricsSummarySchema(BaseModel):
    """Summary statistics for simulated quantities (costs, tokens, etc.)."""

    mean: float = Field(description="Sample mean")
    sample_std: float = Field(description="Unbiased sample standard deviation (ddof=1)")
    p5: float = Field(description="5th percentile (optimistic baseline)")
    p50_median: float = Field(description="50th percentile (median)")
    p95: float = Field(description="95th percentile (stress scenario)")
    min: float = Field(description="Minimum value observed across iterations")
    max: float = Field(description="Maximum value observed across iterations")
    count: int = Field(description="Sample size / iteration count")


class SimulationResponse(BaseModel):
    """Complete results of a Monte Carlo simulation execution."""

    iterations: int = Field(description="Number of simulation iterations evaluated")
    budget_limit: float = Field(description="Budget limit evaluated in USD")
    probability_exceeding_budget: float = Field(
        description="Empirical probability P(Monthly Cost > Budget Limit)"
    )
    cost_metrics: MetricsSummarySchema = Field(
        description="Summary statistics for total monthly cost in USD"
    )
    token_metrics: MetricsSummarySchema = Field(
        description="Summary statistics for total monthly token consumption"
    )
    effective_config: dict[str, Any] = Field(
        description="Effective configuration parameters utilized in the run"
    )
    summary: dict[str, Any] = Field(
        description="Structured summary dictionary compatible with CLI and reporting"
    )
    execution_time_ms: float = Field(
        description="Execution wall-clock time in milliseconds"
    )
    seed: int | None = Field(
        default=None, description="Random seed used for deterministic execution"
    )
