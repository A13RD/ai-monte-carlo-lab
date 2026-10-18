"""Pydantic schemas and serialization models for predictive demand modeling."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from ai_monte_carlo.api.schemas.simulation import SimulationRequest
from ai_monte_carlo.data.demand import DailyDemandConfig
from ai_monte_carlo.services.experiment import ExperimentConfig


class DailyDemandConfigSchema(BaseModel):
    """Configuration for historical synthetic daily demand trace generation."""

    n_days: int = Field(
        default=180, ge=7, le=3650, description="Total days in synthetic history (>= 7)"
    )
    start_date: str = Field(
        default="2026-01-01", description="Chronological start date (YYYY-MM-DD)"
    )
    base_demand: float = Field(
        default=1000.0, ge=0.0, description="Baseline request volume at time t=0"
    )
    trend_slope: float = Field(
        default=2.5, description="Daily secular growth or decline rate"
    )
    weekly_seasonality: list[float] = Field(
        default=[1.05, 1.10, 1.12, 1.08, 0.95, 0.70, 0.60],
        description="7-element multiplicative seasonality multipliers (Mon-Sun)",
    )
    noise_dispersion: float = Field(
        default=0.05, ge=0.0, description="Overdispersion factor for stochastic noise"
    )
    noise_family: str = Field(
        default="poisson",
        description="Noise distribution family ('poisson', 'negative_binomial', 'fixed')",
    )
    seed: int | None = Field(
        default=42, description="Random seed for history generation"
    )

    def to_domain(self) -> DailyDemandConfig:
        return DailyDemandConfig(
            n_days=self.n_days,
            start_date=self.start_date,
            base_demand=self.base_demand,
            trend_slope=self.trend_slope,
            weekly_seasonality=tuple(self.weekly_seasonality),
            noise_dispersion=self.noise_dispersion,
            noise_family=self.noise_family,
            seed=self.seed,
        )


class DemandExperimentRequest(BaseModel):
    """Request payload to execute the full Phase 2 Predictive Experiment pipeline."""

    demand_config: DailyDemandConfigSchema = Field(
        default_factory=DailyDemandConfigSchema,
        description="Parameters for historical demand generation",
    )
    monte_carlo_config: SimulationRequest = Field(
        default_factory=SimulationRequest,
        description="Baseline Monte Carlo parameters receiving forecasted demand",
    )
    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Proportion of history reserved for chronological out-of-sample evaluation",
    )
    forecast_days: int | None = Field(
        default=30,
        ge=1,
        le=365,
        description="Future horizon length in days (defaults to simulation days_per_month if None)",
    )
    refit_full_history: bool = Field(
        default=True,
        description="Whether to refit Poisson GLM on full history before projecting horizon",
    )
    model_alpha: float = Field(
        default=1e-4, ge=0.0, description="L2 regularization penalty for PoissonRegressor"
    )
    seed: int | None = Field(
        default=42, description="Master seed for end-to-end deterministic reproducibility"
    )

    def to_domain(self) -> ExperimentConfig:
        return ExperimentConfig(
            demand_config=self.demand_config.to_domain(),
            monte_carlo_config=self.monte_carlo_config.to_domain(),
            test_size=self.test_size,
            forecast_days=self.forecast_days,
            refit_full_history=self.refit_full_history,
            model_alpha=self.model_alpha,
            seed=self.seed,
        )


class DemandEvaluationMetricsSchema(BaseModel):
    """Out-of-sample forecast accuracy metrics."""

    mae: float = Field(description="Mean Absolute Error (requests)")
    rmse: float = Field(description="Root Mean Squared Error (requests)")
    mean_deviance: float = Field(description="Mean Poisson Deviance")
    sample_size: int = Field(description="Evaluation sample count (days in test partition)")


class ModelComparisonSchema(BaseModel):
    """Comparative performance between candidate Poisson GLM and baseline reference."""

    model_mae: float
    model_rmse: float
    model_deviance: float
    baseline_mae: float
    baseline_rmse: float
    mae_reduction: float = Field(
        description="MAE reduction relative to baseline: baseline_mae - model_mae"
    )
    rmse_reduction: float = Field(
        description="RMSE reduction relative to baseline: baseline_rmse - model_rmse"
    )
    model_beats_baseline: bool = Field(
        description="True if candidate model achieves lower or equal MAE and RMSE"
    )
    test_sample_size: int


class HorizonForecastSchema(BaseModel):
    """Forecasted request demand metrics across future horizon."""

    horizon_days: int
    mean_daily_demand: float
    total_expected_demand: float
    min_daily_demand: float
    max_daily_demand: float
    daily_forecast: list[float] = Field(
        description="Daily predicted demand array over the horizon"
    )


class DemandExperimentResponse(BaseModel):
    """Complete results from the integrated Phase 2 experiment."""

    evaluation: ModelComparisonSchema
    horizon_forecast: HorizonForecastSchema
    monte_carlo_simulation: dict[str, Any]
    learned_coefficients: dict[str, float]
    reproduction: dict[str, Any]
    execution_time_ms: float


class DemandForecastRequest(BaseModel):
    """Request payload to fit Poisson GLM and generate only a horizon forecast."""

    demand_config: DailyDemandConfigSchema = Field(
        default_factory=DailyDemandConfigSchema,
        description="Parameters for synthetic daily demand series",
    )
    horizon_days: int = Field(
        default=30, ge=1, le=365, description="Days to forecast into the future"
    )
    model_alpha: float = Field(
        default=1e-4, ge=0.0, description="L2 regularization penalty for PoissonRegressor"
    )
    seed: int | None = Field(
        default=42, description="Random seed for deterministic reproducibility"
    )


class DemandForecastResponse(BaseModel):
    """Forecasted demand trajectory and learned model coefficients."""

    horizon_days: int
    mean_daily_demand: float
    total_expected_demand: float
    min_daily_demand: float
    max_daily_demand: float
    daily_forecast: list[float]
    learned_coefficients: dict[str, float]
    execution_time_ms: float
