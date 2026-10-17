"""Experiment service module integrating predictive demand modeling with the Monte Carlo engine.

This service coordinates Phase 2 execution:
1. Ingests or generates synthetic daily historical demand traces.
2. Performs chronological train/test splitting to prevent future data leakage.
3. Fits the Poisson GLM and the training-derived baseline reference model.
4. Evaluates out-of-sample forecast accuracy (MAE, RMSE, deviance).
5. Retrains the final model on all historical data and forecasts daily demand for the simulation horizon.
6. Transfers the predicted mean daily demand into a newly derived MonteCarloConfig using `dataclasses.replace`.
7. Executes the Monte Carlo simulation to project consumption, monetary costs, and budget exceedance probabilities.

Important Architectural Note:
    Reducing a daily time-varying forecast vector into an aggregate scalar mean (mean_daily_requests)
    is a mathematical approximation. In Phase 1, the simulation engine models monthly volume via
    homogeneous or overdispersed compound arrival processes across D days. Phase 2 preserves the full
    vector of daily predictions for downstream analysis while maintaining strict backward compatibility.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai_monte_carlo.config import MonteCarloConfig, WorkloadConfig
from ai_monte_carlo.data.demand import DailyDemandConfig, generate_daily_demand
from ai_monte_carlo.models.demand import (
    BaselineDemandModel,
    DemandMetrics,
    HorizonForecast,
    ModelComparison,
    PoissonDemandModel,
    calculate_demand_metrics,
    forecast_demand_horizon,
    temporal_train_test_split,
)
from ai_monte_carlo.simulation.monte_carlo import MonteCarloResult, run_monte_carlo


@dataclass(frozen=True)
class ExperimentConfig:
    """Master configuration for the integrated Demand-Monte Carlo experiment.

    Attributes:
        demand_config: Parameters for synthetic history generation (if historical_df not provided).
        monte_carlo_config: Baseline Monte Carlo configuration to receive forecasted demand.
        test_size: Proportion (float in (0, 1)) or day count (int) for chronological test evaluation.
        forecast_days: Length of future forecast horizon in days. If None, defaults to workload.days_per_month.
        refit_full_history: If True, refits Poisson GLM on full history before forecasting future horizon.
        model_alpha: L2 regularization parameter for PoissonRegressor (default: 1e-4).
        seed: Random seed for deterministic reproducibility.
    """

    demand_config: DailyDemandConfig = field(default_factory=DailyDemandConfig)
    monte_carlo_config: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    test_size: float | int = 0.2
    forecast_days: int | None = None
    refit_full_history: bool = True
    model_alpha: float = 1e-4
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.model_alpha < 0:
            raise ValueError(f"model_alpha must be non-negative, got {self.model_alpha}")
        if self.forecast_days is not None and self.forecast_days < 1:
            raise ValueError(f"forecast_days must be at least 1, got {self.forecast_days}")


@dataclass(frozen=True)
class DemandExperimentResult:
    """Encapsulates all outputs, evaluation metrics, and simulation results from an experiment run.

    Attributes:
        test_metrics: Accuracy metrics of the candidate predictive model on test set.
        baseline_metrics: Accuracy metrics of the training-derived baseline reference on test set.
        model_comparison: Structured comparison between candidate model and baseline.
        horizon_forecast: Daily demand predictions and aggregate statistics across the horizon.
        effective_monte_carlo_config: Derived MonteCarloConfig containing forecasted demand.
        original_monte_carlo_config: Original unmodified MonteCarloConfig provided at input.
        monte_carlo_result: Full MonteCarloResult from simulation run.
        reproduction_info: Metadata required to audit, interpret, and reproduce the experiment.
    """

    test_metrics: DemandMetrics
    baseline_metrics: DemandMetrics
    model_comparison: ModelComparison
    horizon_forecast: HorizonForecast
    effective_monte_carlo_config: MonteCarloConfig
    original_monte_carlo_config: MonteCarloConfig
    monte_carlo_result: MonteCarloResult
    reproduction_info: dict[str, Any]

    def forecast_dataframe(self) -> pd.DataFrame:
        """Returns the daily forecast for the horizon as a pandas DataFrame."""
        return self.horizon_forecast.to_dataframe()

    def summary_dict(self) -> dict[str, Any]:
        """Provides a comprehensive, structured summary dictionary of the experiment."""
        mc_summary = self.monte_carlo_result.summary_dict()
        return {
            "evaluation": {
                "model_mae": self.test_metrics.mae,
                "model_rmse": self.test_metrics.rmse,
                "model_deviance": self.test_metrics.mean_deviance,
                "baseline_mae": self.baseline_metrics.mae,
                "baseline_rmse": self.baseline_metrics.rmse,
                "mae_reduction": self.model_comparison.mae_reduction,
                "rmse_reduction": self.model_comparison.rmse_reduction,
                "model_beats_baseline": self.model_comparison.model_beats_baseline,
                "test_sample_size": self.test_metrics.sample_size,
            },
            "horizon_forecast": {
                "horizon_days": self.horizon_forecast.horizon_days,
                "mean_daily_demand": self.horizon_forecast.mean_daily_demand,
                "total_expected_demand": self.horizon_forecast.total_expected_demand,
                "min_daily_demand": self.horizon_forecast.min_daily_demand,
                "max_daily_demand": self.horizon_forecast.max_daily_demand,
            },
            "monte_carlo_simulation": mc_summary,
            "reproduction": self.reproduction_info,
        }


def run_demand_experiment(
    config: ExperimentConfig | None = None,
    historical_df: pd.DataFrame | None = None,
) -> DemandExperimentResult:
    """Executes the end-to-end Demand Modeling and Monte Carlo simulation workflow.

    Workflow:
    1. Obtains historical daily request series (either from historical_df or generated synthetically).
    2. Validates temporal integrity and chronological monotonicity.
    3. Splits series chronologically into train and test sets (no leakage).
    4. Fits PoissonDemandModel and BaselineDemandModel on the training partition.
    5. Evaluates out-of-sample accuracy on the test partition.
    6. Fits final model on all available history (if configured) and forecasts future horizon.
    7. Builds effective MonteCarloConfig using `dataclasses.replace` with forecasted mean_daily_requests.
    8. Executes `run_monte_carlo` and returns the integrated result.

    Args:
        config: ExperimentConfig settings. Defaults to ExperimentConfig() if None.
        historical_df: Optional pre-existing DataFrame with daily requests. Must have 'requests'
                       and 'date' (or 'time_index').

    Returns:
        DemandExperimentResult containing all metrics, forecast traces, and simulation outputs.
    """
    cfg = config if config is not None else ExperimentConfig()

    # 1. Obtain historical demand data
    if historical_df is not None:
        history = historical_df.copy()
        if "requests" not in history.columns:
            raise ValueError("historical_df must contain 'requests' column.")
        if "time_index" not in history.columns:
            history["time_index"] = np.arange(len(history), dtype=np.int64)
    else:
        history = generate_daily_demand(cfg.demand_config)

    # Validate historical data length
    if len(history) < 7:
        raise ValueError(
            f"Insufficient historical data: expected at least 7 daily records, got {len(history)}"
        )

    # 2. Chronological Train/Test Split
    train_df, test_df = temporal_train_test_split(history, test_size=cfg.test_size)
    if len(train_df) < 4 or len(test_df) < 1:
        raise ValueError(
            f"Split resulted in inadequate sample sizes: train={len(train_df)}, test={len(test_df)}"
        )

    # 3. Model Training on Training Set
    model_eval = PoissonDemandModel(alpha=cfg.model_alpha)
    model_eval.fit(train_df)

    baseline_eval = BaselineDemandModel()
    baseline_eval.fit(train_df)

    # 4. Out-of-sample Evaluation
    y_test = test_df["requests"].to_numpy(dtype=np.float64)
    pred_model_test = model_eval.predict(test_df)
    pred_baseline_test = baseline_eval.predict(test_df)

    test_metrics = calculate_demand_metrics(y_test, pred_model_test)
    baseline_metrics = calculate_demand_metrics(y_test, pred_baseline_test)

    mae_red = round(baseline_metrics.mae - test_metrics.mae, 4)
    rmse_red = round(baseline_metrics.rmse - test_metrics.rmse, 4)
    beats_baseline = bool(test_metrics.mae <= baseline_metrics.mae and test_metrics.rmse <= baseline_metrics.rmse)

    comparison = ModelComparison(
        model_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        mae_reduction=mae_red,
        rmse_reduction=rmse_red,
        model_beats_baseline=beats_baseline,
    )

    # 5. Final Model Fitting for Future Horizon Forecasting
    if cfg.refit_full_history:
        final_model = PoissonDemandModel(alpha=cfg.model_alpha)
        final_model.fit(history)
    else:
        final_model = model_eval

    # Determine horizon parameters
    horizon_days = (
        cfg.forecast_days
        if cfg.forecast_days is not None
        else cfg.monte_carlo_config.workload.days_per_month
    )

    if "date" in history.columns:
        last_date = pd.to_datetime(history["date"].iloc[-1])
        start_date = last_date + pd.Timedelta(days=1)
    else:
        start_date = pd.Timestamp("2026-07-01")

    start_time_idx = int(history["time_index"].iloc[-1]) + 1
    horizon_forecast = forecast_demand_horizon(
        model=final_model,
        start_date=start_date,
        horizon_days=horizon_days,
        start_time_index=start_time_idx,
    )

    # Validate horizon forecast values
    if not np.all(np.isfinite(horizon_forecast.daily_forecast)):
        raise ValueError("Horizon forecast contains non-finite values.")
    if np.any(horizon_forecast.daily_forecast < 0):
        raise ValueError("Horizon forecast contains negative demand values.")
    if horizon_forecast.mean_daily_demand <= 0:
        raise ValueError(
            f"Horizon forecast produced non-positive mean demand: {horizon_forecast.mean_daily_demand}"
        )

    # 6. Immutable Configuration Transfer using dataclasses.replace
    derived_workload = dataclasses.replace(
        cfg.monte_carlo_config.workload,
        mean_daily_requests=horizon_forecast.mean_daily_demand,
        days_per_month=horizon_days,
    )
    effective_mc_config = dataclasses.replace(
        cfg.monte_carlo_config,
        workload=derived_workload,
        seed=cfg.seed if cfg.seed is not None else cfg.monte_carlo_config.seed,
    )

    # 7. Execute Monte Carlo Simulation
    mc_result = run_monte_carlo(effective_mc_config)

    # 8. Reproduction Metadata
    reproduction_info: dict[str, Any] = {
        "seed": cfg.seed,
        "history_days": len(history),
        "train_days": len(train_df),
        "test_days": len(test_df),
        "test_size": cfg.test_size,
        "refit_full_history": cfg.refit_full_history,
        "model_alpha": cfg.model_alpha,
        "forecast_horizon_days": horizon_days,
        "learned_coefficients": final_model.get_coefficients(),
        "approximation_note": (
            "The daily forecast vector was aggregated to a mean daily demand scalar "
            "(mean_daily_requests) for the Phase 1 Monte Carlo simulation engine. "
            "Individual daily forecasts are preserved in horizon_forecast."
        ),
    }

    return DemandExperimentResult(
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        model_comparison=comparison,
        horizon_forecast=horizon_forecast,
        effective_monte_carlo_config=effective_mc_config,
        original_monte_carlo_config=cfg.monte_carlo_config,
        monte_carlo_result=mc_result,
        reproduction_info=reproduction_info,
    )
