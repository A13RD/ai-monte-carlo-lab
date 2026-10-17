"""Tests for experiment coordination service integrating demand forecast and Monte Carlo (Phase 2)."""

import numpy as np
import pandas as pd
import pytest

from ai_monte_carlo.config import MonteCarloConfig, WorkloadConfig
from ai_monte_carlo.data.demand import DailyDemandConfig, generate_daily_demand
from ai_monte_carlo.services.experiment import (
    DemandExperimentResult,
    ExperimentConfig,
    run_demand_experiment,
)
from ai_monte_carlo.simulation.monte_carlo import MonteCarloResult


class TestExperimentServiceIntegration:
    """Test suite covering Requirements 10, 11, and 12."""

    def test_end_to_end_integration_with_run_monte_carlo(self) -> None:
        """Requirement 10: Seamless integration executing run_monte_carlo with forecasted inputs."""
        base_mc_cfg = MonteCarloConfig(n_iterations=200, budget_limit=400.0, seed=42)
        exp_cfg = ExperimentConfig(
            monte_carlo_config=base_mc_cfg,
            test_size=0.2,
            seed=42,
        )

        result = run_demand_experiment(exp_cfg)

        assert isinstance(result, DemandExperimentResult)
        assert isinstance(result.monte_carlo_result, MonteCarloResult)
        assert len(result.monte_carlo_result.cost_samples) == 200
        assert result.monte_carlo_result.cost_metrics.mean > 0
        assert np.isfinite(result.monte_carlo_result.cost_metrics.sample_std)
        assert result.horizon_forecast.mean_daily_demand > 0

    def test_forecast_transferred_as_monte_carlo_input(self) -> None:
        """Requirement 11: Daily forecast is reduced to mean_daily_requests and passed to Monte Carlo."""
        base_mc_cfg = MonteCarloConfig(
            n_iterations=100,
            workload=WorkloadConfig(mean_daily_requests=500.0, days_per_month=30),
            seed=42,
        )
        exp_cfg = ExperimentConfig(
            monte_carlo_config=base_mc_cfg,
            forecast_days=30,
            seed=42,
        )

        result = run_demand_experiment(exp_cfg)

        # Check effective simulation configuration received the predicted mean
        effective_workload = result.effective_monte_carlo_config.workload
        assert effective_workload.mean_daily_requests == result.horizon_forecast.mean_daily_demand
        assert effective_workload.days_per_month == 30
        assert effective_workload.mean_daily_requests != 500.0  # Must be updated from forecast

        # Verify Monte Carlo result reflects effective config
        assert result.monte_carlo_result.config == result.effective_monte_carlo_config

    def test_original_config_immutability_and_preservation(self) -> None:
        """Requirement 12: Original configuration remains strictly unmodified."""
        initial_workload = WorkloadConfig(mean_daily_requests=750.0, days_per_month=30)
        original_mc_cfg = MonteCarloConfig(
            n_iterations=100,
            workload=initial_workload,
            budget_limit=600.0,
            seed=42,
        )

        exp_cfg = ExperimentConfig(
            monte_carlo_config=original_mc_cfg,
            seed=42,
        )

        result = run_demand_experiment(exp_cfg)

        # Verify original instance attributes were not mutated
        assert original_mc_cfg.workload.mean_daily_requests == 750.0
        assert original_mc_cfg.workload.days_per_month == 30
        assert result.original_monte_carlo_config is original_mc_cfg
        assert result.original_monte_carlo_config.workload.mean_daily_requests == 750.0

        # Effective config is a distinct instance with updated demand
        assert result.effective_monte_carlo_config is not original_mc_cfg
        assert (
            result.effective_monte_carlo_config.workload.mean_daily_requests
            == result.horizon_forecast.mean_daily_demand
        )

    def test_custom_historical_dataframe(self) -> None:
        """Tests that experiment service accepts external/custom historical DataFrames."""
        custom_df = generate_daily_demand(
            DailyDemandConfig(
                n_days=60,
                start_date="2025-06-01",
                base_demand=800.0,
                trend_slope=1.5,
                seed=99,
            )
        )
        base_mc_cfg = MonteCarloConfig(n_iterations=100, seed=99)
        exp_cfg = ExperimentConfig(monte_carlo_config=base_mc_cfg, test_size=10, seed=99)

        result = run_demand_experiment(exp_cfg, historical_df=custom_df)

        assert result.reproduction_info["history_days"] == 60
        assert result.reproduction_info["train_days"] == 50
        assert result.reproduction_info["test_days"] == 10
        assert len(result.horizon_forecast.daily_forecast) == 30

    def test_insufficient_history_raises(self) -> None:
        """Ensures that insufficient history (< 7 days) raises ValueError."""
        short_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=5, freq="D"),
                "time_index": [0, 1, 2, 3, 4],
                "day_of_week": [3, 4, 5, 6, 0],
                "requests": [100, 110, 80, 75, 120],
            }
        )
        with pytest.raises(ValueError, match="Insufficient historical data"):
            run_demand_experiment(historical_df=short_df)

    def test_summary_and_forecast_dataframe_methods(self) -> None:
        """Verifies summary_dict and forecast_dataframe utility methods."""
        base_mc_cfg = MonteCarloConfig(n_iterations=100, seed=42)
        exp_cfg = ExperimentConfig(monte_carlo_config=base_mc_cfg, seed=42)

        result = run_demand_experiment(exp_cfg)

        summary = result.summary_dict()
        assert "evaluation" in summary
        assert "horizon_forecast" in summary
        assert "monte_carlo_simulation" in summary
        assert "reproduction" in summary

        f_df = result.forecast_dataframe()
        assert isinstance(f_df, pd.DataFrame)
        assert len(f_df) == result.horizon_forecast.horizon_days
        assert "predicted_requests" in f_df.columns
