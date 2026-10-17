"""Tests for synthetic daily demand time-series generation (Phase 2)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ai_monte_carlo.data.demand import DailyDemandConfig, generate_daily_demand


class TestDailyDemandGeneration:
    """Test suite covering DailyDemandConfig and generate_daily_demand."""

    def test_reproducibility_with_same_seed(self) -> None:
        """Requirement 1: Series generation is strictly reproducible with the same seed."""
        cfg1 = DailyDemandConfig(n_days=60, seed=123)
        cfg2 = DailyDemandConfig(n_days=60, seed=123)

        df1 = generate_daily_demand(cfg1)
        df2 = generate_daily_demand(cfg2)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_yield_different_data(self) -> None:
        """Requirement 2: Different seeds produce distinct stochastic trajectories."""
        cfg1 = DailyDemandConfig(n_days=60, seed=42)
        cfg2 = DailyDemandConfig(n_days=60, seed=999)

        df1 = generate_daily_demand(cfg1)
        df2 = generate_daily_demand(cfg2)

        assert not np.array_equal(df1["requests"].to_numpy(), df2["requests"].to_numpy())

    def test_chronological_dates_and_non_negative_integer_counts(self) -> None:
        """Requirement 3: Dates are chronologically ordered and request counts are non-negative integers."""
        cfg = DailyDemandConfig(n_days=90, start_date="2026-03-01", seed=42)
        df = generate_daily_demand(cfg)

        assert len(df) == 90
        # Check monotonic calendar sequence
        date_diffs = df["date"].diff().dropna()
        assert (date_diffs == pd.Timedelta(days=1)).all()
        assert df["date"].is_monotonic_increasing

        # Check time_index is 0 to N-1
        assert (df["time_index"].to_numpy() == np.arange(90)).all()

        # Check day of week corresponds to actual date
        expected_dow = df["date"].dt.dayofweek.to_numpy()
        assert (df["day_of_week"].to_numpy() == expected_dow).all()

        # Check non-negative integers
        requests = df["requests"].to_numpy()
        assert np.issubdtype(requests.dtype, np.integer)
        assert np.all(requests >= 0)

    def test_invalid_parameters_raise(self) -> None:
        """Requirement 4: Invalid parameters must be caught preventively."""
        # Negative or zero n_days
        with pytest.raises(ValueError, match="n_days must be at least 1"):
            DailyDemandConfig(n_days=0)
        with pytest.raises(ValueError, match="n_days must be at least 1"):
            DailyDemandConfig(n_days=-5)

        # Negative base demand
        with pytest.raises(ValueError, match="base_demand must be non-negative"):
            DailyDemandConfig(base_demand=-100.0)

        # Invalid seasonality tuple length
        with pytest.raises(ValueError, match="weekly_seasonality must contain exactly 7 multipliers"):
            DailyDemandConfig(weekly_seasonality=(1.0, 1.0, 1.0))

        # Negative seasonality factor
        with pytest.raises(ValueError, match="weekly_seasonality multipliers must be non-negative"):
            DailyDemandConfig(weekly_seasonality=(1.0, 1.0, 1.0, -0.5, 1.0, 1.0, 1.0))

        # Negative noise dispersion
        with pytest.raises(ValueError, match="noise_dispersion must be non-negative"):
            DailyDemandConfig(noise_dispersion=-0.1)

        # Unknown noise family
        with pytest.raises(ValueError, match="Unsupported noise_family"):
            DailyDemandConfig(noise_family="lognormal")

        # Negative binomial with zero or negative dispersion
        with pytest.raises(ValueError, match="noise_dispersion must be strictly positive"):
            DailyDemandConfig(noise_family="negative_binomial", noise_dispersion=0.0)

        # Extreme negative trend causing negative expected demand
        with pytest.raises(ValueError, match="causes negative expected demand"):
            DailyDemandConfig(base_demand=100.0, trend_slope=-5.0, n_days=50)

    def test_deterministic_fixed_noise_mode(self) -> None:
        """Tests that fixed mode computes deterministic expectation exactly."""
        seasonality = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        cfg = DailyDemandConfig(
            n_days=10,
            base_demand=500.0,
            trend_slope=10.0,
            weekly_seasonality=seasonality,
            noise_family="fixed",
        )
        df = generate_daily_demand(cfg)

        expected = 500.0 + 10.0 * np.arange(10)
        assert np.array_equal(df["requests"].to_numpy(), expected.astype(np.int64))

    def test_negative_binomial_noise_generation(self) -> None:
        """Tests generation using Negative Binomial discrete noise."""
        cfg = DailyDemandConfig(
            n_days=30,
            base_demand=800.0,
            trend_slope=1.0,
            noise_family="negative_binomial",
            noise_dispersion=0.2,
            seed=77,
        )
        df = generate_daily_demand(cfg)
        assert len(df) == 30
        assert np.all(df["requests"].to_numpy() >= 0)
