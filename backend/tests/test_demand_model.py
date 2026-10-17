"""Tests for demand predictive modeling and evaluation (Phase 2)."""

import numpy as np
import pandas as pd
import pytest

from ai_monte_carlo.data.demand import DailyDemandConfig, generate_daily_demand
from ai_monte_carlo.models.demand import (
    BaselineDemandModel,
    DemandMetrics,
    PoissonDemandModel,
    calculate_demand_metrics,
    extract_demand_features,
    forecast_demand_horizon,
    temporal_train_test_split,
)


class TestTemporalTrainTestSplit:
    """Test suite for Requirement 5: Chronological temporal splitting."""

    def test_chronological_split_strict_ordering_and_no_leakage(self) -> None:
        """Requirement 5: Train strictly precedes test; no overlap or temporal leakage."""
        df = generate_daily_demand(DailyDemandConfig(n_days=100, seed=42))
        train_df, test_df = temporal_train_test_split(df, test_size=0.25)

        assert len(train_df) == 75
        assert len(test_df) == 25
        assert len(train_df) + len(test_df) == len(df)

        # Chronological boundary validation: train max date < test min date
        assert train_df["date"].max() < test_df["date"].min()
        assert train_df["time_index"].max() < test_df["time_index"].min()
        assert train_df["time_index"].max() + 1 == test_df["time_index"].min()

        # Check there is zero overlap between index sets
        train_dates = set(train_df["date"])
        test_dates = set(test_df["date"])
        assert len(train_dates.intersection(test_dates)) == 0

    def test_integer_test_size(self) -> None:
        """Verifies splitting with an exact integer day count."""
        df = generate_daily_demand(DailyDemandConfig(n_days=50, seed=42))
        train_df, test_df = temporal_train_test_split(df, test_size=14)

        assert len(train_df) == 36
        assert len(test_df) == 14
        assert train_df["date"].max() < test_df["date"].min()

    def test_non_monotonic_dates_raise_error(self) -> None:
        """Ensures that unordered data fails prevention checks to avoid subtle leakage."""
        df = generate_daily_demand(DailyDemandConfig(n_days=20, seed=42))
        # Shuffle rows
        shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            temporal_train_test_split(shuffled, test_size=0.2)

    def test_invalid_split_parameters_raise(self) -> None:
        """Tests invalid split bounds."""
        df = generate_daily_demand(DailyDemandConfig(n_days=10, seed=42))

        with pytest.raises(ValueError, match="test_size as float must be in"):
            temporal_train_test_split(df, test_size=0.0)
        with pytest.raises(ValueError, match="test_size as float must be in"):
            temporal_train_test_split(df, test_size=1.0)
        with pytest.raises(ValueError, match="test_size as integer must be between"):
            temporal_train_test_split(df, test_size=10)


class TestPoissonDemandModel:
    """Test suite for Requirements 6 and 9: Poisson GLM predictive model."""

    def test_finite_and_non_negative_predictions(self) -> None:
        """Requirement 6: Predictions must strictly be finite and non-negative."""
        df = generate_daily_demand(DailyDemandConfig(n_days=90, seed=42))
        train_df, test_df = temporal_train_test_split(df, test_size=0.2)

        model = PoissonDemandModel(alpha=1e-4)
        model.fit(train_df)

        preds = model.predict(test_df)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(test_df)
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0.0)

    def test_unfitted_model_raises_error(self) -> None:
        """Calling predict before fit must raise an exception."""
        df = generate_daily_demand(DailyDemandConfig(n_days=10, seed=42))
        model = PoissonDemandModel()
        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            model.predict(df)

    def test_feature_extraction(self) -> None:
        """Verifies feature extraction shapes and columns."""
        df = generate_daily_demand(DailyDemandConfig(n_days=14, seed=42))
        X, names = extract_demand_features(df)

        assert X.shape == (14, 7)  # time_index + 6 day dummies
        assert "time_index" in names
        assert len(names) == 7


class TestDemandEvaluationMetrics:
    """Test suite for Requirement 7: Evaluation metrics calculation."""

    def test_exact_metric_calculations(self) -> None:
        """Requirement 7: Computes MAE, RMSE, and deviance accurately on known data."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 30.0])

        metrics = calculate_demand_metrics(y_true, y_pred)

        # Absolute errors: |10-12|=2, |20-18|=2, |30-30|=0 -> MAE = 4 / 3 = 1.3333
        assert pytest.approx(metrics.mae, abs=1e-3) == 4.0 / 3.0
        # Squared errors: 4 + 4 + 0 = 8 -> RMSE = sqrt(8 / 3) = 1.6330
        assert pytest.approx(metrics.rmse, abs=1e-3) == np.sqrt(8.0 / 3.0)
        assert metrics.sample_size == 3
        assert metrics.mean_actual == 20.0
        assert metrics.mean_predicted == 20.0

    def test_metrics_with_zero_actuals(self) -> None:
        """Ensures that actual zero counts do not cause divide-by-zero or NaN in metrics."""
        y_true = np.array([0.0, 10.0, 0.0])
        y_pred = np.array([1.0, 9.0, 0.5])

        metrics = calculate_demand_metrics(y_true, y_pred)
        assert np.isfinite(metrics.mae)
        assert np.isfinite(metrics.rmse)
        assert np.isfinite(metrics.mean_deviance)

    def test_metrics_invalid_inputs_raise(self) -> None:
        """Validates that NaN, inf, or mismatched inputs raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_demand_metrics([1, 2], [1])

        with pytest.raises(ValueError, match="Cannot compute metrics on empty arrays"):
            calculate_demand_metrics([], [])

        with pytest.raises(ValueError, match="Arrays contain NaN"):
            calculate_demand_metrics([1.0, np.nan], [1.0, 2.0])

        with pytest.raises(ValueError, match="Actuals and predictions must be non-negative"):
            calculate_demand_metrics([-5.0, 10.0], [5.0, 10.0])


class TestBaselineComparison:
    """Test suite for Requirement 8: Comparison against training-derived baseline reference."""

    def test_comparison_against_baseline_reference(self) -> None:
        """Requirement 8: Evaluates candidate model vs simple baseline computed strictly on train."""
        # Generate series with distinct trend and seasonality
        df = generate_daily_demand(
            DailyDemandConfig(
                n_days=180,
                base_demand=1000.0,
                trend_slope=3.0,
                weekly_seasonality=(1.05, 1.10, 1.12, 1.08, 0.95, 0.70, 0.60),
                seed=42,
            )
        )
        train_df, test_df = temporal_train_test_split(df, test_size=0.2)

        # Baseline fitted solely on training data
        baseline = BaselineDemandModel().fit(train_df)
        model = PoissonDemandModel(alpha=1e-4).fit(train_df)

        y_test = test_df["requests"].to_numpy()
        pred_base = baseline.predict(test_df)
        pred_model = model.predict(test_df)

        metrics_base = calculate_demand_metrics(y_test, pred_base)
        metrics_model = calculate_demand_metrics(y_test, pred_model)

        # Because series has trend, Poisson model should capture trend + seasonality
        # while baseline only captures training-period seasonality without trend.
        assert metrics_model.mae < metrics_base.mae
        assert metrics_model.rmse < metrics_base.rmse

    def test_baseline_uses_strictly_training_data(self) -> None:
        """Confirms baseline averages match training day-of-week averages exactly."""
        df = generate_daily_demand(DailyDemandConfig(n_days=30, seed=42))
        train_df, test_df = temporal_train_test_split(df, test_size=10)

        baseline = BaselineDemandModel().fit(train_df)

        for d in range(7):
            d_train_subset = train_df[train_df["day_of_week"] == d]
            if len(d_train_subset) > 0:
                expected_mean = float(d_train_subset["requests"].mean())
                assert pytest.approx(baseline.dow_means[d], rel=1e-5) == expected_mean


class TestFutureHorizonForecasting:
    """Test suite for Requirement 9: Generating future horizon without target values."""

    def test_future_horizon_generation_without_leakage(self) -> None:
        """Requirement 9: Forecast horizon is generated purely from future calendar features."""
        df = generate_daily_demand(DailyDemandConfig(n_days=60, seed=42))
        model = PoissonDemandModel().fit(df)

        last_date = df["date"].iloc[-1]
        start_horizon = last_date + pd.Timedelta(days=1)
        start_idx = int(df["time_index"].iloc[-1]) + 1

        horizon = forecast_demand_horizon(
            model=model,
            start_date=start_horizon,
            horizon_days=30,
            start_time_index=start_idx,
        )

        assert horizon.horizon_days == 30
        assert len(horizon.daily_forecast) == 30
        assert len(horizon.dates) == 30
        assert horizon.dates[0] == start_horizon.strftime("%Y-%m-%d")
        assert horizon.time_indices[0] == start_idx
        assert horizon.time_indices[-1] == start_idx + 29

        # Confirm non-negative and finite
        assert np.all(np.isfinite(horizon.daily_forecast))
        assert np.all(horizon.daily_forecast >= 0)
        assert horizon.mean_daily_demand > 0
        assert horizon.total_expected_demand > 0

        # DataFrame export
        hdf = horizon.to_dataframe()
        assert len(hdf) == 30
        assert "predicted_requests" in hdf.columns
