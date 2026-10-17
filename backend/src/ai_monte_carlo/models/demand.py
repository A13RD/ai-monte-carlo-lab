"""Demand predictive modeling module for AI Monte Carlo Lab.

Provides:
- Temporal feature extraction for count regression without future data leakage.
- Chronological train/test splitting preserving time-series ordering.
- Poisson GLM regression (PoissonRegressor) with log-link for strictly non-negative demand counts.
- Seasonal naive / day-of-week mean baseline reference model calculated strictly on training data.
- Metric calculations (MAE, RMSE, Poisson deviance) and honest model comparisons.
- Future horizon demand forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


@dataclass(frozen=True)
class DemandMetrics:
    """Evaluation metrics for demand forecasting models.

    Attributes:
        mae: Mean Absolute Error (sum |y - y_hat| / n).
        rmse: Root Mean Squared Error (sqrt(sum (y - y_hat)^2 / n)).
        mean_deviance: Mean Poisson deviance (canonical GLM loss for Poisson regression).
        sample_size: Number of evaluated temporal observations (n).
        mean_actual: Mean of actual target values.
        mean_predicted: Mean of predicted target values.
    """

    mae: float
    rmse: float
    mean_deviance: float
    sample_size: int
    mean_actual: float
    mean_predicted: float


@dataclass(frozen=True)
class ModelComparison:
    """Comparative evaluation between candidate model and baseline reference.

    Attributes:
        model_metrics: Evaluation metrics for the candidate predictive model.
        baseline_metrics: Evaluation metrics for the training-derived baseline reference.
        mae_reduction: Absolute reduction in MAE (baseline.mae - model.mae). Positive means improvement.
        rmse_reduction: Absolute reduction in RMSE (baseline.rmse - model.rmse). Positive means improvement.
        model_beats_baseline: True if candidate model achieves lower or equal MAE and RMSE than baseline.
    """

    model_metrics: DemandMetrics
    baseline_metrics: DemandMetrics
    mae_reduction: float
    rmse_reduction: float
    model_beats_baseline: bool


@dataclass(frozen=True)
class HorizonForecast:
    """Predicted demand across a future simulation horizon.

    Attributes:
        dates: List of ISO date strings for the horizon.
        time_indices: Array of integer temporal indices.
        daily_forecast: Array of predicted daily request counts (finite, >= 0).
        mean_daily_demand: Average predicted daily demand across the horizon.
        total_expected_demand: Cumulative sum of predicted demand across the horizon.
        min_daily_demand: Minimum predicted daily demand in the horizon.
        max_daily_demand: Maximum predicted daily demand in the horizon.
        horizon_days: Total number of days in the forecast horizon.
    """

    dates: list[str]
    time_indices: np.ndarray
    daily_forecast: np.ndarray
    mean_daily_demand: float
    total_expected_demand: float
    min_daily_demand: float
    max_daily_demand: float
    horizon_days: int

    def to_dataframe(self) -> pd.DataFrame:
        """Converts horizon predictions to a pandas DataFrame."""
        return pd.DataFrame(
            {
                "date": pd.to_datetime(self.dates),
                "time_index": self.time_indices,
                "predicted_requests": self.daily_forecast,
            }
        )


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float | int = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a time-series DataFrame chronologically into training and test sets.

    Guarantees:
    - Training data strictly precedes test data in time.
    - No shuffling or future data leakage.
    - Zero temporal overlap between sets.

    Args:
        df: Input DataFrame containing at least 'date' or 'time_index'.
        test_size: Fraction of samples for test set (if float in (0, 1))
                   or absolute number of test days (if int >= 1).

    Returns:
        tuple (train_df, test_df) as independent DataFrames.
    """
    if len(df) < 2:
        raise ValueError(f"DataFrame must contain at least 2 rows for temporal splitting, got {len(df)}")

    # Verify chronological monotonicity
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        if not dates.is_monotonic_increasing:
            raise ValueError(
                "DataFrame 'date' column must be strictly monotonically increasing. "
                "Sort by date before performing chronological split."
            )
    elif "time_index" in df.columns:
        time_idx = df["time_index"]
        if not time_idx.is_monotonic_increasing:
            raise ValueError(
                "DataFrame 'time_index' column must be strictly monotonically increasing."
            )

    n_total = len(df)
    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size as float must be in (0.0, 1.0), got {test_size}")
        n_test = int(round(n_total * test_size))
        n_test = max(1, min(n_total - 1, n_test))
    elif isinstance(test_size, (int, np.integer)):
        if test_size < 1 or test_size >= n_total:
            raise ValueError(
                f"test_size as integer must be between 1 and {n_total - 1}, got {test_size}"
            )
        n_test = int(test_size)
    else:
        raise TypeError(f"test_size must be float or int, got {type(test_size)}")

    split_idx = n_total - n_test
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df


def extract_demand_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extracts strictly causal temporal features available at prediction time.

    Features:
    - time_index: Linear continuous trend feature.
    - dow_1 to dow_6: Binary dummy indicators for Tuesday (1) through Sunday (6).
      Monday (0) serves as the reference category to prevent collinearity with the intercept.

    Args:
        df: DataFrame containing 'time_index' and 'day_of_week' (or 'date').

    Returns:
        tuple (X_matrix, feature_names)
    """
    if "time_index" not in df.columns:
        raise ValueError("DataFrame must contain 'time_index' column.")

    if "day_of_week" in df.columns:
        dow = df["day_of_week"].to_numpy(dtype=np.int64)
    elif "date" in df.columns:
        dow = pd.to_datetime(df["date"]).dt.dayofweek.to_numpy(dtype=np.int64)
    else:
        raise ValueError("DataFrame must contain 'day_of_week' or 'date' column.")

    t_idx = df["time_index"].to_numpy(dtype=np.float64)

    # One-hot encoding for days 1 to 6 (reference: Monday = 0)
    dow_dummies = np.zeros((len(df), 6), dtype=np.float64)
    for d in range(1, 7):
        dow_dummies[:, d - 1] = (dow == d).astype(np.float64)

    X = np.column_stack([t_idx, dow_dummies])
    feature_names = ["time_index", "dow_1_tue", "dow_2_wed", "dow_3_thu", "dow_4_fri", "dow_5_sat", "dow_6_sun"]

    return X, feature_names


def calculate_demand_metrics(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
) -> DemandMetrics:
    """Computes statistical evaluation metrics comparing actual vs predicted counts.

    Metrics:
    - MAE: Mean Absolute Error (sum |y - y_hat| / n).
    - RMSE: Root Mean Squared Error (sqrt(sum (y - y_hat)^2 / n)).
    - Poisson Deviance: 2 * sum( y * ln(y / y_hat) - (y - y_hat) ) / n.

    Args:
        y_true: Ground truth observed counts (non-negative).
        y_pred: Model predictions (finite, non-negative).

    Returns:
        DemandMetrics dataclass.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_pred)={len(y_p)}")
    if len(y_t) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    if np.any(np.isnan(y_t)) or np.any(np.isnan(y_p)):
        raise ValueError("Arrays contain NaN values.")
    if np.any(np.isinf(y_t)) or np.any(np.isinf(y_p)):
        raise ValueError("Arrays contain infinite values.")
    if np.any(y_t < 0) or np.any(y_p < 0):
        raise ValueError("Actuals and predictions must be non-negative.")

    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # Poisson deviance: 2 * (y * log(y / y_hat) - (y - y_hat))
    # where y * log(y / y_hat) = 0 when y = 0
    safe_pred = np.maximum(y_p, 1e-9)
    deviance_terms = np.zeros_like(y_t)
    nonzero_mask = y_t > 0
    deviance_terms[nonzero_mask] = y_t[nonzero_mask] * np.log(y_t[nonzero_mask] / safe_pred[nonzero_mask])
    deviance_terms = 2.0 * (deviance_terms - (y_t - safe_pred))
    mean_deviance = float(np.mean(deviance_terms))

    return DemandMetrics(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mean_deviance=round(mean_deviance, 4),
        sample_size=len(y_t),
        mean_actual=round(float(np.mean(y_t)), 4),
        mean_predicted=round(float(np.mean(y_p)), 4),
    )


class BaselineDemandModel:
    """Simple baseline model calculated exclusively from training data.

    Computes the historical average demand per day-of-week on the training set:
        y_hat_baseline(t) = mean_{s in Train, dow(s) = dow(t)} Y_s
    Falling back to overall training mean if a specific day is missing.
    """

    def __init__(self) -> None:
        self.dow_means: dict[int, float] = {}
        self.global_mean: float = 0.0
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> BaselineDemandModel:
        """Fits baseline model using ONLY training data.

        Args:
            train_df: Training DataFrame with 'day_of_week' (or 'date') and 'requests'.
        """
        if "requests" not in train_df.columns:
            raise ValueError("Training DataFrame must contain 'requests' column.")

        if "day_of_week" in train_df.columns:
            dow = train_df["day_of_week"].to_numpy(dtype=np.int64)
        elif "date" in train_df.columns:
            dow = pd.to_datetime(train_df["date"]).dt.dayofweek.to_numpy(dtype=np.int64)
        else:
            raise ValueError("Training DataFrame must contain 'day_of_week' or 'date'.")

        y = train_df["requests"].to_numpy(dtype=np.float64)
        if len(y) == 0:
            raise ValueError("Cannot fit baseline on empty DataFrame.")

        self.global_mean = float(np.mean(y))
        self.dow_means = {}

        for d in range(7):
            mask = dow == d
            if np.any(mask):
                self.dow_means[d] = float(np.mean(y[mask]))
            else:
                self.dow_means[d] = self.global_mean

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts demand for input DataFrame using training day-of-week averages."""
        if not self.is_fitted:
            raise RuntimeError("BaselineDemandModel must be fitted before predict.")

        if "day_of_week" in df.columns:
            dow = df["day_of_week"].to_numpy(dtype=np.int64)
        elif "date" in df.columns:
            dow = pd.to_datetime(df["date"]).dt.dayofweek.to_numpy(dtype=np.int64)
        else:
            raise ValueError("DataFrame must contain 'day_of_week' or 'date'.")

        predictions = np.array([self.dow_means.get(d, self.global_mean) for d in dow], dtype=np.float64)
        return np.maximum(0.0, predictions)


class PoissonDemandModel:
    """Predictive demand model for non-negative count series using Poisson Generalized Linear Model.

    Uses `sklearn.linear_model.PoissonRegressor` with log-link:
        E[Y | X] = exp(beta_0 + beta_1 * time_index + sum_{d=1}^6 gamma_d * I(dow = d))

    Properties:
    - Bounded: Strictly non-negative predictions without post-hoc clipping distortion.
    - Multiplicative structure: Naturally models exponential/linear trend alongside weekly seasonal factors.
    - Loss: Minimizes Poisson deviance, consistent with discrete count arrival processes.
    """

    def __init__(self, alpha: float = 1e-4, max_iter: int = 1000) -> None:
        """Initializes PoissonDemandModel.

        Args:
            alpha: L2 regularization strength (>= 0). Defaults to 1e-4 for mild regularization.
            max_iter: Maximum solver iterations.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")

        self.alpha = alpha
        self.max_iter = max_iter
        self.regressor = PoissonRegressor(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> PoissonDemandModel:
        """Fits the Poisson GLM on historical demand records.

        Args:
            df: DataFrame containing 'requests', 'time_index', and 'day_of_week' (or 'date').
        """
        if "requests" not in df.columns:
            raise ValueError("Input DataFrame must contain 'requests' column.")

        X, self.feature_names = extract_demand_features(df)
        y = df["requests"].to_numpy(dtype=np.float64)

        if len(y) == 0:
            raise ValueError("Cannot fit model on empty DataFrame.")
        if np.any(y < 0):
            raise ValueError("Target 'requests' counts must be non-negative.")

        self.regressor.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generates demand count predictions for given features.

        Guarantees:
        - Output is 1D NumPy array.
        - All predicted values are finite and non-negative.

        Args:
            df: DataFrame with 'time_index' and 'day_of_week' (or 'date').

        Returns:
            1D array of non-negative float predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("PoissonDemandModel must be fitted before predict.")

        X, _ = extract_demand_features(df)
        raw_pred = self.regressor.predict(X)

        if np.any(np.isnan(raw_pred)) or np.any(np.isinf(raw_pred)):
            raise ValueError("Model produced NaN or infinite predictions.")

        return np.maximum(0.0, np.asarray(raw_pred, dtype=np.float64))

    def get_coefficients(self) -> dict[str, float]:
        """Returns learned model coefficients and intercept."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        coefs = {name: float(coef) for name, coef in zip(self.feature_names, self.regressor.coef_)}
        coefs["intercept"] = float(self.regressor.intercept_)
        return coefs


def forecast_demand_horizon(
    model: PoissonDemandModel,
    start_date: str | pd.Timestamp,
    horizon_days: int,
    start_time_index: int,
) -> HorizonForecast:
    """Forecasts daily demand for a future simulation horizon without target values.

    Args:
        model: Fitted PoissonDemandModel.
        start_date: First date of the future forecast horizon.
        horizon_days: Number of days to forecast (>= 1).
        start_time_index: Continuous time index for day 0 of the horizon.

    Returns:
        HorizonForecast object containing daily predictions and summary statistics.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")
    if start_time_index < 0:
        raise ValueError(f"start_time_index must be non-negative, got {start_time_index}")

    dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")
    time_indices = np.arange(start_time_index, start_time_index + horizon_days, dtype=np.int64)
    day_of_weeks = dates.dayofweek.to_numpy(dtype=np.int64)

    horizon_df = pd.DataFrame(
        {
            "date": dates,
            "time_index": time_indices,
            "day_of_week": day_of_weeks,
        }
    )

    preds = model.predict(horizon_df)

    return HorizonForecast(
        dates=[d.strftime("%Y-%m-%d") for d in dates],
        time_indices=time_indices,
        daily_forecast=preds,
        mean_daily_demand=round(float(np.mean(preds)), 4),
        total_expected_demand=round(float(np.sum(preds)), 4),
        min_daily_demand=round(float(np.min(preds)), 4),
        max_daily_demand=round(float(np.max(preds)), 4),
        horizon_days=horizon_days,
    )
