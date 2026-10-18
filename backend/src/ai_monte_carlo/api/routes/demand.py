"""API routes for predictive demand modeling and integrated experiments."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from ai_monte_carlo.api.schemas.demand import (
    DemandExperimentRequest,
    DemandExperimentResponse,
    DemandForecastRequest,
    DemandForecastResponse,
    HorizonForecastSchema,
    ModelComparisonSchema,
)
from ai_monte_carlo.data.demand import generate_daily_demand
from ai_monte_carlo.models.demand import (
    PoissonDemandModel,
    forecast_demand_horizon,
)
from ai_monte_carlo.services.experiment import run_demand_experiment

router = APIRouter(prefix="/api/v1/demand", tags=["Predictive Demand"])


@router.post(
    "/experiment",
    response_model=DemandExperimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute integrated Demand GLM + Monte Carlo experiment",
    description=(
        "Executes Phase 2 pipeline: generates daily historical demand, performs "
        "chronological train/test split (no future data leakage), trains Poisson GLM and "
        "baseline reference, evaluates out-of-sample metrics (MAE, RMSE, deviance), projects "
        "the future horizon, transfers predicted demand to Monte Carlo engine via immutable "
        "configuration transfer, and runs simulation."
    ),
)
def run_experiment(request: DemandExperimentRequest) -> DemandExperimentResponse:
    """Runs the integrated predictive modeling and simulation pipeline."""
    t0 = time.perf_counter()
    try:
        domain_config = request.to_domain()
        result = run_demand_experiment(domain_config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    summary = result.summary_dict()
    eval_s = summary["evaluation"]
    fore_s = summary["horizon_forecast"]

    comparison = ModelComparisonSchema(
        model_mae=eval_s["model_mae"],
        model_rmse=eval_s["model_rmse"],
        model_deviance=eval_s["model_deviance"],
        baseline_mae=eval_s["baseline_mae"],
        baseline_rmse=eval_s["baseline_rmse"],
        mae_reduction=eval_s["mae_reduction"],
        rmse_reduction=eval_s["rmse_reduction"],
        model_beats_baseline=eval_s["model_beats_baseline"],
        test_sample_size=eval_s["test_sample_size"],
    )

    horizon_schema = HorizonForecastSchema(
        horizon_days=fore_s["horizon_days"],
        mean_daily_demand=fore_s["mean_daily_demand"],
        total_expected_demand=fore_s["total_expected_demand"],
        min_daily_demand=fore_s["min_daily_demand"],
        max_daily_demand=fore_s["max_daily_demand"],
        daily_forecast=result.horizon_forecast.daily_forecast.tolist(),
    )

    return DemandExperimentResponse(
        evaluation=comparison,
        horizon_forecast=horizon_schema,
        monte_carlo_simulation=summary["monte_carlo_simulation"],
        learned_coefficients=result.reproduction_info["learned_coefficients"],
        reproduction=result.reproduction_info,
        execution_time_ms=elapsed_ms,
    )


@router.post(
    "/forecast",
    response_model=DemandForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Fit Poisson GLM and forecast demand horizon",
    description=(
        "Trains a Poisson GLM on historical request volume series and projects daily request "
        "volumes over a future horizon without running Monte Carlo."
    ),
)
def run_forecast(request: DemandForecastRequest) -> DemandForecastResponse:
    """Fits Poisson GLM and forecasts future horizon request volumes."""
    t0 = time.perf_counter()
    try:
        demand_cfg = (
            request.demand_config.to_domain()
            if request.demand_config is not None
            else None
        )
        history = generate_daily_demand(demand_cfg)

        model = PoissonDemandModel(alpha=request.model_alpha)
        model.fit(history)

        last_date = pd.to_datetime(history["date"].iloc[-1])
        start_date = last_date + pd.Timedelta(days=1)
        start_time_idx = int(history["time_index"].iloc[-1]) + 1

        forecast_res = forecast_demand_horizon(
            model=model,
            start_date=start_date,
            horizon_days=request.horizon_days,
            start_time_index=start_time_idx,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return DemandForecastResponse(
        horizon_days=forecast_res.horizon_days,
        mean_daily_demand=forecast_res.mean_daily_demand,
        total_expected_demand=forecast_res.total_expected_demand,
        min_daily_demand=forecast_res.min_daily_demand,
        max_daily_demand=forecast_res.max_daily_demand,
        daily_forecast=forecast_res.daily_forecast.tolist(),
        learned_coefficients=model.get_coefficients(),
        execution_time_ms=elapsed_ms,
    )
