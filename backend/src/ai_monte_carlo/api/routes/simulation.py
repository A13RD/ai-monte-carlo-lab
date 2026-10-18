"""API routes for Monte Carlo simulation engine."""

from __future__ import annotations

import dataclasses
import time
from fastapi import APIRouter, HTTPException, status

from ai_monte_carlo.api.schemas.simulation import (
    MetricsSummarySchema,
    SimulationRequest,
    SimulationResponse,
)
from ai_monte_carlo.simulation.monte_carlo import run_monte_carlo

router = APIRouter(prefix="/api/v1/simulation", tags=["Simulation"])


@router.get(
    "/defaults",
    response_model=SimulationRequest,
    summary="Get default simulation configuration",
    description="Returns nominal default parameters for Monte Carlo simulation.",
)
def get_simulation_defaults() -> SimulationRequest:
    """Provides default simulation request configuration."""
    return SimulationRequest()


@router.post(
    "/run",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Monte Carlo simulation",
    description=(
        "Executes a vectorized Monte Carlo simulation modeling request volume, "
        "stochastic token consumption, and monthly financial costs. Computes "
        "sample mean, sample standard deviation (ddof=1), percentiles (P5, P50, P95), "
        "and probability of exceeding budget."
    ),
)
def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """Executes a Monte Carlo simulation run with explicit, validated parameters."""
    t0 = time.perf_counter()
    try:
        domain_config = request.to_domain()
        result = run_monte_carlo(domain_config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    summary = result.summary_dict()

    cost_m = result.cost_metrics
    token_m = result.total_token_metrics

    effective_cfg = {
        "n_iterations": result.config.n_iterations,
        "budget_limit": result.config.budget_limit,
        "pricing": dataclasses.asdict(result.config.pricing),
        "input_tokens": {
            "family": result.config.input_tokens.family.value,
            "mean": result.config.input_tokens.mean,
            "std": result.config.input_tokens.std,
            "min_tokens": result.config.input_tokens.min_tokens,
            "max_tokens": result.config.input_tokens.max_tokens,
        },
        "output_tokens": {
            "family": result.config.output_tokens.family.value,
            "mean": result.config.output_tokens.mean,
            "std": result.config.output_tokens.std,
            "min_tokens": result.config.output_tokens.min_tokens,
            "max_tokens": result.config.output_tokens.max_tokens,
        },
        "workload": {
            "family": result.config.workload.family.value,
            "mean_daily_requests": result.config.workload.mean_daily_requests,
            "dispersion_parameter": result.config.workload.dispersion_parameter,
            "days_per_month": result.config.workload.days_per_month,
        },
        "seed": result.config.seed,
    }

    return SimulationResponse(
        iterations=result.config.n_iterations,
        budget_limit=result.config.budget_limit,
        probability_exceeding_budget=(
            cost_m.prob_exceed_budget if cost_m.prob_exceed_budget is not None else 0.0
        ),
        cost_metrics=MetricsSummarySchema(
            mean=cost_m.mean,
            sample_std=cost_m.sample_std,
            p5=cost_m.p5,
            p50_median=cost_m.p50,
            p95=cost_m.p95,
            min=cost_m.min_value,
            max=cost_m.max_value,
            count=cost_m.sample_size,
        ),
        token_metrics=MetricsSummarySchema(
            mean=token_m.mean,
            sample_std=token_m.sample_std,
            p5=token_m.p5,
            p50_median=token_m.p50,
            p95=token_m.p95,
            min=token_m.min_value,
            max=token_m.max_value,
            count=token_m.sample_size,
        ),
        effective_config=effective_cfg,
        summary=summary,
        execution_time_ms=elapsed_ms,
        seed=request.seed,
    )
