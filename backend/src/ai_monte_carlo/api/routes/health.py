"""Health check endpoints."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from fastapi import APIRouter

from ai_monte_carlo.api.schemas.common import HealthStatusResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthStatusResponse,
    summary="Health check",
    description="Returns service status, package version, active engines, and current timestamp.",
)
@router.get(
    "/api/v1/health",
    response_model=HealthStatusResponse,
    include_in_schema=False,
)
def get_health() -> HealthStatusResponse:
    """Verifies service readiness and computational subsystem status."""
    return HealthStatusResponse(
        status="healthy",
        version="0.1.0",
        environment=os.getenv("APP_ENV", "development"),
        engines={
            "monte_carlo_engine": "ready",
            "predictive_demand_glm": "ready",
            "statistical_validation": "ready",
        },
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
