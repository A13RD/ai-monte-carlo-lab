"""Common API schemas and error response models."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthStatusResponse(BaseModel):
    """Health check response schema indicating system readiness."""

    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="0.1.0", description="API and package version")
    environment: str = Field(default="production", description="Runtime environment")
    engines: dict[str, str] = Field(
        default_factory=lambda: {
            "monte_carlo_engine": "ready",
            "predictive_demand_glm": "ready",
            "statistical_validation": "ready",
        },
        description="Readiness status of internal computational engines",
    )
    timestamp_utc: str = Field(description="ISO-8601 UTC timestamp of the health check")


class ErrorDetail(BaseModel):
    """Structured error details for client-facing exceptions."""

    detail: str = Field(description="Human-readable explanation of the error")
    error_type: str = Field(description="Exception or validation class name")
    context: dict[str, Any] | None = Field(
        default=None, description="Optional extra diagnostic details"
    )
