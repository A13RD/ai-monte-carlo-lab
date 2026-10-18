"""API routes aggregator."""

from fastapi import APIRouter

from ai_monte_carlo.api.routes.demand import router as demand_router
from ai_monte_carlo.api.routes.health import router as health_router
from ai_monte_carlo.api.routes.simulation import router as simulation_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(simulation_router)
api_router.include_router(demand_router)

__all__ = ["api_router", "health_router", "simulation_router", "demand_router"]
