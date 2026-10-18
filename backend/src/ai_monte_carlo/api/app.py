"""FastAPI application factory and middleware configuration for AI Monte Carlo Lab."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_monte_carlo.api.routes import api_router


def create_app() -> FastAPI:
    """Creates and configures an instance of the AI Monte Carlo Lab FastAPI app."""
    app = FastAPI(
        title="AI Monte Carlo Lab API",
        version="0.1.0",
        description=(
            "Statistical simulation and financial cost modeling engine for Large Language Models (LLMs). "
            "Exposes endpoints for stochastic Monte Carlo runs, predictive Poisson GLM demand forecasting, "
            "and out-of-sample statistical evaluation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS configuration to facilitate future frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler for domain ValueErrors
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": str(exc),
                "error_type": "ValueError",
            },
        )

    # Mount API routes
    app.include_router(api_router)

    return app


# Default application instance for ASGI servers (uvicorn ai_monte_carlo.api.app:app)
app = create_app()
