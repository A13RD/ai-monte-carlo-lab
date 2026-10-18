"""Integration and unit tests for AI Monte Carlo Lab RESTful API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_monte_carlo.api.app import create_app

client = TestClient(create_app())


class TestHealthEndpoint:
    """Tests for health check and OpenAPI documentation endpoints."""

    def test_health_check_root_path(self):
        """Verifies GET /health returns 200 and expected health metadata."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "engines" in data
        assert data["engines"]["monte_carlo_engine"] == "ready"
        assert data["engines"]["predictive_demand_glm"] == "ready"
        assert data["engines"]["statistical_validation"] == "ready"
        assert "timestamp_utc" in data

    def test_health_check_versioned_path(self):
        """Verifies GET /api/v1/health returns identical status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_openapi_docs_availability(self):
        """Verifies /openapi.json schema is generated and /docs responds."""
        openapi_resp = client.get("/openapi.json")
        assert openapi_resp.status_code == 200
        schema = openapi_resp.json()
        assert schema["info"]["title"] == "AI Monte Carlo Lab API"
        assert "/api/v1/simulation/run" in schema["paths"]
        assert "/api/v1/demand/experiment" in schema["paths"]

        docs_resp = client.get("/docs")
        assert docs_resp.status_code == 200


class TestSimulationEndpoints:
    """Tests for Monte Carlo simulation endpoints."""

    def test_get_simulation_defaults(self):
        """Verifies GET /api/v1/simulation/defaults returns valid default parameters."""
        response = client.get("/api/v1/simulation/defaults")
        assert response.status_code == 200
        data = response.json()
        assert data["n_iterations"] == 10_000
        assert data["budget_limit"] == 500.0
        assert data["pricing"]["cost_per_million_input"] == 2.50
        assert data["pricing"]["cost_per_million_output"] == 10.00
        assert data["workload"]["family"] == "poisson"

    def test_run_simulation_nominal_execution(self):
        """Verifies POST /api/v1/simulation/run executes real engine and returns valid metrics."""
        payload = {
            "n_iterations": 2_000,
            "budget_limit": 300.0,
            "pricing": {
                "cost_per_million_input": 2.50,
                "cost_per_million_output": 10.00,
                "fixed_monthly_cost": 15.00,
            },
            "input_tokens": {
                "family": "lognormal",
                "mean": 600.0,
                "std": 250.0,
                "min_tokens": 10,
            },
            "output_tokens": {
                "family": "lognormal",
                "mean": 350.0,
                "std": 150.0,
                "min_tokens": 10,
            },
            "workload": {
                "family": "poisson",
                "mean_daily_requests": 500.0,
                "days_per_month": 30,
            },
            "seed": 42,
        }
        response = client.post("/api/v1/simulation/run", json=payload)
        assert response.status_code == 200
        res = response.json()

        assert res["iterations"] == 2_000
        assert res["budget_limit"] == 300.0
        assert 0.0 <= res["probability_exceeding_budget"] <= 1.0
        assert res["execution_time_ms"] > 0

        # Cost metrics validation
        cost_m = res["cost_metrics"]
        assert cost_m["mean"] > 0
        assert cost_m["sample_std"] >= 0
        assert cost_m["p5"] <= cost_m["p50_median"] <= cost_m["p95"]
        assert cost_m["min"] <= cost_m["max"]
        assert cost_m["count"] == 2_000

        # Token metrics validation
        token_m = res["token_metrics"]
        assert token_m["mean"] > 0
        assert token_m["count"] == 2_000

        # Wald's Identity Theoretical check:
        # E[N] = 500 * 30 = 15,000
        # E[C] = 15,000 * (600 * 2.50e-6 + 350 * 10.00e-6) + 15.00
        # E[C] = 15,000 * (0.0015 + 0.0035) + 15 = 15,000 * 0.0050 + 15 = 75 + 15 = 90.00 USD
        expected_cost = 90.00
        assert pytest.approx(cost_m["mean"], rel=0.03) == expected_cost

    def test_simulation_reproducibility_via_seed(self):
        """Verifies deterministic reproducibility when providing explicit seed."""
        payload = {
            "n_iterations": 1_000,
            "budget_limit": 200.0,
            "seed": 9999,
        }
        res1 = client.post("/api/v1/simulation/run", json=payload).json()
        res2 = client.post("/api/v1/simulation/run", json=payload).json()

        assert res1["cost_metrics"]["mean"] == res2["cost_metrics"]["mean"]
        assert res1["cost_metrics"]["sample_std"] == res2["cost_metrics"]["sample_std"]
        assert res1["probability_exceeding_budget"] == res2["probability_exceeding_budget"]

    def test_simulation_different_seeds_yield_different_results(self):
        """Verifies different seeds yield statistically distinct runs."""
        res1 = client.post(
            "/api/v1/simulation/run",
            json={"n_iterations": 500, "seed": 101},
        ).json()
        res2 = client.post(
            "/api/v1/simulation/run",
            json={"n_iterations": 500, "seed": 202},
        ).json()

        assert res1["cost_metrics"]["mean"] != res2["cost_metrics"]["mean"]

    @pytest.mark.parametrize(
        "invalid_patch,expected_status",
        [
            ({"n_iterations": 1}, 422),  # M < 2
            ({"budget_limit": -100.0}, 422),  # negative budget
            ({"pricing": {"cost_per_million_input": -5.0}}, 422),  # negative tariff
            ({"pricing": {"fixed_monthly_cost": -20.0}}, 422),  # negative fixed cost
            ({"input_tokens": {"mean": 0.0}}, 422),  # mean must be > 0
            ({"input_tokens": {"min_tokens": 100, "max_tokens": 50}}, 422),  # max < min
            ({"input_tokens": {"family": "unsupported_distribution"}}, 422),  # bad enum
            ({"workload": {"mean_daily_requests": -10.0}}, 422),  # negative requests
            ({"workload": {"dispersion_parameter": 0.0}}, 422),  # dispersion <= 0
            ({"workload": {"days_per_month": 0}}, 422),  # days < 1
        ],
    )
    def test_simulation_validation_errors(self, invalid_patch, expected_status):
        """Verifies controlled HTTP 422 validation error for out-of-bounds parameters."""
        response = client.post("/api/v1/simulation/run", json=invalid_patch)
        assert response.status_code == expected_status


class TestDemandEndpoints:
    """Tests for predictive demand modeling and experiment endpoints."""

    def test_run_demand_forecast(self):
        """Verifies POST /api/v1/demand/forecast fits Poisson GLM and projects future horizon."""
        payload = {
            "demand_config": {
                "n_days": 90,
                "start_date": "2026-01-01",
                "base_demand": 800.0,
                "trend_slope": 3.0,
                "noise_family": "poisson",
                "seed": 42,
            },
            "horizon_days": 15,
            "seed": 42,
        }
        response = client.post("/api/v1/demand/forecast", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["horizon_days"] == 15
        assert len(data["daily_forecast"]) == 15
        assert data["mean_daily_demand"] > 800.0  # due to positive trend
        assert data["total_expected_demand"] > 0
        assert "time_index" in data["learned_coefficients"]

    def test_run_demand_experiment_end_to_end(self):
        """Verifies POST /api/v1/demand/experiment executes full Phase 2 pipeline."""
        payload = {
            "demand_config": {
                "n_days": 90,
                "start_date": "2026-01-01",
                "base_demand": 500.0,
                "trend_slope": 1.5,
                "noise_family": "poisson",
                "seed": 42,
            },
            "monte_carlo_config": {
                "n_iterations": 1_000,
                "budget_limit": 150.0,
                "seed": 42,
            },
            "test_size": 0.2,
            "forecast_days": 30,
            "seed": 42,
        }
        response = client.post("/api/v1/demand/experiment", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Evaluation checks
        eval_data = data["evaluation"]
        assert eval_data["model_mae"] > 0
        assert eval_data["model_rmse"] > 0
        assert eval_data["baseline_mae"] > 0
        assert eval_data["test_sample_size"] == 18  # 20% of 90 days

        # Horizon forecast checks
        forecast_data = data["horizon_forecast"]
        assert forecast_data["horizon_days"] == 30
        assert len(forecast_data["daily_forecast"]) == 30
        assert forecast_data["mean_daily_demand"] > 0

        # Monte Carlo integrated result checks
        mc_data = data["monte_carlo_simulation"]
        assert "cost_usd" in mc_data
        assert mc_data["cost_usd"]["mean"] > 0
        assert "probability_exceeding_budget" in mc_data

    @pytest.mark.parametrize(
        "invalid_demand_patch",
        [
            {"demand_config": {"n_days": 5}},  # < 7 days
            {"demand_config": {"base_demand": -100.0}},  # negative base demand
            {"test_size": 1.5},  # test_size > 1.0
            {"test_size": 0.0},  # test_size <= 0
            {"forecast_days": 0},  # forecast_days < 1
            {"model_alpha": -0.01},  # negative regularization
        ],
    )
    def test_demand_validation_errors(self, invalid_demand_patch):
        """Verifies validation errors for invalid demand experiment parameters."""
        response = client.post("/api/v1/demand/experiment", json=invalid_demand_patch)
        assert response.status_code == 422
