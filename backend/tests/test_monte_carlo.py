"""Unit and mathematical validation tests for Monte Carlo simulation engine."""

import numpy as np
import pytest

from ai_monte_carlo.config import (
    DistributionType,
    MonteCarloConfig,
    PricingConfig,
    TokenDistributionConfig,
    WorkloadConfig,
    WorkloadDistributionType,
)
from ai_monte_carlo.simulation.monte_carlo import (
    MonteCarloResult,
    run_monte_carlo,
)


class TestMonteCarloSimulation:
    """Mathematical convergence and operational tests for Monte Carlo engine."""

    def test_reproducibility_with_seed(self):
        """Verifies that identical seeds produce bitwise-identical results across runs."""
        config = MonteCarloConfig(n_iterations=500, seed=42)
        res1 = run_monte_carlo(config)
        res2 = run_monte_carlo(config)

        np.testing.assert_array_equal(res1.cost_samples, res2.cost_samples)
        np.testing.assert_array_equal(res1.input_token_samples, res2.input_token_samples)
        np.testing.assert_array_equal(res1.output_token_samples, res2.output_token_samples)
        np.testing.assert_array_equal(res1.request_count_samples, res2.request_count_samples)
        assert res1.cost_metrics == res2.cost_metrics

    def test_mathematical_convergence_walds_identity(self):
        """Tests theoretical expectation convergence via Wald's Identity on compound Poisson process.

        Given:
            E[N] = 100 requests/day * 30 days = 3000 monthly requests.
            E[T_in] = 500 tokens
            E[T_out] = 250 tokens
            R_in = $2.00 / 1M tokens
            R_out = $10.00 / 1M tokens
            Fixed cost = $15.00

            Theoretical E[Cost]:
                E[Cost] = E[N] * ( (E[T_in] * R_in + E[T_out] * R_out) / 1e6 ) + Fixed
                E[Cost] = 3000 * ( (500 * 2.0 + 250 * 10.0) / 1e6 ) + 15.00
                        = 3000 * ( (1000 + 2500) / 1e6 ) + 15.00
                        = 3000 * ( 0.0035 ) + 15.00
                        = $10.50 + $15.00 = $25.50
        """
        config = MonteCarloConfig(
            n_iterations=5_000,
            seed=12345,
            pricing=PricingConfig(
                cost_per_million_input=2.00,
                cost_per_million_output=10.00,
                fixed_monthly_cost=15.00,
            ),
            input_tokens=TokenDistributionConfig(
                family=DistributionType.NORMAL, mean=500.0, std=50.0, min_tokens=10
            ),
            output_tokens=TokenDistributionConfig(
                family=DistributionType.NORMAL, mean=250.0, std=30.0, min_tokens=10
            ),
            workload=WorkloadConfig(
                family=WorkloadDistributionType.POISSON,
                mean_daily_requests=100.0,
                days_per_month=30,
            ),
        )

        result = run_monte_carlo(config)
        expected_cost = 25.50
        simulated_mean = result.cost_metrics.mean

        # Simulated mean should be within 1.5% of theoretical expectation
        assert simulated_mean == pytest.approx(expected_cost, rel=0.015)

    def test_percentile_ordering_and_monotonicity(self):
        """Validates that P5 <= P50 <= P95 and values stay within [min, max]."""
        config = MonteCarloConfig(n_iterations=1_000, seed=777)
        res = run_monte_carlo(config)
        m = res.cost_metrics

        assert m.min_value <= m.p5
        assert m.p5 <= m.p50
        assert m.p50 <= m.p95
        assert m.p95 <= m.max_value
        assert m.sample_std > 0.0

    def test_probability_of_exceeding_budget(self):
        """Verifies boundary conditions and median calibration for P(Cost > Budget)."""
        config = MonteCarloConfig(n_iterations=2_000, seed=888, budget_limit=0.0)
        res = run_monte_carlo(config)

        # Budget = 0.0 -> Probability must be 1.0 (since costs are strictly positive)
        assert res.cost_metrics.prob_exceed_budget == pytest.approx(1.0, rel=1e-6)

        # Very high budget -> Probability must be 0.0
        config_high = MonteCarloConfig(
            n_iterations=1_000, seed=888, budget_limit=1_000_000.0
        )
        res_high = run_monte_carlo(config_high)
        assert res_high.cost_metrics.prob_exceed_budget == pytest.approx(0.0, abs=1e-6)

        # Setting budget to the empirical median P50
        median_cost = res.cost_metrics.p50
        config_median = MonteCarloConfig(
            n_iterations=2_000, seed=888, budget_limit=median_cost
        )
        res_median = run_monte_carlo(config_median)
        # Probability of exceeding median should be very close to 0.50
        assert res_median.cost_metrics.prob_exceed_budget == pytest.approx(0.50, abs=0.03)

    def test_fixed_workload_distribution(self):
        """Verifies simulation under deterministic fixed request workload."""
        config = MonteCarloConfig(
            n_iterations=200,
            seed=42,
            workload=WorkloadConfig(
                family=WorkloadDistributionType.FIXED,
                mean_daily_requests=50.0,
                days_per_month=30,
            ),
        )
        result = run_monte_carlo(config)
        expected_requests = 1500

        assert (result.request_count_samples == expected_requests).all()
        assert result.request_count_metrics.sample_std == pytest.approx(0.0, abs=1e-6)

    def test_zero_request_demand(self):
        """Verifies edge case where request volume is zero."""
        fixed_fee = 50.0
        config = MonteCarloConfig(
            n_iterations=100,
            seed=42,
            pricing=PricingConfig(fixed_monthly_cost=fixed_fee),
            workload=WorkloadConfig(
                family=WorkloadDistributionType.FIXED,
                mean_daily_requests=0.0,
            ),
        )
        result = run_monte_carlo(config)

        assert (result.input_token_samples == 0).all()
        assert (result.output_token_samples == 0).all()
        assert (result.cost_samples == fixed_fee).all()
        assert result.cost_metrics.mean == pytest.approx(fixed_fee, rel=1e-6)
        assert result.cost_metrics.sample_std == pytest.approx(0.0, abs=1e-6)

    def test_result_export_methods(self):
        """Verifies dataframe conversion and summary dictionary export."""
        config = MonteCarloConfig(n_iterations=50, seed=42)
        result = run_monte_carlo(config)

        df = result.to_dataframe()
        assert len(df) == 50
        assert "monthly_cost_usd" in df.columns
        assert "total_tokens" in df.columns

        summary = result.summary_dict()
        assert summary["iterations"] == 50
        assert "cost_usd" in summary
        assert "mean" in summary["cost_usd"]
        assert "p5" in summary["cost_usd"]
        assert "p50_median" in summary["cost_usd"]
        assert "p95" in summary["cost_usd"]
