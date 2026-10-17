"""Unit tests for analytical reference comparison module."""

import math
import numpy as np
import pytest

from ai_monte_carlo.analysis.analytical import (
    AnalyticalComparisonResult,
    AnalyticalCostMoments,
    calculate_nominal_analytical_moments,
    calculate_per_request_cost_moments,
    calculate_request_count_moments,
    compare_simulation_with_analytical,
)
from ai_monte_carlo.config import (
    DistributionType,
    MonteCarloConfig,
    PricingConfig,
    TokenDistributionConfig,
    WorkloadConfig,
    WorkloadDistributionType,
)
from ai_monte_carlo.simulation.monte_carlo import run_monte_carlo


class TestAnalyticalFormulas:
    """Tests for exact theoretical moment calculations."""

    def test_manual_controlled_case_exact_values(self):
        """Verifies exact manual calculation of E[Z], Var(Z), E[C], and Var(C).

        Setup:
            Requests N: Fixed at 1,000 monthly requests (Var(N) = 0).
            Input tokens: mean=500, std=0 (Var(T_in) = 0).
            Output tokens: mean=250, std=0 (Var(T_out) = 0).
            Pricing: R_in = $2.00 / 1M, R_out = $10.00 / 1M, Fixed cost = $15.00.

            Manual calculations:
                E[Z] = (500 * 2.0 + 250 * 10.0) / 1e6 = (1000 + 2500) / 1e6 = $0.0035 USD/req
                Var(Z) = 0
                E[C] = 1,000 * 0.0035 + 15.00 = $3.50 + $15.00 = $18.50 USD
                Var(C) = 1,000 * 0 + 0 * (0.0035)^2 = 0
        """
        config = MonteCarloConfig(
            n_iterations=100,
            pricing=PricingConfig(
                cost_per_million_input=2.00,
                cost_per_million_output=10.00,
                fixed_monthly_cost=15.00,
            ),
            input_tokens=TokenDistributionConfig(
                family=DistributionType.NORMAL, mean=500.0, std=0.0
            ),
            output_tokens=TokenDistributionConfig(
                family=DistributionType.NORMAL, mean=250.0, std=0.0
            ),
            workload=WorkloadConfig(
                family=WorkloadDistributionType.FIXED,
                mean_daily_requests=1000.0 / 30.0,
                days_per_month=30,
            ),
        )

        moments = calculate_nominal_analytical_moments(config)

        assert moments.expected_requests == pytest.approx(1000.0, abs=1e-4)
        assert moments.variance_requests == pytest.approx(0.0, abs=1e-4)
        assert moments.expected_cost_per_request == pytest.approx(0.0035, abs=1e-6)
        assert moments.variance_cost_per_request == pytest.approx(0.0, abs=1e-10)
        assert moments.expected_monthly_cost == pytest.approx(18.50, abs=1e-4)
        assert moments.variance_monthly_cost == pytest.approx(0.0, abs=1e-6)
        assert moments.std_monthly_cost == pytest.approx(0.0, abs=1e-6)

    def test_workload_distribution_moments(self):
        """Verifies E[N] and Var(N) for Poisson, Negative Binomial, and Fixed distributions."""
        # 1. Poisson: Var = Mean
        wl_poisson = WorkloadConfig(
            family=WorkloadDistributionType.POISSON,
            mean_daily_requests=100.0,
            days_per_month=30,
        )
        exp_p, var_p, _ = calculate_request_count_moments(wl_poisson)
        assert exp_p == 3000.0
        assert var_p == 3000.0

        # 2. Negative Binomial: Var = mu + alpha * mu^2
        wl_nb = WorkloadConfig(
            family=WorkloadDistributionType.NEGATIVE_BINOMIAL,
            mean_daily_requests=100.0,
            dispersion_parameter=0.05,
            days_per_month=30,
        )
        exp_nb, var_nb, _ = calculate_request_count_moments(wl_nb)
        assert exp_nb == 3000.0
        expected_nb_var = 3000.0 + 0.05 * (3000.0**2)  # 3000 + 0.05 * 9,000,000 = 453,000
        assert var_nb == pytest.approx(expected_nb_var, rel=1e-6)

        # 3. Fixed: Var = 0
        wl_fixed = WorkloadConfig(
            family=WorkloadDistributionType.FIXED,
            mean_daily_requests=50.0,
            days_per_month=30,
        )
        exp_fix, var_fix, _ = calculate_request_count_moments(wl_fixed)
        assert exp_fix == 1500.0
        assert var_fix == 0.0


class TestSimulationAnalyticalComparison:
    """Tests for statistical comparison between simulation and analytical models."""

    def test_statistical_consistency_hypothesis_test(self):
        """Verifies that simulation mean is statistically consistent with analytical expectation (|z| <= 3.0)."""
        config = MonteCarloConfig(
            n_iterations=2_000,
            seed=42,
            pricing=PricingConfig(
                cost_per_million_input=2.50,
                cost_per_million_output=10.00,
                fixed_monthly_cost=20.00,
            ),
            input_tokens=TokenDistributionConfig(
                family=DistributionType.NORMAL, mean=500.0, std=50.0, min_tokens=10
            ),
            output_tokens=TokenDistributionConfig(
                family=DistributionType.NORMAL, mean=300.0, std=30.0, min_tokens=10
            ),
            workload=WorkloadConfig(
                family=WorkloadDistributionType.POISSON,
                mean_daily_requests=50.0,
                days_per_month=30,
            ),
        )

        sim_result = run_monte_carlo(config)
        comp = compare_simulation_with_analytical(sim_result, z_threshold=3.0)

        assert comp.is_mean_consistent is True
        assert abs(comp.z_score_mean) <= 3.0
        assert comp.cost_mean_relative_error < 0.02
        assert comp.mc_standard_error > 0.0

        # Check standard deviation agreement
        assert comp.cost_std_rel_error < 0.08

        # Check request count moments
        assert comp.requests_mean_rel_error < 0.01

    def test_error_detection_on_tariff_unit_mismatch(self):
        """Verifies that a tariff scale error (e.g. omitting 1e6 divisor) produces an extreme z-score failure."""
        config = MonteCarloConfig(n_iterations=500, seed=42)
        sim_result = run_monte_carlo(config)

        # Create a corrupted configuration with 100x pricing error
        corrupted_pricing = PricingConfig(
            cost_per_million_input=config.pricing.cost_per_million_input * 100.0,
            cost_per_million_output=config.pricing.cost_per_million_output * 100.0,
        )
        import dataclasses
        corrupted_config = dataclasses.replace(config, pricing=corrupted_pricing)

        comp = compare_simulation_with_analytical(sim_result, config=corrupted_config, z_threshold=3.0)
        assert comp.is_mean_consistent is False
        assert abs(comp.z_score_mean) > 10.0

    def test_summary_dictionary_structure(self):
        """Verifies structure and keys of the analytical comparison summary dictionary."""
        config = MonteCarloConfig(n_iterations=200, seed=42)
        sim_result = run_monte_carlo(config)
        comp = compare_simulation_with_analytical(sim_result)

        summary = comp.summary_dict()
        assert "is_mean_consistent" in summary
        assert "z_score" in summary
        assert "cost_comparison" in summary
        assert "request_count_comparison" in summary
        assert "validity_note" in summary
        assert len(comp.theoretical_moments.assumptions) >= 3

