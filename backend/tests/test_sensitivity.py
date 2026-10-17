"""Unit tests for One-At-a-Time sensitivity analysis module."""

import copy
import pytest

from ai_monte_carlo.analysis.sensitivity import (
    SUPPORTED_PARAMETERS,
    ParameterSensitivitySummary,
    ParameterVariation,
    SensitivityAnalysisResult,
    _get_parameter_value,
    _set_parameter_value,
    run_sensitivity_analysis,
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


class TestSensitivityAnalysis:
    """Tests for parameter sensitivity perturbations, ranking, and immutability."""

    @pytest.fixture
    def base_config(self) -> MonteCarloConfig:
        """Standard fixture for base simulation configuration."""
        return MonteCarloConfig(
            n_iterations=200,
            budget_limit=100.0,
            pricing=PricingConfig(
                cost_per_million_input=2.00,
                cost_per_million_output=8.00,
                fixed_monthly_cost=0.0,
            ),
            input_tokens=TokenDistributionConfig(
                family=DistributionType.NORMAL, mean=500.0, std=100.0
            ),
            output_tokens=TokenDistributionConfig(
                family=DistributionType.NORMAL, mean=250.0, std=50.0
            ),
            workload=WorkloadConfig(
                family=WorkloadDistributionType.FIXED,
                mean_daily_requests=50.0,
                days_per_month=30,
            ),
            seed=42,
        )

    def test_base_configuration_preservation_and_immutability(self, base_config):
        """Verifies that base_config is strictly preserved and not mutated during analysis."""
        original_copy = copy.deepcopy(base_config)
        result = run_sensitivity_analysis(
            base_config=base_config,
            parameters=["pricing.cost_per_million_input", "pricing.fixed_monthly_cost"],
            perturbation_fraction=0.20,
        )

        assert base_config == original_copy
        assert base_config.pricing.cost_per_million_input == original_copy.pricing.cost_per_million_input
        assert base_config.pricing.fixed_monthly_cost == 0.0

    def test_parameter_value_get_and_set_immutability(self, base_config):
        """Verifies _get_parameter_value and immutable derived config creation."""
        val = _get_parameter_value(base_config, "input_tokens.mean")
        assert val == 500.0

        derived = _set_parameter_value(base_config, "input_tokens.mean", 600.0)
        assert _get_parameter_value(derived, "input_tokens.mean") == 600.0
        assert _get_parameter_value(base_config, "input_tokens.mean") == 500.0
        assert base_config is not derived

    def test_zero_base_parameter_handling(self, base_config):
        """Verifies that parameters with base value 0.0 (fixed_monthly_cost) receive additive perturbation."""
        result = run_sensitivity_analysis(
            base_config=base_config,
            parameters=["pricing.fixed_monthly_cost"],
            zero_base_step=30.0,
        )

        variations = [v for v in result.variations if v.parameter_name == "pricing.fixed_monthly_cost"]
        assert len(variations) == 2

        high_var = [v for v in variations if "additive" in v.variation_label][0]
        assert high_var.base_value == 0.0
        assert high_var.perturbed_value == 30.0
        assert high_var.delta_parameter == 30.0
        # In a deterministic fixed workload, adding $30 fixed cost increases mean cost by exactly $30
        assert high_var.delta_mean_cost == pytest.approx(30.0, rel=1e-3)

    def test_tornado_ranking_order(self, base_config):
        """Verifies that tornado summaries are strictly ordered by descending cost swing."""
        result = run_sensitivity_analysis(
            base_config=base_config,
            parameters=[
                "pricing.cost_per_million_input",
                "pricing.cost_per_million_output",
                "output_tokens.mean",
            ],
            perturbation_fraction=0.20,
        )

        summaries = result.parameter_summaries
        assert len(summaries) == 3
        # Check ranks 1, 2, 3
        assert [s.rank_impact for s in summaries] == [1, 2, 3]
        # Check strictly decreasing or equal swing
        for i in range(len(summaries) - 1):
            assert summaries[i].swing_mean_cost >= summaries[i + 1].swing_mean_cost

    def test_elasticity_computation(self, base_config):
        """Verifies point elasticity calculation: for linear tariff pricing, elasticity should be close to tariff share."""
        result = run_sensitivity_analysis(
            base_config=base_config,
            parameters=["pricing.cost_per_million_input"],
            perturbation_fraction=0.10,
        )

        var = [v for v in result.variations if v.variation_label == "+10%"][0]
        assert var.elasticity_mean is not None
        # Input cost is (500*2) / (500*2 + 250*8) = 1000 / 3000 = 1/3 ~ 0.333
        assert var.elasticity_mean == pytest.approx(0.333, rel=0.05)

    def test_reproducibility_with_controlled_seeds(self, base_config):
        """Verifies that repeated sensitivity runs with identical seeds produce identical results."""
        res1 = run_sensitivity_analysis(base_config, parameters=["workload.mean_daily_requests"])
        res2 = run_sensitivity_analysis(base_config, parameters=["workload.mean_daily_requests"])

        assert res1.summary_dict() == res2.summary_dict()
        assert res1.to_dataframe().equals(res2.to_dataframe())

    def test_dataframe_and_tornado_exports(self, base_config):
        """Verifies export to pandas DataFrames and dictionary format."""
        result = run_sensitivity_analysis(
            base_config=base_config,
            parameters=["input_tokens.mean", "output_tokens.mean"],
        )

        df = result.to_dataframe()
        assert len(df) == 4  # 2 parameters * 2 variations
        assert "delta_mean_cost" in df.columns
        assert "normalized_impact" in df.columns

        tornado_df = result.tornado_dataframe()
        assert len(tornado_df) == 2
        assert "swing_mean_cost" in tornado_df.columns
        assert "rank" in tornado_df.columns

    def test_invalid_parameters_and_out_of_range_handling(self, base_config):
        """Verifies error handling for unsupported parameters, invalid bounds, and negative perturbations."""
        with pytest.raises(ValueError, match="Unsupported sensitivity parameter"):
            run_sensitivity_analysis(base_config, parameters=["unsupported_param"])

        with pytest.raises(ValueError, match="perturbation_fraction must be in"):
            run_sensitivity_analysis(base_config, perturbation_fraction=1.5)

        with pytest.raises(ValueError, match="zero_base_step must be strictly positive"):
            run_sensitivity_analysis(base_config, zero_base_step=-10.0)

        with pytest.raises(ValueError, match="cannot be negative"):
            _set_parameter_value(base_config, "pricing.cost_per_million_input", -5.0)

