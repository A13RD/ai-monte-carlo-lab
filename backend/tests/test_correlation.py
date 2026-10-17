"""Unit tests for correlation analysis and joint token sampling module."""

import numpy as np
import pandas as pd
import pytest

from ai_monte_carlo.analysis.correlation import (
    DEFAULT_CORRELATION_PAIRS,
    CorrelationAnalysisResult,
    CorrelationPairResult,
    analyze_dataframe_correlations,
    calculate_bivariate_correlations,
    generate_synthetic_requests_with_correlation,
    sample_joint_tokens,
)
from ai_monte_carlo.config import DistributionType, TokenDistributionConfig


class TestCorrelationCalculations:
    """Tests for Pearson and Spearman correlation estimations on known data."""

    def test_perfect_positive_linear_correlation(self):
        """Verifies r = 1.0 and rho = 1.0 for strictly positive linear relation y = 2x + 3."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 3.0
        res = calculate_bivariate_correlations(x, y, "x", "y")

        assert res.pearson_r == pytest.approx(1.0, abs=1e-5)
        assert res.spearman_rho == pytest.approx(1.0, abs=1e-5)
        assert res.pearson_p_value < 1e-4

    def test_perfect_negative_linear_correlation(self):
        """Verifies r = -1.0 and rho = -1.0 for strictly negative linear relation y = -4x + 10."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -4.0 * x + 10.0
        res = calculate_bivariate_correlations(x, y, "x", "y")

        assert res.pearson_r == pytest.approx(-1.0, abs=1e-5)
        assert res.spearman_rho == pytest.approx(-1.0, abs=1e-5)

    def test_monotonic_nonlinear_relationship(self):
        """Verifies that monotonic nonlinear function has rho = 1.0 but Pearson r < 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.exp(x)  # Strictly monotonic convex non-linear
        res = calculate_bivariate_correlations(x, y, "x", "y")

        # Spearman handles ranks: must be strictly 1.0
        assert res.spearman_rho == pytest.approx(1.0, abs=1e-5)
        # Pearson linear correlation must be strictly less than 1.0 due to exponential curvature
        assert res.pearson_r < 0.95

    def test_constant_vector_handles_zero_variance(self):
        """Verifies constant array does not trigger ZeroDivisionError and reports 0.0 correlation."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        res = calculate_bivariate_correlations(x, y, "x", "y")

        assert res.pearson_r == 0.0
        assert res.spearman_rho == 0.0
        assert res.covariance == 0.0

    def test_input_validation_and_errors(self):
        """Verifies array length mismatch and insufficient sample size raise ValueError."""
        with pytest.raises(ValueError, match="Insufficient observations"):
            calculate_bivariate_correlations([1.0, 2.0], [1.0, 2.0])

        with pytest.raises(ValueError, match="length mismatch"):
            calculate_bivariate_correlations([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])


class TestDataFrameCorrelationAnalysis:
    """Tests for DataFrame telemetry correlation scanning."""

    def test_synthetic_data_correlations_and_methodological_notes(self):
        """Verifies correlation scanning over synthetic requests DataFrame."""
        df = pd.DataFrame(
            {
                "input_tokens": [100, 200, 300, 400, 500],
                "output_tokens": [50, 40, 80, 20, 60],
                "total_tokens": [150, 240, 380, 420, 560],
                "latency_ms": [190.0, 210.0, 240.0, 260.0, 290.0],
                "cost_total_usd": [0.001, 0.002, 0.003, 0.004, 0.005],
            }
        )

        result = analyze_dataframe_correlations(df)
        assert len(result.pair_results) == len(DEFAULT_CORRELATION_PAIRS)
        assert result.sample_size == 5

        # Check export formats
        summary = result.summary_dict()
        assert "pairs" in summary
        assert len(summary["pairs"]) == len(DEFAULT_CORRELATION_PAIRS)

        table_df = result.to_dataframe()
        assert "pearson_r" in table_df.columns
        assert "spearman_rho" in table_df.columns
        assert "relationship_nature" in table_df.columns


class TestJointTokenSampling:
    """Tests for optional bivariate token generator with controlled correlation rho."""

    @pytest.fixture
    def token_configs(self):
        cfg_in = TokenDistributionConfig(
            family=DistributionType.LOGNORMAL, mean=500.0, std=200.0, min_tokens=10
        )
        cfg_out = TokenDistributionConfig(
            family=DistributionType.LOGNORMAL, mean=250.0, std=100.0, min_tokens=10
        )
        return cfg_in, cfg_out

    def test_zero_correlation_recovers_independent_behavior(self, token_configs):
        """Verifies correlation_rho=0.0 produces near-zero sample correlation on large samples."""
        cfg_in, cfg_out = token_configs
        rng = np.random.default_rng(42)

        t_in, t_out = sample_joint_tokens(rng, cfg_in, cfg_out, size=5_000, correlation_rho=0.0)
        corr_res = calculate_bivariate_correlations(t_in, t_out)

        # Correlation should be close to 0
        assert abs(corr_res.pearson_r) < 0.05
        assert (t_in >= 10).all()
        assert (t_out >= 10).all()

    def test_high_positive_correlation_sampling(self, token_configs):
        """Verifies high positive correlation rho=0.8 produces observed sample correlation close to 0.8."""
        cfg_in, cfg_out = token_configs
        rng = np.random.default_rng(42)

        t_in, t_out = sample_joint_tokens(rng, cfg_in, cfg_out, size=10_000, correlation_rho=0.80)
        corr_res = calculate_bivariate_correlations(t_in, t_out)

        # Observed correlation should be within [0.70, 0.88]
        assert corr_res.pearson_r == pytest.approx(0.80, abs=0.08)
        assert corr_res.spearman_rho > 0.70

    def test_negative_correlation_sampling(self, token_configs):
        """Verifies negative correlation rho=-0.60 produces observed negative correlation."""
        cfg_in, cfg_out = token_configs
        rng = np.random.default_rng(42)

        t_in, t_out = sample_joint_tokens(rng, cfg_in, cfg_out, size=10_000, correlation_rho=-0.60)
        corr_res = calculate_bivariate_correlations(t_in, t_out)

        assert corr_res.pearson_r == pytest.approx(-0.60, abs=0.08)
        assert corr_res.spearman_rho < -0.50

    def test_reproducibility_with_seed(self, token_configs):
        """Verifies identical seeds generate bitwise identical bivariate token samples."""
        cfg_in, cfg_out = token_configs
        rng1 = np.random.default_rng(777)
        rng2 = np.random.default_rng(777)

        in1, out1 = sample_joint_tokens(rng1, cfg_in, cfg_out, size=500, correlation_rho=0.5)
        in2, out2 = sample_joint_tokens(rng2, cfg_in, cfg_out, size=500, correlation_rho=0.5)

        np.testing.assert_array_equal(in1, in2)
        np.testing.assert_array_equal(out1, out2)

    def test_invalid_correlation_rho_raises_error(self, token_configs):
        """Verifies error raised when correlation_rho is outside [-1.0, 1.0]."""
        cfg_in, cfg_out = token_configs
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="correlation_rho must be in"):
            sample_joint_tokens(rng, cfg_in, cfg_out, size=100, correlation_rho=1.5)

        with pytest.raises(ValueError, match="correlation_rho must be in"):
            sample_joint_tokens(rng, cfg_in, cfg_out, size=100, correlation_rho=-1.2)

    def test_synthetic_requests_generation_with_correlation(self):
        """Verifies DataFrame generated with generate_synthetic_requests_with_correlation."""
        df = generate_synthetic_requests_with_correlation(n_requests=100, correlation_rho=0.5, seed=42)
        assert len(df) == 100
        assert "input_tokens" in df.columns
        assert "output_tokens" in df.columns
        assert "latency_ms" in df.columns
        assert "cost_total_usd" in df.columns

