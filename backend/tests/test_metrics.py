"""Unit and mathematical validation tests for statistical metrics module."""

import math
import numpy as np
import pytest

from ai_monte_carlo.simulation.metrics import (
    SimulationMetricsSummary,
    calculate_mean,
    calculate_percentiles,
    calculate_probability_exceeding,
    calculate_sample_std,
    summarize_metrics,
)


class TestSampleMean:
    """Mathematical tests for sample mean estimator."""

    def test_sample_mean_exact(self):
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert calculate_mean(samples) == pytest.approx(30.0, rel=1e-9)

    def test_sample_mean_single_element(self):
        assert calculate_mean([42.5]) == pytest.approx(42.5, rel=1e-9)

    def test_sample_mean_empty_raises(self):
        with pytest.raises(ValueError, match="Insufficient observations"):
            calculate_mean([])

    def test_sample_mean_with_nan_raises(self):
        with pytest.raises(ValueError, match="NaN values"):
            calculate_mean([1.0, np.nan, 3.0])

    def test_sample_mean_with_inf_raises(self):
        with pytest.raises(ValueError, match="Infinite values"):
            calculate_mean([1.0, np.inf, 3.0])


class TestSampleStd:
    """Mathematical tests for sample standard deviation with Bessel's correction (ddof=1)."""

    def test_sample_std_bessels_correction(self):
        # Sample: [2, 4, 4, 4, 5, 5, 7, 9]
        # Mean = 40 / 8 = 5.0
        # Sum of squared deviations = 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32
        # Sample variance s^2 = 32 / (8 - 1) = 32 / 7 = 4.5714285714...
        # s = sqrt(32/7) = 2.138089935...
        samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        expected_s = math.sqrt(32.0 / 7.0)
        assert calculate_sample_std(samples, ddof=1) == pytest.approx(expected_s, rel=1e-9)

    def test_sample_std_constant_is_zero(self):
        samples = [7.0, 7.0, 7.0, 7.0]
        assert calculate_sample_std(samples, ddof=1) == pytest.approx(0.0, abs=1e-9)

    def test_sample_std_single_element_raises_for_ddof_1(self):
        with pytest.raises(ValueError, match="Insufficient observations"):
            calculate_sample_std([10.0], ddof=1)

    def test_sample_std_negative_ddof_raises(self):
        with pytest.raises(ValueError, match="ddof must be non-negative"):
            calculate_sample_std([10.0, 20.0], ddof=-1)


class TestPercentiles:
    """Mathematical tests for percentiles P5, P50, P95."""

    def test_percentiles_monotonicity_and_values(self):
        # 101 points from 0 to 100
        samples = np.linspace(0.0, 100.0, 101)
        res = calculate_percentiles(samples, percentiles=[5.0, 50.0, 95.0])

        assert res["p5"] == pytest.approx(5.0, abs=1e-7)
        assert res["p50"] == pytest.approx(50.0, abs=1e-7)
        assert res["p95"] == pytest.approx(95.0, abs=1e-7)
        assert res["p5"] < res["p50"] < res["p95"]

    def test_p50_equals_median(self):
        samples = [12.0, 45.0, 7.0, 89.0, 23.0]  # sorted: 7, 12, 23, 45, 89 -> median 23
        res = calculate_percentiles(samples, percentiles=[50.0])
        assert res["p50"] == pytest.approx(23.0, abs=1e-9)

    def test_invalid_percentile_range_raises(self):
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            calculate_percentiles([1.0, 2.0, 3.0], percentiles=[105.0])


class TestProbabilityExceeding:
    """Mathematical tests for empirical probability P(X > threshold)."""

    def test_probability_exceeding_discrete_case(self):
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        # X > 30.0 gives [40, 50] -> 2 out of 5 = 0.4
        assert calculate_probability_exceeding(samples, threshold=30.0) == pytest.approx(0.4, rel=1e-9)

    def test_probability_exceeding_all_above(self):
        samples = [10.0, 20.0, 30.0]
        assert calculate_probability_exceeding(samples, threshold=5.0) == pytest.approx(1.0, rel=1e-9)

    def test_probability_exceeding_all_below(self):
        samples = [10.0, 20.0, 30.0]
        assert calculate_probability_exceeding(samples, threshold=50.0) == pytest.approx(0.0, rel=1e-9)

    def test_probability_with_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="finite real number"):
            calculate_probability_exceeding([1.0, 2.0], threshold=float("nan"))


class TestSummarizeMetrics:
    """Tests for complete summary metrics structure."""

    def test_summarize_metrics_structure(self):
        samples = [100.0, 200.0, 300.0, 400.0, 500.0]
        summary = summarize_metrics(samples, threshold=250.0)

        assert isinstance(summary, SimulationMetricsSummary)
        assert summary.sample_size == 5
        assert summary.mean == pytest.approx(300.0, rel=1e-9)
        assert summary.p50 == pytest.approx(300.0, rel=1e-9)
        assert summary.p5 <= summary.p50 <= summary.p95
        assert summary.prob_exceed_budget == pytest.approx(0.6, rel=1e-9)
        assert summary.min_value == 100.0
        assert summary.max_value == 500.0

    def test_summarize_metrics_requires_at_least_two_samples(self):
        with pytest.raises(ValueError, match="Insufficient observations: expected at least 2"):
            summarize_metrics([50.0])
