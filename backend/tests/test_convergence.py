"""Unit tests for Monte Carlo convergence analysis module."""

import math
import numpy as np
import pytest

from ai_monte_carlo.analysis.convergence import (
    ConvergenceAnalysisResult,
    ConvergenceCheckpoint,
    analyze_convergence,
    calculate_wilson_score_interval,
    generate_adaptive_checkpoints,
)
from ai_monte_carlo.config import MonteCarloConfig
from ai_monte_carlo.simulation.monte_carlo import run_monte_carlo


class TestConvergenceCheckpoints:
    """Tests for adaptive checkpoint generation and sorting."""

    def test_default_adaptive_checkpoints_sorting_and_deduplication(self):
        """Verifies checkpoint generation is strictly increasing, deduplicated, and bounded."""
        checkpoints = generate_adaptive_checkpoints(total_size=7_500)
        assert checkpoints == [100, 500, 1_000, 5_000, 7_500]
        assert checkpoints[-1] == 7_500
        assert all(checkpoints[i] < checkpoints[i + 1] for i in range(len(checkpoints) - 1))

    def test_exact_match_with_nominal_checkpoint(self):
        """Verifies no duplicate checkpoint is created when total_size matches a nominal checkpoint."""
        checkpoints = generate_adaptive_checkpoints(total_size=5_000)
        assert checkpoints == [100, 500, 1_000, 5_000]
        assert len(checkpoints) == len(set(checkpoints))

    def test_small_sample_size_adaptive_steps(self):
        """Verifies that for M < 100, valid fractional checkpoints are generated."""
        checkpoints = generate_adaptive_checkpoints(total_size=50)
        assert checkpoints[-1] == 50
        assert all(2 <= cp <= 50 for cp in checkpoints)
        assert len(checkpoints) >= 2

    def test_custom_user_checkpoints_filtering(self):
        """Verifies filtering of custom checkpoints outside [2, M] and deduplication."""
        custom = [50, 10, 500, 1, 10_000, 50]  # Contains 1 (<2), 10000 (>M), duplicates
        checkpoints = generate_adaptive_checkpoints(total_size=1_000, preferred_checkpoints=custom)
        assert checkpoints == [10, 50, 500, 1_000]

    def test_total_size_under_two_raises_error(self):
        """Verifies error raised when total sample size is less than 2."""
        with pytest.raises(ValueError, match="total_size must be at least 2"):
            generate_adaptive_checkpoints(total_size=1)


class TestWilsonScoreInterval:
    """Tests for Wilson score confidence intervals on binomial proportions."""

    def test_wilson_interval_known_values(self):
        """Verifies Wilson interval on 50% successes with n=100 at 95% confidence."""
        # For p_hat = 0.5, n=100, z=1.96:
        # denominator = 1 + 1.96^2/100 = 1.038416
        # center = 0.5
        # spread = 1.96 / 1.038416 * sqrt(0.25/100 + 1.96^2/40000) ~ 0.098
        lower, upper = calculate_wilson_score_interval(count_success=50, n_trials=100, confidence_level=0.95)
        assert lower == pytest.approx(0.4038, abs=0.01)
        assert upper == pytest.approx(0.5962, abs=0.01)
        assert 0.0 <= lower <= 0.5 <= upper <= 1.0

    def test_wilson_interval_boundary_zero_successes(self):
        """Verifies that 0 successes does not degenerate to [0, 0] and stays within [0, 1]."""
        lower, upper = calculate_wilson_score_interval(count_success=0, n_trials=100, confidence_level=0.95)
        assert lower == 0.0
        assert upper > 0.0
        assert upper < 0.05  # Rule of three bound: ~3/100 = 0.03

    def test_wilson_interval_boundary_all_successes(self):
        """Verifies that 100% successes does not degenerate to [1, 1] and stays within [0, 1]."""
        lower, upper = calculate_wilson_score_interval(count_success=100, n_trials=100, confidence_level=0.95)
        assert lower > 0.95
        assert upper == 1.0

    def test_wilson_interval_invalid_inputs(self):
        """Verifies validation of trials, counts, and confidence levels."""
        with pytest.raises(ValueError, match="n_trials must be at least 1"):
            calculate_wilson_score_interval(count_success=0, n_trials=0)
        with pytest.raises(ValueError, match="count_success must be between"):
            calculate_wilson_score_interval(count_success=15, n_trials=10)
        with pytest.raises(ValueError, match="confidence_level must be in"):
            calculate_wilson_score_interval(count_success=5, n_trials=10, confidence_level=1.5)


class TestConvergenceAnalysisEstimators:
    """Tests for cumulative convergence estimators, standard errors, and confidence intervals."""

    def test_deterministic_known_sample(self):
        """Tests convergence calculations on controlled deterministic array."""
        # Samples: [100, 200, 300, 400]
        # Mean at n=4: 250
        # Sample std at n=4: sqrt((150^2 + 50^2 + 50^2 + 150^2) / 3) = sqrt(50000 / 3) ~ 129.0994
        # SE = 129.0994 / sqrt(4) = 64.5497
        samples = np.array([100.0, 200.0, 300.0, 400.0])
        result = analyze_convergence(
            samples,
            budget_limit=250.0,
            checkpoints=[2, 4],
            confidence_level=0.95,
            method_ci="normal",
        )

        assert len(result.checkpoints) == 2
        cp4 = result.checkpoints[1]

        assert cp4.checkpoint_size == 4
        assert cp4.mean == pytest.approx(250.0, abs=1e-4)
        assert cp4.sample_std == pytest.approx(np.std(samples, ddof=1), abs=1e-4)
        expected_se = np.std(samples, ddof=1) / math.sqrt(4)
        assert cp4.se_mean == pytest.approx(expected_se, abs=1e-4)

        # Budget exceedance: 300, 400 > 250 -> 2/4 = 0.5
        assert cp4.prob_exceed_budget == pytest.approx(0.50, abs=1e-4)
        assert cp4.se_prob_exceed == pytest.approx(math.sqrt(0.25 / 4), abs=1e-4)
        assert cp4.ci_prob_lower is not None
        assert cp4.ci_prob_upper is not None
        assert cp4.ci_prob_lower < 0.50 < cp4.ci_prob_upper

    def test_student_t_vs_normal_confidence_intervals(self):
        """Verifies Student-t CI is strictly wider than Normal CI for finite samples."""
        samples = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        res_t = analyze_convergence(samples, checkpoints=[6], method_ci="t")
        res_z = analyze_convergence(samples, checkpoints=[6], method_ci="normal")

        width_t = res_t.checkpoints[0].ci_mean_upper - res_t.checkpoints[0].ci_mean_lower
        width_z = res_z.checkpoints[0].ci_mean_upper - res_z.checkpoints[0].ci_mean_lower

        # For df = 5, t_crit ~ 2.571 > z_crit ~ 1.960 -> width_t > width_z
        assert width_t > width_z

    def test_standard_error_decay_rate(self):
        """Verifies Monte Carlo standard error formula and monotonic reduction with large sample increase."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=100.0, scale=20.0, size=10_000)

        res = analyze_convergence(data, checkpoints=[100, 500, 1_000, 5_000, 10_000], method_ci="normal")

        # 1. Verify exact mathematical formula at every checkpoint: SE = s / sqrt(n)
        for cp in res.checkpoints:
            expected_se = cp.sample_std / math.sqrt(cp.checkpoint_size)
            assert cp.se_mean == pytest.approx(expected_se, rel=1e-5)

        # 2. Verify asymptotic reduction
        se_100 = res.checkpoints[0].se_mean
        se_10000 = res.checkpoints[-1].se_mean
        assert se_10000 < se_100
        # Ratio should reflect order of magnitude improvement (~10x)
        ratio = se_100 / se_10000
        assert 5.0 < ratio < 15.0

    def test_reproducibility_with_controlled_seeds(self):
        """Verifies bitwise identical convergence results given identical simulation results."""
        config = MonteCarloConfig(n_iterations=500, seed=123)
        sim1 = run_monte_carlo(config)
        sim2 = run_monte_carlo(config)

        res1 = analyze_convergence(sim1)
        res2 = analyze_convergence(sim2)

        assert res1.summary_dict() == res2.summary_dict()
        assert res1.to_dataframe().equals(res2.to_dataframe())

    def test_export_methods_and_diagnostics(self):
        """Verifies DataFrame export, summary dictionary, and relative SE diagnostic."""
        samples = np.linspace(10.0, 20.0, 200)
        res = analyze_convergence(samples, budget_limit=15.0)

        df = res.to_dataframe()
        assert len(df) == len(res.checkpoints)
        assert "ci_mean_width" in df.columns
        assert "se_mean" in df.columns

        summary = res.summary_dict()
        assert summary["total_samples"] == 200
        assert "final_estimate" in summary
        assert res.relative_se_final() > 0.0

    def test_invalid_arguments_raise_errors(self):
        """Verifies input validation for sample counts, methods, and confidence levels."""
        with pytest.raises(ValueError, match="Insufficient observations"):
            analyze_convergence([42.0])

        with pytest.raises(ValueError, match="confidence_level must be in"):
            analyze_convergence([1.0, 2.0, 3.0], confidence_level=1.2)

        with pytest.raises(ValueError, match="method_ci must be 't' or 'normal'"):
            analyze_convergence([1.0, 2.0, 3.0], method_ci="invalid")
