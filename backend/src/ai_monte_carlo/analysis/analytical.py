"""Analytical Reference Comparison Module for AI Monte Carlo Lab.

Provides exact theoretical moment formulas for compound random sums:
- Single request cost moments: E[Z] and Var(Z)
- Monthly workload count moments: E[N] and Var(N) (Poisson, Negative Binomial, Fixed, Normal)
- Monthly total cost moments via Wald's Identity and the Law of Total Variance (Eve's Law):
    E[C] = E[N] * E[Z] + C_fixed
    Var(C) = E[N] * Var(Z) + Var(N) * (E[Z])^2
- Rigorous statistical comparison against empirical Monte Carlo simulation samples
  using Monte Carlo standard error and z-score hypothesis testing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ai_monte_carlo.config import (
    DistributionType,
    MonteCarloConfig,
    PricingConfig,
    TokenDistributionConfig,
    WorkloadConfig,
    WorkloadDistributionType,
)
from ai_monte_carlo.simulation.monte_carlo import MonteCarloResult


@dataclass(frozen=True)
class AnalyticalCostMoments:
    """Theoretical expectation and variance derived from compound stochastic process theory.

    Attributes:
        expected_requests: Theoretical expected monthly request count E[N].
        variance_requests: Theoretical variance of monthly request count Var(N).
        expected_cost_per_request: Theoretical expected inference cost per request E[Z] in USD.
        variance_cost_per_request: Theoretical variance of inference cost per request Var(Z) in USD^2.
        expected_monthly_cost: Theoretical expected total monthly cost E[C] in USD (Wald's identity).
        variance_monthly_cost: Theoretical total monthly cost variance Var(C) (Law of Total Variance).
        std_monthly_cost: Theoretical total monthly cost standard deviation SD(C) in USD.
        distribution_validity: Assessment of exactness ('exact', 'exact_nominal_discrete_approx', etc.).
        assumptions: Explicit list of mathematical assumptions underlying the formula.
    """

    expected_requests: float
    variance_requests: float
    expected_cost_per_request: float
    variance_cost_per_request: float
    expected_monthly_cost: float
    variance_monthly_cost: float
    std_monthly_cost: float
    distribution_validity: str
    assumptions: list[str]


@dataclass(frozen=True)
class AnalyticalComparisonResult:
    """Statistical hypothesis test and comparative metrics between simulation and theory.

    Attributes:
        theoretical_moments: AnalyticalCostMoments computed from configuration.
        simulated_mean_cost: Empirical sample mean cost bar{C} from Monte Carlo.
        simulated_std_cost: Empirical sample standard deviation s_C from Monte Carlo.
        mc_standard_error: Monte Carlo standard error of the mean SE = s_C / sqrt(M).
        cost_mean_abs_error: Absolute difference |bar{C} - E[C]| in USD.
        cost_mean_relative_error: Relative difference |bar{C} - E[C]| / E[C].
        z_score_mean: Standardized difference z = (bar{C} - E[C]) / SE.
        z_threshold: Critical threshold used for statistical consistency (e.g. 3.0 for 99.7%).
        is_mean_consistent: True if |z_score_mean| <= z_threshold.
        simulated_mean_requests: Empirical sample mean monthly request count bar{N}.
        simulated_var_requests: Empirical sample variance of monthly request count s_N^2.
        requests_mean_rel_error: Relative error for monthly request count mean.
        requests_var_rel_error: Relative error for monthly request count variance.
        cost_std_rel_error: Relative difference between simulated and theoretical standard deviation.
        sample_size: Number of Monte Carlo iterations M evaluated.
    """

    theoretical_moments: AnalyticalCostMoments
    simulated_mean_cost: float
    simulated_std_cost: float
    mc_standard_error: float
    cost_mean_abs_error: float
    cost_mean_relative_error: float
    z_score_mean: float
    z_threshold: float
    is_mean_consistent: bool
    simulated_mean_requests: float
    simulated_var_requests: float
    requests_mean_rel_error: float
    requests_var_rel_error: float
    cost_std_rel_error: float
    sample_size: int

    def summary_dict(self) -> dict[str, Any]:
        """Provides a structured dictionary summary of the theoretical validation."""
        return {
            "is_mean_consistent": self.is_mean_consistent,
            "z_score": round(self.z_score_mean, 4),
            "z_threshold": self.z_threshold,
            "sample_size": self.sample_size,
            "cost_comparison": {
                "theoretical_mean": round(self.theoretical_moments.expected_monthly_cost, 4),
                "simulated_mean": round(self.simulated_mean_cost, 4),
                "absolute_error": round(self.cost_mean_abs_error, 4),
                "relative_error_pct": round(self.cost_mean_relative_error * 100, 4),
                "mc_standard_error": round(self.mc_standard_error, 6),
                "theoretical_std": round(self.theoretical_moments.std_monthly_cost, 4),
                "simulated_std": round(self.simulated_std_cost, 4),
                "std_relative_error_pct": round(self.cost_std_rel_error * 100, 4),
            },
            "request_count_comparison": {
                "theoretical_mean": round(self.theoretical_moments.expected_requests, 2),
                "simulated_mean": round(self.simulated_mean_requests, 2),
                "theoretical_var": round(self.theoretical_moments.variance_requests, 2),
                "simulated_var": round(self.simulated_var_requests, 2),
            },
            "validity_note": self.theoretical_moments.distribution_validity,
        }


def calculate_request_count_moments(
    workload: WorkloadConfig,
) -> tuple[float, float, str]:
    """Calculates theoretical monthly request count expectation E[N] and variance Var(N).

    Supports:
    - POISSON: E[N] = mu_N, Var(N) = mu_N
    - NEGATIVE_BINOMIAL: E[N] = mu_N, Var(N) = mu_N + alpha * mu_N^2
    - FIXED: E[N] = round(mu_N), Var(N) = 0
    - NORMAL: E[N] = mu_N, Var(N) = mu_N + alpha * mu_N^2 (continuous approximation)

    Args:
        workload: WorkloadConfig parameters.

    Returns:
        tuple (E[N], Var(N), validity_description).
    """
    mu_n = workload.expected_monthly_requests

    if workload.family == WorkloadDistributionType.POISSON:
        exp_n = mu_n
        var_n = mu_n
        validity = "Exact: Poisson process possesses reproduction property over sum of days."

    elif workload.family == WorkloadDistributionType.NEGATIVE_BINOMIAL:
        exp_n = mu_n
        # Overdispersion parameter alpha: Var = mu + alpha * mu^2
        var_n = mu_n + workload.dispersion_parameter * (mu_n**2)
        validity = (
            "Exact under specified parametrization: r = 1/alpha, p = 1 / (1 + alpha * mu_N)."
        )

    elif workload.family == WorkloadDistributionType.FIXED:
        exp_n = float(round(mu_n))
        var_n = 0.0
        validity = "Exact: Deterministic constant workload count."

    elif workload.family == WorkloadDistributionType.NORMAL:
        exp_n = mu_n
        var_n = mu_n + workload.dispersion_parameter * (mu_n**2)
        validity = "Approximation: Continuous Gaussian clipped at zero lower bound."

    else:
        raise ValueError(f"Unsupported workload family: {workload.family}")

    return exp_n, var_n, validity


def calculate_per_request_cost_moments(
    input_tokens: TokenDistributionConfig,
    output_tokens: TokenDistributionConfig,
    pricing: PricingConfig,
    token_covariance: float = 0.0,
) -> tuple[float, float]:
    """Calculates theoretical expectation E[Z] and variance Var(Z) for single request cost.

    Definition:
        Z = (T_in * R_in + T_out * R_out) / 10^6

    Formulas:
        E[Z] = (E[T_in] * R_in + E[T_out] * R_out) / 10^6
        Var(Z) = (R_in^2 * Var(T_in) + R_out^2 * Var(T_out) + 2 * R_in * R_out * Cov(T_in, T_out)) / 10^12

    Args:
        input_tokens: Input prompt token distribution config.
        output_tokens: Output completion token distribution config.
        pricing: Pricing config with rates per 1M tokens.
        token_covariance: Covariance between input and output tokens (0.0 for independent draws).

    Returns:
        tuple (E[Z], Var(Z)) in USD and USD^2.
    """
    r_in = pricing.cost_per_million_input
    r_out = pricing.cost_per_million_output

    mu_in = input_tokens.mean
    mu_out = output_tokens.mean

    var_in = input_tokens.std**2
    var_out = output_tokens.std**2

    # Expected cost per request
    expected_z = (mu_in * r_in + mu_out * r_out) / 1_000_000.0

    # Variance of cost per request
    var_numerator = (
        (r_in**2) * var_in
        + (r_out**2) * var_out
        + 2.0 * r_in * r_out * token_covariance
    )
    variance_z = var_numerator / 1_000_000_000_000.0

    return expected_z, variance_z


def calculate_nominal_analytical_moments(
    config: MonteCarloConfig,
    token_covariance: float = 0.0,
) -> AnalyticalCostMoments:
    """Computes nominal theoretical moments for monthly cost based on compound random sum theory.

    Compound Random Sum Formulas:
        E[C] = E[N] * E[Z] + C_fixed
        Var(C) = E[N] * Var(Z) + Var(N) * (E[Z])^2
        SD(C) = sqrt(Var(C))

    Caveat on Discretization and Truncation:
        Continuous distribution moments (mean and std in TokenDistributionConfig) are exact
        prior to integer rounding (round) and clipping [min_tokens, max_tokens]. If significant
        clipping occurs, the nominal formulas serve as a close approximation.

    Args:
        config: Full MonteCarloConfig.
        token_covariance: Covariance between prompt and completion tokens (default: 0.0).

    Returns:
        AnalyticalCostMoments with exact theoretical metrics.
    """
    exp_n, var_n, count_validity = calculate_request_count_moments(config.workload)
    exp_z, var_z = calculate_per_request_cost_moments(
        input_tokens=config.input_tokens,
        output_tokens=config.output_tokens,
        pricing=config.pricing,
        token_covariance=token_covariance,
    )

    # Compound sum formulas
    c_fixed = config.pricing.fixed_monthly_cost
    expected_c = exp_n * exp_z + c_fixed
    variance_c = exp_n * var_z + var_n * (exp_z**2)
    std_c = math.sqrt(variance_c)

    assumptions = [
        "Independencia entre el conteo mensual de solicitudes N y las dimensiones de cada solicitud Z_i.",
        "Independencia mutua e identica distribucion (i.i.d.) de los pares de tokens entre diferentes solicitudes.",
        (
            "Independencia entre tokens de entrada y tokens de salida por solicitud (Cov=0)"
            if token_covariance == 0.0
            else f"Covarianza controlada entre tokens de entrada y salida: Cov={token_covariance:.4f}."
        ),
        "Momentos nominales continuos de las configuraciones de tokens como aproximacion de la muestra discretizada.",
    ]

    return AnalyticalCostMoments(
        expected_requests=round(exp_n, 4),
        variance_requests=round(var_n, 4),
        expected_cost_per_request=round(exp_z, 8),
        variance_cost_per_request=round(var_z, 14),
        expected_monthly_cost=round(expected_c, 6),
        variance_monthly_cost=round(variance_c, 6),
        std_monthly_cost=round(std_c, 6),
        distribution_validity=f"{count_validity} Nota de redondeo: los momentos de tokens son nominales previos a truncamiento.",
        assumptions=assumptions,
    )


def compare_simulation_with_analytical(
    result: MonteCarloResult,
    config: MonteCarloConfig | None = None,
    z_threshold: float = 3.0,
    token_covariance: float = 0.0,
) -> AnalyticalComparisonResult:
    """Performs rigorous statistical hypothesis testing between simulation output and analytical expectation.

    Evaluation:
    - Monte Carlo Standard Error: SE = s_C / sqrt(M)
    - Standardized error (z-score): z = (bar{C} - E[C]) / SE
    - Significance test: Checks whether |z| <= z_threshold. Under Central Limit Theorem,
      for z_threshold = 3.0, the probability of exceeding the threshold by random chance
      under the null hypothesis is approximately 0.27% (p = 0.0027).

    Args:
        result: MonteCarloResult from simulation run.
        config: MonteCarloConfig. If None, inherits result.config.
        z_threshold: Critical z-score bound (default: 3.0).
        token_covariance: Target covariance between input and output tokens (default: 0.0).

    Returns:
        AnalyticalComparisonResult containing error metrics, z-scores, and test verdict.
    """
    cfg = config if config is not None else result.config
    m_samples = len(result.cost_samples)

    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be strictly positive, got {z_threshold}")

    # 1. Theoretical moments
    theor = calculate_nominal_analytical_moments(cfg, token_covariance=token_covariance)

    # 2. Simulated moments
    sim_mean_c = result.cost_metrics.mean
    sim_std_c = result.cost_metrics.sample_std

    # Monte Carlo standard error of the mean
    mc_se = sim_std_c / math.sqrt(m_samples)

    # Cost comparison
    abs_err_cost = abs(sim_mean_c - theor.expected_monthly_cost)
    rel_err_cost = (
        (abs_err_cost / theor.expected_monthly_cost)
        if theor.expected_monthly_cost != 0.0
        else abs_err_cost
    )

    z_score = (
        (sim_mean_c - theor.expected_monthly_cost) / mc_se
        if mc_se > 0.0
        else 0.0
    )
    is_consistent = bool(abs(z_score) <= z_threshold)

    # Request count moments comparison
    sim_mean_n = float(np.mean(result.request_count_samples))
    sim_var_n = float(np.var(result.request_count_samples, ddof=1))

    rel_err_n_mean = (
        abs(sim_mean_n - theor.expected_requests) / theor.expected_requests
        if theor.expected_requests != 0.0
        else 0.0
    )
    rel_err_n_var = (
        abs(sim_var_n - theor.variance_requests) / theor.variance_requests
        if theor.variance_requests != 0.0
        else (0.0 if sim_var_n == 0.0 else 1.0)
    )

    # Standard deviation relative error
    rel_err_std = (
        abs(sim_std_c - theor.std_monthly_cost) / theor.std_monthly_cost
        if theor.std_monthly_cost != 0.0
        else (0.0 if sim_std_c == 0.0 else 1.0)
    )

    return AnalyticalComparisonResult(
        theoretical_moments=theor,
        simulated_mean_cost=round(sim_mean_c, 6),
        simulated_std_cost=round(sim_std_c, 6),
        mc_standard_error=round(mc_se, 6),
        cost_mean_abs_error=round(abs_err_cost, 6),
        cost_mean_relative_error=round(rel_err_cost, 6),
        z_score_mean=round(z_score, 4),
        z_threshold=round(z_threshold, 2),
        is_mean_consistent=is_consistent,
        simulated_mean_requests=round(sim_mean_n, 4),
        simulated_var_requests=round(sim_var_n, 4),
        requests_mean_rel_error=round(rel_err_n_mean, 6),
        requests_var_rel_error=round(rel_err_n_var, 6),
        cost_std_rel_error=round(rel_err_std, 6),
        sample_size=m_samples,
    )

