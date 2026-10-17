"""Sensitivity Analysis Module for AI Monte Carlo Lab.

Implements One-At-a-Time (OAT) parameter perturbation analysis,
evaluating the relative and absolute impact of key inputs on:
- Expected monthly cost (E[Cost])
- 95th Percentile cost (P95)
- Probability of exceeding budget (P(Cost > Budget))

Generates structured tabular outputs, impact rankings, and tornado-diagram data
while strictly preserving configuration immutability.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ai_monte_carlo.config import MonteCarloConfig
from ai_monte_carlo.simulation.monte_carlo import MonteCarloResult, run_monte_carlo


# Identifiers of supported sensitivity parameters
SUPPORTED_PARAMETERS = (
    "workload.mean_daily_requests",
    "input_tokens.mean",
    "input_tokens.std",
    "output_tokens.mean",
    "output_tokens.std",
    "pricing.cost_per_million_input",
    "pricing.cost_per_million_output",
    "workload.dispersion_parameter",
    "pricing.fixed_monthly_cost",
)


@dataclass(frozen=True)
class ParameterVariation:
    """Detailed outcome of a single parameter perturbation.

    Attributes:
        parameter_name: Dot-notated parameter path (e.g. 'input_tokens.mean').
        variation_label: Description ('base', '+20%', '-20%', '+25 USD').
        base_value: Original parameter value in baseline configuration.
        perturbed_value: Modified parameter value.
        delta_parameter: Absolute difference (perturbed_value - base_value).
        pct_delta_parameter: Relative percentage change in parameter (or None if base is 0).
        mean_cost: Simulated sample mean cost in USD.
        delta_mean_cost: Absolute change in mean cost (mean_cost - base_mean_cost).
        pct_delta_mean_cost: Relative percentage change in mean cost.
        p95_cost: Simulated P95 monthly cost in USD.
        delta_p95_cost: Absolute change in P95 cost (p95_cost - base_p95_cost).
        pct_delta_p95_cost: Relative percentage change in P95 cost.
        prob_exceed_budget: Empirical probability P(Cost > Budget).
        delta_prob_exceed: Absolute change in probability of exceedance (points).
        elasticity_mean: Point elasticity (% delta mean cost / % delta parameter). None if base is 0.
        normalized_impact: Absolute relative impact on mean (|delta_mean_cost| / base_mean_cost).
    """

    parameter_name: str
    variation_label: str
    base_value: float
    perturbed_value: float
    delta_parameter: float
    pct_delta_parameter: float | None
    mean_cost: float
    delta_mean_cost: float
    pct_delta_mean_cost: float
    p95_cost: float
    delta_p95_cost: float
    pct_delta_p95_cost: float
    prob_exceed_budget: float | None
    delta_prob_exceed: float | None
    elasticity_mean: float | None
    normalized_impact: float


@dataclass(frozen=True)
class ParameterSensitivitySummary:
    """Aggregated sensitivity metric for a specific parameter across high/low bounds.

    Attributes:
        parameter_name: Name of parameter evaluated.
        base_value: Nominal baseline parameter value.
        low_value: Perturbed lower value.
        high_value: Perturbed upper value.
        cost_at_low: Expected cost at lower parameter perturbation.
        cost_at_high: Expected cost at upper parameter perturbation.
        swing_mean_cost: Absolute cost range |cost_at_high - cost_at_low|.
        swing_p95_cost: Absolute P95 cost range |p95_at_high - p95_at_low|.
        max_abs_elasticity: Maximum absolute point elasticity observed for this parameter.
        rank_impact: 1-indexed rank based on swing_mean_cost.
    """

    parameter_name: str
    base_value: float
    low_value: float
    high_value: float
    cost_at_low: float
    cost_at_high: float
    swing_mean_cost: float
    swing_p95_cost: float
    max_abs_elasticity: float | None
    rank_impact: int


@dataclass(frozen=True)
class SensitivityAnalysisResult:
    """Complete collection of sensitivity perturbations, rankings, and base evaluation.

    Attributes:
        base_config: Original unmodified baseline MonteCarloConfig.
        base_result: Simulation result executed on base_config.
        variations: List of all individual ParameterVariation evaluations.
        parameter_summaries: Ranked list of ParameterSensitivitySummary (tornado ranking).
        perturbation_fraction: Relative perturbation step used (e.g. 0.20 for +/-20%).
        zero_base_step: Additive step applied to parameters starting at 0.0.
    """

    base_config: MonteCarloConfig
    base_result: MonteCarloResult
    variations: list[ParameterVariation]
    parameter_summaries: list[ParameterSensitivitySummary]
    perturbation_fraction: float
    zero_base_step: float

    def to_dataframe(self) -> pd.DataFrame:
        """Exports all evaluated perturbations to a pandas DataFrame."""
        records = []
        for v in self.variations:
            records.append(
                {
                    "parameter": v.parameter_name,
                    "label": v.variation_label,
                    "base_value": v.base_value,
                    "perturbed_value": v.perturbed_value,
                    "delta_param": v.delta_parameter,
                    "pct_delta_param": v.pct_delta_parameter,
                    "mean_cost": v.mean_cost,
                    "delta_mean_cost": v.delta_mean_cost,
                    "pct_delta_mean_cost": v.pct_delta_mean_cost,
                    "p95_cost": v.p95_cost,
                    "delta_p95_cost": v.delta_p95_cost,
                    "pct_delta_p95_cost": v.pct_delta_p95_cost,
                    "prob_exceed": v.prob_exceed_budget,
                    "delta_prob_exceed": v.delta_prob_exceed,
                    "elasticity_mean": v.elasticity_mean,
                    "normalized_impact": v.normalized_impact,
                }
            )
        return pd.DataFrame(records)

    def tornado_dataframe(self) -> pd.DataFrame:
        """Exports ranked parameter swings for rendering a Tornado diagram."""
        records = []
        for s in self.parameter_summaries:
            records.append(
                {
                    "rank": s.rank_impact,
                    "parameter": s.parameter_name,
                    "base_value": s.base_value,
                    "low_value": s.low_value,
                    "high_value": s.high_value,
                    "cost_at_low": s.cost_at_low,
                    "cost_at_high": s.cost_at_high,
                    "base_cost": self.base_result.cost_metrics.mean,
                    "swing_mean_cost": s.swing_mean_cost,
                    "swing_p95_cost": s.swing_p95_cost,
                    "max_abs_elasticity": s.max_abs_elasticity,
                }
            )
        return pd.DataFrame(records)

    def summary_dict(self) -> dict[str, Any]:
        """Provides a structured dictionary summary of the sensitivity analysis."""
        return {
            "baseline": {
                "mean_cost": self.base_result.cost_metrics.mean,
                "p95_cost": self.base_result.cost_metrics.p95,
                "prob_exceed": self.base_result.cost_metrics.prob_exceed_budget,
                "iterations": self.base_config.n_iterations,
            },
            "perturbation_settings": {
                "relative_fraction": self.perturbation_fraction,
                "zero_base_step": self.zero_base_step,
            },
            "top_drivers": [
                {
                    "rank": s.rank_impact,
                    "parameter": s.parameter_name,
                    "swing_mean_cost": s.swing_mean_cost,
                    "max_elasticity": s.max_abs_elasticity,
                }
                for s in self.parameter_summaries[:5]
            ],
        }


def _get_parameter_value(config: MonteCarloConfig, param_name: str) -> float:
    """Retrieves current numeric parameter value from a MonteCarloConfig."""
    parts = param_name.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid parameter path format: '{param_name}'. Expected 'group.attribute'.")

    group_name, attr_name = parts
    if not hasattr(config, group_name):
        raise ValueError(f"Config has no group named '{group_name}'")

    group = getattr(config, group_name)
    if not hasattr(group, attr_name):
        raise ValueError(f"Group '{group_name}' has no attribute '{attr_name}'")

    val = getattr(group, attr_name)
    if not isinstance(val, (int, float, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(val)}")

    return float(val)


def _set_parameter_value(
    config: MonteCarloConfig, param_name: str, new_value: float
) -> MonteCarloConfig:
    """Creates a new derived MonteCarloConfig with the specified parameter updated.

    Guarantees:
    - Never mutates the original configuration.
    - Uses dataclasses.replace to build immutable instances.
    """
    parts = param_name.split(".")
    group_name, attr_name = parts

    group = getattr(config, group_name)
    # Validate non-negativity for known parameters
    if attr_name in {"mean", "std", "mean_daily_requests", "cost_per_million_input", "cost_per_million_output", "fixed_monthly_cost", "dispersion_parameter"}:
        if new_value < 0:
            raise ValueError(f"Parameter '{param_name}' cannot be negative, got {new_value}")

    # Derived sub-dataclass
    new_group = dataclasses.replace(group, **{attr_name: new_value})
    # Derived root configuration
    return dataclasses.replace(config, **{group_name: new_group})


def run_sensitivity_analysis(
    base_config: MonteCarloConfig,
    parameters: Sequence[str] | None = None,
    perturbation_fraction: float = 0.20,
    zero_base_step: float = 25.0,
    base_result: MonteCarloResult | None = None,
) -> SensitivityAnalysisResult:
    """Executes a One-At-a-Time (OAT) sensitivity analysis on a Monte Carlo model.

    Evaluates high (+fraction) and low (-fraction) variations of each parameter,
    comparing expected cost, P95, and exceedance probability against the baseline.

    Args:
        base_config: Base MonteCarloConfig to perturb. Must not be mutated.
        parameters: Sequence of parameter names to test. Defaults to SUPPORTED_PARAMETERS.
        perturbation_fraction: Relative perturbation step (must be in (0, 1), default 0.20).
        zero_base_step: Additive perturbation step for parameters with base value == 0.0 (default 25.0).
        base_result: Optional pre-computed MonteCarloResult for base_config to avoid recomputation.

    Returns:
        SensitivityAnalysisResult with all variations and ranked tornado summaries.
    """
    if not (0.0 < perturbation_fraction < 1.0):
        raise ValueError(
            f"perturbation_fraction must be in (0, 1), got {perturbation_fraction}"
        )
    if zero_base_step <= 0:
        raise ValueError(f"zero_base_step must be strictly positive, got {zero_base_step}")

    target_params = list(parameters) if parameters is not None else list(SUPPORTED_PARAMETERS)
    for p in target_params:
        if p not in SUPPORTED_PARAMETERS:
            raise ValueError(f"Unsupported sensitivity parameter: '{p}'. Supported: {SUPPORTED_PARAMETERS}")

    # 1. Run or retrieve baseline simulation
    if base_result is not None:
        b_res = base_result
    else:
        b_res = run_monte_carlo(base_config)

    b_mean = b_res.cost_metrics.mean
    b_p95 = b_res.cost_metrics.p95
    b_prob = b_res.cost_metrics.prob_exceed_budget

    variations_list: list[ParameterVariation] = []
    summary_data_map: dict[str, dict[str, Any]] = {}

    for param_name in target_params:
        base_val = _get_parameter_value(base_config, param_name)

        # Determine perturbation values
        if base_val == 0.0:
            # Special case: Parameter starting at 0 (e.g. fixed_monthly_cost = 0)
            low_val = 0.0
            high_val = zero_base_step
            low_label = "base (0.0)"
            high_label = f"+{zero_base_step:.1f} (additive)"
            pert_specs = [(low_val, low_label), (high_val, high_label)]
        else:
            low_val = base_val * (1.0 - perturbation_fraction)
            high_val = base_val * (1.0 + perturbation_fraction)

            # Prevent violation of bounds (e.g. std <= 0 or dispersion <= 0)
            if param_name in {"input_tokens.std", "output_tokens.std", "workload.dispersion_parameter"}:
                low_val = max(1e-4, low_val)
            if param_name in {"input_tokens.mean", "output_tokens.mean", "workload.mean_daily_requests"}:
                low_val = max(1.0, low_val)

            low_label = f"-{int(perturbation_fraction * 100)}%"
            high_label = f"+{int(perturbation_fraction * 100)}%"
            pert_specs = [(low_val, low_label), (high_val, high_label)]

        param_runs: dict[str, MonteCarloResult] = {}

        for p_val, label in pert_specs:
            derived_cfg = _set_parameter_value(base_config, param_name, p_val)
            # Re-run Monte Carlo with identical seed (Common Random Numbers)
            sim_res = run_monte_carlo(derived_cfg)
            param_runs[label] = sim_res

            v_mean = sim_res.cost_metrics.mean
            v_p95 = sim_res.cost_metrics.p95
            v_prob = sim_res.cost_metrics.prob_exceed_budget

            delta_param = p_val - base_val
            pct_delta_param = (delta_param / base_val * 100.0) if base_val != 0.0 else None

            delta_mean = v_mean - b_mean
            pct_delta_mean = (delta_mean / b_mean * 100.0) if b_mean != 0.0 else 0.0

            delta_p95 = v_p95 - b_p95
            pct_delta_p95 = (delta_p95 / b_p95 * 100.0) if b_p95 != 0.0 else 0.0

            delta_prob = (v_prob - b_prob) if (v_prob is not None and b_prob is not None) else None

            # Elasticity: (% delta Mean Cost) / (% delta Parameter)
            if pct_delta_param is not None and pct_delta_param != 0.0:
                elasticity = pct_delta_mean / pct_delta_param
            else:
                elasticity = None

            norm_impact = abs(delta_mean) / b_mean if b_mean != 0.0 else abs(delta_mean)

            var_obj = ParameterVariation(
                parameter_name=param_name,
                variation_label=label,
                base_value=round(base_val, 6),
                perturbed_value=round(p_val, 6),
                delta_parameter=round(delta_param, 6),
                pct_delta_parameter=round(pct_delta_param, 2) if pct_delta_param is not None else None,
                mean_cost=round(v_mean, 6),
                delta_mean_cost=round(delta_mean, 6),
                pct_delta_mean_cost=round(pct_delta_mean, 2),
                p95_cost=round(v_p95, 6),
                delta_p95_cost=round(delta_p95, 6),
                pct_delta_p95_cost=round(pct_delta_p95, 2),
                prob_exceed_budget=round(v_prob, 6) if v_prob is not None else None,
                delta_prob_exceed=round(delta_prob, 6) if delta_prob is not None else None,
                elasticity_mean=round(elasticity, 4) if elasticity is not None else None,
                normalized_impact=round(norm_impact, 4),
            )
            variations_list.append(var_obj)

        # Extract low and high costs for tornado
        low_cost = param_runs[pert_specs[0][1]].cost_metrics.mean
        high_cost = param_runs[pert_specs[1][1]].cost_metrics.mean
        low_p95 = param_runs[pert_specs[0][1]].cost_metrics.p95
        high_p95 = param_runs[pert_specs[1][1]].cost_metrics.p95

        swing_mean = abs(high_cost - low_cost)
        swing_p95 = abs(high_p95 - low_p95)

        # Collect elasticities
        elasts = [
            abs(v.elasticity_mean)
            for v in variations_list
            if v.parameter_name == param_name and v.elasticity_mean is not None
        ]
        max_elast = max(elasts) if elasts else None

        summary_data_map[param_name] = {
            "base_val": base_val,
            "low_val": pert_specs[0][0],
            "high_val": pert_specs[1][0],
            "low_cost": low_cost,
            "high_cost": high_cost,
            "swing_mean": swing_mean,
            "swing_p95": swing_p95,
            "max_elast": max_elast,
        }

    # Rank parameters by descending swing on mean cost
    sorted_params = sorted(
        summary_data_map.items(),
        key=lambda item: item[1]["swing_mean"],
        reverse=True,
    )

    summaries: list[ParameterSensitivitySummary] = []
    for rank, (p_name, s_data) in enumerate(sorted_params, start=1):
        summaries.append(
            ParameterSensitivitySummary(
                parameter_name=p_name,
                base_value=round(s_data["base_val"], 6),
                low_value=round(s_data["low_val"], 6),
                high_value=round(s_data["high_val"], 6),
                cost_at_low=round(s_data["low_cost"], 6),
                cost_at_high=round(s_data["high_cost"], 6),
                swing_mean_cost=round(s_data["swing_mean"], 6),
                swing_p95_cost=round(s_data["swing_p95"], 6),
                max_abs_elasticity=round(s_data["max_elast"], 4) if s_data["max_elast"] is not None else None,
                rank_impact=rank,
            )
        )

    return SensitivityAnalysisResult(
        base_config=base_config,
        base_result=b_res,
        variations=variations_list,
        parameter_summaries=summaries,
        perturbation_fraction=perturbation_fraction,
        zero_base_step=zero_base_step,
    )

