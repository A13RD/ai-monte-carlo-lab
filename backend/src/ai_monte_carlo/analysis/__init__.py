"""Analysis package for AI Monte Carlo Lab.

Provides Phase 3 statistical validation:
- Convergence analysis of Monte Carlo estimators
- One-At-a-Time (OAT) sensitivity analysis and tornado rankings
- Correlation estimation and joint bivariate sampling
- Analytical reference moment calculations and hypothesis testing
"""

from ai_monte_carlo.analysis.analytical import (
    AnalyticalComparisonResult,
    AnalyticalCostMoments,
    calculate_nominal_analytical_moments,
    calculate_per_request_cost_moments,
    calculate_request_count_moments,
    compare_simulation_with_analytical,
)
from ai_monte_carlo.analysis.convergence import (
    ConvergenceAnalysisResult,
    ConvergenceCheckpoint,
    analyze_convergence,
    calculate_wilson_score_interval,
    generate_adaptive_checkpoints,
)
from ai_monte_carlo.analysis.correlation import (
    DEFAULT_CORRELATION_PAIRS,
    CorrelationAnalysisResult,
    CorrelationPairResult,
    analyze_dataframe_correlations,
    calculate_bivariate_correlations,
    generate_synthetic_requests_with_correlation,
    sample_joint_tokens,
)
from ai_monte_carlo.analysis.sensitivity import (
    SUPPORTED_PARAMETERS,
    ParameterSensitivitySummary,
    ParameterVariation,
    SensitivityAnalysisResult,
    run_sensitivity_analysis,
)

__all__ = [
    # Convergence
    "ConvergenceCheckpoint",
    "ConvergenceAnalysisResult",
    "calculate_wilson_score_interval",
    "generate_adaptive_checkpoints",
    "analyze_convergence",
    # Sensitivity
    "SUPPORTED_PARAMETERS",
    "ParameterVariation",
    "ParameterSensitivitySummary",
    "SensitivityAnalysisResult",
    "run_sensitivity_analysis",
    # Correlation
    "DEFAULT_CORRELATION_PAIRS",
    "CorrelationPairResult",
    "CorrelationAnalysisResult",
    "calculate_bivariate_correlations",
    "analyze_dataframe_correlations",
    "sample_joint_tokens",
    "generate_synthetic_requests_with_correlation",
    # Analytical
    "AnalyticalCostMoments",
    "AnalyticalComparisonResult",
    "calculate_request_count_moments",
    "calculate_per_request_cost_moments",
    "calculate_nominal_analytical_moments",
    "compare_simulation_with_analytical",
]

