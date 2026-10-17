"""Correlation Analysis and Correlated Workload Generation Module.

Provides:
- Pearson (linear) and Spearman (rank-order monotonic) correlation estimation with p-values.
- Methodological pair-by-pair interpretations for synthetic LLM telemetry:
  * Statistical association vs generator-constructed dependency vs real-world causality.
- Optional bivariate joint sampling mechanism for (input_tokens, output_tokens)
  via Gaussian copula / bivariate log-normal sampling, preserving default
  independent behavior (correlation_rho = 0.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ai_monte_carlo.config import (
    DistributionType,
    LatencyConfig,
    MonteCarloConfig,
    PricingConfig,
    TokenDistributionConfig,
)
from ai_monte_carlo.simulation.generators import sample_latency, sample_tokens
from ai_monte_carlo.simulation.metrics import _validate_numeric_array, summarize_metrics
from ai_monte_carlo.simulation.monte_carlo import (
    MonteCarloResult,
    _aggregate_compound_tokens,
)


# Standard telemetry variable pairs recommended for investigation
DEFAULT_CORRELATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("input_tokens", "output_tokens"),
    ("input_tokens", "latency_ms"),
    ("output_tokens", "latency_ms"),
    ("total_tokens", "cost_total_usd"),
    ("latency_ms", "cost_total_usd"),
)


@dataclass(frozen=True)
class CorrelationPairResult:
    """Statistical correlation metrics and analytical rationale for a variable pair.

    Attributes:
        variable_x: Identifier of first variable.
        variable_y: Identifier of second variable.
        pearson_r: Pearson sample correlation coefficient in [-1, 1].
        pearson_p_value: Two-tailed p-value for testing non-correlation under Pearson.
        spearman_rho: Spearman rank correlation coefficient in [-1, 1].
        spearman_p_value: Two-tailed p-value for testing non-correlation under Spearman.
        covariance: Sample covariance (ddof=1).
        sample_size: Number of paired observations n.
        relationship_nature: 'independent_by_construction', 'structural_affine', 'deterministic_linear', etc.
        methodological_note: In-depth interpretation distinguishing association from causality.
    """

    variable_x: str
    variable_y: str
    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    covariance: float
    sample_size: int
    relationship_nature: str
    methodological_note: str


@dataclass(frozen=True)
class CorrelationAnalysisResult:
    """Collection of pairwise correlations and full correlation matrices.

    Attributes:
        pair_results: List of CorrelationPairResult for evaluated pairs.
        pearson_matrix: Full square Pearson correlation DataFrame across numeric columns.
        spearman_matrix: Full square Spearman correlation DataFrame across numeric columns.
        sample_size: Number of records analyzed.
    """

    pair_results: list[CorrelationPairResult]
    pearson_matrix: pd.DataFrame
    spearman_matrix: pd.DataFrame
    sample_size: int

    def to_dataframe(self) -> pd.DataFrame:
        """Converts pairwise correlation findings into a clean summary DataFrame."""
        records = []
        for p in self.pair_results:
            records.append(
                {
                    "variable_x": p.variable_x,
                    "variable_y": p.variable_y,
                    "pearson_r": p.pearson_r,
                    "pearson_p": p.pearson_p_value,
                    "spearman_rho": p.spearman_rho,
                    "spearman_p": p.spearman_p_value,
                    "sample_cov": p.covariance,
                    "relationship_nature": p.relationship_nature,
                }
            )
        return pd.DataFrame(records)

    def summary_dict(self) -> dict[str, Any]:
        """Provides a structured dictionary summary of all evaluated pairs."""
        return {
            "sample_size": self.sample_size,
            "pairs": [
                {
                    "pair": f"{p.variable_x} vs {p.variable_y}",
                    "pearson_r": p.pearson_r,
                    "spearman_rho": p.spearman_rho,
                    "nature": p.relationship_nature,
                    "interpretation": p.methodological_note,
                }
                for p in self.pair_results
            ],
        }


def calculate_bivariate_correlations(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    var_name_x: str = "x",
    var_name_y: str = "y",
) -> CorrelationPairResult:
    """Calculates Pearson and Spearman correlations with exact p-values for two vectors.

    Args:
        x: First numeric variable sequence.
        y: Second numeric variable sequence.
        var_name_x: Label for variable x.
        var_name_y: Label for variable y.

    Returns:
        CorrelationPairResult with correlation values, p-values, and contextual documentation.
    """
    arr_x = _validate_numeric_array(x, min_elements=3)
    arr_y = _validate_numeric_array(y, min_elements=3)

    if len(arr_x) != len(arr_y):
        raise ValueError(f"Array length mismatch: len(x)={len(arr_x)} vs len(y)={len(arr_y)}")

    n = len(arr_x)

    # Check for zero variance / constant arrays
    std_x = np.std(arr_x, ddof=1)
    std_y = np.std(arr_y, ddof=1)

    if std_x == 0.0 or std_y == 0.0:
        cov = 0.0
        r, p_val_r = 0.0, 1.0
        rho, p_val_rho = 0.0, 1.0
    else:
        cov_matrix = np.cov(arr_x, arr_y, ddof=1)
        cov = float(cov_matrix[0, 1])

        # Pearson
        res_pearson = stats.pearsonr(arr_x, arr_y)
        r = float(res_pearson.statistic)
        p_val_r = float(res_pearson.pvalue)

        # Spearman
        res_spearman = stats.spearmanr(arr_x, arr_y)
        rho = float(res_spearman.statistic)
        p_val_rho = float(res_spearman.pvalue)

    # Build contextual methodological interpretation
    pair_key = (var_name_x, var_name_y)
    nature, note = _get_methodological_rationale(pair_key, r, rho)

    return CorrelationPairResult(
        variable_x=var_name_x,
        variable_y=var_name_y,
        pearson_r=round(r, 6),
        pearson_p_value=round(p_val_r, 6),
        spearman_rho=round(rho, 6),
        spearman_p_value=round(p_val_rho, 6),
        covariance=round(cov, 6),
        sample_size=n,
        relationship_nature=nature,
        methodological_note=note,
    )


def _get_methodological_rationale(
    pair: tuple[str, str], r: float, rho: float
) -> tuple[str, str]:
    """Generates rigorous statistical interpretation distinguishing association from causality."""
    pair_set = {pair[0], pair[1]}

    if pair_set == {"input_tokens", "output_tokens"}:
        nature = "independent_by_construction"
        note = (
            "En el generador estandar de telemetria sintetica, los tokens de entrada y salida se muestrean "
            "mediante llamadas independientes e incorrelacionadas a RNG. La correlacion observada "
            f"(r={r:.4f}, rho={rho:.4f}) refleja unicamente fluctuacion estocastica muestral alrededor de 0. "
            "No existe causalidad fisica ni relacion de dependencia construida en este generador base."
        )
    elif pair_set == {"input_tokens", "latency_ms"}:
        nature = "structural_affine_dependency"
        note = (
            "La latencia se define formalmente en el modelo como una transformacion afin L = beta_0 + "
            "alpha_in * T_in + alpha_out * T_out + epsilon. La asociacion positiva observada "
            f"(r={r:.4f}) esta directamente construida por el termino alpha_in ({r:.2f} contribucion marginal). "
            "Aunque matematicamente inducida por diseno, refleja un principio causal en inferencia LLM: "
            "mayor longitud de prompt prolonga la fase de prefill/procesamiento de contexto."
        )
    elif pair_set == {"output_tokens", "latency_ms"}:
        nature = "dominant_structural_affine_dependency"
        note = (
            "La latencia exhibe una fuerte correlacion positiva con los tokens de salida "
            f"(r={r:.4f}, rho={rho:.4f}) debido a que alpha_out (0.80 ms/token) domina sobre alpha_in "
            "(0.05 ms/token). Esta estructura modela con realismo la decodificacion autorregresiva de tokens "
            "secuenciales en transformadores, donde cada nuevo token generado impone una pasada por la red neuronal."
        )
    elif pair_set == {"total_tokens", "cost_total_usd"}:
        nature = "deterministic_linear_combination"
        note = (
            "El costo total es una combinacion lineal exacta determinista de los tokens (C = R_in * T_in / 1e6 + "
            f"R_out * T_out / 1e6). Por consiguiente, la correlacion es cuasi-perfecta (r={r:.4f}). "
            "No es un hallazgo empirico, sino una identidad contable tarifaria del sistema."
        )
    elif pair_set == {"latency_ms", "cost_total_usd"}:
        nature = "confounded_spurious_association"
        note = (
            "La correlacion positiva observada entre latencia y costo total "
            f"(r={r:.4f}) es un claro ejemplo de variable confusora latente: ninguna de las dos causa a la otra, "
            "sino que ambas son funciones crecientes de los tokens procesados (T_in y T_out). La asociacion "
            "estadistica es genuina, pero la relacion causal directa entre latencia y tarifa es inexistente."
        )
    else:
        nature = "empirical_observation"
        note = (
            f"Asociacion muestral calculada sobre registros sinteticos (r={r:.4f}, rho={rho:.4f}). "
            "Debe interpretarse dentro del contexto parametrico del generador."
        )

    return nature, note


def analyze_dataframe_correlations(
    df: pd.DataFrame,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> CorrelationAnalysisResult:
    """Computes bivariate and matrix correlations across numerical fields in a DataFrame.

    Args:
        df: Input pandas DataFrame (e.g. from generate_synthetic_requests).
        pairs: Specific pairs of column names to evaluate. Defaults to DEFAULT_CORRELATION_PAIRS.

    Returns:
        CorrelationAnalysisResult with pairwise details and complete matrices.
    """
    if len(df) < 3:
        raise ValueError(f"DataFrame must contain at least 3 rows, got {len(df)}")

    target_pairs = list(pairs) if pairs is not None else list(DEFAULT_CORRELATION_PAIRS)

    # Filter numeric columns for correlation matrices
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("DataFrame contains no numerical columns.")

    pearson_mat = numeric_df.corr(method="pearson")
    spearman_mat = numeric_df.corr(method="spearman")

    pair_results: list[CorrelationPairResult] = []
    for col_x, col_y in target_pairs:
        if col_x not in df.columns or col_y not in df.columns:
            continue
        p_res = calculate_bivariate_correlations(
            x=df[col_x].to_numpy(dtype=np.float64),
            y=df[col_y].to_numpy(dtype=np.float64),
            var_name_x=col_x,
            var_name_y=col_y,
        )
        pair_results.append(p_res)

    return CorrelationAnalysisResult(
        pair_results=pair_results,
        pearson_matrix=pearson_mat,
        spearman_matrix=spearman_mat,
        sample_size=len(df),
    )


# ---------------------------------------------------------------------------
# Optional Correlated Token Sampling (Joint Bivariate Modeling)
# ---------------------------------------------------------------------------

def sample_joint_tokens(
    rng: np.random.Generator,
    input_config: TokenDistributionConfig,
    output_config: TokenDistributionConfig,
    size: int,
    correlation_rho: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates joint (input_tokens, output_tokens) draws with controlled correlation rho.

    Methodology:
        - When correlation_rho == 0.0 (default):
          Strictly preserves independent sampling using native sample_tokens from Phase 1.
        - When correlation_rho != 0.0:
          Applies a Gaussian copula / bivariate normal coupling:
          1. Samples bivariate standard normal (Z_in, Z_out) with covariance matrix [[1, rho], [rho, 1]].
          2. For Log-Normal distributions: uses exact moment scaling:
             Y_in = mu_log_in + sigma_log_in * Z_in
             Y_out = mu_log_out + sigma_log_out * Z_out
             Tokens = clip(round(exp(Y)))
             This preserves the exact marginal LogNormal distributions while establishing the joint correlation.
          3. For other distributions: maps uniform quantiles U = Phi(Z) to marginal inverse CDFs.

    Args:
        rng: NumPy random generator.
        input_config: Token distribution config for input prompt.
        output_config: Token distribution config for output completion.
        size: Number of joint request draws (>= 0).
        correlation_rho: Target Pearson correlation parameter in [-1.0, 1.0] (default: 0.0).

    Returns:
        tuple (input_tokens_array, output_tokens_array) of 1D int64 arrays.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    if not (-1.0 <= correlation_rho <= 1.0):
        raise ValueError(f"correlation_rho must be in [-1.0, 1.0], got {correlation_rho}")

    # Baseline default: Exact independent sampling
    if correlation_rho == 0.0:
        in_tokens = sample_tokens(rng, input_config, size=size)
        out_tokens = sample_tokens(rng, output_config, size=size)
        return in_tokens, out_tokens

    # Bivariate correlated sampling via Gaussian Copula / Bivariate Normal
    # Construct correlation covariance matrix
    cov_matrix = np.array([[1.0, correlation_rho], [correlation_rho, 1.0]], dtype=np.float64)
    # Draw bivariate standard normal pairs: shape (size, 2)
    z_draws = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_matrix, size=size)
    z_in = z_draws[:, 0]
    z_out = z_draws[:, 1]

    # Process Input Tokens
    in_tokens = _transform_standard_normal_to_marginal(
        z=z_in,
        config=input_config,
    )

    # Process Output Tokens
    out_tokens = _transform_standard_normal_to_marginal(
        z=z_out,
        config=output_config,
    )

    return in_tokens, out_tokens


def _transform_standard_normal_to_marginal(
    z: np.ndarray,
    config: TokenDistributionConfig,
) -> np.ndarray:
    """Transforms a standard normal vector Z ~ N(0, 1) into marginal token integers."""
    min_tok = config.min_tokens
    max_tok = config.max_tokens if config.max_tokens is not None else np.iinfo(np.int64).max

    if config.family == DistributionType.LOGNORMAL:
        if config.std == 0.0:
            return np.full(len(z), int(np.clip(np.round(config.mean), min_tok, max_tok)), dtype=np.int64)
        variance_ratio = (config.std**2) / (config.mean**2)
        sigma_log = math.sqrt(math.log(1.0 + variance_ratio))
        mu_log = math.log(config.mean) - 0.5 * (sigma_log**2)
        y = mu_log + sigma_log * z
        continuous = np.exp(y)

    elif config.family == DistributionType.NORMAL:
        continuous = config.mean + config.std * z

    elif config.family == DistributionType.GAMMA:
        # Probability Integral Transform: U = Phi(Z)
        u = stats.norm.cdf(z)
        # Avoid exact 0.0 and 1.0 for Gamma ppf
        u = np.clip(u, 1e-9, 1.0 - 1e-9)
        shape = (config.mean**2) / (config.std**2)
        scale = (config.std**2) / config.mean
        continuous = stats.gamma.ppf(u, a=shape, scale=scale)

    elif config.family == DistributionType.UNIFORM:
        u = stats.norm.cdf(z)
        low = max(config.min_tokens, int(round(config.mean - math.sqrt(3.0) * config.std)))
        high = max(low + 1, int(round(config.mean + math.sqrt(3.0) * config.std)))
        continuous = low + u * (high - low)

    else:
        raise ValueError(f"Unsupported distribution family: {config.family}")

    discrete = np.round(continuous).astype(np.int64)
    return np.clip(discrete, a_min=min_tok, a_max=max_tok)


def generate_synthetic_requests_with_correlation(
    n_requests: int,
    correlation_rho: float = 0.0,
    token_in_config: TokenDistributionConfig | None = None,
    token_out_config: TokenDistributionConfig | None = None,
    latency_config: LatencyConfig | None = None,
    pricing_config: PricingConfig | None = None,
    start_timestamp: str | pd.Timestamp = "2026-01-01 00:00:00",
    mean_interval_seconds: float = 2.5,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generates synthetic requests with optional joint token correlation rho.

    Maintains full backward compatibility: if correlation_rho == 0.0, produces
    identical independent sampling behavior.
    """
    if n_requests < 1:
        raise ValueError(f"n_requests must be at least 1, got {n_requests}")
    if not (-1.0 <= correlation_rho <= 1.0):
        raise ValueError(f"correlation_rho must be in [-1.0, 1.0], got {correlation_rho}")

    t_in_cfg = token_in_config if token_in_config is not None else TokenDistributionConfig()
    t_out_cfg = token_out_config if token_out_config is not None else TokenDistributionConfig(
        mean=300.0, std=150.0
    )
    lat_cfg = latency_config if latency_config is not None else LatencyConfig()
    price_cfg = pricing_config if pricing_config is not None else PricingConfig()

    rng = np.random.default_rng(seed)

    # Sample joint tokens
    input_tokens, output_tokens = sample_joint_tokens(
        rng=rng,
        input_config=t_in_cfg,
        output_config=t_out_cfg,
        size=n_requests,
        correlation_rho=correlation_rho,
    )
    total_tokens = input_tokens + output_tokens

    # Sample latencies
    latencies = sample_latency(rng, input_tokens, output_tokens, lat_cfg)

    # Inter-arrivals
    inter_arrivals_sec = rng.exponential(scale=mean_interval_seconds, size=n_requests)
    inter_arrivals_sec[0] = 0.0
    cumulative_seconds = np.cumsum(inter_arrivals_sec)
    base_time = pd.to_datetime(start_timestamp)
    timestamps = base_time + pd.to_timedelta(cumulative_seconds, unit="s")

    # Pricing
    cost_in = (input_tokens / 1_000_000.0) * price_cfg.cost_per_million_input
    cost_out = (output_tokens / 1_000_000.0) * price_cfg.cost_per_million_output
    cost_total = cost_in + cost_out

    width = max(6, len(str(n_requests)))
    request_ids = [f"req_{i:0{width}d}" for i in range(1, n_requests + 1)]

    return pd.DataFrame(
        {
            "request_id": request_ids,
            "timestamp": timestamps,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latencies,
            "cost_input_usd": np.round(cost_in, 6),
            "cost_output_usd": np.round(cost_out, 6),
            "cost_total_usd": np.round(cost_total, 6),
        }
    )
