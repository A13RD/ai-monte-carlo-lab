"""Stochastic generation functions for AI Monte Carlo Lab.

Implements pure, reproducible random sampling for:
- Token counts (Log-Normal, Gamma, Truncated Normal, Uniform)
- Workload and request volume (Poisson, Negative Binomial, Normal, Fixed)
- Latency (multivariate function of tokens with stochastic jitter)

All functions strictly use `numpy.random.Generator` from `default_rng(seed)`.
"""

from __future__ import annotations

import numpy as np

from ai_monte_carlo.config import (
    DistributionType,
    LatencyConfig,
    TokenDistributionConfig,
    WorkloadDistributionType,
)


def sample_tokens_lognormal(
    rng: np.random.Generator,
    size: int,
    mean: float,
    std: float,
    min_tokens: int = 1,
    max_tokens: int | None = None,
) -> np.ndarray:
    """Samples integer token counts from a Log-Normal distribution parameterized by natural moments.

    Derivation:
        Given target natural mean mu_X and standard deviation sigma_X:
            sigma_log^2 = ln(1 + (sigma_X / mu_X)^2)
            mu_log = ln(mu_X) - 0.5 * sigma_log^2
        Then X ~ LogNormal(mu_log, sigma_log), discretized and bounded.

    Args:
        rng: NumPy random generator instance.
        size: Number of token values to draw. Must be >= 0.
        mean: Expected token count in natural scale (> 0).
        std: Standard deviation of token count in natural scale (>= 0).
        min_tokens: Minimum token count bound (>= 1).
        max_tokens: Maximum token count bound (optional).

    Returns:
        1D NumPy array of integers representing token counts.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if mean <= 0:
        raise ValueError(f"mean must be strictly positive, got {mean}")
    if std < 0:
        raise ValueError(f"std must be non-negative, got {std}")
    if min_tokens < 1:
        raise ValueError(f"min_tokens must be >= 1, got {min_tokens}")
    if max_tokens is not None and max_tokens < min_tokens:
        raise ValueError(f"max_tokens ({max_tokens}) cannot be less than min_tokens ({min_tokens})")

    if size == 0:
        return np.empty(0, dtype=np.int64)

    if std == 0.0:
        constant_val = int(np.clip(np.round(mean), min_tokens, max_tokens if max_tokens else np.iinfo(np.int64).max))
        return np.full(size, constant_val, dtype=np.int64)

    variance = std**2
    variance_ratio = variance / (mean**2)
    sigma_log = np.sqrt(np.log(1.0 + variance_ratio))
    mu_log = np.log(mean) - 0.5 * (sigma_log**2)

    continuous_draws = rng.lognormal(mean=mu_log, sigma=sigma_log, size=size)
    discrete_draws = np.round(continuous_draws).astype(np.int64)

    return np.clip(
        discrete_draws,
        a_min=min_tokens,
        a_max=max_tokens if max_tokens is not None else np.iinfo(np.int64).max,
    )


def sample_tokens_gamma(
    rng: np.random.Generator,
    size: int,
    mean: float,
    std: float,
    min_tokens: int = 1,
    max_tokens: int | None = None,
) -> np.ndarray:
    """Samples integer token counts from a Gamma distribution using method of moments.

    Derivation:
        For X ~ Gamma(shape=k, scale=theta):
            E[X] = k * theta = mean
            Var(X) = k * theta^2 = std^2
        Hence:
            theta = std^2 / mean
            k = mean^2 / std^2

    Args:
        rng: NumPy random generator instance.
        size: Number of token values to draw (>= 0).
        mean: Expected token count (> 0).
        std: Standard deviation of token count (> 0).
        min_tokens: Minimum token count bound (>= 1).
        max_tokens: Maximum token count bound (optional).

    Returns:
        1D NumPy array of integers representing token counts.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if mean <= 0:
        raise ValueError(f"mean must be strictly positive, got {mean}")
    if std <= 0:
        raise ValueError(f"std for Gamma must be strictly positive, got {std}")
    if min_tokens < 1:
        raise ValueError(f"min_tokens must be >= 1, got {min_tokens}")
    if max_tokens is not None and max_tokens < min_tokens:
        raise ValueError(f"max_tokens ({max_tokens}) cannot be less than min_tokens ({min_tokens})")

    if size == 0:
        return np.empty(0, dtype=np.int64)

    shape = (mean**2) / (std**2)
    scale = (std**2) / mean

    continuous_draws = rng.gamma(shape=shape, scale=scale, size=size)
    discrete_draws = np.round(continuous_draws).astype(np.int64)

    return np.clip(
        discrete_draws,
        a_min=min_tokens,
        a_max=max_tokens if max_tokens is not None else np.iinfo(np.int64).max,
    )


def sample_tokens_normal(
    rng: np.random.Generator,
    size: int,
    mean: float,
    std: float,
    min_tokens: int = 1,
    max_tokens: int | None = None,
) -> np.ndarray:
    """Samples integer token counts from a truncated Normal distribution.

    Args:
        rng: NumPy random generator instance.
        size: Number of token values to draw (>= 0).
        mean: Normal mean.
        std: Normal standard deviation (>= 0).
        min_tokens: Lower truncation bound (>= 1).
        max_tokens: Upper truncation bound (optional).

    Returns:
        1D NumPy array of integers representing token counts.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if min_tokens < 1:
        raise ValueError(f"min_tokens must be >= 1, got {min_tokens}")
    if max_tokens is not None and max_tokens < min_tokens:
        raise ValueError(f"max_tokens ({max_tokens}) cannot be less than min_tokens ({min_tokens})")

    if size == 0:
        return np.empty(0, dtype=np.int64)

    continuous_draws = rng.normal(loc=mean, scale=std, size=size)
    discrete_draws = np.round(continuous_draws).astype(np.int64)

    return np.clip(
        discrete_draws,
        a_min=min_tokens,
        a_max=max_tokens if max_tokens is not None else np.iinfo(np.int64).max,
    )


def sample_tokens(
    rng: np.random.Generator,
    config: TokenDistributionConfig,
    size: int,
) -> np.ndarray:
    """Dispatches token sampling according to the provided TokenDistributionConfig.

    Args:
        rng: NumPy random generator instance.
        config: TokenDistributionConfig specifying distribution family and moments.
        size: Number of token values to sample.

    Returns:
        1D NumPy array of sampled token integers.
    """
    if config.family == DistributionType.LOGNORMAL:
        return sample_tokens_lognormal(
            rng=rng,
            size=size,
            mean=config.mean,
            std=config.std,
            min_tokens=config.min_tokens,
            max_tokens=config.max_tokens,
        )
    elif config.family == DistributionType.GAMMA:
        return sample_tokens_gamma(
            rng=rng,
            size=size,
            mean=config.mean,
            std=config.std,
            min_tokens=config.min_tokens,
            max_tokens=config.max_tokens,
        )
    elif config.family == DistributionType.NORMAL:
        return sample_tokens_normal(
            rng=rng,
            size=size,
            mean=config.mean,
            std=config.std,
            min_tokens=config.min_tokens,
            max_tokens=config.max_tokens,
        )
    elif config.family == DistributionType.UNIFORM:
        low = max(config.min_tokens, int(round(config.mean - np.sqrt(3.0) * config.std)))
        high = max(low + 1, int(round(config.mean + np.sqrt(3.0) * config.std)))
        if config.max_tokens:
            high = min(high, config.max_tokens)
        draws = rng.integers(low=low, high=high + 1, size=size)
        return draws.astype(np.int64)
    else:
        raise ValueError(f"Unsupported distribution family: {config.family}")


def sample_requests_count(
    rng: np.random.Generator,
    family: WorkloadDistributionType | str,
    mean_requests: float,
    size: int = 1,
    dispersion: float = 0.1,
) -> np.ndarray:
    """Samples request counts over a given time horizon.

    Supports:
        - Poisson (standard arrival process without overdispersion, Var = mean)
        - Negative Binomial (overdispersed bursty arrivals, Var = mean + alpha * mean^2)
        - Normal (continuous approximation clipped at 0)
        - Fixed (deterministic constant count)

    Args:
        rng: NumPy random generator instance.
        family: Workload distribution family.
        mean_requests: Expected number of requests (>= 0).
        size: Number of realizations to sample (>= 0).
        dispersion: Overdispersion coefficient alpha for Negative Binomial (> 0).

    Returns:
        1D NumPy array of integer request counts.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if mean_requests < 0:
        raise ValueError(f"mean_requests must be non-negative, got {mean_requests}")
    if size == 0:
        return np.empty(0, dtype=np.int64)

    if mean_requests == 0.0:
        return np.zeros(size, dtype=np.int64)

    fam = WorkloadDistributionType(family)

    if fam == WorkloadDistributionType.POISSON:
        return rng.poisson(lam=mean_requests, size=size).astype(np.int64)

    elif fam == WorkloadDistributionType.NEGATIVE_BINOMIAL:
        if dispersion <= 0:
            raise ValueError(f"dispersion must be strictly positive, got {dispersion}")
        # Var = mean + dispersion * mean^2
        # p = mean / Var = 1 / (1 + dispersion * mean)
        # r = mean^2 / (Var - mean) = 1 / dispersion
        r = 1.0 / dispersion
        p = 1.0 / (1.0 + dispersion * mean_requests)
        return rng.negative_binomial(n=r, p=p, size=size).astype(np.int64)

    elif fam == WorkloadDistributionType.NORMAL:
        variance = mean_requests + dispersion * (mean_requests**2)
        scale = np.sqrt(variance)
        continuous = rng.normal(loc=mean_requests, scale=scale, size=size)
        return np.clip(np.round(continuous).astype(np.int64), a_min=0, a_max=None)

    elif fam == WorkloadDistributionType.FIXED:
        return np.full(size, int(round(mean_requests)), dtype=np.int64)

    else:
        raise ValueError(f"Unsupported workload distribution: {family}")


def sample_latency(
    rng: np.random.Generator,
    input_tokens: np.ndarray,
    output_tokens: np.ndarray,
    config: LatencyConfig,
) -> np.ndarray:
    """Simulates request latency based on token counts and stochastic jitter.

    Model:
        L_i = max(min_latency, base + alpha_in * T_in,i + alpha_out * T_out,i + epsilon_i)
        where epsilon_i ~ Normal(0, jitter_std_ms^2).

    Args:
        rng: NumPy random generator instance.
        input_tokens: 1D array of prompt token counts.
        output_tokens: 1D array of completion token counts.
        config: LatencyConfig containing model parameters.

    Returns:
        1D NumPy array of latency values in milliseconds, rounded to 2 decimals.
    """
    if len(input_tokens) != len(output_tokens):
        raise ValueError(
            f"Token array lengths do not match: input={len(input_tokens)}, output={len(output_tokens)}"
        )
    if len(input_tokens) == 0:
        return np.empty(0, dtype=np.float64)

    n = len(input_tokens)
    jitter = rng.normal(loc=0.0, scale=config.jitter_std_ms, size=n)

    latency = (
        config.base_latency_ms
        + (config.per_input_token_ms * input_tokens)
        + (config.per_output_token_ms * output_tokens)
        + jitter
    )

    bounded_latency = np.clip(latency, a_min=config.min_latency_ms, a_max=None)
    return np.round(bounded_latency, 2)
