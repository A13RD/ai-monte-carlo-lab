"""Unit tests for synthetic dataset generation module."""

import numpy as np
import pandas as pd
import pytest

from ai_monte_carlo.config import (
    DistributionType,
    LatencyConfig,
    PricingConfig,
    TokenDistributionConfig,
)
from ai_monte_carlo.data.synthetic import (
    generate_synthetic_requests,
    save_synthetic_requests,
)


class TestSyntheticDataGeneration:
    """Tests for reproducible synthetic dataset generator."""

    def test_reproducibility_with_seed(self):
        """Verifies that two runs with the same seed yield identical DataFrames."""
        df1 = generate_synthetic_requests(n_requests=50, seed=42)
        df2 = generate_synthetic_requests(n_requests=50, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_yield_different_data(self):
        """Verifies that different seeds produce distinct stochastic traces."""
        df1 = generate_synthetic_requests(n_requests=50, seed=42)
        df2 = generate_synthetic_requests(n_requests=50, seed=999)

        assert not df1["input_tokens"].equals(df2["input_tokens"])
        assert not df1["latency_ms"].equals(df2["latency_ms"])

    def test_expected_columns_and_types(self):
        """Verifies structure and types of the generated DataFrame."""
        df = generate_synthetic_requests(n_requests=25, seed=123)

        expected_columns = [
            "request_id",
            "timestamp",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "latency_ms",
            "cost_input_usd",
            "cost_output_usd",
            "cost_total_usd",
        ]
        assert list(df.columns) == expected_columns
        assert len(df) == 25

        assert pd.api.types.is_string_dtype(df["request_id"]) or pd.api.types.is_object_dtype(df["request_id"])
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_integer_dtype(df["input_tokens"])
        assert pd.api.types.is_integer_dtype(df["output_tokens"])
        assert pd.api.types.is_integer_dtype(df["total_tokens"])
        assert pd.api.types.is_float_dtype(df["latency_ms"])
        assert pd.api.types.is_float_dtype(df["cost_total_usd"])

    def test_domain_integrity_and_bounds(self):
        """Validates that simulated values obey physical and statistical constraints."""
        min_tok = 5
        min_lat = 25.0
        t_in = TokenDistributionConfig(min_tokens=min_tok)
        lat_cfg = LatencyConfig(min_latency_ms=min_lat)

        df = generate_synthetic_requests(
            n_requests=100,
            token_in_config=t_in,
            latency_config=lat_cfg,
            seed=42,
        )

        # Tokens must be >= min_tokens
        assert (df["input_tokens"] >= min_tok).all()
        assert (df["output_tokens"] >= 1).all()

        # Additive token integrity
        assert (df["total_tokens"] == df["input_tokens"] + df["output_tokens"]).all()

        # Latency bounds
        assert (df["latency_ms"] >= min_lat).all()

        # Chronological timestamps
        assert df["timestamp"].is_monotonic_increasing

    def test_pricing_accuracy(self):
        """Verifies exact calculation of token cost per million."""
        pricing = PricingConfig(cost_per_million_input=5.0, cost_per_million_output=15.0)
        df = generate_synthetic_requests(n_requests=20, pricing_config=pricing, seed=7)

        expected_input_cost = np.round((df["input_tokens"] / 1e6) * 5.0, 6)
        expected_output_cost = np.round((df["output_tokens"] / 1e6) * 15.0, 6)
        expected_total_cost = np.round(expected_input_cost + expected_output_cost, 6)

        np.testing.assert_allclose(df["cost_input_usd"], expected_input_cost, atol=1e-6)
        np.testing.assert_allclose(df["cost_output_usd"], expected_output_cost, atol=1e-6)
        np.testing.assert_allclose(df["cost_total_usd"], expected_total_cost, atol=1e-6)

    def test_invalid_parameters_raise(self):
        """Verifies validation errors on invalid request counts or intervals."""
        with pytest.raises(ValueError, match="n_requests must be at least 1"):
            generate_synthetic_requests(n_requests=0)

        with pytest.raises(ValueError, match="mean_interval_seconds must be strictly positive"):
            generate_synthetic_requests(n_requests=10, mean_interval_seconds=-1.0)

    def test_save_synthetic_requests(self, tmp_path):
        """Verifies export to CSV file."""
        df = generate_synthetic_requests(n_requests=10, seed=1)
        csv_file = tmp_path / "synthetic_trace.csv"

        save_synthetic_requests(df, csv_file)
        assert csv_file.exists()

        reloaded = pd.read_csv(csv_file)
        assert len(reloaded) == 10
        assert "request_id" in reloaded.columns
