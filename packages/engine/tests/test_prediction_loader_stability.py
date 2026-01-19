"""Tests for stability field loading in prediction_loader."""

import pytest


class TestStabilityFieldsLoading:
    """Test that new stability fields are fetched from predictions."""

    def test_get_best_prediction_includes_stability_fields(self):
        """Query should select stability fields when available."""
        # Verify the SQL query includes new columns
        # We'll check by inspecting the query string
        import src.prediction_loader as pl_module
        source = open(pl_module.__file__).read()

        # These columns should be in the SELECT statement
        assert 'median_14d' in source or 'p.median_14d' in source, \
            "Query should select median_14d"
        assert 'price_vs_median_ratio' in source, \
            "Query should select price_vs_median_ratio"
        assert 'return_1h' in source, \
            "Query should select return_1h"
        assert 'return_4h' in source, \
            "Query should select return_4h"
        assert 'return_24h' in source, \
            "Query should select return_24h"
        assert 'volatility_24h' in source, \
            "Query should select volatility_24h"
