"""Tests for stability field loading in prediction_loader."""

import pytest


class TestStabilityFieldsLoading:
    """Test that new stability fields are fetched from predictions."""

    @pytest.mark.skip(reason="Stability fields (median_14d, return_*, volatility_24h) not available on Ampere predictions table")
    def test_get_best_prediction_includes_stability_fields(self):
        """Query should select stability fields when available.

        NOTE: This test is skipped because the Ampere production database
        does not have these columns in the predictions table. The PatchTST
        model outputs only: time, item_id, hour_offset, offset_pct,
        fill_probability, expected_value, buy_price, sell_price,
        current_high, current_low, confidence, model_version.

        If stability fields are added to the predictions table in the future,
        this test can be re-enabled.
        """
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
