"""Tests for model registry and lifecycle management."""

from src.model_registry import (
    ModelStatus,
    can_serve_prediction,
    is_active_status,
    is_deprecated_status,
    should_include_for_recommendations,
)


class TestModelStatus:
    """Test ModelStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert ModelStatus.ACTIVE == "ACTIVE"
        assert ModelStatus.DEPRECATED == "DEPRECATED"
        assert ModelStatus.SUNSET == "SUNSET"
        assert ModelStatus.ARCHIVED == "ARCHIVED"

    def test_status_is_string_enum(self):
        """Test that ModelStatus values are strings."""
        assert isinstance(ModelStatus.ACTIVE, str)
        assert isinstance(ModelStatus.DEPRECATED, str)


class TestShouldIncludeForRecommendations:
    """Test the should_include_for_recommendations function."""

    def test_active_included(self):
        """Active models should be included in recommendations."""
        assert should_include_for_recommendations("ACTIVE") is True
        assert should_include_for_recommendations(ModelStatus.ACTIVE) is True

    def test_deprecated_excluded(self):
        """Deprecated models should not be included in recommendations."""
        assert should_include_for_recommendations("DEPRECATED") is False
        assert should_include_for_recommendations(ModelStatus.DEPRECATED) is False

    def test_sunset_excluded(self):
        """Sunset models should not be included in recommendations."""
        assert should_include_for_recommendations("SUNSET") is False
        assert should_include_for_recommendations(ModelStatus.SUNSET) is False

    def test_archived_excluded(self):
        """Archived models should not be included in recommendations."""
        assert should_include_for_recommendations("ARCHIVED") is False
        assert should_include_for_recommendations(ModelStatus.ARCHIVED) is False

    def test_unknown_status_excluded(self):
        """Unknown status values should not be included."""
        assert should_include_for_recommendations("UNKNOWN") is False
        assert should_include_for_recommendations("") is False


class TestCanServePrediction:
    """Test the can_serve_prediction function."""

    def test_active_can_serve_new_trade(self):
        """Active models can serve new trade predictions."""
        assert can_serve_prediction("ACTIVE", is_existing_trade=False) is True
        assert can_serve_prediction(ModelStatus.ACTIVE, is_existing_trade=False) is True

    def test_deprecated_cannot_serve_new_trade(self):
        """Deprecated models cannot serve new trade predictions."""
        assert can_serve_prediction("DEPRECATED", is_existing_trade=False) is False
        assert (
            can_serve_prediction(ModelStatus.DEPRECATED, is_existing_trade=False)
            is False
        )

    def test_deprecated_can_serve_existing_trade(self):
        """Deprecated models CAN serve existing trade lookups."""
        assert can_serve_prediction("DEPRECATED", is_existing_trade=True) is True
        assert (
            can_serve_prediction(ModelStatus.DEPRECATED, is_existing_trade=True) is True
        )

    def test_sunset_can_serve_existing_trade(self):
        """Sunset models CAN serve existing trade lookups."""
        assert can_serve_prediction("SUNSET", is_existing_trade=True) is True

    def test_archived_can_serve_existing_trade(self):
        """Archived models CAN serve existing trade lookups."""
        assert can_serve_prediction("ARCHIVED", is_existing_trade=True) is True

    def test_active_can_serve_existing_trade(self):
        """Active models can serve existing trade lookups."""
        assert can_serve_prediction("ACTIVE", is_existing_trade=True) is True

    def test_default_is_new_trade(self):
        """Default behavior is for new trades (not existing)."""
        # Default is_existing_trade=False
        assert can_serve_prediction("DEPRECATED") is False
        assert can_serve_prediction("ACTIVE") is True


class TestIsActiveStatus:
    """Test the is_active_status function."""

    def test_active_is_active(self):
        """ACTIVE status returns True."""
        assert is_active_status("ACTIVE") is True
        assert is_active_status(ModelStatus.ACTIVE) is True

    def test_deprecated_is_not_active(self):
        """DEPRECATED status returns False."""
        assert is_active_status("DEPRECATED") is False
        assert is_active_status(ModelStatus.DEPRECATED) is False

    def test_sunset_is_not_active(self):
        """SUNSET status returns False."""
        assert is_active_status("SUNSET") is False

    def test_archived_is_not_active(self):
        """ARCHIVED status returns False."""
        assert is_active_status("ARCHIVED") is False


class TestIsDeprecatedStatus:
    """Test the is_deprecated_status function."""

    def test_deprecated_is_deprecated(self):
        """DEPRECATED status returns True."""
        assert is_deprecated_status("DEPRECATED") is True
        assert is_deprecated_status(ModelStatus.DEPRECATED) is True

    def test_sunset_is_deprecated(self):
        """SUNSET status returns True (considered deprecated)."""
        assert is_deprecated_status("SUNSET") is True
        assert is_deprecated_status(ModelStatus.SUNSET) is True

    def test_archived_is_deprecated(self):
        """ARCHIVED status returns True (considered deprecated)."""
        assert is_deprecated_status("ARCHIVED") is True
        assert is_deprecated_status(ModelStatus.ARCHIVED) is True

    def test_active_is_not_deprecated(self):
        """ACTIVE status returns False."""
        assert is_deprecated_status("ACTIVE") is False
        assert is_deprecated_status(ModelStatus.ACTIVE) is False
