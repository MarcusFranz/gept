"""
Item Selector - Intelligent Item Selection for Training
=======================================================

Selects items for daily training based on:
1. Volume threshold (minimum 24h trading volume)
2. Data quality (minimum rows for training)
3. Model staleness (prioritize oldest models)
4. Performance degradation (AUC drift detection)
5. New items (items without any trained model)
6. High-value items (Issue #120): Items >= 10M gp with relaxed volume thresholds

Usage:
    from src.training.item_selector import ItemSelector

    selector = ItemSelector()
    items = selector.select_items_for_training(max_items=50)

    # For discovery mode (1st of month)
    all_items = selector.select_items_for_training(max_items=400, discovery=True)

    # Include high-value items (enabled by default)
    items = selector.select_items_for_training(include_high_value=True)

CLI:
    python -m src.training.item_selector --run-id 20260111_050000 --output items.json
    python -m src.training.item_selector --dry-run  # Just show what would be selected
    python -m src.training.item_selector --high-value-only  # Only select high-value items
"""

import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db_utils import get_db_connection, get_db_cursor

logger = logging.getLogger(__name__)


@dataclass
class HighValueConfig:
    """Configuration for high-value item selection (Issue #120)."""
    enabled: bool = True
    min_price_gp: int = 10_000_000  # 10M gp
    min_24h_volume: int = 100  # Relaxed from 10,000
    min_training_rows: int = 5000
    max_items_per_run: int = 20
    force_include_items: List[int] = field(default_factory=list)
    min_history_days: int = 30

    # Priority weights for high-value items
    weight_no_model: int = 100
    weight_high_price: int = 50
    weight_low_manipulation_risk: int = 30


@dataclass
class SelectionConfig:
    """Configuration for item selection."""
    min_24h_volume: int = 10000
    min_training_rows: int = 5000
    max_model_age_days: int = 30
    max_calibration_error: float = 0.15
    max_items_per_run: int = 50
    discovery_max_items: int = 3000  # Enough for all qualifying items

    # Priority weights
    weight_no_model: int = 100
    weight_auc_degradation: int = 80
    weight_calibration_error: int = 60
    weight_model_staleness: int = 40
    weight_high_volume: int = 20

    # High-value item configuration
    high_value: HighValueConfig = field(default_factory=HighValueConfig)


@dataclass
class SelectedItem:
    """An item selected for training."""
    item_id: int
    item_name: str
    reason: str
    priority_score: float
    current_model_id: Optional[int] = None
    current_model_auc: Optional[float] = None
    model_age_days: Optional[float] = None
    calibration_error: Optional[float] = None
    volume_24h: Optional[int] = None
    data_rows: Optional[int] = None
    is_high_value: bool = False  # Issue #120: Flag for high-value items
    item_price: Optional[int] = None  # GE value in gp


@dataclass
class SelectionResult:
    """Result of item selection process."""
    run_id: str
    timestamp: str
    config: Dict[str, Any]
    items: List[Dict[str, Any]]
    total_eligible: int
    total_selected: int
    selection_reasons: Dict[str, int] = field(default_factory=dict)
    # Issue #120: High-value item tracking
    high_value_selected: int = 0
    high_value_eligible: int = 0


class ItemSelector:
    """Intelligent item selector for training pipeline."""

    def __init__(self, config: Optional[SelectionConfig] = None, config_path: Optional[str] = None):
        """
        Initialize item selector.

        Args:
            config: SelectionConfig object with selection parameters
            config_path: Path to training_config.yaml (loads item_selection section)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # Try default config path
            default_path = Path(__file__).parent.parent.parent / 'config' / 'training_config.yaml'
            if default_path.exists():
                self.config = self._load_config(str(default_path))
            else:
                self.config = SelectionConfig()

    def _load_config(self, config_path: str) -> SelectionConfig:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)

        item_config = yaml_config.get('item_selection', {})
        weights = item_config.get('priority_weights', {})

        # Load high-value configuration (Issue #120)
        hv_config = item_config.get('high_value', {})
        hv_weights = hv_config.get('priority_weights', {})
        high_value = HighValueConfig(
            enabled=hv_config.get('enabled', True),
            min_price_gp=hv_config.get('min_price_gp', 10_000_000),
            min_24h_volume=hv_config.get('min_24h_volume', 100),
            min_training_rows=hv_config.get('min_training_rows', 5000),
            max_items_per_run=hv_config.get('max_items_per_run', 20),
            force_include_items=hv_config.get('force_include_items', []),
            min_history_days=hv_config.get('min_history_days', 30),
            weight_no_model=hv_weights.get('no_model', 100),
            weight_high_price=hv_weights.get('high_price', 50),
            weight_low_manipulation_risk=hv_weights.get('low_manipulation_risk', 30),
        )

        return SelectionConfig(
            min_24h_volume=item_config.get('min_24h_volume', 10000),
            min_training_rows=item_config.get('min_training_rows', 5000),
            max_model_age_days=item_config.get('max_model_age_days', 30),
            max_calibration_error=item_config.get('max_calibration_error', 0.15),
            max_items_per_run=item_config.get('max_items_per_run', 50),
            discovery_max_items=item_config.get('discovery', {}).get('max_items', 400),
            weight_no_model=weights.get('no_model', 100),
            weight_auc_degradation=weights.get('auc_degradation', 80),
            weight_calibration_error=weights.get('calibration_error', 60),
            weight_model_staleness=weights.get('model_staleness', 40),
            weight_high_volume=weights.get('high_volume', 20),
            high_value=high_value,
        )

    def select_items_for_training(
        self,
        max_items: Optional[int] = None,
        discovery: bool = False,
        force_items: Optional[List[int]] = None,
        include_high_value: bool = True,
        high_value_only: bool = False
    ) -> List[SelectedItem]:
        """
        Select items that need training.

        Args:
            max_items: Maximum items to select (defaults to config value)
            discovery: If True, use discovery mode (larger item count)
            force_items: List of item_ids to force-include
            include_high_value: Include high-value items (Issue #120)
            high_value_only: Only select high-value items

        Returns:
            List of SelectedItem objects ordered by priority
        """
        if max_items is None:
            max_items = self.config.discovery_max_items if discovery else self.config.max_items_per_run

        logger.info(f"Selecting items for training (max={max_items}, discovery={discovery}, "
                    f"include_high_value={include_high_value}, high_value_only={high_value_only})")

        scored_items = []

        # Get regular candidate items (unless high_value_only)
        if not high_value_only:
            candidates = self._get_candidate_items()
            logger.info(f"Found {len(candidates)} regular candidate items")

            for candidate in candidates:
                score, reason = self._score_item(candidate, discovery)
                if score > 0:
                    scored_items.append(SelectedItem(
                        item_id=candidate['item_id'],
                        item_name=candidate['item_name'],
                        reason=reason,
                        priority_score=score,
                        current_model_id=candidate.get('model_id'),
                        current_model_auc=candidate.get('mean_auc'),
                        model_age_days=candidate.get('model_age_days'),
                        calibration_error=candidate.get('calibration_error'),
                        volume_24h=candidate.get('volume_24h'),
                        data_rows=candidate.get('data_rows'),
                        is_high_value=False,
                    ))

        # Get high-value items (Issue #120)
        if include_high_value and self.config.high_value.enabled:
            hv_candidates = self._get_high_value_candidate_items()
            logger.info(f"Found {len(hv_candidates)} high-value candidate items")

            # Track existing item IDs to avoid duplicates
            existing_ids = {item.item_id for item in scored_items}

            for candidate in hv_candidates:
                if candidate['item_id'] in existing_ids:
                    continue  # Skip if already selected as regular item

                score, reason = self._score_high_value_item(candidate)
                if score > 0:
                    scored_items.append(SelectedItem(
                        item_id=candidate['item_id'],
                        item_name=candidate['item_name'],
                        reason=f"high_value,{reason}",
                        priority_score=score,
                        current_model_id=candidate.get('model_id'),
                        current_model_auc=candidate.get('mean_auc'),
                        model_age_days=candidate.get('model_age_days'),
                        calibration_error=candidate.get('calibration_error'),
                        volume_24h=candidate.get('volume_24h'),
                        data_rows=candidate.get('data_rows'),
                        is_high_value=True,
                        item_price=candidate.get('item_price'),
                    ))

        # Sort by priority score descending
        scored_items.sort(key=lambda x: x.priority_score, reverse=True)

        # Add forced items at the top (if not already included)
        if force_items:
            existing_ids = {item.item_id for item in scored_items}
            for item_id in force_items:
                if item_id not in existing_ids:
                    item_info = self._get_item_info(item_id)
                    if item_info:
                        scored_items.insert(0, SelectedItem(
                            item_id=item_id,
                            item_name=item_info.get('name', f'Item {item_id}'),
                            reason='forced',
                            priority_score=1000,
                        ))

        # Add force-included high-value items
        if self.config.high_value.force_include_items:
            existing_ids = {item.item_id for item in scored_items}
            for item_id in self.config.high_value.force_include_items:
                if item_id not in existing_ids:
                    item_info = self._get_item_info(item_id)
                    if item_info:
                        scored_items.insert(0, SelectedItem(
                            item_id=item_id,
                            item_name=item_info.get('name', f'Item {item_id}'),
                            reason='forced_high_value',
                            priority_score=1000,
                            is_high_value=True,
                        ))

        # Limit to max_items
        selected = scored_items[:max_items]

        # Count high-value items selected
        hv_count = sum(1 for item in selected if item.is_high_value)
        logger.info(f"Selected {len(selected)} items for training ({hv_count} high-value)")

        return selected

    def _get_candidate_items(self) -> List[Dict[str, Any]]:
        """
        Query database for candidate items with their metrics.

        Returns items with:
        - Recent trading volume
        - Current model info (if any)
        - Recent performance metrics (if any)
        - Data row counts
        """
        query = """
        WITH item_volume AS (
            -- Calculate 24h trading volume
            SELECT
                item_id,
                SUM(high_price_volume + low_price_volume) as volume_24h
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY item_id
        ),
        item_data_counts AS (
            -- Count total data rows per item (last 6 months)
            SELECT
                item_id,
                COUNT(*) as data_rows
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '6 months'
            GROUP BY item_id
        ),
        active_models AS (
            -- Get active model per item
            SELECT
                mr.item_id,
                mr.id as model_id,
                mr.mean_auc,
                mr.trained_at,
                EXTRACT(EPOCH FROM NOW() - mr.trained_at) / 86400 as model_age_days
            FROM model_registry mr
            WHERE mr.status = 'ACTIVE'
        ),
        latest_performance AS (
            -- Get latest performance metrics
            SELECT DISTINCT ON (item_id)
                item_id,
                calibration_error,
                estimated_auc
            FROM model_performance
            WHERE window_hours = 24
            ORDER BY item_id, time DESC
        )
        SELECT
            i.item_id,
            i.name as item_name,
            COALESCE(v.volume_24h, 0) as volume_24h,
            COALESCE(dc.data_rows, 0) as data_rows,
            am.model_id,
            am.mean_auc,
            am.trained_at,
            am.model_age_days,
            lp.calibration_error,
            lp.estimated_auc
        FROM items i
        LEFT JOIN item_volume v ON v.item_id = i.item_id
        LEFT JOIN item_data_counts dc ON dc.item_id = i.item_id
        LEFT JOIN active_models am ON am.item_id = i.item_id
        LEFT JOIN latest_performance lp ON lp.item_id = i.item_id
        WHERE
            -- Basic filters
            i.item_id > 0
            -- Volume threshold
            AND COALESCE(v.volume_24h, 0) >= %(min_volume)s
            -- Data quality
            AND COALESCE(dc.data_rows, 0) >= %(min_rows)s
        ORDER BY
            -- Priority: no model > degraded > stale > high volume
            CASE WHEN am.model_id IS NULL THEN 0 ELSE 1 END,
            COALESCE(lp.calibration_error, 0) DESC,
            COALESCE(am.model_age_days, 999) DESC,
            v.volume_24h DESC
        """

        candidates = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query, {
                        'min_volume': self.config.min_24h_volume,
                        'min_rows': self.config.min_training_rows,
                    })
                    columns = [desc[0] for desc in cur.description]
                    for row in cur.fetchall():
                        candidates.append(dict(zip(columns, row)))
                except Exception as e:
                    # If model_registry doesn't exist yet, fall back to simpler query
                    logger.warning(f"Full query failed ({e}), using fallback query")
                    candidates = self._get_candidate_items_fallback()

        return candidates

    def _get_candidate_items_fallback(self) -> List[Dict[str, Any]]:
        """Fallback query when model_registry doesn't exist yet."""
        query = """
        WITH item_volume AS (
            SELECT
                item_id,
                SUM(high_price_volume + low_price_volume) as volume_24h
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY item_id
        ),
        item_data_counts AS (
            SELECT
                item_id,
                COUNT(*) as data_rows
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '6 months'
            GROUP BY item_id
        )
        SELECT
            i.item_id,
            i.name as item_name,
            COALESCE(v.volume_24h, 0) as volume_24h,
            COALESCE(dc.data_rows, 0) as data_rows,
            NULL::bigint as model_id,
            NULL::float as mean_auc,
            NULL::timestamp as trained_at,
            NULL::float as model_age_days,
            NULL::float as calibration_error,
            NULL::float as estimated_auc
        FROM items i
        LEFT JOIN item_volume v ON v.item_id = i.item_id
        LEFT JOIN item_data_counts dc ON dc.item_id = i.item_id
        WHERE
            i.item_id > 0
            AND COALESCE(v.volume_24h, 0) >= %(min_volume)s
            AND COALESCE(dc.data_rows, 0) >= %(min_rows)s
        ORDER BY v.volume_24h DESC
        """

        candidates = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {
                    'min_volume': self.config.min_24h_volume,
                    'min_rows': self.config.min_training_rows,
                })
                columns = [desc[0] for desc in cur.description]
                for row in cur.fetchall():
                    candidates.append(dict(zip(columns, row)))

        return candidates

    def _score_item(self, item: Dict[str, Any], discovery: bool) -> tuple:
        """
        Calculate priority score for an item.

        Returns:
            (score, reason) tuple
        """
        score = 0
        reasons = []

        # No model exists - highest priority
        if item.get('model_id') is None:
            score += self.config.weight_no_model
            reasons.append('no_model')

        # Performance degradation
        if item.get('calibration_error') and item['calibration_error'] > self.config.max_calibration_error:
            score += self.config.weight_calibration_error
            reasons.append(f'calibration_error:{item["calibration_error"]:.3f}')

        # Model staleness
        if item.get('model_age_days') and item['model_age_days'] > self.config.max_model_age_days:
            # More stale = higher priority
            staleness_factor = min(item['model_age_days'] / self.config.max_model_age_days, 3.0)
            score += self.config.weight_model_staleness * staleness_factor
            reasons.append(f'stale:{item["model_age_days"]:.0f}d')

        # High volume bonus (normalized)
        if item.get('volume_24h'):
            volume_factor = min(item['volume_24h'] / 1000000, 1.0)  # Cap at 1M volume
            score += self.config.weight_high_volume * volume_factor

        # Discovery mode: include items even without specific triggers
        if discovery and score == 0 and item.get('model_id') is not None:
            # Include for discovery if model exists but might benefit from refresh
            model_age = item.get('model_age_days', 0)
            if model_age > 7:  # At least a week old
                score += 10
                reasons.append('discovery')

        reason = ','.join(reasons) if reasons else 'none'
        return score, reason

    def _get_high_value_candidate_items(self) -> List[Dict[str, Any]]:
        """
        Query database for high-value items (Issue #120).

        Returns items with:
        - Price >= min_price_gp threshold
        - Relaxed volume threshold
        - Minimum data history
        """
        hv = self.config.high_value
        query = """
        WITH recent_prices AS (
            SELECT
                item_id,
                AVG(avg_high_price) as ge_price
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '24 hours'
              AND avg_high_price > 0
            GROUP BY item_id
        ),
        item_volume AS (
            SELECT
                item_id,
                SUM(high_price_volume + low_price_volume) as volume_24h
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY item_id
        ),
        item_data_counts AS (
            SELECT
                item_id,
                COUNT(*) as data_rows,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '6 months'
            GROUP BY item_id
        ),
        active_models AS (
            SELECT
                mr.item_id,
                mr.id as model_id,
                mr.mean_auc,
                mr.trained_at,
                EXTRACT(EPOCH FROM NOW() - mr.trained_at) / 86400 as model_age_days
            FROM model_registry mr
            WHERE mr.status = 'ACTIVE'
        ),
        latest_performance AS (
            SELECT DISTINCT ON (item_id)
                item_id,
                calibration_error,
                estimated_auc
            FROM model_performance
            WHERE window_hours = 24
            ORDER BY item_id, time DESC
        )
        SELECT
            i.item_id,
            i.name as item_name,
            COALESCE(rp.ge_price, 0)::bigint as item_price,
            8 as buy_limit,
            COALESCE(v.volume_24h, 0) as volume_24h,
            COALESCE(dc.data_rows, 0) as data_rows,
            dc.first_seen,
            dc.last_seen,
            EXTRACT(EPOCH FROM (dc.last_seen - dc.first_seen)) / 86400 as history_days,
            am.model_id,
            am.mean_auc,
            am.trained_at,
            am.model_age_days,
            lp.calibration_error,
            lp.estimated_auc
        FROM items i
        JOIN recent_prices rp ON rp.item_id = i.item_id
        LEFT JOIN item_volume v ON v.item_id = i.item_id
        LEFT JOIN item_data_counts dc ON dc.item_id = i.item_id
        LEFT JOIN active_models am ON am.item_id = i.item_id
        LEFT JOIN latest_performance lp ON lp.item_id = i.item_id
        WHERE
            -- Basic filters
            i.item_id > 0
            -- High-value threshold (GE price >= min_price_gp)
            AND COALESCE(rp.ge_price, 0) >= %(min_price)s
            -- Relaxed volume threshold for high-value items
            AND COALESCE(v.volume_24h, 0) >= %(min_volume)s
            -- Data quality
            AND COALESCE(dc.data_rows, 0) >= %(min_rows)s
            -- Minimum history requirement
            AND EXTRACT(EPOCH FROM (dc.last_seen - dc.first_seen)) / 86400 >= %(min_history_days)s
        ORDER BY
            -- Priority: no model > high price > volume
            CASE WHEN am.model_id IS NULL THEN 0 ELSE 1 END,
            rp.ge_price DESC,
            v.volume_24h DESC
        LIMIT %(max_items)s
        """

        candidates = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query, {
                        'min_price': hv.min_price_gp,
                        'min_volume': hv.min_24h_volume,
                        'min_rows': hv.min_training_rows,
                        'min_history_days': hv.min_history_days,
                        'max_items': hv.max_items_per_run * 2,  # Get extra for filtering
                    })
                    columns = [desc[0] for desc in cur.description]
                    for row in cur.fetchall():
                        candidates.append(dict(zip(columns, row)))
                except Exception as e:
                    logger.warning(f"High-value query failed ({e}), using fallback")
                    candidates = self._get_high_value_candidate_items_fallback()

        return candidates

    def _get_high_value_candidate_items_fallback(self) -> List[Dict[str, Any]]:
        """Fallback query when model_registry doesn't exist yet."""
        hv = self.config.high_value
        query = """
        WITH recent_prices AS (
            SELECT
                item_id,
                AVG(avg_high_price) as ge_price
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '24 hours'
              AND avg_high_price > 0
            GROUP BY item_id
        ),
        item_volume AS (
            SELECT
                item_id,
                SUM(high_price_volume + low_price_volume) as volume_24h
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY item_id
        ),
        item_data_counts AS (
            SELECT
                item_id,
                COUNT(*) as data_rows,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM price_data_5min
            WHERE timestamp > NOW() - INTERVAL '6 months'
            GROUP BY item_id
        )
        SELECT
            i.item_id,
            i.name as item_name,
            COALESCE(rp.ge_price, 0)::bigint as item_price,
            8 as buy_limit,
            COALESCE(v.volume_24h, 0) as volume_24h,
            COALESCE(dc.data_rows, 0) as data_rows,
            dc.first_seen,
            dc.last_seen,
            EXTRACT(EPOCH FROM (dc.last_seen - dc.first_seen)) / 86400 as history_days,
            NULL::bigint as model_id,
            NULL::float as mean_auc,
            NULL::timestamp as trained_at,
            NULL::float as model_age_days,
            NULL::float as calibration_error,
            NULL::float as estimated_auc
        FROM items i
        JOIN recent_prices rp ON rp.item_id = i.item_id
        LEFT JOIN item_volume v ON v.item_id = i.item_id
        LEFT JOIN item_data_counts dc ON dc.item_id = i.item_id
        WHERE
            i.item_id > 0
            AND COALESCE(rp.ge_price, 0) >= %(min_price)s
            AND COALESCE(v.volume_24h, 0) >= %(min_volume)s
            AND COALESCE(dc.data_rows, 0) >= %(min_rows)s
        ORDER BY rp.ge_price DESC, v.volume_24h DESC
        LIMIT %(max_items)s
        """

        candidates = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {
                    'min_price': hv.min_price_gp,
                    'min_volume': hv.min_24h_volume,
                    'min_rows': hv.min_training_rows,
                    'max_items': hv.max_items_per_run * 2,
                })
                columns = [desc[0] for desc in cur.description]
                for row in cur.fetchall():
                    candidates.append(dict(zip(columns, row)))

        return candidates

    def _score_high_value_item(self, item: Dict[str, Any]) -> tuple:
        """
        Calculate priority score for a high-value item.

        Returns:
            (score, reason) tuple
        """
        hv = self.config.high_value
        score = 0
        reasons = []

        # No model exists - highest priority
        if item.get('model_id') is None:
            score += hv.weight_no_model
            reasons.append('no_model')

        # Higher price = higher priority (normalized to 1B gp max)
        if item.get('item_price'):
            price_factor = min(item['item_price'] / 1_000_000_000, 1.0)
            score += hv.weight_high_price * price_factor
            reasons.append(f'price:{item["item_price"] / 1_000_000:.0f}M')

        # Default score if no specific triggers
        if score == 0:
            score = 10  # Minimum score for eligible high-value items
            reasons.append('eligible')

        reason = ','.join(reasons) if reasons else 'none'
        return score, reason

    def _get_item_info(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get basic item info by ID."""
        with get_db_cursor() as cur:
            cur.execute(
                "SELECT item_id, name FROM items WHERE item_id = %s",
                (item_id,)
            )
            row = cur.fetchone()
            if row:
                return {'item_id': row[0], 'name': row[1]}
        return None

    def create_selection_result(
        self,
        run_id: str,
        items: List[SelectedItem],
        total_eligible: int,
        high_value_eligible: int = 0
    ) -> SelectionResult:
        """Create a SelectionResult object for persistence."""
        # Count reasons
        reason_counts = {}
        for item in items:
            for reason_part in item.reason.split(','):
                base_reason = reason_part.split(':')[0]
                reason_counts[base_reason] = reason_counts.get(base_reason, 0) + 1

        # Count high-value items selected
        high_value_selected = sum(1 for item in items if item.is_high_value)

        return SelectionResult(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            config=asdict(self.config),
            items=[asdict(item) for item in items],
            total_eligible=total_eligible,
            total_selected=len(items),
            selection_reasons=reason_counts,
            high_value_selected=high_value_selected,
            high_value_eligible=high_value_eligible,
        )

    def save_selection(self, result: SelectionResult, output_path: str) -> None:
        """Save selection result to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        logger.info(f"Saved selection to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Select items for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.training.item_selector --run-id 20260111_050000 --output items.json
    python -m src.training.item_selector --discovery --max-items 400
    python -m src.training.item_selector --dry-run --verbose
    python -m src.training.item_selector --high-value-only  # Only high-value items
    python -m src.training.item_selector --no-high-value    # Exclude high-value items
        """
    )
    parser.add_argument('--run-id', help='Training run ID (default: timestamp)')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--max-items', type=int, help='Maximum items to select')
    parser.add_argument('--discovery', action='store_true', help='Discovery mode (more items)')
    parser.add_argument('--force-items', help='Comma-separated item IDs to force-include')
    parser.add_argument('--dry-run', action='store_true', help='Show selection without saving')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--config', help='Path to training_config.yaml')
    # High-value item options (Issue #120)
    parser.add_argument('--high-value-only', action='store_true',
                        help='Only select high-value items (Issue #120)')
    parser.add_argument('--no-high-value', action='store_true',
                        help='Exclude high-value items from selection')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Generate run_id if not provided
    run_id = args.run_id or datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    # Parse force items
    force_items = None
    if args.force_items:
        force_items = [int(x.strip()) for x in args.force_items.split(',')]

    # Initialize selector
    selector = ItemSelector(config_path=args.config)

    # Determine high-value inclusion
    include_high_value = not args.no_high_value
    high_value_only = args.high_value_only

    # Select items
    items = selector.select_items_for_training(
        max_items=args.max_items,
        discovery=args.discovery,
        force_items=force_items,
        include_high_value=include_high_value,
        high_value_only=high_value_only,
    )

    # Create result
    # Get total eligible count for stats
    total_eligible = len(selector._get_candidate_items())
    hv_eligible = len(selector._get_high_value_candidate_items()) if include_high_value else 0
    result = selector.create_selection_result(run_id, items, total_eligible, hv_eligible)

    # Display results
    print(f"\n{'=' * 60}")
    print(f"Item Selection Results - Run {run_id}")
    print(f"{'=' * 60}")
    print(f"Total eligible items: {total_eligible}")
    print(f"High-value eligible: {hv_eligible}")
    print(f"Items selected: {len(items)} ({result.high_value_selected} high-value)")
    print("\nSelection reasons:")
    for reason, count in sorted(result.selection_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    if args.verbose:
        print(f"\n{'=' * 60}")
        print("Selected Items:")
        print(f"{'=' * 60}")
        for item in items[:20]:  # Show first 20
            hv_flag = "[HV]" if item.is_high_value else ""
            print(f"  {item.item_id:>6} | {item.item_name[:30]:<30} | "
                  f"score={item.priority_score:.1f} | {hv_flag} {item.reason}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more")

    # Save if output specified and not dry run
    if args.output and not args.dry_run:
        selector.save_selection(result, args.output)
        print(f"\nSaved to: {args.output}")
    elif args.dry_run:
        print("\n[DRY RUN] No files written")

    # Return for programmatic use
    return result


if __name__ == '__main__':
    main()
