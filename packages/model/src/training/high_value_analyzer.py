"""
High-Value Item Analyzer - Data Quality and Risk Assessment
============================================================

Analyzes high-value items (Issue #120) to assess:
1. Data quality (completeness, gaps, consistency)
2. Manipulation risk (volume spikes, spread anomalies)
3. Training viability (sufficient history, positive samples)

Usage:
    from src.training.high_value_analyzer import HighValueAnalyzer

    analyzer = HighValueAnalyzer()
    assessment = analyzer.analyze_item(item_id=20997)  # Twisted bow

    if assessment.training_viable:
        print(f"Item {item_id} is viable for training")
        print(f"Data quality: {assessment.data_quality_score:.2f}")
        print(f"Manipulation risk: {assessment.manipulation_risk}")

CLI:
    python -m src.training.high_value_analyzer --item-id 20997
    python -m src.training.high_value_analyzer --analyze-all --output assessments.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db_utils import get_db_connection

logger = logging.getLogger(__name__)


@dataclass
class HighValueAssessment:
    """Assessment result for a high-value item."""
    item_id: int
    item_name: str
    item_price: int

    # Data quality metrics
    data_quality_score: float  # 0-1 score
    completeness: float  # Fraction of expected intervals present
    history_days: float  # Days of history available
    data_rows: int  # Total price data rows

    # Manipulation risk
    manipulation_risk: str  # 'low', 'medium', 'high'
    volume_stability: float  # 0-1, higher is more stable
    spread_stability: float  # 0-1, higher is more stable
    volume_spike_count: int  # Number of volume anomalies detected

    # Training viability
    training_viable: bool
    viability_reason: str

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Metadata
    assessed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class HighValueAnalyzer:
    """Analyzer for high-value item training viability."""

    # Thresholds for assessment
    MIN_HISTORY_DAYS = 30
    MIN_COMPLETENESS = 0.5
    MIN_DATA_ROWS = 5000
    VOLUME_SPIKE_THRESHOLD = 5.0  # Standard deviations

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def analyze_item(self, item_id: int) -> Optional[HighValueAssessment]:
        """
        Analyze a single item for training viability.

        Args:
            item_id: The item ID to analyze

        Returns:
            HighValueAssessment object or None if item not found
        """
        # Fetch item data
        item_data = self._fetch_item_data(item_id)
        if not item_data:
            logger.warning(f"Item {item_id} not found")
            return None

        # Calculate data quality metrics
        completeness = item_data.get('completeness', 0)
        history_days = item_data.get('history_days', 0)
        data_rows = item_data.get('data_rows', 0)

        # Calculate data quality score (weighted average)
        completeness_score = min(completeness / 0.8, 1.0)  # 80% completeness = 1.0
        history_score = min(history_days / 180, 1.0)  # 180 days = 1.0
        rows_score = min(data_rows / 50000, 1.0)  # 50k rows = 1.0
        data_quality_score = (
            completeness_score * 0.4 +
            history_score * 0.3 +
            rows_score * 0.3
        )

        # Analyze volume patterns for manipulation risk
        volume_stats = self._analyze_volume_patterns(item_id)
        volume_stability = volume_stats.get('stability', 0.5)
        volume_spike_count = volume_stats.get('spike_count', 0)

        # Analyze spread patterns
        spread_stats = self._analyze_spread_patterns(item_id)
        spread_stability = spread_stats.get('stability', 0.5)

        # Determine manipulation risk level
        if volume_spike_count > 10 or volume_stability < 0.3 or spread_stability < 0.3:
            manipulation_risk = 'high'
        elif volume_spike_count > 5 or volume_stability < 0.5 or spread_stability < 0.5:
            manipulation_risk = 'medium'
        else:
            manipulation_risk = 'low'

        # Determine training viability
        warnings = []
        training_viable = True
        viability_reason = "Meets all requirements"

        if history_days < self.MIN_HISTORY_DAYS:
            training_viable = False
            viability_reason = f"Insufficient history ({history_days:.0f} days < {self.MIN_HISTORY_DAYS})"
        elif completeness < self.MIN_COMPLETENESS:
            training_viable = False
            viability_reason = f"Low data completeness ({completeness:.1%} < {self.MIN_COMPLETENESS:.0%})"
        elif data_rows < self.MIN_DATA_ROWS:
            training_viable = False
            viability_reason = f"Insufficient data rows ({data_rows} < {self.MIN_DATA_ROWS})"
        elif manipulation_risk == 'high':
            warnings.append("High manipulation risk - model may be unreliable")
            viability_reason = "Viable but high manipulation risk"

        if manipulation_risk == 'medium':
            warnings.append("Medium manipulation risk - monitor model performance closely")

        if completeness < 0.7:
            warnings.append(f"Data completeness below 70% ({completeness:.1%})")

        return HighValueAssessment(
            item_id=item_id,
            item_name=item_data.get('item_name', f'Item {item_id}'),
            item_price=item_data.get('item_price', 0),
            data_quality_score=round(data_quality_score, 3),
            completeness=round(completeness, 4),
            history_days=round(history_days, 1),
            data_rows=data_rows,
            manipulation_risk=manipulation_risk,
            volume_stability=round(volume_stability, 3),
            spread_stability=round(spread_stability, 3),
            volume_spike_count=volume_spike_count,
            training_viable=training_viable,
            viability_reason=viability_reason,
            warnings=warnings,
        )

    def analyze_all_high_value_items(
        self,
        min_price_gp: int = 10_000_000
    ) -> List[HighValueAssessment]:
        """
        Analyze all high-value items in the database.

        Args:
            min_price_gp: Minimum price to consider high-value

        Returns:
            List of HighValueAssessment objects
        """
        # Get all high-value item IDs
        item_ids = self._get_high_value_item_ids(min_price_gp)
        logger.info(f"Analyzing {len(item_ids)} high-value items")

        assessments = []
        for item_id in item_ids:
            assessment = self.analyze_item(item_id)
            if assessment:
                assessments.append(assessment)

        # Sort by price descending
        assessments.sort(key=lambda x: x.item_price, reverse=True)

        return assessments

    def _fetch_item_data(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Fetch basic item data and statistics."""
        query = """
        WITH item_stats AS (
            SELECT
                item_id,
                COUNT(*) as data_rows,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen,
                EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 86400 as history_days
            FROM price_data_5min
            WHERE item_id = %(item_id)s
                AND timestamp > NOW() - INTERVAL '6 months'
            GROUP BY item_id
        ),
        expected_intervals AS (
            SELECT
                EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 300 as expected_count
            FROM price_data_5min
            WHERE item_id = %(item_id)s
                AND timestamp > NOW() - INTERVAL '6 months'
        )
        SELECT
            i.item_id,
            i.name as item_name,
            COALESCE(i.value, 0) as item_price,
            COALESCE(s.data_rows, 0) as data_rows,
            COALESCE(s.history_days, 0) as history_days,
            COALESCE(s.data_rows::float / NULLIF(e.expected_count, 0), 0) as completeness
        FROM items i
        LEFT JOIN item_stats s ON s.item_id = i.item_id
        LEFT JOIN expected_intervals e ON true
        WHERE i.item_id = %(item_id)s
        """

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {'item_id': item_id})
                row = cur.fetchone()
                if row:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row))
        return None

    def _analyze_volume_patterns(self, item_id: int) -> Dict[str, Any]:
        """Analyze volume patterns for manipulation detection."""
        query = """
        WITH daily_volumes AS (
            SELECT
                DATE(timestamp) as date,
                SUM(high_price_volume + low_price_volume) as daily_vol
            FROM price_data_5min
            WHERE item_id = %(item_id)s
                AND timestamp > NOW() - INTERVAL '90 days'
            GROUP BY DATE(timestamp)
        ),
        vol_stats AS (
            SELECT
                AVG(daily_vol) as avg_vol,
                STDDEV(daily_vol) as stddev_vol,
                MIN(daily_vol) as min_vol,
                MAX(daily_vol) as max_vol
            FROM daily_volumes
        )
        SELECT
            vs.avg_vol,
            vs.stddev_vol,
            vs.min_vol,
            vs.max_vol,
            COALESCE(vs.stddev_vol / NULLIF(vs.avg_vol, 0), 1) as cv,
            (
                SELECT COUNT(*)
                FROM daily_volumes dv, vol_stats vs2
                WHERE dv.daily_vol > vs2.avg_vol + %(threshold)s * vs2.stddev_vol
            ) as spike_count
        FROM vol_stats vs
        """

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {
                    'item_id': item_id,
                    'threshold': self.VOLUME_SPIKE_THRESHOLD
                })
                row = cur.fetchone()
                if row:
                    cv = row[4] or 1.0  # Coefficient of variation
                    stability = max(0, 1 - min(cv, 1))  # Lower CV = higher stability
                    return {
                        'avg_vol': row[0] or 0,
                        'stddev_vol': row[1] or 0,
                        'stability': stability,
                        'spike_count': row[5] or 0,
                    }
        return {'stability': 0.5, 'spike_count': 0}

    def _analyze_spread_patterns(self, item_id: int) -> Dict[str, Any]:
        """Analyze spread patterns for manipulation detection."""
        query = """
        WITH spreads AS (
            SELECT
                timestamp,
                CASE
                    WHEN avg_high_price > 0 AND avg_low_price > 0
                    THEN (avg_high_price - avg_low_price)::float / avg_high_price
                    ELSE NULL
                END as spread_pct
            FROM price_data_5min
            WHERE item_id = %(item_id)s
                AND timestamp > NOW() - INTERVAL '90 days'
                AND avg_high_price > 0
                AND avg_low_price > 0
        ),
        spread_stats AS (
            SELECT
                AVG(spread_pct) as avg_spread,
                STDDEV(spread_pct) as stddev_spread
            FROM spreads
            WHERE spread_pct IS NOT NULL
        )
        SELECT
            ss.avg_spread,
            ss.stddev_spread,
            COALESCE(ss.stddev_spread / NULLIF(ss.avg_spread, 0), 1) as cv
        FROM spread_stats ss
        """

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {'item_id': item_id})
                row = cur.fetchone()
                if row:
                    cv = row[2] or 1.0
                    stability = max(0, 1 - min(cv, 1))
                    return {
                        'avg_spread': row[0] or 0,
                        'stability': stability,
                    }
        return {'stability': 0.5}

    def _get_high_value_item_ids(self, min_price_gp: int) -> List[int]:
        """Get all item IDs meeting the high-value threshold."""
        query = """
        SELECT item_id
        FROM items
        WHERE value >= %(min_price)s
            AND item_id > 0
        ORDER BY value DESC
        """

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {'min_price': min_price_gp})
                return [row[0] for row in cur.fetchall()]

    def save_assessments(
        self,
        assessments: List[HighValueAssessment],
        output_path: str
    ) -> None:
        """Save assessments to JSON file."""
        data = {
            'assessed_at': datetime.utcnow().isoformat(),
            'total_items': len(assessments),
            'viable_count': sum(1 for a in assessments if a.training_viable),
            'risk_distribution': {
                'low': sum(1 for a in assessments if a.manipulation_risk == 'low'),
                'medium': sum(1 for a in assessments if a.manipulation_risk == 'medium'),
                'high': sum(1 for a in assessments if a.manipulation_risk == 'high'),
            },
            'assessments': [asdict(a) for a in assessments],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(assessments)} assessments to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze high-value items for training viability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.training.high_value_analyzer --item-id 20997
    python -m src.training.high_value_analyzer --analyze-all --output assessments.json
    python -m src.training.high_value_analyzer --analyze-all --min-price 100000000
        """
    )
    parser.add_argument('--item-id', type=int, help='Analyze a specific item')
    parser.add_argument('--analyze-all', action='store_true', help='Analyze all high-value items')
    parser.add_argument('--min-price', type=int, default=10_000_000,
                        help='Minimum price for high-value (default: 10M)')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    analyzer = HighValueAnalyzer()

    if args.item_id:
        # Analyze single item
        assessment = analyzer.analyze_item(args.item_id)
        if assessment:
            print("\n" + "=" * 60)
            print(f"High-Value Item Assessment: {assessment.item_name}")
            print("=" * 60)
            print(f"Item ID: {assessment.item_id}")
            print(f"Price: {assessment.item_price:,} gp")
            print("\nData Quality:")
            print(f"  Score: {assessment.data_quality_score:.2f}")
            print(f"  Completeness: {assessment.completeness:.1%}")
            print(f"  History: {assessment.history_days:.0f} days")
            print(f"  Data rows: {assessment.data_rows:,}")
            print("\nManipulation Risk:")
            print(f"  Level: {assessment.manipulation_risk.upper()}")
            print(f"  Volume stability: {assessment.volume_stability:.2f}")
            print(f"  Spread stability: {assessment.spread_stability:.2f}")
            print(f"  Volume spikes: {assessment.volume_spike_count}")
            print("\nTraining Viability:")
            print(f"  Viable: {'YES' if assessment.training_viable else 'NO'}")
            print(f"  Reason: {assessment.viability_reason}")
            if assessment.warnings:
                print("\nWarnings:")
                for warning in assessment.warnings:
                    print(f"  - {warning}")
        else:
            print(f"Item {args.item_id} not found")

    elif args.analyze_all:
        # Analyze all high-value items
        assessments = analyzer.analyze_all_high_value_items(args.min_price)

        # Print summary
        viable_count = sum(1 for a in assessments if a.training_viable)
        print("\n" + "=" * 60)
        print("High-Value Item Analysis Summary")
        print("=" * 60)
        print(f"Total items analyzed: {len(assessments)}")
        print(f"Training viable: {viable_count}")
        print("\nRisk distribution:")
        for risk in ['low', 'medium', 'high']:
            count = sum(1 for a in assessments if a.manipulation_risk == risk)
            print(f"  {risk}: {count}")

        if args.verbose:
            print("\n" + "=" * 60)
            print("Viable Items (top 20 by price):")
            print("=" * 60)
            viable = [a for a in assessments if a.training_viable][:20]
            for a in viable:
                risk_flag = f"[{a.manipulation_risk[0].upper()}]"
                print(f"  {a.item_id:>6} | {a.item_name[:30]:<30} | "
                      f"{a.item_price / 1_000_000:.0f}M | {risk_flag}")

        if args.output:
            analyzer.save_assessments(assessments, args.output)
            print(f"\nSaved to: {args.output}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
