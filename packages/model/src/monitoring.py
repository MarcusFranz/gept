#!/usr/bin/env python3
"""
Model Monitoring and Alerting Module

Monitors model health and sends alerts for:
1. Calibration drift - Model predictions not matching actual fill rates
2. Prediction volume anomalies - Too few or too many predictions
3. Model degradation - Performance dropping over time
4. System health - DB connectivity, inference time

Usage:
    python monitoring.py --check all     # Run all checks
    python monitoring.py --check drift   # Check calibration drift only
    python monitoring.py --send-alert    # Send Discord webhook on issues

Cron setup (run every hour):
    0 * * * * cd /path/to/GePT\ Model && python3 src/monitoring.py --check all >> logs/monitoring.log 2>&1
"""

import sys
import json
import argparse
import logging
import os
from datetime import datetime
from typing import List, Dict

import pandas as pd
import requests

# Centralized database connection management
from db_utils import get_simple_connection

# Alert thresholds
DRIFT_THRESHOLD = 0.25  # 25% relative calibration error
MIN_PREDICTIONS_PER_HOUR = 5000  # Alert if below this
MAX_PREDICTIONS_PER_HOUR = 10000  # Alert if above this
MAX_INFERENCE_TIME_SEC = 120  # Alert if inference takes longer

# Discord webhook (set via environment variable)
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class Alert:
    """Represents an alert to be sent."""

    SEVERITY_INFO = 'info'
    SEVERITY_WARNING = 'warning'
    SEVERITY_CRITICAL = 'critical'

    def __init__(self, title: str, message: str, severity: str = 'warning'):
        self.title = title
        self.message = message
        self.severity = severity
        self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'message': self.message,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat()
        }

    def __str__(self):
        icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}[self.severity]
        return f"{icon} [{self.severity.upper()}] {self.title}: {self.message}"


class ModelMonitor:
    """Monitors model health and performance."""

    def __init__(self):
        self.conn = None
        self.logger = logging.getLogger(__name__)
        self.alerts: List[Alert] = []

    def connect(self):
        """Open database connection (non-pooled for monitoring job)."""
        self.conn = get_simple_connection()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def add_alert(self, title: str, message: str, severity: str = 'warning'):
        """Add an alert to the queue."""
        alert = Alert(title, message, severity)
        self.alerts.append(alert)
        self.logger.warning(str(alert))

    def check_db_connectivity(self) -> bool:
        """Verify database connectivity."""
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            return True
        except Exception as e:
            self.add_alert(
                "Database Connectivity",
                f"Cannot connect to database: {e}",
                Alert.SEVERITY_CRITICAL
            )
            return False

    def check_calibration_drift(self, hours: int = 24) -> Dict:
        """
        Check for calibration drift in recent predictions.

        Compares predicted probabilities to actual fill rates.
        """
        query = """
            SELECT
                CASE
                    WHEN predicted_probability < 0.01 THEN '0-1%'
                    WHEN predicted_probability < 0.05 THEN '1-5%'
                    WHEN predicted_probability < 0.10 THEN '5-10%'
                    WHEN predicted_probability < 0.20 THEN '10-20%'
                    ELSE '20%+'
                END as bucket,
                COUNT(*) as count,
                AVG(predicted_probability) as avg_predicted,
                AVG(both_would_fill::int) as actual_rate
            FROM actual_fills
            WHERE time > NOW() - make_interval(hours => %s)
            GROUP BY bucket
            HAVING COUNT(*) >= 50
            ORDER BY bucket
        """

        try:
            df = pd.read_sql(query, self.conn, params=[hours])
        except Exception as e:
            self.logger.debug(f"Could not fetch calibration data: {e}")
            return {'status': 'no_data'}

        if len(df) == 0:
            return {'status': 'no_data'}

        results = {'status': 'ok', 'buckets': []}

        for _, row in df.iterrows():
            error = row['actual_rate'] - row['avg_predicted']
            rel_error = abs(error) / max(row['avg_predicted'], 0.01)

            bucket_result = {
                'bucket': row['bucket'],
                'count': int(row['count']),
                'predicted': float(row['avg_predicted']),
                'actual': float(row['actual_rate']),
                'error': float(error),
                'relative_error': float(rel_error)
            }
            results['buckets'].append(bucket_result)

            # Check for drift
            if rel_error > DRIFT_THRESHOLD and row['count'] >= 100:
                self.add_alert(
                    "Calibration Drift",
                    f"Bucket {row['bucket']}: predicted {row['avg_predicted']:.1%}, "
                    f"actual {row['actual_rate']:.1%} ({error:+.1%})",
                    Alert.SEVERITY_WARNING
                )
                results['status'] = 'drift_detected'

        return results

    def check_prediction_volume(self, hours: int = 1) -> Dict:
        """Check that prediction volume is within expected range."""
        query = """
            SELECT COUNT(*) as count
            FROM predictions
            WHERE time > NOW() - make_interval(hours => %s)
        """

        try:
            cur = self.conn.cursor()
            cur.execute(query, [hours])
            count = cur.fetchone()[0]
            cur.close()
        except Exception as e:
            self.add_alert(
                "Prediction Volume Check Failed",
                f"Could not query prediction count: {e}",
                Alert.SEVERITY_WARNING
            )
            return {'status': 'error'}

        result = {
            'status': 'ok',
            'count': count,
            'hours': hours,
            'per_hour': count / hours if hours > 0 else 0
        }

        if count == 0:
            self.add_alert(
                "No Predictions",
                f"No predictions generated in last {hours} hour(s)",
                Alert.SEVERITY_CRITICAL
            )
            result['status'] = 'no_predictions'
        elif count < MIN_PREDICTIONS_PER_HOUR * hours:
            self.add_alert(
                "Low Prediction Volume",
                f"Only {count:,} predictions in last {hours}h (expected {MIN_PREDICTIONS_PER_HOUR * hours:,}+)",
                Alert.SEVERITY_WARNING
            )
            result['status'] = 'low_volume'
        elif count > MAX_PREDICTIONS_PER_HOUR * hours:
            self.add_alert(
                "High Prediction Volume",
                f"{count:,} predictions in last {hours}h (expected <{MAX_PREDICTIONS_PER_HOUR * hours:,})",
                Alert.SEVERITY_INFO
            )
            result['status'] = 'high_volume'

        return result

    def check_high_probability_predictions(self, hours: int = 1) -> Dict:
        """Check for suspicious high-probability predictions (>30%)."""
        query = """
            SELECT COUNT(*) as count,
                   AVG(fill_probability) as avg_prob
            FROM predictions
            WHERE time > NOW() - make_interval(hours => %s)
              AND fill_probability > 0.30
        """

        try:
            cur = self.conn.cursor()
            cur.execute(query, [hours])
            row = cur.fetchone()
            count, avg_prob = row[0], row[1]
            cur.close()
        except Exception:
            return {'status': 'error'}

        result = {
            'status': 'ok',
            'high_prob_count': count,
            'avg_high_prob': float(avg_prob) if avg_prob else 0
        }

        if count > 0:
            # These predictions are broken per calibration_analysis.md
            self.add_alert(
                "High Probability Predictions",
                f"{count} predictions with >30% probability in last {hours}h. "
                f"These should be filtered (see calibration_analysis.md)",
                Alert.SEVERITY_WARNING
            )
            result['status'] = 'high_prob_detected'

        return result

    def check_latest_inference_time(self) -> Dict:
        """Check when last inference was run and its duration."""
        query = """
            SELECT MAX(time) as latest_time
            FROM predictions
        """

        try:
            cur = self.conn.cursor()
            cur.execute(query)
            latest = cur.fetchone()[0]
            cur.close()
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

        if latest is None:
            return {'status': 'no_predictions'}

        age_minutes = (datetime.now(latest.tzinfo) - latest).total_seconds() / 60

        result = {
            'status': 'ok',
            'latest_time': latest.isoformat(),
            'age_minutes': age_minutes
        }

        if age_minutes > 10:  # More than 2 inference cycles
            self.add_alert(
                "Stale Predictions",
                f"Last prediction was {age_minutes:.0f} minutes ago (expected <10 min)",
                Alert.SEVERITY_CRITICAL
            )
            result['status'] = 'stale'

        return result

    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get overall performance summary."""
        query = """
            SELECT
                COUNT(*) as total_predictions,
                SUM(both_would_fill::int) as total_fills,
                AVG(both_would_fill::int) as fill_rate,
                AVG(predicted_probability) as avg_predicted,
                AVG(CASE WHEN both_would_fill THEN predicted_probability ELSE NULL END) as avg_filled_pred
            FROM actual_fills
            WHERE time > NOW() - make_interval(hours => %s)
        """

        try:
            df = pd.read_sql(query, self.conn, params=[hours])
            row = df.iloc[0]
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

        return {
            'status': 'ok',
            'total_predictions': int(row['total_predictions']) if row['total_predictions'] else 0,
            'total_fills': int(row['total_fills']) if row['total_fills'] else 0,
            'fill_rate': float(row['fill_rate']) if row['fill_rate'] else 0,
            'avg_predicted': float(row['avg_predicted']) if row['avg_predicted'] else 0,
            'hours': hours
        }

    def run_all_checks(self) -> Dict:
        """Run all monitoring checks."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }

        # Database connectivity
        if not self.check_db_connectivity():
            results['checks']['db'] = {'status': 'failed'}
            return results
        results['checks']['db'] = {'status': 'ok'}

        # Calibration drift
        results['checks']['calibration'] = self.check_calibration_drift(hours=24)

        # Prediction volume
        results['checks']['volume'] = self.check_prediction_volume(hours=1)

        # High probability filter
        results['checks']['high_prob'] = self.check_high_probability_predictions(hours=1)

        # Inference recency
        results['checks']['recency'] = self.check_latest_inference_time()

        # Performance summary
        results['checks']['performance'] = self.get_performance_summary(hours=24)

        # Overall status
        results['alert_count'] = len(self.alerts)
        results['alerts'] = [a.to_dict() for a in self.alerts]

        critical_count = sum(1 for a in self.alerts if a.severity == Alert.SEVERITY_CRITICAL)
        warning_count = sum(1 for a in self.alerts if a.severity == Alert.SEVERITY_WARNING)

        if critical_count > 0:
            results['overall_status'] = 'critical'
        elif warning_count > 0:
            results['overall_status'] = 'warning'
        else:
            results['overall_status'] = 'healthy'

        return results


def send_discord_alert(alerts: List[Alert], webhook_url: str):
    """Send alerts to Discord webhook."""
    if not webhook_url:
        return

    if not alerts:
        return

    # Build message
    message_parts = []
    for alert in alerts:
        icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}[alert.severity]
        message_parts.append(f"{icon} **{alert.title}**\n{alert.message}")

    message = "\n\n".join(message_parts)

    # Determine embed color
    critical = any(a.severity == Alert.SEVERITY_CRITICAL for a in alerts)
    color = 0xFF0000 if critical else 0xFFA500  # Red for critical, orange for warning

    payload = {
        "embeds": [{
            "title": "GE Flipping Model Alert",
            "description": message,
            "color": color,
            "timestamp": datetime.now().isoformat()
        }]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Discord alert sent successfully")
    except Exception as e:
        logging.error(f"Failed to send Discord alert: {e}")


def main():
    parser = argparse.ArgumentParser(description='Monitor model health')
    parser.add_argument('--check', choices=['all', 'drift', 'volume', 'recency'],
                        default='all', help='Which checks to run')
    parser.add_argument('--send-alert', action='store_true',
                        help='Send Discord webhook on alerts')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    if not args.json:
        logger.info("="*60)
        logger.info("MODEL MONITORING CHECK")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("="*60)

    monitor = ModelMonitor()

    try:
        monitor.connect()

        if args.check == 'all':
            results = monitor.run_all_checks()
        elif args.check == 'drift':
            results = {'calibration': monitor.check_calibration_drift()}
        elif args.check == 'volume':
            results = {'volume': monitor.check_prediction_volume()}
        elif args.check == 'recency':
            results = {'recency': monitor.check_latest_inference_time()}

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            # Print human-readable results
            logger.info(f"\nOverall Status: {results.get('overall_status', 'unknown').upper()}")
            logger.info(f"Alerts: {results.get('alert_count', len(monitor.alerts))}")

            if monitor.alerts:
                logger.info("\nAlerts:")
                for alert in monitor.alerts:
                    logger.info(f"  {alert}")

            # Print check results
            if 'checks' in results:
                logger.info("\nCheck Results:")
                for check_name, check_result in results['checks'].items():
                    status = check_result.get('status', 'unknown')
                    logger.info(f"  {check_name}: {status}")

        # Send Discord alert if requested
        if args.send_alert and monitor.alerts:
            send_discord_alert(monitor.alerts, DISCORD_WEBHOOK_URL)

    except Exception as e:
        logger.exception(f"Monitoring error: {e}")
        sys.exit(1)
    finally:
        monitor.close()

    if not args.json:
        logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
