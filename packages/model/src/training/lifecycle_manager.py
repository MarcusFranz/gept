"""
Model Lifecycle Manager - State Machine for Model Lifecycle
============================================================

Manages model lifecycle transitions:
    PENDING → ACTIVE → DEPRECATED → SUNSET → ARCHIVED

States:
    PENDING:    Trained and validated, awaiting activation
    ACTIVE:     Live in production, generating predictions
    DEPRECATED: Replaced by newer model, no new trades
    SUNSET:     In 48-hour grace period before archival
    ARCHIVED:   Removed from production, preserved for history

Usage:
    from src.training.lifecycle_manager import ModelLifecycleManager

    lifecycle = ModelLifecycleManager()

    # Activate a validated model (atomic swap with old model)
    lifecycle.activate_model(model_id)

    # Process lifecycle (run as part of daily pipeline)
    lifecycle.process_lifecycle()

    # Rollback to previous model
    lifecycle.rollback_model(item_id)

CLI:
    python -m src.training.lifecycle_manager deploy --run-id 20260111_050000
    python -m src.training.lifecycle_manager process-lifecycle
    python -m src.training.lifecycle_manager rollback --item-id 2
"""

import sys
import yaml
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db_utils import get_db_connection

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle states."""
    PENDING = 'PENDING'
    ACTIVE = 'ACTIVE'
    DEPRECATED = 'DEPRECATED'
    SUNSET = 'SUNSET'
    ARCHIVED = 'ARCHIVED'


@dataclass
class LifecycleConfig:
    """Configuration for lifecycle management."""
    sunset_grace_period_hours: int = 48
    archive_cleanup_days: int = 90
    deprecation_auc_threshold: float = 0.52


@dataclass
class ActivationResult:
    """Result of model activation."""
    model_id: int
    item_id: int
    item_name: str
    success: bool
    old_model_id: Optional[int] = None
    reason: str = ''


@dataclass
class LifecycleProcessResult:
    """Result of lifecycle processing."""
    timestamp: str
    deprecated_count: int
    sunset_count: int
    archived_count: int
    details: List[Dict[str, Any]]


class ModelLifecycleManager:
    """Manages model lifecycle transitions."""

    def __init__(
        self,
        config: Optional[LifecycleConfig] = None,
        config_path: Optional[str] = None,
        models_dir: str = 'models'
    ):
        """
        Initialize lifecycle manager.

        Args:
            config: LifecycleConfig object
            config_path: Path to training_config.yaml
            models_dir: Base directory containing models
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            default_path = Path(__file__).parent.parent.parent / 'config' / 'training_config.yaml'
            if default_path.exists():
                self.config = self._load_config(str(default_path))
            else:
                self.config = LifecycleConfig()

        self.models_dir = Path(models_dir)

    def _load_config(self, config_path: str) -> LifecycleConfig:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)

        lc_config = yaml_config.get('lifecycle', {})

        return LifecycleConfig(
            sunset_grace_period_hours=lc_config.get('sunset_grace_period_hours', 48),
            archive_cleanup_days=lc_config.get('archive_cleanup_days', 90),
            deprecation_auc_threshold=lc_config.get('deprecation_auc_threshold', 0.52),
        )

    def activate_model(self, model_id: int) -> ActivationResult:
        """
        Activate a PENDING model, replacing any existing ACTIVE model.

        This is an atomic operation:
        1. Mark old ACTIVE model as DEPRECATED
        2. Mark new model as ACTIVE
        3. Update model lineage references

        Args:
            model_id: ID of the model to activate

        Returns:
            ActivationResult
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get model info
                cur.execute("""
                    SELECT id, item_id, item_name, status, run_id, model_path
                    FROM model_registry
                    WHERE id = %s
                """, (model_id,))
                row = cur.fetchone()

                if not row:
                    return ActivationResult(
                        model_id=model_id,
                        item_id=0,
                        item_name='',
                        success=False,
                        reason='model_not_found'
                    )

                _, item_id, item_name, status, run_id, model_path = row

                if status != ModelStatus.PENDING.value:
                    return ActivationResult(
                        model_id=model_id,
                        item_id=item_id,
                        item_name=item_name,
                        success=False,
                        reason=f'wrong_status:{status}'
                    )

                # Find current active model for this item
                cur.execute("""
                    SELECT id, run_id
                    FROM model_registry
                    WHERE item_id = %s AND status = 'ACTIVE'
                """, (item_id,))
                old_model = cur.fetchone()
                old_model_id = old_model[0] if old_model else None

                # Atomic swap
                try:
                    # Deprecate old model (if exists)
                    if old_model_id:
                        cur.execute("""
                            UPDATE model_registry
                            SET status = 'DEPRECATED',
                                status_changed_at = NOW(),
                                deprecated_at = NOW(),
                                replaced_by_model_id = %s,
                                status_reason = 'Replaced by newer model'
                            WHERE id = %s
                        """, (model_id, old_model_id))
                        logger.info(f"Deprecated old model {old_model_id} for item {item_id}")

                    # Activate new model
                    cur.execute("""
                        UPDATE model_registry
                        SET status = 'ACTIVE',
                            status_changed_at = NOW(),
                            activated_at = NOW(),
                            replaces_model_id = %s
                        WHERE id = %s
                    """, (old_model_id, model_id))

                    conn.commit()
                    logger.info(f"Activated model {model_id} for item {item_id} ({item_name})")

                    return ActivationResult(
                        model_id=model_id,
                        item_id=item_id,
                        item_name=item_name,
                        success=True,
                        old_model_id=old_model_id,
                        reason='success'
                    )

                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to activate model {model_id}: {e}")
                    return ActivationResult(
                        model_id=model_id,
                        item_id=item_id,
                        item_name=item_name,
                        success=False,
                        reason=f'db_error:{str(e)}'
                    )

    def deploy_run(self, run_id: str) -> List[ActivationResult]:
        """
        Deploy all validated models from a training run.

        Args:
            run_id: Training run ID

        Returns:
            List of ActivationResult for each model
        """
        results = []

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get all PENDING models from this run
                cur.execute("""
                    SELECT id, item_id, item_name
                    FROM model_registry
                    WHERE run_id = %s AND status = 'PENDING' AND validation_passed = TRUE
                    ORDER BY item_id
                """, (run_id,))

                pending_models = cur.fetchall()

        logger.info(f"Deploying {len(pending_models)} models from run {run_id}")

        for model_id, item_id, item_name in pending_models:
            result = self.activate_model(model_id)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        logger.info(f"Deployed {successful}/{len(results)} models")

        return results

    def process_lifecycle(self) -> LifecycleProcessResult:
        """
        Process model lifecycle transitions.

        Called as part of daily pipeline to:
        1. Move DEPRECATED models to SUNSET after grace period intent
        2. Move SUNSET models to ARCHIVED after 48h grace period

        Returns:
            LifecycleProcessResult with counts and details
        """
        details = []
        deprecated_count = 0
        sunset_count = 0
        archived_count = 0

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Move DEPRECATED to SUNSET (immediately start grace period)
                cur.execute("""
                    UPDATE model_registry
                    SET status = 'SUNSET',
                        status_changed_at = NOW(),
                        sunset_at = NOW(),
                        status_reason = 'Starting sunset grace period'
                    WHERE status = 'DEPRECATED'
                      AND deprecated_at IS NOT NULL
                    RETURNING id, item_id, item_name
                """)
                deprecated_to_sunset = cur.fetchall()
                sunset_count = len(deprecated_to_sunset)

                for model_id, item_id, item_name in deprecated_to_sunset:
                    details.append({
                        'action': 'deprecated_to_sunset',
                        'model_id': model_id,
                        'item_id': item_id,
                        'item_name': item_name,
                    })
                    logger.info(f"Model {model_id} ({item_name}): DEPRECATED -> SUNSET")

                # Move SUNSET to ARCHIVED (after grace period)
                cur.execute("""
                    UPDATE model_registry
                    SET status = 'ARCHIVED',
                        status_changed_at = NOW(),
                        archived_at = NOW(),
                        status_reason = 'Grace period expired'
                    WHERE status = 'SUNSET'
                      AND sunset_at IS NOT NULL
                      AND sunset_at < NOW() - make_interval(hours => %s)
                    RETURNING id, item_id, item_name
                """, (self.config.sunset_grace_period_hours,))
                sunset_to_archived = cur.fetchall()
                archived_count = len(sunset_to_archived)

                for model_id, item_id, item_name in sunset_to_archived:
                    details.append({
                        'action': 'sunset_to_archived',
                        'model_id': model_id,
                        'item_id': item_id,
                        'item_name': item_name,
                    })
                    logger.info(f"Model {model_id} ({item_name}): SUNSET -> ARCHIVED")

                conn.commit()

        logger.info(f"Lifecycle processing: {sunset_count} to sunset, {archived_count} archived")

        return LifecycleProcessResult(
            timestamp=datetime.utcnow().isoformat(),
            deprecated_count=deprecated_count,
            sunset_count=sunset_count,
            archived_count=archived_count,
            details=details,
        )

    def rollback_model(self, item_id: int) -> ActivationResult:
        """
        Rollback to the previous model for an item.

        Finds the most recent DEPRECATED/SUNSET/ARCHIVED model and reactivates it.

        Args:
            item_id: Item ID to rollback

        Returns:
            ActivationResult
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get current active model
                cur.execute("""
                    SELECT id, replaces_model_id
                    FROM model_registry
                    WHERE item_id = %s AND status = 'ACTIVE'
                """, (item_id,))
                current = cur.fetchone()

                if not current:
                    return ActivationResult(
                        model_id=0,
                        item_id=item_id,
                        item_name='',
                        success=False,
                        reason='no_active_model'
                    )

                current_id, previous_id = current

                if not previous_id:
                    # Find most recent non-active model
                    cur.execute("""
                        SELECT id, item_name
                        FROM model_registry
                        WHERE item_id = %s
                          AND status IN ('DEPRECATED', 'SUNSET', 'ARCHIVED')
                          AND id != %s
                        ORDER BY trained_at DESC
                        LIMIT 1
                    """, (item_id, current_id))
                    prev = cur.fetchone()
                    if prev:
                        previous_id = prev[0]

                if not previous_id:
                    return ActivationResult(
                        model_id=current_id,
                        item_id=item_id,
                        item_name='',
                        success=False,
                        reason='no_previous_model'
                    )

                # Get previous model info
                cur.execute("""
                    SELECT item_name, run_id
                    FROM model_registry WHERE id = %s
                """, (previous_id,))
                prev_info = cur.fetchone()
                item_name = prev_info[0] if prev_info else ''

                try:
                    # Deprecate current model
                    cur.execute("""
                        UPDATE model_registry
                        SET status = 'DEPRECATED',
                            status_changed_at = NOW(),
                            deprecated_at = NOW(),
                            status_reason = 'Manual rollback'
                        WHERE id = %s
                    """, (current_id,))

                    # Reactivate previous model
                    cur.execute("""
                        UPDATE model_registry
                        SET status = 'ACTIVE',
                            status_changed_at = NOW(),
                            activated_at = NOW(),
                            status_reason = 'Rollback from newer model'
                        WHERE id = %s
                    """, (previous_id,))

                    conn.commit()
                    logger.info(f"Rolled back item {item_id}: {current_id} -> {previous_id}")

                    return ActivationResult(
                        model_id=previous_id,
                        item_id=item_id,
                        item_name=item_name,
                        success=True,
                        old_model_id=current_id,
                        reason='rollback_success'
                    )

                except Exception as e:
                    conn.rollback()
                    logger.error(f"Rollback failed: {e}")
                    return ActivationResult(
                        model_id=previous_id,
                        item_id=item_id,
                        item_name=item_name,
                        success=False,
                        reason=f'rollback_error:{str(e)}'
                    )

    def get_model_status(self, item_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get model status for one or all items.

        Args:
            item_id: Specific item ID or None for all items

        Returns:
            List of model status dicts
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if item_id:
                    cur.execute("""
                        SELECT
                            id, item_id, item_name, run_id, status,
                            mean_auc, trained_at, activated_at,
                            deprecated_at, sunset_at, archived_at,
                            status_reason
                        FROM model_registry
                        WHERE item_id = %s
                        ORDER BY trained_at DESC
                    """, (item_id,))
                else:
                    cur.execute("""
                        SELECT
                            id, item_id, item_name, run_id, status,
                            mean_auc, trained_at, activated_at,
                            deprecated_at, sunset_at, archived_at,
                            status_reason
                        FROM model_registry
                        WHERE status = 'ACTIVE'
                        ORDER BY item_id
                    """)

                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_lifecycle_summary(self) -> Dict[str, Any]:
        """Get summary of model lifecycle status."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        status,
                        COUNT(*) as count,
                        AVG(mean_auc) as avg_auc,
                        MIN(trained_at) as oldest,
                        MAX(trained_at) as newest
                    FROM model_registry
                    GROUP BY status
                    ORDER BY
                        CASE status
                            WHEN 'ACTIVE' THEN 1
                            WHEN 'PENDING' THEN 2
                            WHEN 'DEPRECATED' THEN 3
                            WHEN 'SUNSET' THEN 4
                            WHEN 'ARCHIVED' THEN 5
                        END
                """)

                summary = {}
                for row in cur.fetchall():
                    status, count, avg_auc, oldest, newest = row
                    summary[status] = {
                        'count': count,
                        'avg_auc': float(avg_auc) if avg_auc else None,
                        'oldest': oldest.isoformat() if oldest else None,
                        'newest': newest.isoformat() if newest else None,
                    }

                return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Manage model lifecycle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy models from a training run')
    deploy_parser.add_argument('--run-id', required=True, help='Training run ID')

    # Process lifecycle command
    subparsers.add_parser('process-lifecycle',
                          help='Process lifecycle transitions')

    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous model')
    rollback_parser.add_argument('--item-id', type=int, required=True, help='Item ID')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show model status')
    status_parser.add_argument('--item-id', type=int, help='Specific item ID')

    # Summary command
    subparsers.add_parser('summary', help='Show lifecycle summary')

    # Common args
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--config', help='Path to training_config.yaml')
    parser.add_argument('--models-dir', default='models', help='Models directory')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    if not args.command:
        parser.print_help()
        return

    manager = ModelLifecycleManager(
        config_path=args.config,
        models_dir=args.models_dir
    )

    if args.command == 'deploy':
        results = manager.deploy_run(args.run_id)

        print(f"\n{'='*60}")
        print(f"Deployment Results - Run {args.run_id}")
        print(f"{'='*60}")

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print("\nFailed deployments:")
            for r in failed:
                print(f"  {r.item_id:>6} | {r.item_name[:30]:<30} | {r.reason}")

    elif args.command == 'process-lifecycle':
        result = manager.process_lifecycle()

        print(f"\n{'='*60}")
        print("Lifecycle Processing Results")
        print(f"{'='*60}")
        print(f"Moved to SUNSET: {result.sunset_count}")
        print(f"Moved to ARCHIVED: {result.archived_count}")

        if result.details:
            print("\nDetails:")
            for d in result.details:
                print(f"  {d['action']}: {d['item_id']} ({d['item_name']})")

    elif args.command == 'rollback':
        result = manager.rollback_model(args.item_id)

        print(f"\n{'='*60}")
        print(f"Rollback Result - Item {args.item_id}")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"New active model: {result.model_id}")
        print(f"Previous model: {result.old_model_id}")
        print(f"Reason: {result.reason}")

    elif args.command == 'status':
        models = manager.get_model_status(args.item_id)

        print(f"\n{'='*60}")
        print("Model Status")
        print(f"{'='*60}")

        for m in models:
            print(f"{m['id']:>6} | {m['item_id']:>6} | {m['item_name'][:25]:<25} | "
                  f"{m['status']:<10} | AUC={m['mean_auc'] or 0:.4f}")

    elif args.command == 'summary':
        summary = manager.get_lifecycle_summary()

        print(f"\n{'='*60}")
        print("Lifecycle Summary")
        print(f"{'='*60}")

        for status, stats in summary.items():
            print(f"\n{status}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg AUC: {stats['avg_auc']:.4f}" if stats['avg_auc'] else "  Avg AUC: N/A")
            print(f"  Oldest: {stats['oldest']}" if stats['oldest'] else "  Oldest: N/A")
            print(f"  Newest: {stats['newest']}" if stats['newest'] else "  Newest: N/A")


if __name__ == '__main__':
    main()
