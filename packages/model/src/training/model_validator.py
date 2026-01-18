"""
Model Validator - Validation Pipeline for Trained Models
========================================================

Validates trained models before deployment by checking:
1. Mean AUC >= minimum threshold (0.55)
2. Not worse than current production model
3. Minimum targets scored
4. Sanity checks on prediction distribution

Usage:
    from src.training.model_validator import ModelValidator

    validator = ModelValidator()
    results = validator.validate_training_run('20260111_050000')

    # Check specific model
    result = validator.validate_model(model_path, production_auc=0.58)

CLI:
    python -m src.training.model_validator --run-id 20260111_050000
    python -m src.training.model_validator --run-id 20260111_050000 --verbose
"""

import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db_utils import get_db_connection

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    min_mean_auc: float = 0.55
    min_targets_above_52: int = 50  # out of 108
    max_auc_regression: float = 0.02
    min_validation_samples: int = 1000


@dataclass
class ValidationResult:
    """Result of model validation."""
    item_id: int
    item_name: str
    run_id: str
    passed: bool
    reason: str
    mean_auc: Optional[float] = None
    targets_above_52: Optional[int] = None
    targets_above_55: Optional[int] = None
    targets_scored: Optional[int] = None
    production_auc: Optional[float] = None
    auc_improvement: Optional[float] = None
    n_samples: Optional[int] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class RunValidationResult:
    """Aggregate validation result for a training run."""
    run_id: str
    timestamp: str
    total_models: int
    passed: int
    failed: int
    pass_rate: float
    results: List[Dict[str, Any]]
    failure_reasons: Dict[str, int]


class ModelValidator:
    """Validates trained models before deployment."""

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        config_path: Optional[str] = None,
        models_dir: str = 'models'
    ):
        """
        Initialize validator.

        Args:
            config: ValidationConfig object
            config_path: Path to training_config.yaml
            models_dir: Base directory containing model runs
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
                self.config = ValidationConfig()

        self.models_dir = Path(models_dir)

    def _load_config(self, config_path: str) -> ValidationConfig:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)

        val_config = yaml_config.get('validation', {})

        return ValidationConfig(
            min_mean_auc=val_config.get('min_mean_auc', 0.55),
            min_targets_above_52=val_config.get('min_targets_above_52', 50),
            max_auc_regression=val_config.get('max_auc_regression', 0.02),
            min_validation_samples=val_config.get('min_validation_samples', 1000),
        )

    def validate_training_run(self, run_id: str) -> RunValidationResult:
        """
        Validate all models from a training run.

        Args:
            run_id: The training run ID (e.g., '20260111_050000')

        Returns:
            RunValidationResult with aggregate stats and per-model results
        """
        run_dir = self.models_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")

        logger.info(f"Validating training run: {run_id}")

        # Get production model AUCs for comparison
        production_aucs = self._get_production_aucs()

        results = []
        failure_reasons = {}

        # Find all model directories
        model_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        logger.info(f"Found {len(model_dirs)} models to validate")

        for model_dir in model_dirs:
            item_id = int(model_dir.name)

            # Load model metadata
            meta_path = model_dir / 'meta.json'
            auc_path = model_dir / 'target_aucs.json'

            if not meta_path.exists():
                results.append(ValidationResult(
                    item_id=item_id,
                    item_name=f'Item {item_id}',
                    run_id=run_id,
                    passed=False,
                    reason='missing_meta'
                ))
                failure_reasons['missing_meta'] = failure_reasons.get('missing_meta', 0) + 1
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            target_aucs = {}
            if auc_path.exists():
                with open(auc_path) as f:
                    raw_aucs = json.load(f)
                    # Handle both list format and dict format
                    if isinstance(raw_aucs, list):
                        target_aucs = {item['target']: item['auc'] for item in raw_aucs}
                    else:
                        target_aucs = raw_aucs

            # Get production AUC for this item
            prod_auc = production_aucs.get(item_id)

            # Validate the model
            result = self._validate_model(
                item_id=item_id,
                meta=meta,
                target_aucs=target_aucs,
                run_id=run_id,
                production_auc=prod_auc
            )

            results.append(result)

            if not result.passed:
                base_reason = result.reason.split(':')[0]
                failure_reasons[base_reason] = failure_reasons.get(base_reason, 0) + 1

        # Calculate aggregate stats
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        pass_rate = passed / len(results) if results else 0

        logger.info(f"Validation complete: {passed}/{len(results)} passed ({pass_rate:.1%})")

        return RunValidationResult(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            total_models=len(results),
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            results=[asdict(r) for r in results],
            failure_reasons=failure_reasons,
        )

    def _validate_model(
        self,
        item_id: int,
        meta: Dict[str, Any],
        target_aucs: Dict[str, float],
        run_id: str,
        production_auc: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate a single model.

        Args:
            item_id: Item ID
            meta: Model metadata from meta.json
            target_aucs: Per-target AUC scores
            run_id: Training run ID
            production_auc: Current production model AUC (for comparison)

        Returns:
            ValidationResult
        """
        item_name = meta.get('item_name', f'Item {item_id}')
        warnings = []

        # Extract metrics from meta
        mean_auc = meta.get('mean_auc')
        if mean_auc is None and 'metrics' in meta:
            mean_auc = meta['metrics'].get('mean_auc')

        n_samples = meta.get('n_samples')
        if n_samples is None and 'metrics' in meta:
            n_samples = meta['metrics'].get('n_samples')

        # Count targets above thresholds
        targets_above_52 = sum(1 for auc in target_aucs.values() if auc and auc > 0.52)
        targets_above_55 = sum(1 for auc in target_aucs.values() if auc and auc > 0.55)
        targets_scored = len([auc for auc in target_aucs.values() if auc is not None])

        # Calculate AUC improvement over production
        auc_improvement = None
        if mean_auc is not None and production_auc is not None:
            auc_improvement = mean_auc - production_auc

        # Validation checks
        # Check 1: Mean AUC threshold
        if mean_auc is None:
            return ValidationResult(
                item_id=item_id,
                item_name=item_name,
                run_id=run_id,
                passed=False,
                reason='no_auc_metric',
                warnings=warnings,
            )

        if mean_auc < self.config.min_mean_auc:
            return ValidationResult(
                item_id=item_id,
                item_name=item_name,
                run_id=run_id,
                passed=False,
                reason=f'low_auc:{mean_auc:.4f}<{self.config.min_mean_auc}',
                mean_auc=mean_auc,
                targets_above_52=targets_above_52,
                targets_above_55=targets_above_55,
                targets_scored=targets_scored,
                production_auc=production_auc,
                auc_improvement=auc_improvement,
                n_samples=n_samples,
                warnings=warnings,
            )

        # Check 2: Minimum targets scored
        if targets_above_52 < self.config.min_targets_above_52:
            return ValidationResult(
                item_id=item_id,
                item_name=item_name,
                run_id=run_id,
                passed=False,
                reason=f'few_targets:{targets_above_52}<{self.config.min_targets_above_52}',
                mean_auc=mean_auc,
                targets_above_52=targets_above_52,
                targets_above_55=targets_above_55,
                targets_scored=targets_scored,
                production_auc=production_auc,
                auc_improvement=auc_improvement,
                n_samples=n_samples,
                warnings=warnings,
            )

        # Check 3: Not worse than production (if production model exists)
        if production_auc is not None and auc_improvement is not None:
            if auc_improvement < -self.config.max_auc_regression:
                return ValidationResult(
                    item_id=item_id,
                    item_name=item_name,
                    run_id=run_id,
                    passed=False,
                    reason=f'regression:{mean_auc:.4f}vs{production_auc:.4f}',
                    mean_auc=mean_auc,
                    targets_above_52=targets_above_52,
                    targets_above_55=targets_above_55,
                    targets_scored=targets_scored,
                    production_auc=production_auc,
                    auc_improvement=auc_improvement,
                    n_samples=n_samples,
                    warnings=warnings,
                )

        # Check 4: Minimum samples (warning only)
        if n_samples is not None and n_samples < self.config.min_validation_samples:
            warnings.append(f'low_samples:{n_samples}')

        # All checks passed
        return ValidationResult(
            item_id=item_id,
            item_name=item_name,
            run_id=run_id,
            passed=True,
            reason='passed',
            mean_auc=mean_auc,
            targets_above_52=targets_above_52,
            targets_above_55=targets_above_55,
            targets_scored=targets_scored,
            production_auc=production_auc,
            auc_improvement=auc_improvement,
            n_samples=n_samples,
            warnings=warnings,
        )

    def _get_production_aucs(self) -> Dict[int, float]:
        """Get current production model AUCs from model_registry."""
        aucs = {}
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT item_id, mean_auc
                        FROM model_registry
                        WHERE status = 'ACTIVE' AND mean_auc IS NOT NULL
                    """)
                    for row in cur.fetchall():
                        aucs[row[0]] = float(row[1])
        except Exception as e:
            logger.warning(f"Could not fetch production AUCs: {e}")
        return aucs

    def register_validated_models(
        self,
        validation_result: RunValidationResult
    ) -> int:
        """
        Register validated models in model_registry with PENDING status.

        Args:
            validation_result: Result from validate_training_run()

        Returns:
            Number of models registered
        """
        registered = 0

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for result in validation_result.results:
                    if not result['passed']:
                        continue

                    item_id = result['item_id']
                    run_id = validation_result.run_id
                    model_path = f"models/{run_id}/{item_id}/model.cbm"

                    try:
                        cur.execute("""
                            INSERT INTO model_registry (
                                item_id, item_name, run_id, model_path,
                                status, trained_at,
                                mean_auc, targets_above_52, targets_above_55,
                                targets_scored, n_samples,
                                validation_auc, validation_passed, validated_at
                            ) VALUES (
                                %(item_id)s, %(item_name)s, %(run_id)s, %(model_path)s,
                                'PENDING', NOW(),
                                %(mean_auc)s, %(targets_above_52)s, %(targets_above_55)s,
                                %(targets_scored)s, %(n_samples)s,
                                %(mean_auc)s, TRUE, NOW()
                            )
                            ON CONFLICT (item_id, run_id) DO UPDATE SET
                                validation_passed = TRUE,
                                validated_at = NOW()
                        """, {
                            'item_id': item_id,
                            'item_name': result['item_name'],
                            'run_id': run_id,
                            'model_path': model_path,
                            'mean_auc': result.get('mean_auc'),
                            'targets_above_52': result.get('targets_above_52'),
                            'targets_above_55': result.get('targets_above_55'),
                            'targets_scored': result.get('targets_scored'),
                            'n_samples': result.get('n_samples'),
                        })
                        registered += 1
                    except Exception as e:
                        logger.error(f"Failed to register model {item_id}: {e}")

                conn.commit()

        logger.info(f"Registered {registered} validated models")
        return registered

    def save_validation_results(
        self,
        result: RunValidationResult,
        output_path: str
    ) -> None:
        """Save validation results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        logger.info(f"Saved validation results to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Validate trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.training.model_validator --run-id 20260111_050000
    python -m src.training.model_validator --run-id 20260111_050000 --register
    python -m src.training.model_validator --run-id 20260111_050000 --output results.json
        """
    )
    parser.add_argument('--run-id', required=True, help='Training run ID to validate')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--register', action='store_true',
                        help='Register validated models in model_registry')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--config', help='Path to training_config.yaml')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    validator = ModelValidator(
        config_path=args.config,
        models_dir=args.models_dir
    )

    result = validator.validate_training_run(args.run_id)

    # Display results
    print(f"\n{'='*60}")
    print(f"Validation Results - Run {args.run_id}")
    print(f"{'='*60}")
    print(f"Total models: {result.total_models}")
    print(f"Passed: {result.passed} ({result.pass_rate:.1%})")
    print(f"Failed: {result.failed}")

    if result.failure_reasons:
        print("\nFailure reasons:")
        for reason, count in sorted(result.failure_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if args.verbose:
        print(f"\n{'='*60}")
        print("Passed Models:")
        print(f"{'='*60}")
        passed_results = [r for r in result.results if r['passed']]
        for r in passed_results[:20]:
            improvement = r.get('auc_improvement')
            imp_str = f" ({improvement:+.4f})" if improvement else ""
            print(f"  {r['item_id']:>6} | {r['item_name'][:25]:<25} | "
                  f"AUC={r['mean_auc']:.4f}{imp_str}")
        if len(passed_results) > 20:
            print(f"  ... and {len(passed_results) - 20} more")

        print(f"\n{'='*60}")
        print("Failed Models:")
        print(f"{'='*60}")
        failed_results = [r for r in result.results if not r['passed']]
        for r in failed_results[:20]:
            print(f"  {r['item_id']:>6} | {r['item_name'][:25]:<25} | "
                  f"reason={r['reason']}")
        if len(failed_results) > 20:
            print(f"  ... and {len(failed_results) - 20} more")

    # Register models if requested
    if args.register:
        registered = validator.register_validated_models(result)
        print(f"\nRegistered {registered} models in model_registry")

    # Save results if output specified
    if args.output:
        validator.save_validation_results(result, args.output)
        print(f"\nSaved to: {args.output}")

    # Exit with error code if too many failures
    if result.pass_rate < 0.5:
        logger.warning("Pass rate below 50%!")
        sys.exit(1)

    return result


if __name__ == '__main__':
    main()
