#!/usr/bin/env python3
"""
Export Model Registry to JSON

Exports the model_registry table to a JSON file that can be used by
the gept-recommendation-engine or other external systems.

This script solves the discrepancy described in Issue #119 where the
recommendation engine only had 3 items in its registry.json while
the inference engine has models for 300+ items.

Usage:
    # Export ACTIVE models to file
    python scripts/export_model_registry.py -o registry.json

    # Export to stdout (for piping)
    python scripts/export_model_registry.py

    # Dry run - show stats without writing
    python scripts/export_model_registry.py --dry-run

    # Include all statuses, not just ACTIVE
    python scripts/export_model_registry.py --all-statuses

    # Filter by specific run_id
    python scripts/export_model_registry.py --run-id 20260111_142024

Environment:
    Requires DB_PASS environment variable to be set.
    See .env.example for all database configuration options.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_utils import get_db_connection  # noqa: E402


def fetch_models(
    status: Optional[str] = 'ACTIVE',
    run_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch models from the model_registry table.

    Args:
        status: Filter by status ('ACTIVE', 'PENDING', etc.) or None for all
        run_id: Filter by specific training run ID

    Returns:
        List of model dictionaries
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Build query with optional filters
            query = """
                SELECT
                    id as model_id,
                    item_id,
                    item_name,
                    run_id,
                    model_path,
                    status,
                    mean_auc,
                    targets_above_52,
                    targets_above_55,
                    targets_scored,
                    n_features,
                    n_samples,
                    iterations_used,
                    validation_auc,
                    validation_passed,
                    trained_at,
                    activated_at,
                    training_duration_seconds
                FROM model_registry
                WHERE 1=1
            """
            params = []

            if status is not None:
                query += " AND status = %s"
                params.append(status)

            if run_id is not None:
                query += " AND run_id = %s"
                params.append(run_id)

            query += " ORDER BY item_id"

            cur.execute(query, params)

            columns = [desc[0] for desc in cur.description]
            models = []

            for row in cur.fetchall():
                model = dict(zip(columns, row))
                # Convert datetime objects to ISO format strings
                for key in ['trained_at', 'activated_at']:
                    if model.get(key) is not None:
                        model[key] = model[key].isoformat()
                # Convert Decimal to float for JSON serialization
                for key in ['mean_auc', 'validation_auc', 'training_duration_seconds']:
                    if model.get(key) is not None:
                        model[key] = float(model[key])
                models.append(model)

            return models


def fetch_status_summary() -> Dict[str, int]:
    """Fetch count of models by status."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT status, COUNT(*) as count
                FROM model_registry
                GROUP BY status
            """)
            return {row[0]: row[1] for row in cur.fetchall()}


def build_registry(
    models: List[Dict[str, Any]],
    status_summary: Dict[str, int]
) -> Dict[str, Any]:
    """
    Build the registry JSON structure.

    Args:
        models: List of model dictionaries from database
        status_summary: Count of models by status

    Returns:
        Complete registry dictionary
    """
    # Group models by item_id
    items = {}
    run_ids = set()

    for model in models:
        item_id = str(model['item_id'])
        run_ids.add(model['run_id'])

        items[item_id] = {
            'item_id': model['item_id'],
            'item_name': model['item_name'],
            'model_id': model['model_id'],
            'run_id': model['run_id'],
            'model_path': model['model_path'],
            'status': model['status'],
            'mean_auc': model['mean_auc'],
            'targets_above_52': model['targets_above_52'],
            'targets_above_55': model['targets_above_55'],
            'targets_scored': model['targets_scored'],
            'n_features': model['n_features'],
            'n_samples': model['n_samples'],
            'validation_passed': model['validation_passed'],
            'trained_at': model['trained_at'],
            'activated_at': model['activated_at'],
        }

    return {
        'metadata': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'total_items': len(items),
            'total_models': len(models),
            'source': 'model_registry table',
            'run_ids': sorted(run_ids),
        },
        'statistics': {
            'models_by_status': status_summary,
            'total_in_registry': sum(status_summary.values()),
        },
        'items': items,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Export model registry to JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export ACTIVE models to file
    python scripts/export_model_registry.py -o registry.json

    # Export to stdout
    python scripts/export_model_registry.py

    # Show stats only (dry run)
    python scripts/export_model_registry.py --dry-run

    # Export all models regardless of status
    python scripts/export_model_registry.py --all-statuses -o full_registry.json

    # Export models from specific training run
    python scripts/export_model_registry.py --run-id 20260111_142024
        """
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show statistics without writing output'
    )
    parser.add_argument(
        '--all-statuses',
        action='store_true',
        help='Include all statuses, not just ACTIVE'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        help='Filter by specific training run ID'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        default=True,
        help='Pretty-print JSON output (default: True)'
    )
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Compact JSON output (no indentation)'
    )

    args = parser.parse_args()

    # Determine status filter
    status_filter = None if args.all_statuses else 'ACTIVE'

    try:
        # Fetch data from database
        print("Fetching models from database...", file=sys.stderr)
        models = fetch_models(status=status_filter, run_id=args.run_id)
        status_summary = fetch_status_summary()

        # Build registry structure
        registry = build_registry(models, status_summary)

        # Print summary
        print("\nRegistry Summary:", file=sys.stderr)
        print(f"  Total items exported: {registry['metadata']['total_items']}", file=sys.stderr)
        print(f"  Run IDs: {', '.join(registry['metadata']['run_ids']) or 'None'}", file=sys.stderr)
        print("\nModels by status in database:", file=sys.stderr)
        for status, count in sorted(status_summary.items()):
            marker = "*" if status == 'ACTIVE' else " "
            print(f"  {marker} {status}: {count}", file=sys.stderr)

        if args.dry_run:
            print("\n[Dry run - no output written]", file=sys.stderr)
            return 0

        # Generate JSON output
        indent = None if args.compact else 2
        json_output = json.dumps(registry, indent=indent, default=str)

        # Write output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_output)
            print(f"\nWritten to: {output_path}", file=sys.stderr)
        else:
            print(json_output)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
