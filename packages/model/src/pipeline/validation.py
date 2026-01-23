"""
Validation utilities for the training pipeline.
Provides functions to validate data at each stage boundary.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ItemValidationResult:
    """Validation result for a single item."""
    item_id: int
    passed: bool
    reason: Optional[str] = None
    rows_5min: int = 0
    gap_pct: float = 0.0
    outlier_count: int = 0
    date_range: Optional[tuple] = None


@dataclass
class ValidationReport:
    """Complete validation report for a stage."""
    stage: str
    timestamp: str
    passed: int
    failed: int
    items: Dict[int, ItemValidationResult]
    summary: Dict[str, Any]

    def to_json(self, path: Path) -> None:
        data = {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "failed": self.failed,
            "summary": self.summary,
            "items": {str(k): asdict(v) for k, v in self.items.items()},
            "failures": [
                {"item_id": v.item_id, "reason": v.reason}
                for v in self.items.values() if not v.passed
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def validate_parquet_file(path: Path, item_id: int, max_gap_pct: float = 0.05) -> ItemValidationResult:
    """Validate a single parquet file for data quality."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return ItemValidationResult(item_id=item_id, passed=False, reason=f"Failed to read: {e}")

    required_cols = ["timestamp", "avg_high_price", "avg_low_price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return ItemValidationResult(item_id=item_id, passed=False, reason=f"Missing columns: {missing}")

    nan_count = df["avg_high_price"].isna().sum() + df["avg_low_price"].isna().sum()
    if nan_count > 0:
        return ItemValidationResult(item_id=item_id, passed=False, reason=f"NaN values: {nan_count}")

    neg_count = (df["avg_high_price"] < 0).sum() + (df["avg_low_price"] < 0).sum()
    if neg_count > 0:
        return ItemValidationResult(item_id=item_id, passed=False, reason=f"Negative prices: {neg_count}")

    gap_pct = df["gap"].sum() / len(df) if "gap" in df.columns and len(df) > 0 else 0.0
    if gap_pct > max_gap_pct:
        return ItemValidationResult(item_id=item_id, passed=False, reason=f"Gap pct {gap_pct:.1%} > {max_gap_pct:.1%}")

    outlier_count = df["outlier"].sum() if "outlier" in df.columns else 0

    return ItemValidationResult(
        item_id=item_id,
        passed=True,
        rows_5min=len(df),
        gap_pct=gap_pct,
        outlier_count=outlier_count,
    )


def validate_stage1_output(output_dir: Path, max_gap_pct: float = 0.05) -> ValidationReport:
    """Validate all Stage 1 outputs."""
    cleaned_dir = output_dir / "5min"
    if not cleaned_dir.exists():
        raise FileNotFoundError(f"Directory not found: {cleaned_dir}")

    items = {}
    passed = failed = 0

    for pq_file in cleaned_dir.glob("item_*.parquet"):
        item_id = int(pq_file.stem.replace("item_", ""))
        result = validate_parquet_file(pq_file, item_id, max_gap_pct)
        items[item_id] = result
        if result.passed:
            passed += 1
        else:
            failed += 1
            logger.warning(f"Item {item_id} failed: {result.reason}")

    summary = {
        "total_items": len(items),
        "total_rows": sum(r.rows_5min for r in items.values()),
    }

    return ValidationReport(
        stage="stage1_extract",
        timestamp=datetime.utcnow().isoformat(),
        passed=passed,
        failed=failed,
        items=items,
        summary=summary,
    )


def validate_feature_chunk(chunk_path: Path) -> Dict[str, Any]:
    """Validate a single feature chunk."""
    data = np.load(chunk_path)
    issues = []

    for key in ["recent", "medium", "long"]:
        if key not in data:
            issues.append(f"Missing: {key}")
            continue
        arr = data[key]
        if np.isnan(arr).any():
            issues.append(f"{key}: has NaN")
        if np.isinf(arr).any():
            issues.append(f"{key}: has Inf")

    return {"path": str(chunk_path), "valid": len(issues) == 0, "issues": issues}


def validate_stage3_output(features_dir: Path, n_samples: int = 10) -> ValidationReport:
    """Validate Stage 3 outputs (spot-check chunks)."""
    import random

    all_chunks = list((features_dir / "train").glob("chunk_*.npz"))
    all_chunks += list((features_dir / "val").glob("chunk_*.npz"))

    chunks_to_check = random.sample(all_chunks, min(n_samples, len(all_chunks)))

    items = {}
    passed = failed = 0

    for chunk_path in chunks_to_check:
        result = validate_feature_chunk(chunk_path)
        chunk_id = int(chunk_path.stem.replace("chunk_", ""))
        if result["valid"]:
            passed += 1
            items[chunk_id] = ItemValidationResult(item_id=chunk_id, passed=True)
        else:
            failed += 1
            items[chunk_id] = ItemValidationResult(item_id=chunk_id, passed=False, reason="; ".join(result["issues"]))

    return ValidationReport(
        stage="stage3_precompute",
        timestamp=datetime.utcnow().isoformat(),
        passed=passed,
        failed=failed,
        items=items,
        summary={"chunks_validated": len(chunks_to_check), "total_chunks": len(all_chunks)},
    )
