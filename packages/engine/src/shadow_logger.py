"""Shadow mode logging for ML ranker comparison.

This module logs the rankings from both the heuristic scoring and ML ranker
side-by-side, enabling analysis of which approach produces better recommendations.

Data is logged to a rolling file and can be analyzed offline to validate
the ML ranker before switching to it as the primary scoring method.
"""

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ShadowLogger:
    """Logs comparative rankings between heuristic and ML scoring.

    Writes CSV files with recommendation data including both scoring methods,
    allowing offline analysis of ranking quality differences.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        """Initialize shadow logger.

        Args:
            log_dir: Directory for shadow log files.
                    Defaults to ~/gept/logs/shadow_ranking
            enabled: Whether shadow logging is enabled
            sample_rate: Fraction of requests to log (0.0-1.0)
        """
        self.enabled = enabled
        self.sample_rate = sample_rate

        if log_dir is None:
            # Try common locations
            candidates = [
                os.path.expanduser("~/gept/logs/shadow_ranking"),
                "/opt/recommendation-engine/logs/shadow_ranking",
            ]
            for candidate in candidates:
                if os.path.isdir(os.path.dirname(candidate)):
                    log_dir = candidate
                    break
            else:
                log_dir = candidates[0]

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._current_date = None
        self._current_file = None
        self._csv_writer = None

    def _get_log_file(self) -> Path:
        """Get the current log file path (rotates daily)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"shadow_ranking_{today}.csv"

    def _ensure_writer(self):
        """Ensure CSV writer is open for current date."""
        log_file = self._get_log_file()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._current_date != today or self._current_file is None:
            # Close previous file if open
            if self._current_file is not None:
                self._current_file.close()

            # Open new file (append mode)
            file_exists = log_file.exists()
            self._current_file = open(log_file, "a", newline="")
            self._csv_writer = csv.writer(self._current_file)
            self._current_date = today

            # Write header if new file
            if not file_exists:
                self._csv_writer.writerow([
                    "timestamp",
                    "request_id",
                    "user_id",
                    "style",
                    "risk",
                    "capital",
                    "slots",
                    # Item details
                    "item_id",
                    "item_name",
                    "buy_price",
                    "sell_price",
                    "expected_value",
                    "fill_probability",
                    # Rankings
                    "heuristic_rank",
                    "heuristic_score",
                    "ml_rank",
                    "ml_score",
                    # Agreement metrics
                    "rank_diff",
                    "in_heuristic_top5",
                    "in_ml_top5",
                ])

    def log_rankings(
        self,
        request_id: str,
        user_id: Optional[str],
        style: str,
        risk: str,
        capital: int,
        slots: int,
        candidates: list[dict],
    ):
        """Log comparative rankings for a recommendation request.

        Args:
            request_id: Unique identifier for this request
            user_id: User identifier (or anonymous)
            style: Trading style
            risk: Risk level
            capital: Total capital
            slots: Number of GE slots
            candidates: List of candidate dicts with both heuristic and ML scores
        """
        if not self.enabled:
            return

        # Sample rate check
        if self.sample_rate < 1.0:
            import random
            if random.random() > self.sample_rate:
                return

        try:
            self._ensure_writer()

            timestamp = datetime.now(timezone.utc).isoformat()

            # Sort by heuristic score
            heuristic_sorted = sorted(
                candidates,
                key=lambda x: x.get("_score", 0) or 0,
                reverse=True,
            )
            heuristic_ranks = {
                c.get("item_id"): i + 1
                for i, c in enumerate(heuristic_sorted)
            }

            # Sort by ML score (if available)
            ml_candidates = [c for c in candidates if c.get("ml_score") is not None]
            if ml_candidates:
                ml_sorted = sorted(
                    ml_candidates,
                    key=lambda x: x.get("ml_score", 0) or 0,
                    reverse=True,
                )
                ml_ranks = {
                    c.get("item_id"): i + 1
                    for i, c in enumerate(ml_sorted)
                }
            else:
                ml_ranks = {}

            # Get top-5 sets
            heuristic_top5 = set(list(heuristic_ranks.keys())[:5])
            ml_top5 = set(list(ml_ranks.keys())[:5]) if ml_ranks else set()

            # Log each candidate
            for cand in candidates:
                item_id = cand.get("item_id")
                h_rank = heuristic_ranks.get(item_id)
                ml_rank = ml_ranks.get(item_id)
                ml_score = cand.get("ml_score")

                self._csv_writer.writerow([
                    timestamp,
                    request_id,
                    user_id or "anonymous",
                    style,
                    risk,
                    capital,
                    slots,
                    item_id,
                    cand.get("item_name", ""),
                    cand.get("buy_price", 0),
                    cand.get("sell_price", 0),
                    cand.get("expected_value", 0),
                    cand.get("fill_probability", 0),
                    h_rank,
                    cand.get("_score", 0),
                    ml_rank,
                    ml_score,
                    abs(h_rank - ml_rank) if ml_rank else None,
                    1 if item_id in heuristic_top5 else 0,
                    1 if item_id in ml_top5 else 0,
                ])

            self._current_file.flush()

        except Exception as e:
            logger.warning(f"Shadow logging failed: {e}")

    def log_comparison_summary(
        self,
        request_id: str,
        candidates: list[dict],
    ) -> dict:
        """Log and return a summary of ranking comparison.

        Args:
            request_id: Unique request identifier
            candidates: List of candidates with both scores

        Returns:
            Summary dict with agreement metrics
        """
        # Calculate agreement metrics
        heuristic_sorted = sorted(
            candidates,
            key=lambda x: x.get("_score", 0) or 0,
            reverse=True,
        )

        ml_candidates = [c for c in candidates if c.get("ml_score") is not None]
        if not ml_candidates:
            return {
                "ml_available": False,
                "num_candidates": len(candidates),
            }

        ml_sorted = sorted(
            ml_candidates,
            key=lambda x: x.get("ml_score", 0) or 0,
            reverse=True,
        )

        # Top-1 agreement
        h_top1 = heuristic_sorted[0].get("item_id") if heuristic_sorted else None
        ml_top1 = ml_sorted[0].get("item_id") if ml_sorted else None
        top1_agree = h_top1 == ml_top1

        # Top-5 overlap
        h_top5 = set(c.get("item_id") for c in heuristic_sorted[:5])
        ml_top5 = set(c.get("item_id") for c in ml_sorted[:5])
        top5_overlap = len(h_top5 & ml_top5)

        # Rank correlation (Spearman)
        from scipy import stats
        h_ranks = {c.get("item_id"): i for i, c in enumerate(heuristic_sorted)}
        ml_ranks = {c.get("item_id"): i for i, c in enumerate(ml_sorted)}
        common_items = set(h_ranks.keys()) & set(ml_ranks.keys())

        if len(common_items) >= 3:
            h_vals = [h_ranks[item] for item in common_items]
            ml_vals = [ml_ranks[item] for item in common_items]
            correlation, _ = stats.spearmanr(h_vals, ml_vals)
        else:
            correlation = None

        summary = {
            "ml_available": True,
            "num_candidates": len(candidates),
            "num_ml_scored": len(ml_candidates),
            "top1_agree": top1_agree,
            "top5_overlap": top5_overlap,
            "rank_correlation": correlation,
            "h_top1_item": h_top1,
            "ml_top1_item": ml_top1,
        }

        logger.debug(
            f"Shadow comparison [{request_id}]: "
            f"top1_agree={top1_agree}, top5_overlap={top5_overlap}/5, "
            f"correlation={correlation:.3f if correlation else 'N/A'}"
        )

        return summary

    def close(self):
        """Close any open file handles."""
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None
            self._csv_writer = None
