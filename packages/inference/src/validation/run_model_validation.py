#!/usr/bin/env python3
"""
Model validation pipeline for predictions and recommendations.

Compares historical predictions against realized market conditions from
`price_data_5min`, then stores:
1) row-level realized outcomes for each prediction
2) model-level summary snapshots with development steering signals
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json, execute_values

GE_TAX_RATE = 0.02


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


logger = logging.getLogger(__name__)


def get_db_connection():
    password = (os.getenv("DB_PASS") or os.getenv("DB_PASSWORD") or "").strip()
    if not password:
        raise RuntimeError(
            "DB_PASS is required (or DB_PASSWORD as alias) for model validation."
        )
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "osrs_data"),
        user=os.getenv("DB_USER", "osrs_user"),
        password=password,
    )


def table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        row = cur.fetchone()
    return bool(row and row[0])


def table_has_column(conn, table_name: str, column_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
            LIMIT 1
            """,
            (table_name, column_name),
        )
        row = cur.fetchone()
    return bool(row)


def detect_predictions_model_column(conn) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'predictions'
              AND column_name IN ('model_id', 'model_version')
            ORDER BY CASE column_name WHEN 'model_id' THEN 0 ELSE 1 END
            LIMIT 1
            """
        )
        row = cur.fetchone()
    return row[0] if row else None


def ensure_validation_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_validation_prediction_outcomes (
                prediction_time TIMESTAMPTZ NOT NULL,
                item_id INTEGER NOT NULL,
                hour_offset INTEGER NOT NULL,
                offset_pct DECIMAL(5,4) NOT NULL,
                model_id TEXT NOT NULL DEFAULT 'unknown',
                evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                predicted_fill_probability DECIMAL(7,6) NOT NULL,
                predicted_expected_value DECIMAL(8,6) NOT NULL,
                predicted_buy_price DECIMAL(12,2) NOT NULL,
                predicted_sell_price DECIMAL(12,2) NOT NULL,
                actual_low_price DECIMAL(12,2),
                actual_high_price DECIMAL(12,2),
                buy_would_fill BOOLEAN NOT NULL,
                sell_would_fill BOOLEAN NOT NULL,
                realized_fill BOOLEAN NOT NULL,
                realized_edge DECIMAL(8,6),
                is_recommendation_candidate BOOLEAN NOT NULL,
                recommendation_success BOOLEAN,
                PRIMARY KEY (prediction_time, item_id, hour_offset, offset_pct, model_id)
            );

            CREATE INDEX IF NOT EXISTS idx_mvpo_model_time
                ON model_validation_prediction_outcomes(model_id, prediction_time DESC);

            CREATE INDEX IF NOT EXISTS idx_mvpo_recommendation_success
                ON model_validation_prediction_outcomes(recommendation_success, prediction_time DESC)
                WHERE is_recommendation_candidate = TRUE;

            CREATE TABLE IF NOT EXISTS model_validation_snapshots (
                id BIGSERIAL PRIMARY KEY,
                generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                window_start TIMESTAMPTZ NOT NULL,
                window_end TIMESTAMPTZ NOT NULL,
                model_id TEXT NOT NULL,
                total_evaluated INTEGER NOT NULL,
                recommendation_candidates INTEGER NOT NULL,
                mean_predicted_fill_probability DECIMAL(7,6),
                realized_fill_rate DECIMAL(7,6),
                calibration_error DECIMAL(7,6),
                brier_score DECIMAL(7,6),
                recommendation_success_rate DECIMAL(7,6),
                expected_value_mean DECIMAL(8,6),
                realized_edge_mean DECIMAL(8,6),
                expected_minus_realized_edge DECIMAL(8,6),
                trade_outcomes_count INTEGER NOT NULL DEFAULT 0,
                trade_outcomes_win_rate DECIMAL(7,6),
                trade_outcomes_avg_profit DECIMAL(14,2),
                feedback_count INTEGER NOT NULL DEFAULT 0,
                feedback_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
                horizon_metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
                offset_metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
                development_signals JSONB NOT NULL DEFAULT '[]'::jsonb
            );

            CREATE INDEX IF NOT EXISTS idx_mvs_model_generated
                ON model_validation_snapshots(model_id, generated_at DESC);
            """
        )
    conn.commit()


def safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def fetch_matured_predictions(
    conn,
    *,
    window_start: datetime,
    window_end: datetime,
    evaluated_at: datetime,
    model_filter: Optional[str],
    model_column: Optional[str],
    max_rows: int,
) -> list[dict[str, Any]]:
    if model_filter and not model_column:
        raise RuntimeError(
            "Cannot filter by model_id because predictions has neither model_id nor model_version."
        )

    model_select = f"p.{model_column}::text" if model_column else "'unknown'::text"
    model_where = f"AND p.{model_column} = %s" if model_filter and model_column else ""

    params: list[Any] = [window_start, window_end, evaluated_at]
    if model_filter and model_column:
        params.append(model_filter)
    params.append(int(max_rows))

    sql = f"""
    WITH matured AS (
        SELECT
            p.time AS prediction_time,
            p.item_id,
            p.hour_offset,
            p.offset_pct::double precision AS offset_pct,
            {model_select} AS model_id,
            p.fill_probability::double precision AS predicted_fill_probability,
            p.expected_value::double precision AS predicted_expected_value,
            p.buy_price::double precision AS predicted_buy_price,
            p.sell_price::double precision AS predicted_sell_price
        FROM predictions p
        WHERE p.time >= %s
          AND p.time < %s
          AND p.time + make_interval(hours => p.hour_offset) <= %s
          AND p.fill_probability IS NOT NULL
          AND p.expected_value IS NOT NULL
          AND p.buy_price IS NOT NULL
          AND p.sell_price IS NOT NULL
          {model_where}
        ORDER BY p.time DESC
        LIMIT %s
    )
    SELECT
        m.prediction_time,
        m.item_id,
        m.hour_offset,
        m.offset_pct,
        COALESCE(NULLIF(TRIM(m.model_id), ''), 'unknown') AS model_id,
        m.predicted_fill_probability,
        m.predicted_expected_value,
        m.predicted_buy_price,
        m.predicted_sell_price,
        MIN(d.avg_low_price)::double precision AS actual_low_price,
        MAX(d.avg_high_price)::double precision AS actual_high_price,
        COUNT(*)::integer AS samples_in_window
    FROM matured m
    JOIN price_data_5min d
      ON d.item_id = m.item_id
     AND d.timestamp > m.prediction_time
     AND d.timestamp <= m.prediction_time + make_interval(hours => m.hour_offset)
     AND d.avg_low_price IS NOT NULL
     AND d.avg_high_price IS NOT NULL
    GROUP BY
        m.prediction_time,
        m.item_id,
        m.hour_offset,
        m.offset_pct,
        m.model_id,
        m.predicted_fill_probability,
        m.predicted_expected_value,
        m.predicted_buy_price,
        m.predicted_sell_price
    HAVING COUNT(*) > 0
    ORDER BY m.prediction_time DESC
    """

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "prediction_time": row[0],
                "item_id": int(row[1]),
                "hour_offset": int(row[2]),
                "offset_pct": float(row[3]),
                "model_id": str(row[4]),
                "predicted_fill_probability": float(row[5]),
                "predicted_expected_value": float(row[6]),
                "predicted_buy_price": float(row[7]),
                "predicted_sell_price": float(row[8]),
                "actual_low_price": float(row[9]) if row[9] is not None else None,
                "actual_high_price": float(row[10]) if row[10] is not None else None,
                "samples_in_window": int(row[11]),
            }
        )
    return out


def realized_edge_from_market(actual_low: Optional[float], actual_high: Optional[float]) -> Optional[float]:
    if actual_low is None or actual_high is None or actual_low <= 0:
        return None
    return ((actual_high - actual_low) / actual_low) - GE_TAX_RATE


def build_outcomes(
    matured_rows: list[dict[str, Any]],
    *,
    evaluated_at: datetime,
    min_fill_probability: float,
    min_expected_value: float,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for row in matured_rows:
        actual_low = row["actual_low_price"]
        actual_high = row["actual_high_price"]
        predicted_buy = row["predicted_buy_price"]
        predicted_sell = row["predicted_sell_price"]

        buy_would_fill = bool(actual_low is not None and predicted_buy >= actual_low)
        sell_would_fill = bool(actual_high is not None and predicted_sell <= actual_high)
        realized_fill = bool(buy_would_fill and sell_would_fill)

        realized_edge = realized_edge_from_market(actual_low, actual_high)
        is_candidate = (
            row["predicted_fill_probability"] >= min_fill_probability
            and row["predicted_expected_value"] >= min_expected_value
        )
        recommendation_success = (
            bool(is_candidate and realized_fill)
            if realized_edge is None
            else bool(is_candidate and realized_fill and realized_edge > 0.0)
        )

        outcomes.append(
            {
                "prediction_time": row["prediction_time"],
                "item_id": row["item_id"],
                "hour_offset": row["hour_offset"],
                "offset_pct": row["offset_pct"],
                "model_id": row["model_id"],
                "evaluated_at": evaluated_at,
                "predicted_fill_probability": row["predicted_fill_probability"],
                "predicted_expected_value": row["predicted_expected_value"],
                "predicted_buy_price": row["predicted_buy_price"],
                "predicted_sell_price": row["predicted_sell_price"],
                "actual_low_price": actual_low,
                "actual_high_price": actual_high,
                "buy_would_fill": buy_would_fill,
                "sell_would_fill": sell_would_fill,
                "realized_fill": realized_fill,
                "realized_edge": realized_edge,
                "is_recommendation_candidate": is_candidate,
                "recommendation_success": recommendation_success,
            }
        )
    return outcomes


def compute_core_metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(outcomes)
    predicted_probs = [float(r["predicted_fill_probability"]) for r in outcomes]
    actual_fills = [1.0 if r["realized_fill"] else 0.0 for r in outcomes]

    mean_predicted = safe_mean(predicted_probs)
    realized_fill_rate = safe_mean(actual_fills)
    calibration_error = (
        abs(mean_predicted - realized_fill_rate)
        if mean_predicted is not None and realized_fill_rate is not None
        else None
    )
    brier_score = (
        safe_mean([(p - y) ** 2 for p, y in zip(predicted_probs, actual_fills, strict=False)])
        if total > 0
        else None
    )

    candidates = [r for r in outcomes if r["is_recommendation_candidate"]]
    successes = [r for r in candidates if r["recommendation_success"]]
    expected_values = [float(r["predicted_expected_value"]) for r in candidates]
    realized_edges = [
        float(r["realized_edge"])
        for r in candidates
        if r["realized_edge"] is not None
    ]
    expected_mean = safe_mean(expected_values)
    realized_edge_mean = safe_mean(realized_edges)
    expected_minus_realized = (
        expected_mean - realized_edge_mean
        if expected_mean is not None and realized_edge_mean is not None
        else None
    )

    return {
        "total_evaluated": total,
        "recommendation_candidates": len(candidates),
        "mean_predicted_fill_probability": round_or_none(mean_predicted, 6),
        "realized_fill_rate": round_or_none(realized_fill_rate, 6),
        "calibration_error": round_or_none(calibration_error, 6),
        "brier_score": round_or_none(brier_score, 6),
        "recommendation_success_rate": round_or_none(
            safe_rate(len(successes), len(candidates)),
            6,
        ),
        "expected_value_mean": round_or_none(expected_mean, 6),
        "realized_edge_mean": round_or_none(realized_edge_mean, 6),
        "expected_minus_realized_edge": round_or_none(expected_minus_realized, 6),
    }


def build_bucket_metrics(
    outcomes: list[dict[str, Any]],
    *,
    bucket_key: str,
    bucket_label: str,
    value_digits: int,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[row[bucket_key]].append(row)

    metrics: list[dict[str, Any]] = []
    for raw_key in sorted(grouped.keys()):
        key_value = int(raw_key) if value_digits == 0 else round(float(raw_key), value_digits)
        core = compute_core_metrics(grouped[raw_key])
        metrics.append(
            {
                bucket_label: key_value,
                "count": core["total_evaluated"],
                "recommendation_candidates": core["recommendation_candidates"],
                "mean_predicted_fill_probability": core["mean_predicted_fill_probability"],
                "realized_fill_rate": core["realized_fill_rate"],
                "calibration_error": core["calibration_error"],
                "brier_score": core["brier_score"],
                "recommendation_success_rate": core["recommendation_success_rate"],
                "expected_value_mean": core["expected_value_mean"],
                "realized_edge_mean": core["realized_edge_mean"],
                "expected_minus_realized_edge": core["expected_minus_realized_edge"],
            }
        )
    return metrics


def load_trade_outcome_summary(
    conn,
    *,
    window_start: datetime,
    window_end: datetime,
    model_id: Optional[str],
) -> dict[str, Any]:
    if not table_exists(conn, "trade_outcomes"):
        return {"count": 0, "win_rate": None, "avg_profit": None}

    has_model = table_has_column(conn, "trade_outcomes", "model_id")
    model_filter = ""
    params: list[Any] = [window_start, window_end]

    if model_id and has_model:
        model_filter = "AND model_id = %s"
        params.append(model_id)

    sql = f"""
    SELECT
        COUNT(*)::integer AS n,
        AVG(CASE WHEN actual_profit > 0 THEN 1.0 ELSE 0.0 END)::double precision AS win_rate,
        AVG(actual_profit)::double precision AS avg_profit
    FROM trade_outcomes
    WHERE reported_at >= %s
      AND reported_at < %s
      {model_filter}
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
        return {
            "count": int(row[0] or 0),
            "win_rate": round_or_none(float(row[1]), 6) if row[1] is not None else None,
            "avg_profit": round_or_none(float(row[2]), 2) if row[2] is not None else None,
        }
    except Exception as exc:
        logger.warning("Trade outcome summary unavailable: %s", exc)
        return {"count": 0, "win_rate": None, "avg_profit": None}


def load_feedback_summary(
    conn,
    *,
    window_start: datetime,
    window_end: datetime,
) -> tuple[int, dict[str, int]]:
    if not table_exists(conn, "recommendation_feedback"):
        return 0, {}

    sql = """
    SELECT feedback_type, COUNT(*)::integer AS n
    FROM recommendation_feedback
    WHERE submitted_at >= %s
      AND submitted_at < %s
    GROUP BY feedback_type
    ORDER BY n DESC
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (window_start, window_end))
            rows = cur.fetchall()
        breakdown = {str(r[0]): int(r[1]) for r in rows}
        return sum(breakdown.values()), breakdown
    except Exception as exc:
        logger.warning("Feedback summary unavailable: %s", exc)
        return 0, {}


def build_development_signals(
    *,
    summary: dict[str, Any],
    horizon_metrics: list[dict[str, Any]],
    offset_metrics: list[dict[str, Any]],
    feedback_breakdown: dict[str, int],
    trade_outcomes: dict[str, Any],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    total_evaluated = int(summary["total_evaluated"])
    if total_evaluated < 200:
        signals.append(
            {
                "priority": "medium",
                "signal": "low_sample_size",
                "evidence": f"Only {total_evaluated} matured predictions in this window.",
                "action": "Increase validation window or prediction coverage before making major model decisions.",
            }
        )

    cal_err = summary["calibration_error"]
    pred = summary["mean_predicted_fill_probability"]
    actual = summary["realized_fill_rate"]
    if cal_err is not None and cal_err >= 0.08:
        direction = "overconfident" if (pred or 0.0) > (actual or 0.0) else "underconfident"
        signals.append(
            {
                "priority": "high",
                "signal": "calibration_drift",
                "evidence": f"Calibration error={cal_err:.3f} ({direction}).",
                "action": "Recalibrate probabilities (isotonic/Platt) and validate by horizon buckets.",
            }
        )

    brier = summary["brier_score"]
    if brier is not None and brier >= 0.20:
        signals.append(
            {
                "priority": "high",
                "signal": "high_brier_score",
                "evidence": f"Brier score={brier:.3f}.",
                "action": "Review feature drift and probability clipping bounds before retraining.",
            }
        )

    rec_success = summary["recommendation_success_rate"]
    rec_candidates = int(summary["recommendation_candidates"])
    if rec_success is not None and rec_candidates >= 50 and rec_success < 0.45:
        signals.append(
            {
                "priority": "high",
                "signal": "recommendation_precision_low",
                "evidence": f"Recommendation success rate={rec_success:.3f} across {rec_candidates} candidates.",
                "action": "Tighten recommendation threshold (min EV/min fill probability) and down-weight weak horizons.",
            }
        )

    edge_gap = summary["expected_minus_realized_edge"]
    if edge_gap is not None and edge_gap > 0.008:
        signals.append(
            {
                "priority": "medium",
                "signal": "expected_value_overestimation",
                "evidence": f"Expected-realized edge gap={edge_gap:.3f}.",
                "action": "Adjust EV formula with slippage penalty or retrain on realized spread targets.",
            }
        )

    if horizon_metrics:
        worst_horizon = max(
            [m for m in horizon_metrics if m["calibration_error"] is not None],
            key=lambda x: (float(x["calibration_error"]), x["count"]),
            default=None,
        )
        if worst_horizon and worst_horizon["count"] >= 40 and float(worst_horizon["calibration_error"]) >= 0.10:
            signals.append(
                {
                    "priority": "medium",
                    "signal": "horizon_specific_drift",
                    "evidence": (
                        f"Hour {worst_horizon['hour_offset']} has calibration error="
                        f"{float(worst_horizon['calibration_error']):.3f}."
                    ),
                    "action": "Retrain with horizon-specific loss weights or separate heads for long horizons.",
                }
            )

    if offset_metrics:
        weakest_offset = min(
            [m for m in offset_metrics if m["recommendation_success_rate"] is not None],
            key=lambda x: (float(x["recommendation_success_rate"]), -x["count"]),
            default=None,
        )
        if weakest_offset and weakest_offset["count"] >= 30:
            signals.append(
                {
                    "priority": "low",
                    "signal": "offset_underperformance",
                    "evidence": (
                        f"Offset {float(weakest_offset['offset_pct']):.4f} has recommendation "
                        f"success={float(weakest_offset['recommendation_success_rate']):.3f}."
                    ),
                    "action": "Reduce ranking weight for this offset or retrain offset embedding.",
                }
            )

    if trade_outcomes.get("count", 0) >= 20:
        win_rate = trade_outcomes.get("win_rate")
        avg_profit = trade_outcomes.get("avg_profit")
        if (win_rate is not None and win_rate < 0.50) or (avg_profit is not None and avg_profit < 0):
            signals.append(
                {
                    "priority": "medium",
                    "signal": "live_trade_outcomes_soft",
                    "evidence": (
                        f"Live trade outcomes win_rate={win_rate}, avg_profit={avg_profit} "
                        f"from {trade_outcomes['count']} reports."
                    ),
                    "action": "Review post-recommendation execution assumptions (price buffer, fill timing, crowding).",
                }
            )

    if feedback_breakdown:
        top_feedback = sorted(feedback_breakdown.items(), key=lambda x: x[1], reverse=True)[0]
        top_key, top_count = top_feedback
        mapped_action = {
            "price_too_high": "Shift buy targets closer to market mid or shorten horizons.",
            "price_too_low": "Allow slightly wider offsets for high-liquidity items.",
            "filled_slowly": "Bias ranking toward higher fill probability.",
            "did_not_fill": "Increase minimum fill threshold and penalize low-volume items.",
            "spread_too_wide": "Reduce EV threshold for conservative recommendation profiles.",
            "price_manipulation": "Increase anti-manipulation filters and monitor concentration spikes.",
        }.get(top_key, "Review recurring feedback themes in recommendation post-processing.")
        signals.append(
            {
                "priority": "low",
                "signal": "user_feedback_theme",
                "evidence": f"Most frequent feedback='{top_key}' ({top_count} reports).",
                "action": mapped_action,
            }
        )

    return signals


def persist_prediction_outcomes(conn, outcomes: list[dict[str, Any]]) -> int:
    if not outcomes:
        return 0

    sql = """
    INSERT INTO model_validation_prediction_outcomes (
        prediction_time,
        item_id,
        hour_offset,
        offset_pct,
        model_id,
        evaluated_at,
        predicted_fill_probability,
        predicted_expected_value,
        predicted_buy_price,
        predicted_sell_price,
        actual_low_price,
        actual_high_price,
        buy_would_fill,
        sell_would_fill,
        realized_fill,
        realized_edge,
        is_recommendation_candidate,
        recommendation_success
    ) VALUES %s
    ON CONFLICT (prediction_time, item_id, hour_offset, offset_pct, model_id)
    DO UPDATE SET
        evaluated_at = EXCLUDED.evaluated_at,
        predicted_fill_probability = EXCLUDED.predicted_fill_probability,
        predicted_expected_value = EXCLUDED.predicted_expected_value,
        predicted_buy_price = EXCLUDED.predicted_buy_price,
        predicted_sell_price = EXCLUDED.predicted_sell_price,
        actual_low_price = EXCLUDED.actual_low_price,
        actual_high_price = EXCLUDED.actual_high_price,
        buy_would_fill = EXCLUDED.buy_would_fill,
        sell_would_fill = EXCLUDED.sell_would_fill,
        realized_fill = EXCLUDED.realized_fill,
        realized_edge = EXCLUDED.realized_edge,
        is_recommendation_candidate = EXCLUDED.is_recommendation_candidate,
        recommendation_success = EXCLUDED.recommendation_success
    """
    rows = [
        (
            o["prediction_time"],
            o["item_id"],
            o["hour_offset"],
            o["offset_pct"],
            o["model_id"],
            o["evaluated_at"],
            o["predicted_fill_probability"],
            o["predicted_expected_value"],
            o["predicted_buy_price"],
            o["predicted_sell_price"],
            o["actual_low_price"],
            o["actual_high_price"],
            o["buy_would_fill"],
            o["sell_would_fill"],
            o["realized_fill"],
            o["realized_edge"],
            o["is_recommendation_candidate"],
            o["recommendation_success"],
        )
        for o in outcomes
    ]

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=2000)
    conn.commit()
    return len(rows)


def persist_snapshot(
    conn,
    *,
    window_start: datetime,
    window_end: datetime,
    model_id: str,
    summary: dict[str, Any],
    trade_outcomes: dict[str, Any],
    feedback_count: int,
    feedback_breakdown: dict[str, int],
    horizon_metrics: list[dict[str, Any]],
    offset_metrics: list[dict[str, Any]],
    development_signals: list[dict[str, Any]],
) -> int:
    sql = """
    INSERT INTO model_validation_snapshots (
        window_start,
        window_end,
        model_id,
        total_evaluated,
        recommendation_candidates,
        mean_predicted_fill_probability,
        realized_fill_rate,
        calibration_error,
        brier_score,
        recommendation_success_rate,
        expected_value_mean,
        realized_edge_mean,
        expected_minus_realized_edge,
        trade_outcomes_count,
        trade_outcomes_win_rate,
        trade_outcomes_avg_profit,
        feedback_count,
        feedback_breakdown,
        horizon_metrics,
        offset_metrics,
        development_signals
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
    )
    RETURNING id
    """
    params = (
        window_start,
        window_end,
        model_id,
        summary["total_evaluated"],
        summary["recommendation_candidates"],
        summary["mean_predicted_fill_probability"],
        summary["realized_fill_rate"],
        summary["calibration_error"],
        summary["brier_score"],
        summary["recommendation_success_rate"],
        summary["expected_value_mean"],
        summary["realized_edge_mean"],
        summary["expected_minus_realized_edge"],
        trade_outcomes.get("count", 0),
        trade_outcomes.get("win_rate"),
        trade_outcomes.get("avg_profit"),
        feedback_count,
        Json(feedback_breakdown),
        Json(horizon_metrics),
        Json(offset_metrics),
        Json(development_signals),
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    conn.commit()
    return int(row[0])


def group_outcomes_by_model(outcomes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[row["model_id"]].append(row)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate model predictions/recommendations against realized market outcomes.",
    )
    parser.add_argument("--window-hours", type=int, default=168, help="Validation lookback window in hours.")
    parser.add_argument(
        "--freshness-buffer-minutes",
        type=int,
        default=10,
        help="Ignore latest market data points by this buffer to avoid in-flight candles.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Optional model id/version filter (matches predictions.model_id when present).",
    )
    parser.add_argument("--max-rows", type=int, default=200000, help="Maximum matured prediction rows to evaluate.")
    parser.add_argument("--min-fill-probability", type=float, default=0.03, help="Recommendation candidate minimum fill probability.")
    parser.add_argument("--min-expected-value", type=float, default=0.005, help="Recommendation candidate minimum expected value.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print metrics without writing to DB.")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write summary JSON.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    now = datetime.now(timezone.utc)
    evaluated_at = now - timedelta(minutes=max(args.freshness_buffer_minutes, 0))
    window_end = evaluated_at
    window_start = evaluated_at - timedelta(hours=max(args.window_hours, 1))
    model_filter = args.model_id.strip() or None

    logger.info(
        "model_validation_start window_start=%s window_end=%s model_filter=%s dry_run=%s",
        window_start.isoformat(),
        window_end.isoformat(),
        model_filter or "ALL",
        args.dry_run,
    )

    conn = get_db_connection()
    snapshots_written: list[tuple[str, int]] = []
    try:
        ensure_validation_schema(conn)
        model_column = detect_predictions_model_column(conn)
        logger.info("predictions_model_column=%s", model_column or "none")

        matured_rows = fetch_matured_predictions(
            conn,
            window_start=window_start,
            window_end=window_end,
            evaluated_at=evaluated_at,
            model_filter=model_filter,
            model_column=model_column,
            max_rows=max(args.max_rows, 1),
        )
        if not matured_rows:
            logger.warning("No matured predictions found in selected window.")
            return 0

        outcomes = build_outcomes(
            matured_rows,
            evaluated_at=evaluated_at,
            min_fill_probability=args.min_fill_probability,
            min_expected_value=args.min_expected_value,
        )

        if not args.dry_run:
            written = persist_prediction_outcomes(conn, outcomes)
            logger.info("prediction_outcomes_upserted=%s", written)
        else:
            logger.info("dry_run=true prediction_outcomes_upsert_skipped count=%s", len(outcomes))

        feedback_count, feedback_breakdown = load_feedback_summary(
            conn,
            window_start=window_start,
            window_end=window_end,
        )
        trade_outcomes_summary = load_trade_outcome_summary(
            conn,
            window_start=window_start,
            window_end=window_end,
            model_id=model_filter,
        )

        grouped = group_outcomes_by_model(outcomes)
        summary_payload: dict[str, Any] = {
            "generated_at": evaluated_at.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "model_filter": model_filter,
            "models": [],
        }

        for model_id, model_rows in sorted(grouped.items()):
            summary = compute_core_metrics(model_rows)
            horizon_metrics = build_bucket_metrics(
                model_rows,
                bucket_key="hour_offset",
                bucket_label="hour_offset",
                value_digits=0,
            )
            offset_metrics = build_bucket_metrics(
                model_rows,
                bucket_key="offset_pct",
                bucket_label="offset_pct",
                value_digits=4,
            )
            development_signals = build_development_signals(
                summary=summary,
                horizon_metrics=horizon_metrics,
                offset_metrics=offset_metrics,
                feedback_breakdown=feedback_breakdown,
                trade_outcomes=trade_outcomes_summary,
            )

            snapshot_id = -1
            if not args.dry_run:
                snapshot_id = persist_snapshot(
                    conn,
                    window_start=window_start,
                    window_end=window_end,
                    model_id=model_id,
                    summary=summary,
                    trade_outcomes=trade_outcomes_summary,
                    feedback_count=feedback_count,
                    feedback_breakdown=feedback_breakdown,
                    horizon_metrics=horizon_metrics,
                    offset_metrics=offset_metrics,
                    development_signals=development_signals,
                )
                snapshots_written.append((model_id, snapshot_id))

            logger.info(
                "model=%s total=%s rec_candidates=%s fill_rate=%s rec_success=%s brier=%s snapshot_id=%s",
                model_id,
                summary["total_evaluated"],
                summary["recommendation_candidates"],
                summary["realized_fill_rate"],
                summary["recommendation_success_rate"],
                summary["brier_score"],
                snapshot_id if snapshot_id >= 0 else "dry-run",
            )

            summary_payload["models"].append(
                {
                    "model_id": model_id,
                    "snapshot_id": snapshot_id if snapshot_id >= 0 else None,
                    "summary": summary,
                    "development_signals": development_signals,
                    "feedback_count": feedback_count,
                    "feedback_breakdown": feedback_breakdown,
                    "trade_outcomes": trade_outcomes_summary,
                }
            )

        if args.output_json:
            out_path = os.path.abspath(args.output_json)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
            logger.info("wrote_json_summary=%s", out_path)

        if snapshots_written:
            logger.info("validation_snapshots_written=%s", snapshots_written)
        logger.info("model_validation_complete models=%s rows=%s", len(grouped), len(outcomes))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
