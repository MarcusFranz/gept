from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class GePT3SnapshotArrays:
    """Fixed-size cross-item snapshots for GePT3 training."""

    x: np.ndarray  # (n_snapshots, snapshot_size, n_features) float32
    item_id: np.ndarray  # (n_snapshots, snapshot_size) int32
    mask: np.ndarray  # (n_snapshots, snapshot_size) bool
    y_sell: np.ndarray  # (n_snapshots, snapshot_size, n_horizons, n_offsets) uint8
    ts: np.ndarray  # (n_snapshots,) int64 unix seconds
    feature_cols: list[str]
    horizons_hours: np.ndarray  # (n_horizons,) int32
    offsets: np.ndarray  # (n_offsets,) float32
    y_buy: np.ndarray | None = None  # (n_snapshots, snapshot_size, n_horizons, n_offsets) uint8
    y_price_high: np.ndarray | None = None  # (n_snapshots, snapshot_size, n_horizons) float32
    y_price_low: np.ndarray | None = None  # (n_snapshots, snapshot_size, n_horizons) float32
    y_regime: np.ndarray | None = None  # (n_snapshots, snapshot_size) uint8


def build_snapshot_dataset_from_row_npz(
    row_npz_path: str | Path,
    snapshot_size: int = 128,
    min_items_per_snapshot: int = 64,
    max_snapshots: int = 6000,
    select_mode: Literal["uniform", "random"] = "uniform",
    row_pick_mode: Literal["random", "first"] = "random",
    seed: int = 1337,
) -> GePT3SnapshotArrays:
    """Convert row-wise sell-fill dataset into fixed-size cross-item snapshots."""
    arr = np.load(str(row_npz_path), allow_pickle=True)
    x = arr["x"].astype(np.float32, copy=False)
    item_id = arr["item_id"].astype(np.int32, copy=False)
    ts = arr["ts"].astype(np.int64, copy=False)
    y_sell = arr["y_sell"].astype(np.uint8, copy=False) if "y_sell" in arr.files else arr["y"].astype(np.uint8, copy=False)
    y_buy = arr["y_buy"].astype(np.uint8, copy=False) if "y_buy" in arr.files else None
    y_price_high = arr["y_price_high"].astype(np.float32, copy=False) if "y_price_high" in arr.files else None
    y_price_low = arr["y_price_low"].astype(np.float32, copy=False) if "y_price_low" in arr.files else None
    y_regime = arr["y_regime"].astype(np.uint8, copy=False) if "y_regime" in arr.files else None

    feature_cols = [str(v) for v in arr["feature_cols"].tolist()]
    horizons = arr["horizons_hours"].astype(np.int32, copy=False)
    offsets = arr["offsets"].astype(np.float32, copy=False)

    order = np.argsort(ts, kind="mergesort")
    ts_s = ts[order]
    uniq_ts, starts, counts = np.unique(ts_s, return_index=True, return_counts=True)

    candidates = np.where(counts >= int(min_items_per_snapshot))[0]
    if len(candidates) == 0:
        raise RuntimeError(
            f"No timestamps with at least {min_items_per_snapshot} items. "
            f"Try lowering --min-items-per-snapshot."
        )

    rng = np.random.default_rng(seed)
    if max_snapshots > 0 and len(candidates) > max_snapshots:
        if select_mode == "uniform":
            pos = np.linspace(0, len(candidates) - 1, max_snapshots, dtype=np.int64)
            candidates = candidates[pos]
        elif select_mode == "random":
            candidates = np.sort(rng.choice(candidates, size=max_snapshots, replace=False))
        else:
            raise ValueError(f"Unsupported select_mode={select_mode}")

    n_snap = len(candidates)
    n_feat = x.shape[1]
    n_h = y_sell.shape[1]
    n_o = y_sell.shape[2]

    out_x = np.zeros((n_snap, snapshot_size, n_feat), dtype=np.float32)
    out_item = np.zeros((n_snap, snapshot_size), dtype=np.int32)
    out_mask = np.zeros((n_snap, snapshot_size), dtype=bool)
    out_y_sell = np.zeros((n_snap, snapshot_size, n_h, n_o), dtype=np.uint8)
    out_y_buy = np.zeros((n_snap, snapshot_size, n_h, n_o), dtype=np.uint8) if y_buy is not None else None
    out_y_price_high = np.zeros((n_snap, snapshot_size, n_h), dtype=np.float32) if y_price_high is not None else None
    out_y_price_low = np.zeros((n_snap, snapshot_size, n_h), dtype=np.float32) if y_price_low is not None else None
    out_y_regime = np.zeros((n_snap, snapshot_size), dtype=np.uint8) if y_regime is not None else None
    out_ts = np.zeros((n_snap,), dtype=np.int64)

    for si, cidx in enumerate(candidates):
        start = int(starts[cidx])
        count = int(counts[cidx])
        rows = order[start : start + count]

        k = min(snapshot_size, count)
        if row_pick_mode == "random":
            pick = rng.choice(rows, size=k, replace=False)
        elif row_pick_mode == "first":
            pick = rows[:k]
        else:
            raise ValueError(f"Unsupported row_pick_mode={row_pick_mode}")

        # Stable per-snapshot ordering by item_id helps training determinism.
        pick = pick[np.argsort(item_id[pick], kind="mergesort")]

        out_x[si, :k] = x[pick]
        out_item[si, :k] = item_id[pick]
        out_mask[si, :k] = True
        out_y_sell[si, :k] = y_sell[pick]
        if out_y_buy is not None and y_buy is not None:
            out_y_buy[si, :k] = y_buy[pick]
        if out_y_price_high is not None and y_price_high is not None:
            out_y_price_high[si, :k] = y_price_high[pick]
        if out_y_price_low is not None and y_price_low is not None:
            out_y_price_low[si, :k] = y_price_low[pick]
        if out_y_regime is not None and y_regime is not None:
            out_y_regime[si, :k] = y_regime[pick]
        out_ts[si] = int(uniq_ts[cidx])

    return GePT3SnapshotArrays(
        x=out_x,
        item_id=out_item,
        mask=out_mask,
        y_sell=out_y_sell,
        y_buy=out_y_buy,
        y_price_high=out_y_price_high,
        y_price_low=out_y_price_low,
        y_regime=out_y_regime,
        ts=out_ts,
        feature_cols=feature_cols,
        horizons_hours=horizons,
        offsets=offsets,
    )


def save_snapshot_npz(ds: GePT3SnapshotArrays, out_path: str | Path) -> None:
    payload = {
        "x": ds.x,
        "item_id": ds.item_id,
        "mask": ds.mask.astype(np.uint8),
        "y_sell": ds.y_sell,
        "ts": ds.ts,
        "feature_cols": np.asarray(ds.feature_cols, dtype=object),
        "horizons_hours": ds.horizons_hours.astype(np.int32, copy=False),
        "offsets": ds.offsets.astype(np.float32, copy=False),
    }
    if ds.y_buy is not None:
        payload["y_buy"] = ds.y_buy
    if ds.y_price_high is not None:
        payload["y_price_high"] = ds.y_price_high
    if ds.y_price_low is not None:
        payload["y_price_low"] = ds.y_price_low
    if ds.y_regime is not None:
        payload["y_regime"] = ds.y_regime
    np.savez_compressed(str(out_path), **payload)


def load_snapshot_npz(path: str | Path) -> GePT3SnapshotArrays:
    arr = np.load(str(path), allow_pickle=True)
    return GePT3SnapshotArrays(
        x=arr["x"].astype(np.float32, copy=False),
        item_id=arr["item_id"].astype(np.int32, copy=False),
        mask=arr["mask"].astype(bool, copy=False),
        y_sell=arr["y_sell"].astype(np.uint8, copy=False),
        y_buy=arr["y_buy"].astype(np.uint8, copy=False) if "y_buy" in arr.files else None,
        y_price_high=arr["y_price_high"].astype(np.float32, copy=False) if "y_price_high" in arr.files else None,
        y_price_low=arr["y_price_low"].astype(np.float32, copy=False) if "y_price_low" in arr.files else None,
        y_regime=arr["y_regime"].astype(np.uint8, copy=False) if "y_regime" in arr.files else None,
        ts=arr["ts"].astype(np.int64, copy=False),
        feature_cols=[str(v) for v in arr["feature_cols"].tolist()],
        horizons_hours=arr["horizons_hours"].astype(np.int32, copy=False),
        offsets=arr["offsets"].astype(np.float32, copy=False),
    )


def infer_n_items(item_id: np.ndarray, mask: np.ndarray) -> int:
    """Compute embedding size from active items."""
    valid = item_id[mask]
    if valid.size == 0:
        return 1
    return int(valid.max()) + 1


def time_split_indices(
    ts: Sequence[int] | np.ndarray,
    val_days: int,
    test_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.asarray(ts, dtype=np.int64)
    if len(ts) == 0:
        raise ValueError("Empty ts")

    max_ts = int(ts.max())
    test_cut = max_ts - int(test_days) * 86400
    val_cut = test_cut - int(val_days) * 86400

    train_idx = np.where(ts < val_cut)[0]
    val_idx = np.where((ts >= val_cut) & (ts < test_cut))[0]
    test_idx = np.where(ts >= test_cut)[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Split produced empty partition(s): train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )
    return train_idx, val_idx, test_idx
