from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class GePT3HybridSnapshotArrays:
    """Fixed-size cross-item snapshots with per-item temporal history."""

    x_seq: np.ndarray  # (n_snapshots, snapshot_size, seq_len, n_features) float32
    item_id: np.ndarray  # (n_snapshots, snapshot_size) int32
    mask: np.ndarray  # (n_snapshots, snapshot_size) bool
    y_sell: np.ndarray  # (n_snapshots, snapshot_size, n_horizons, n_offsets) uint8
    ts: np.ndarray  # (n_snapshots,) int64 unix seconds
    feature_cols: list[str]
    horizons_hours: np.ndarray  # (n_horizons,) int32
    offsets: np.ndarray  # (n_offsets,) float32
    seq_len: int
    y_buy: np.ndarray | None = None  # (n_snapshots, snapshot_size, n_horizons, n_offsets) uint8
    y_price_high: np.ndarray | None = None  # (n_snapshots, snapshot_size, n_horizons) float32
    y_price_low: np.ndarray | None = None  # (n_snapshots, snapshot_size, n_horizons) float32
    y_regime: np.ndarray | None = None  # (n_snapshots, snapshot_size) uint8


def _build_item_local_positions(item_id: np.ndarray, seq_len: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    n = len(item_id)
    row_local_pos = np.full((n,), -1, dtype=np.int32)
    item_to_rows: dict[int, np.ndarray] = {}

    # Group rows by item id.
    order = np.argsort(item_id, kind="mergesort")
    item_sorted = item_id[order]
    uniq, starts, counts = np.unique(item_sorted, return_index=True, return_counts=True)
    for ui, it in enumerate(uniq.tolist()):
        s = int(starts[ui])
        c = int(counts[ui])
        rows = order[s : s + c]
        item_to_rows[int(it)] = rows
        row_local_pos[rows] = np.arange(c, dtype=np.int32)

    return row_local_pos, item_to_rows


def build_hybrid_snapshot_dataset_from_row_npz(
    row_npz_path: str | Path,
    seq_len: int = 96,
    snapshot_size: int = 128,
    min_items_per_snapshot: int = 24,
    max_snapshots: int = 12000,
    select_mode: Literal["uniform", "random"] = "uniform",
    row_pick_mode: Literal["random", "first"] = "random",
    seed: int = 1337,
) -> GePT3HybridSnapshotArrays:
    """Convert row-wise dataset into temporal hybrid snapshots."""
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

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

    row_local_pos, item_to_rows = _build_item_local_positions(item_id=item_id, seq_len=seq_len)
    eligible = row_local_pos >= (seq_len - 1)
    eligible_rows = np.where(eligible)[0]
    if eligible_rows.size == 0:
        raise RuntimeError("No rows have enough per-item history for seq_len.")

    order = eligible_rows[np.argsort(ts[eligible_rows], kind="mergesort")]
    ts_s = ts[order]
    uniq_ts, starts, counts = np.unique(ts_s, return_index=True, return_counts=True)

    candidates = np.where(counts >= int(min_items_per_snapshot))[0]
    if len(candidates) == 0:
        raise RuntimeError(
            f"No timestamps with at least {min_items_per_snapshot} seq-eligible items. "
            f"Try lowering --min-items-per-snapshot or seq_len."
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

    out_x_seq = np.zeros((n_snap, snapshot_size, seq_len, n_feat), dtype=np.float32)
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

        # Stable ordering by item_id helps determinism.
        pick = pick[np.argsort(item_id[pick], kind="mergesort")]

        for j, r in enumerate(pick.tolist()):
            it = int(item_id[r])
            pos = int(row_local_pos[r])
            rows_item = item_to_rows[it]
            seq_rows = rows_item[pos - seq_len + 1 : pos + 1]
            out_x_seq[si, j] = x[seq_rows]
            out_item[si, j] = it
            out_mask[si, j] = True
            out_y_sell[si, j] = y_sell[r]
            if out_y_buy is not None and y_buy is not None:
                out_y_buy[si, j] = y_buy[r]
            if out_y_price_high is not None and y_price_high is not None:
                out_y_price_high[si, j] = y_price_high[r]
            if out_y_price_low is not None and y_price_low is not None:
                out_y_price_low[si, j] = y_price_low[r]
            if out_y_regime is not None and y_regime is not None:
                out_y_regime[si, j] = y_regime[r]

        out_ts[si] = int(uniq_ts[cidx])

    return GePT3HybridSnapshotArrays(
        x_seq=out_x_seq,
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
        seq_len=int(seq_len),
    )


def save_hybrid_snapshot_npz(ds: GePT3HybridSnapshotArrays, out_path: str | Path) -> None:
    payload = {
        "x_seq": ds.x_seq,
        "item_id": ds.item_id,
        "mask": ds.mask.astype(np.uint8),
        "y_sell": ds.y_sell,
        "ts": ds.ts,
        "feature_cols": np.asarray(ds.feature_cols, dtype=object),
        "horizons_hours": ds.horizons_hours.astype(np.int32, copy=False),
        "offsets": ds.offsets.astype(np.float32, copy=False),
        "seq_len": np.asarray([int(ds.seq_len)], dtype=np.int32),
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


def load_hybrid_snapshot_npz(path: str | Path) -> GePT3HybridSnapshotArrays:
    arr = np.load(str(path), allow_pickle=True)
    seq_len = int(arr["seq_len"].astype(np.int32, copy=False)[0]) if "seq_len" in arr.files else int(arr["x_seq"].shape[2])
    return GePT3HybridSnapshotArrays(
        x_seq=arr["x_seq"].astype(np.float32, copy=False),
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
        seq_len=seq_len,
    )
