#!/usr/bin/env python3
"""Train GePT3 hybrid model (temporal per-item encoder + cross-item attention)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Allow `import src...` when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.gept3.artifact import GePT3Artifact
from src.gept3.config import GePT3ModelConfig
from src.gept3.dataset import infer_n_items, time_split_indices
from src.gept3.hybrid_dataset import GePT3HybridSnapshotArrays, load_hybrid_snapshot_npz
from src.gept3.hybrid_model import GePT3HybridModel


def _pick_device(device_arg: str) -> torch.device:
    forced = (device_arg or "auto").strip().lower()
    if forced != "auto":
        if forced == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested --device=cuda but CUDA is not available")
        if forced == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Requested --device=mps but MPS is not available")
        return torch.device(forced)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _iter_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    idx = indices.copy()
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i : i + batch_size]


def _compute_feature_norm(ds: GePT3HybridSnapshotArrays, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_feat = ds.x_seq.shape[3]
    sum_v = np.zeros((n_feat,), dtype=np.float64)
    sumsq_v = np.zeros((n_feat,), dtype=np.float64)
    n = 0

    chunk = 128
    for i in range(0, len(indices), chunk):
        bidx = indices[i : i + chunk]
        xb = ds.x_seq[bidx]  # (B, I, T, F)
        mb = ds.mask[bidx]  # (B, I)
        vals = xb[mb]  # (N_items, T, F)
        if vals.size == 0:
            continue
        vals = vals.reshape(-1, n_feat)
        vals64 = vals.astype(np.float64, copy=False)
        sum_v += vals64.sum(axis=0)
        sumsq_v += (vals64 * vals64).sum(axis=0)
        n += vals64.shape[0]

    if n == 0:
        raise ValueError("No valid feature rows found while computing normalization")

    mean = (sum_v / n).astype(np.float32)
    var = (sumsq_v / n) - np.square(mean.astype(np.float64))
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def _batch_to_tensors(
    ds: GePT3HybridSnapshotArrays,
    batch_idx: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    x_seq = (
        (ds.x_seq[batch_idx] - feature_mean.reshape(1, 1, 1, -1)) / feature_std.reshape(1, 1, 1, -1)
    ).astype(np.float32, copy=False)
    item_id = ds.item_id[batch_idx].astype(np.int64, copy=False)
    mask = ds.mask[batch_idx].astype(bool, copy=False)

    targets: dict[str, torch.Tensor] = {
        "y_sell": torch.from_numpy(ds.y_sell[batch_idx].astype(np.float32, copy=False)).to(device)
    }
    if ds.y_buy is not None:
        targets["y_buy"] = torch.from_numpy(ds.y_buy[batch_idx].astype(np.float32, copy=False)).to(device)
    if ds.y_price_high is not None:
        targets["y_price_high"] = torch.from_numpy(ds.y_price_high[batch_idx].astype(np.float32, copy=False)).to(device)
    if ds.y_price_low is not None:
        targets["y_price_low"] = torch.from_numpy(ds.y_price_low[batch_idx].astype(np.float32, copy=False)).to(device)
    if ds.y_regime is not None:
        targets["y_regime"] = torch.from_numpy(ds.y_regime[batch_idx].astype(np.float32, copy=False)).to(device)

    x_t = torch.from_numpy(x_seq).to(device)
    item_t = torch.from_numpy(item_id).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    return x_t, item_t, mask_t, targets


def _masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = mask.float()
    for _ in range(raw.dim() - 2):
        w = w.unsqueeze(-1)
    w = w.expand_as(raw)
    denom = w.sum().clamp(min=1.0)
    return (raw * w).sum() / denom


def _masked_quantile_loss(
    pred: torch.Tensor,  # (B, I, H, Q)
    target: torch.Tensor,  # (B, I, H)
    quantiles: torch.Tensor,  # (Q,)
    mask: torch.Tensor,  # (B, I)
) -> torch.Tensor:
    err = target.unsqueeze(-1) - pred
    q = quantiles.view(1, 1, 1, -1)
    raw = torch.maximum((q - 1.0) * err, q * err)
    w = mask.unsqueeze(-1).unsqueeze(-1).float().expand_as(raw)
    denom = w.sum().clamp(min=1.0)
    return (raw * w).sum() / denom


def _init_prob_state() -> dict[str, float]:
    return {
        "sum_brier": 0.0,
        "sum_logloss": 0.0,
        "sum_y": 0.0,
        "sum_p": 0.0,
        "n": 0.0,
        "n_ge_080": 0.0,
        "sum_y_ge_080": 0.0,
    }


def _update_prob_state(state: dict[str, float], prob_v: torch.Tensor, y_v: torch.Tensor, track_080: bool = False) -> None:
    if prob_v.numel() == 0:
        return
    eps = 1e-6
    state["sum_brier"] += torch.square(prob_v - y_v).sum().item()
    p_clip = prob_v.clamp(min=eps, max=1.0 - eps)
    state["sum_logloss"] += (-(y_v * torch.log(p_clip) + (1.0 - y_v) * torch.log(1.0 - p_clip))).sum().item()
    state["sum_y"] += y_v.sum().item()
    state["sum_p"] += prob_v.sum().item()
    state["n"] += float(prob_v.numel())

    if track_080:
        hi = prob_v >= 0.80
        n_hi = int(hi.sum().item())
        if n_hi > 0:
            state["n_ge_080"] += float(n_hi)
            state["sum_y_ge_080"] += y_v[hi].sum().item()


def _finalize_prob_state(state: dict[str, float], include_080: bool = False) -> dict[str, Any]:
    n = int(state["n"])
    if n == 0:
        out: dict[str, Any] = {
            "n": 0,
            "brier": float("nan"),
            "logloss": float("nan"),
            "pos_rate": float("nan"),
            "mean_pred": float("nan"),
        }
        if include_080:
            out["n_pred_ge_080"] = 0
            out["empirical_pos_given_pred_ge_080"] = float("nan")
        return out

    out = {
        "n": n,
        "brier": float(state["sum_brier"] / state["n"]),
        "logloss": float(state["sum_logloss"] / state["n"]),
        "pos_rate": float(state["sum_y"] / state["n"]),
        "mean_pred": float(state["sum_p"] / state["n"]),
    }
    if include_080:
        n_hi = int(state["n_ge_080"])
        out["n_pred_ge_080"] = n_hi
        out["empirical_pos_given_pred_ge_080"] = (
            float(state["sum_y_ge_080"] / state["n_ge_080"]) if n_hi > 0 else float("nan")
        )
    return out


def _autocast_context(device: torch.device, precision: str):
    p = precision.strip().lower()
    if device.type != "cuda":
        return nullcontext()
    if p == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if p == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _resolve_run_dir(out_dir: str, run_name: str) -> Path:
    base = Path(out_dir)
    if run_name:
        return base / run_name
    return base / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _default_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "latest.pt"


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _restore_checkpoint(
    path: Path,
    model: GePT3HybridModel,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)

    scaler_state = ckpt.get("scaler_state_dict")
    if scaler is not None and scaler_state:
        scaler.load_state_dict(scaler_state)

    np_state = ckpt.get("numpy_rng_state")
    if np_state:
        rng.bit_generator.state = np_state

    torch_rng_state = ckpt.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.random.set_rng_state(torch_rng_state)

    if device.type == "cuda":
        cuda_state_all = ckpt.get("cuda_rng_state_all")
        if cuda_state_all:
            try:
                torch.cuda.set_rng_state_all(cuda_state_all)
            except Exception:
                pass

    return int(ckpt.get("epoch", 0))


@torch.no_grad()
def _eval_model(
    model: GePT3HybridModel,
    ds: GePT3HybridSnapshotArrays,
    indices: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    seed: int,
    eval_buy: bool,
    eval_price: bool,
    eval_regime: bool,
    price_quantiles: list[float],
    precision: str,
) -> dict[str, Any]:
    model.eval()
    rng = np.random.default_rng(seed)
    eval_idx = indices.copy()
    rng.shuffle(eval_idx)

    sell_state = _init_prob_state()
    buy_state = _init_prob_state() if eval_buy else None
    regime_state = _init_prob_state() if eval_regime else None

    price_sum_abs_hi = 0.0
    price_sum_sq_hi = 0.0
    price_n_hi = 0
    price_sum_abs_lo = 0.0
    price_sum_sq_lo = 0.0
    price_n_lo = 0
    median_q_idx = int(np.argmin(np.abs(np.asarray(price_quantiles, dtype=np.float32) - 0.5)))

    batch_count = 0
    for i in range(0, len(eval_idx), batch_size):
        if max_batches > 0 and batch_count >= max_batches:
            break
        batch = eval_idx[i : i + batch_size]
        x_t, item_t, mask_t, targets = _batch_to_tensors(ds, batch, feature_mean, feature_std, device)
        with _autocast_context(device, precision):
            out = model(x_t, item_t, mask_t)

        sell_prob = torch.sigmoid(out["sell_logits"])
        sell_mask = mask_t.unsqueeze(-1).unsqueeze(-1).expand_as(sell_prob)
        _update_prob_state(sell_state, sell_prob[sell_mask], targets["y_sell"][sell_mask], track_080=True)

        if buy_state is not None and "buy_logits" in out and "y_buy" in targets:
            buy_prob = torch.sigmoid(out["buy_logits"])
            buy_mask = mask_t.unsqueeze(-1).unsqueeze(-1).expand_as(buy_prob)
            _update_prob_state(buy_state, buy_prob[buy_mask], targets["y_buy"][buy_mask], track_080=False)

        if eval_price and "price_high_deltas" in out and "price_low_deltas" in out:
            if "y_price_high" in targets and "y_price_low" in targets:
                pred_hi = out["price_high_deltas"][..., median_q_idx]
                pred_lo = out["price_low_deltas"][..., median_q_idx]
                y_hi = targets["y_price_high"]
                y_lo = targets["y_price_low"]

                m = mask_t.unsqueeze(-1).expand_as(y_hi)
                if int(m.sum().item()) > 0:
                    diff_hi = (pred_hi - y_hi)[m]
                    diff_lo = (pred_lo - y_lo)[m]
                    price_sum_abs_hi += torch.abs(diff_hi).sum().item()
                    price_sum_sq_hi += torch.square(diff_hi).sum().item()
                    price_n_hi += int(diff_hi.numel())
                    price_sum_abs_lo += torch.abs(diff_lo).sum().item()
                    price_sum_sq_lo += torch.square(diff_lo).sum().item()
                    price_n_lo += int(diff_lo.numel())

        if regime_state is not None and "regime_logit" in out and "y_regime" in targets:
            regime_prob = torch.sigmoid(out["regime_logit"])
            regime_mask = mask_t
            _update_prob_state(
                regime_state, regime_prob[regime_mask], targets["y_regime"][regime_mask], track_080=False
            )

        batch_count += 1

    out_metrics: dict[str, Any] = {
        "sell": _finalize_prob_state(sell_state, include_080=True),
    }
    if buy_state is not None:
        out_metrics["buy"] = _finalize_prob_state(buy_state, include_080=False)
    if regime_state is not None:
        out_metrics["regime"] = _finalize_prob_state(regime_state, include_080=False)
    if eval_price:
        out_metrics["price"] = {
            "n_high": int(price_n_hi),
            "high_mae": float(price_sum_abs_hi / price_n_hi) if price_n_hi > 0 else float("nan"),
            "high_rmse": float(np.sqrt(price_sum_sq_hi / price_n_hi)) if price_n_hi > 0 else float("nan"),
            "n_low": int(price_n_lo),
            "low_mae": float(price_sum_abs_lo / price_n_lo) if price_n_lo > 0 else float("nan"),
            "low_rmse": float(np.sqrt(price_sum_sq_lo / price_n_lo)) if price_n_lo > 0 else float("nan"),
            "median_quantile": float(price_quantiles[median_q_idx]),
        }
    return out_metrics


def train(args: argparse.Namespace) -> int:
    ds = load_hybrid_snapshot_npz(args.dataset)
    train_idx, val_idx, test_idx = time_split_indices(ds.ts, val_days=args.val_days, test_days=args.test_days)

    feature_mean, feature_std = _compute_feature_norm(ds, train_idx)
    n_items = infer_n_items(ds.item_id, ds.mask)

    if args.enable_all_heads:
        args.enable_buy_head = True
        args.enable_price_head = True
        args.enable_regime_head = True

    train_buy_head = bool(args.enable_buy_head and ds.y_buy is not None)
    train_price_head = bool(args.enable_price_head and ds.y_price_high is not None and ds.y_price_low is not None)
    train_regime_head = bool(args.enable_regime_head and ds.y_regime is not None)

    if args.enable_buy_head and not train_buy_head:
        print("warning: buy head requested but y_buy labels are missing; disabling buy loss.", flush=True)
    if args.enable_price_head and not train_price_head:
        print("warning: price head requested but y_price labels are missing; disabling price loss.", flush=True)
    if args.enable_regime_head and not train_regime_head:
        print("warning: regime head requested but y_regime labels are missing; disabling regime loss.", flush=True)

    price_quantiles = tuple(np.linspace(0.1, 0.9, int(args.n_price_quantiles), dtype=np.float32).tolist())
    model_cfg = GePT3ModelConfig(
        n_items=n_items,
        n_features=int(ds.x_seq.shape[3]),
        n_horizons=int(ds.y_sell.shape[2]),
        n_offsets=int(ds.y_sell.shape[3]),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        temporal_len=int(ds.seq_len),
        temporal_n_heads=args.temporal_n_heads,
        temporal_n_layers=args.temporal_n_layers,
        n_price_quantiles=args.n_price_quantiles,
        price_quantiles=price_quantiles,
        enable_sell_head=True,
        enable_buy_head=train_buy_head,
        enable_price_head=train_price_head,
        enable_regime_head=train_regime_head,
    )
    model = GePT3HybridModel(model_cfg)

    device = _pick_device(args.device)
    print(f"training_device={device}", flush=True)
    print(f"precision={args.precision}", flush=True)
    print(
        f"train_heads sell=True buy={train_buy_head} price={train_price_head} regime={train_regime_head}",
        flush=True,
    )
    print(f"seq_len={ds.seq_len}", flush=True)
    print(f"grad_accum_steps={max(1, int(args.grad_accum_steps))}", flush=True)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    q_t = torch.tensor(list(model_cfg.price_quantiles), dtype=torch.float32, device=device)

    use_scaler = (device.type == "cuda") and (args.precision.lower() == "fp16")
    if use_scaler:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    run_dir = _resolve_run_dir(args.out_dir, args.run_name)
    checkpoint_path = Path(args.resume_from).expanduser() if args.resume_from else _default_checkpoint_path(run_dir)
    start_epoch = 1
    last_val_metrics: dict[str, Any] | None = None

    if args.resume_from and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if (args.auto_resume or args.resume_from) and checkpoint_path.exists():
        restored_epoch = _restore_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=opt,
            scaler=scaler,
            device=device,
            rng=rng,
        )
        start_epoch = max(1, restored_epoch + 1)
        print(
            f"resumed_checkpoint={checkpoint_path} restored_epoch={restored_epoch} next_epoch={start_epoch}",
            flush=True,
        )
    else:
        print(f"resume_skipped checkpoint={checkpoint_path} exists={checkpoint_path.exists()}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = perf_counter()
        model.train(True)
        total_losses: list[float] = []
        sell_losses: list[float] = []
        buy_losses: list[float] = []
        price_losses: list[float] = []
        regime_losses: list[float] = []
        n_train_batches = int(np.ceil(len(train_idx) / max(1, args.batch_size)))
        grad_accum_steps = max(1, int(args.grad_accum_steps))
        opt.zero_grad(set_to_none=True)
        opt_steps = 0

        for bi, batch in enumerate(_iter_batches(train_idx, args.batch_size, rng), start=1):
            x_t, item_t, mask_t, targets = _batch_to_tensors(ds, batch, feature_mean, feature_std, device)

            with _autocast_context(device, args.precision):
                out = model(x_t, item_t, mask_t)
                sell_loss = _masked_bce_loss(out["sell_logits"], targets["y_sell"], mask_t)
                loss = args.loss_w_sell * sell_loss

                if train_buy_head:
                    buy_loss = _masked_bce_loss(out["buy_logits"], targets["y_buy"], mask_t)
                    loss = loss + (args.loss_w_buy * buy_loss)
                    buy_losses.append(float(buy_loss.detach().cpu().item()))

                if train_price_head:
                    price_high_loss = _masked_quantile_loss(out["price_high_deltas"], targets["y_price_high"], q_t, mask_t)
                    price_low_loss = _masked_quantile_loss(out["price_low_deltas"], targets["y_price_low"], q_t, mask_t)
                    price_loss = 0.5 * (price_high_loss + price_low_loss)
                    loss = loss + (args.loss_w_price * price_loss)
                    price_losses.append(float(price_loss.detach().cpu().item()))

                if train_regime_head:
                    regime_loss = _masked_bce_loss(out["regime_logit"], targets["y_regime"], mask_t)
                    loss = loss + (args.loss_w_regime * regime_loss)
                    regime_losses.append(float(regime_loss.detach().cpu().item()))

            loss_for_backward = loss / float(grad_accum_steps)
            if use_scaler:
                assert scaler is not None
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            should_step = (bi % grad_accum_steps == 0) or (bi == n_train_batches)
            if should_step:
                if use_scaler:
                    assert scaler is not None
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                opt_steps += 1
                if device.type == "mps" and int(args.mps_empty_cache_every_steps) > 0:
                    if opt_steps % int(args.mps_empty_cache_every_steps) == 0:
                        torch.mps.empty_cache()
                if int(args.gc_every_steps) > 0 and opt_steps % int(args.gc_every_steps) == 0:
                    gc.collect()

            total_losses.append(float(loss.detach().cpu().item()))
            sell_losses.append(float(sell_loss.detach().cpu().item()))

            if args.log_every_batches > 0 and (
                bi == 1 or bi % int(args.log_every_batches) == 0 or bi == n_train_batches
            ):
                elapsed = perf_counter() - epoch_t0
                avg_total = float(np.mean(total_losses)) if total_losses else float("nan")
                print(
                    f"epoch={epoch} batch={bi}/{n_train_batches} train_total_running={avg_total:.4f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

        train_total = float(np.mean(total_losses)) if total_losses else float("nan")
        train_sell = float(np.mean(sell_losses)) if sell_losses else float("nan")
        train_buy_loss = float(np.mean(buy_losses)) if buy_losses else float("nan")
        train_price_loss = float(np.mean(price_losses)) if price_losses else float("nan")
        train_regime_loss = float(np.mean(regime_losses)) if regime_losses else float("nan")

        val_metrics = _eval_model(
            model=model,
            ds=ds,
            indices=val_idx,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_size=args.eval_batch_size,
            device=device,
            max_batches=args.eval_max_batches,
            seed=args.seed + epoch,
            eval_buy=bool(model_cfg.enable_buy_head),
            eval_price=bool(model_cfg.enable_price_head),
            eval_regime=bool(model_cfg.enable_regime_head),
            price_quantiles=list(model_cfg.price_quantiles),
            precision=args.precision,
        )

        sell = val_metrics["sell"]
        msg = (
            f"epoch={epoch} train_total={train_total:.4f} train_sell={train_sell:.4f} "
            f"train_buy={train_buy_loss:.4f} train_price={train_price_loss:.4f} train_regime={train_regime_loss:.4f} "
            f"val_sell_brier={sell['brier']:.4f} val_sell_logloss={sell['logloss']:.4f} "
            f"val_empirical_pos_at_080={sell['empirical_pos_given_pred_ge_080']:.3f}"
        )
        if "buy" in val_metrics:
            msg += f" val_buy_brier={val_metrics['buy']['brier']:.4f}"
        if "price" in val_metrics:
            msg += (
                f" val_price_hi_mae={val_metrics['price']['high_mae']:.4f}"
                f" val_price_lo_mae={val_metrics['price']['low_mae']:.4f}"
            )
        if "regime" in val_metrics:
            msg += f" val_regime_brier={val_metrics['regime']['brier']:.4f}"
        print(msg, flush=True)
        last_val_metrics = val_metrics

        if int(args.checkpoint_every_epochs) > 0 and epoch % int(args.checkpoint_every_epochs) == 0:
            ckpt_payload: dict[str, Any] = {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "run_name": args.run_name,
                "dataset": str(args.dataset),
                "args": vars(args),
            }
            _save_checkpoint(checkpoint_path, ckpt_payload)
            print(f"saved_checkpoint={checkpoint_path} epoch={epoch}", flush=True)

    if last_val_metrics is None:
        print("no_training_epochs_executed; running val eval from current checkpoint/model state", flush=True)
        last_val_metrics = _eval_model(
            model=model,
            ds=ds,
            indices=val_idx,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_size=args.eval_batch_size,
            device=device,
            max_batches=args.eval_max_batches,
            seed=args.seed + max(1, args.epochs),
            eval_buy=bool(model_cfg.enable_buy_head),
            eval_price=bool(model_cfg.enable_price_head),
            eval_regime=bool(model_cfg.enable_regime_head),
            price_quantiles=list(model_cfg.price_quantiles),
            precision=args.precision,
        )

    test_metrics = _eval_model(
        model=model,
        ds=ds,
        indices=test_idx,
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=args.eval_batch_size,
        device=device,
        max_batches=args.eval_max_batches,
        seed=args.seed + 999,
        eval_buy=bool(model_cfg.enable_buy_head),
        eval_price=bool(model_cfg.enable_price_head),
        eval_regime=bool(model_cfg.enable_regime_head),
        price_quantiles=list(model_cfg.price_quantiles),
        precision=args.precision,
    )
    test_sell = test_metrics["sell"]
    print(
        f"test_sell_brier={test_sell['brier']:.4f} test_sell_logloss={test_sell['logloss']:.4f} "
        f"test_sell_pos_rate={test_sell['pos_rate']:.3f} test_sell_mean_pred={test_sell['mean_pred']:.3f} "
        f"test_sell_empirical_pos_at_080={test_sell['empirical_pos_given_pred_ge_080']:.3f}",
        flush=True,
    )
    if "buy" in test_metrics:
        print(
            f"test_buy_brier={test_metrics['buy']['brier']:.4f} test_buy_logloss={test_metrics['buy']['logloss']:.4f}",
            flush=True,
        )
    if "price" in test_metrics:
        print(
            f"test_price_hi_mae={test_metrics['price']['high_mae']:.4f} "
            f"test_price_lo_mae={test_metrics['price']['low_mae']:.4f}",
            flush=True,
        )
    if "regime" in test_metrics:
        print(
            f"test_regime_brier={test_metrics['regime']['brier']:.4f} "
            f"test_regime_logloss={test_metrics['regime']['logloss']:.4f}",
            flush=True,
        )

    artifact = GePT3Artifact(
        model_config=model_cfg,
        feature_cols=list(ds.feature_cols),
        horizons_hours=[int(v) for v in ds.horizons_hours.tolist()],
        offsets=[float(v) for v in ds.offsets.tolist()],
        feature_mean=feature_mean.astype(np.float32).tolist(),
        feature_std=feature_std.astype(np.float32).tolist(),
        model_state_dict=model.state_dict(),
        notes={
            "task": "hybrid_multitask",
            "dataset": str(args.dataset),
            "n_snapshots": int(ds.x_seq.shape[0]),
            "snapshot_size": int(ds.x_seq.shape[1]),
            "seq_len": int(ds.seq_len),
            "heads": {
                "sell": True,
                "buy": bool(model_cfg.enable_buy_head),
                "price": bool(model_cfg.enable_price_head),
                "regime": bool(model_cfg.enable_regime_head),
            },
        },
    )
    artifact.save_dir(run_dir)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "n_snapshots": int(ds.x_seq.shape[0]),
        "snapshot_size": int(ds.x_seq.shape[1]),
        "seq_len": int(ds.seq_len),
        "n_features": int(ds.x_seq.shape[3]),
        "n_items": int(n_items),
        "n_horizons": int(ds.y_sell.shape[2]),
        "n_offsets": int(ds.y_sell.shape[3]),
        "train_snapshots": int(len(train_idx)),
        "val_snapshots": int(len(val_idx)),
        "test_snapshots": int(len(test_idx)),
        "heads": {
            "sell": True,
            "buy": bool(model_cfg.enable_buy_head),
            "price": bool(model_cfg.enable_price_head),
            "regime": bool(model_cfg.enable_regime_head),
        },
        "precision": args.precision,
        "loss_weights": {
            "sell": float(args.loss_w_sell),
            "buy": float(args.loss_w_buy),
            "price": float(args.loss_w_price),
            "regime": float(args.loss_w_regime),
        },
        "val_metrics": last_val_metrics,
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    try:
        scripted = torch.jit.script(model.cpu())
        scripted.save(str(run_dir / "model_torchscript.pt"))
        print(f"saved_torchscript={run_dir / 'model_torchscript.pt'}")
    except Exception as exc:
        print(f"torchscript_export_failed: {exc}")

    print(f"saved_artifact_dir={run_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to GePT3 hybrid snapshot dataset NPZ")
    ap.add_argument("--out-dir", default="packages/inference/models/gept3", help="Output artifact directory")
    ap.add_argument("--run-name", default="", help="Optional run subdirectory name")

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--eval-max-batches", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--temporal-n-heads", type=int, default=4)
    ap.add_argument("--temporal-n-layers", type=int, default=2)
    ap.add_argument("--ff-mult", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--n-price-quantiles", type=int, default=5)

    ap.add_argument("--val-days", type=int, default=7)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--log-every-batches", type=int, default=50)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--gc-every-steps", type=int, default=0)
    ap.add_argument("--mps-empty-cache-every-steps", type=int, default=0)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=1)
    ap.add_argument("--auto-resume", action="store_true")
    ap.add_argument("--resume-from", default="", help="Optional checkpoint path to resume from")

    ap.add_argument("--enable-buy-head", action="store_true")
    ap.add_argument("--enable-price-head", action="store_true")
    ap.add_argument("--enable-regime-head", action="store_true")
    ap.add_argument("--enable-all-heads", action="store_true")

    ap.add_argument("--loss-w-sell", type=float, default=1.0)
    ap.add_argument("--loss-w-buy", type=float, default=0.35)
    ap.add_argument("--loss-w-price", type=float, default=0.20)
    ap.add_argument("--loss-w-regime", type=float, default=0.10)

    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
