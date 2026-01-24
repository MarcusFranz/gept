"""
Train Trade Timing Optimizer
============================

Trains the timing layer that decides best buy/sell windows.

IMPORTANT: Buy must fill BEFORE sell can start!
- Buy window: 0 to buy_horizon
- Sell window: buy_horizon to buy_horizon + sell_duration
"""

import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple

from src.pipeline.model import PatchTSTModel
from src.pipeline.config import ModelConfig
from src.pipeline.per_item_head import PerItemHead, PatchTSTWithItemHead, extract_item_samples
from src.pipeline.trade_optimizer import (
    TradeTimingHead,
    TradeOptimizerLoss,
    interpret_recommendation,
    HORIZONS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_prediction_model(base_path: str, head_path: str, device: torch.device):
    """Load base model and per-item head."""
    config = ModelConfig(n_items=1000)
    base = PatchTSTModel(config)
    state_dict = torch.load(base_path, map_location=device, weights_only=True)
    base.load_state_dict(state_dict)
    base.to(device)

    if head_path:
        head = PerItemHead(d_model=config.d_model, hidden_dim=64)
        head_state = torch.load(head_path, map_location=device, weights_only=True)
        head.load_state_dict(head_state)
        head.to(device)
        return PatchTSTWithItemHead(base, head, freeze_base=True)
    return base


def generate_predictions(model, data: Dict, device: torch.device, batch_size: int = 256):
    """Generate predictions for all samples."""
    n_samples = len(data['recent'])
    all_high = []
    all_low = []

    model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            recent = torch.tensor(data['recent'][i:end], dtype=torch.float32, device=device)
            medium = torch.tensor(data['medium'][i:end], dtype=torch.float32, device=device)
            long = torch.tensor(data['long'][i:end], dtype=torch.float32, device=device)
            item_ids = torch.tensor(data['item_ids'][i:end], dtype=torch.long, device=device)

            out = model(recent, medium, long, item_ids)
            all_high.append(out['high_quantiles'].cpu())
            all_low.append(out['low_quantiles'].cpu())

    return torch.cat(all_high), torch.cat(all_low)


def compute_sequential_labels(targets: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute optimal timing labels with SEQUENTIAL constraint.

    Buy must complete before sell can start:
    - Buy fills within buy_horizon
    - Sell starts AFTER buy fills, takes sell_duration more hours
    - Total time = buy_horizon + sell_duration

    We look for the best (buy_horizon, sell_total_horizon) pair where
    sell_total_horizon > buy_horizon.
    """
    n_samples = len(targets)
    n_horizons = len(HORIZONS)

    buy_labels = np.zeros(n_samples, dtype=np.int64)
    sell_labels = np.zeros(n_samples, dtype=np.int64)
    margin_labels = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        high_actual = targets[i, :, 0]  # max high at each horizon
        low_actual = targets[i, :, 1]   # min low at each horizon

        best_margin = -np.inf

        # buy_idx: when buy completes (0-6 maps to 1h, 2h, 4h, 8h, 12h, 24h, 48h)
        for buy_idx in range(n_horizons):
            buy_price = 1.0 + low_actual[buy_idx]  # buy at min low up to this point

            # sell_idx: when sell completes (must be AFTER buy)
            for sell_idx in range(buy_idx + 1, n_horizons):
                sell_price = 1.0 + high_actual[sell_idx]  # sell at max high by this point

                margin = (sell_price - buy_price) / buy_price

                # Time adjustment - total time is sell horizon
                total_hours = HORIZONS[sell_idx]
                adjusted = margin / np.sqrt(total_hours)

                if adjusted > best_margin:
                    best_margin = adjusted
                    buy_labels[i] = buy_idx
                    sell_labels[i] = sell_idx
                    margin_labels[i] = margin

    return {
        'buy_labels': buy_labels,
        'sell_labels': sell_labels,
        'margin_labels': margin_labels,
    }


def compute_sequential_timing(
    high_quantiles: np.ndarray,
    low_quantiles: np.ndarray,
    quantile_idx: int = 2,
) -> Dict:
    """
    Compute optimal timing with sequential constraint.
    Sell horizon must be strictly AFTER buy horizon.
    """
    n_horizons = len(HORIZONS)
    high_prices = high_quantiles[:, quantile_idx]
    low_prices = low_quantiles[:, quantile_idx]

    best_margin = -np.inf
    best_buy_idx = 0
    best_sell_idx = 1

    for buy_idx in range(n_horizons - 1):  # can't buy at last horizon
        for sell_idx in range(buy_idx + 1, n_horizons):  # sell must be AFTER buy
            buy_price = 1.0 + low_prices[buy_idx]
            sell_price = 1.0 + high_prices[sell_idx]
            margin = (sell_price - buy_price) / buy_price

            total_hours = HORIZONS[sell_idx]
            adjusted = margin / np.sqrt(total_hours)

            if adjusted > best_margin:
                best_margin = adjusted
                best_buy_idx = buy_idx
                best_sell_idx = sell_idx

    raw_margin = (
        (1.0 + high_prices[best_sell_idx]) - (1.0 + low_prices[best_buy_idx])
    ) / (1.0 + low_prices[best_buy_idx])

    return {
        'buy_horizon_idx': best_buy_idx,
        'sell_horizon_idx': best_sell_idx,
        'buy_horizon_hours': HORIZONS[best_buy_idx],
        'sell_horizon_hours': HORIZONS[best_sell_idx],
        'expected_margin': raw_margin,
        'buy_price_pct': low_prices[best_buy_idx],
        'sell_price_pct': high_prices[best_sell_idx],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--item-head', default=None)
    parser.add_argument('--data', required=True)
    parser.add_argument('--item-idx', type=int, default=0)
    parser.add_argument('--output', default='/tmp/trade_optimizer.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load prediction model
    logger.info("Loading prediction model...")
    pred_model = load_prediction_model(args.base_model, args.item_head, device)

    # Load and filter data
    logger.info(f"Loading data from {args.data}")
    data = np.load(args.data)
    item_data = extract_item_samples(dict(data), args.item_idx)
    n_samples = len(item_data['recent'])
    logger.info(f"Samples for item {args.item_idx}: {n_samples}")

    # Generate predictions
    logger.info("Generating predictions from price model...")
    high_preds, low_preds = generate_predictions(pred_model, item_data, device)

    # Compute training labels with SEQUENTIAL constraint
    logger.info("Computing optimal timing labels (buy before sell)...")
    labels = compute_sequential_labels(item_data['targets'])

    # Log label distribution
    buy_counts = np.bincount(labels['buy_labels'], minlength=7)
    sell_counts = np.bincount(labels['sell_labels'], minlength=7)
    logger.info(f"Optimal buy horizons:  {dict(zip(HORIZONS, buy_counts))}")
    logger.info(f"Optimal sell horizons: {dict(zip(HORIZONS, sell_counts))}")
    logger.info(f"Avg actual margin: {labels['margin_labels'].mean()*100:.2f}%")

    # Split data
    n_val = int(n_samples * 0.2)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            high_preds[idx],
            low_preds[idx],
            torch.tensor(labels['buy_labels'][idx], dtype=torch.long),
            torch.tensor(labels['sell_labels'][idx], dtype=torch.long),
            torch.tensor(labels['margin_labels'][idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)

    # Create optimizer model
    optimizer_model = TradeTimingHead(hidden_dim=64, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in optimizer_model.parameters())
    logger.info(f"Trade optimizer parameters: {n_params:,}")

    # Training
    optim = torch.optim.AdamW(optimizer_model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)
    loss_fn = TradeOptimizerLoss(margin_weight=0.5)

    best_val_loss = float('inf')
    best_state = None

    logger.info("Training trade optimizer...")
    for epoch in range(args.epochs):
        optimizer_model.train()
        train_loss = 0
        for high, low, buy_lbl, sell_lbl, margin_lbl in train_loader:
            high, low = high.to(device), low.to(device)
            buy_lbl, sell_lbl = buy_lbl.to(device), sell_lbl.to(device)
            margin_lbl = margin_lbl.to(device)

            optim.zero_grad()
            out = optimizer_model(high, low)
            loss = loss_fn(out, buy_lbl, sell_lbl, margin_lbl)
            loss.backward()
            optim.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        optimizer_model.eval()
        val_loss = 0
        buy_correct = sell_correct = total = 0

        with torch.no_grad():
            for high, low, buy_lbl, sell_lbl, margin_lbl in val_loader:
                high, low = high.to(device), low.to(device)
                buy_lbl, sell_lbl = buy_lbl.to(device), sell_lbl.to(device)
                margin_lbl = margin_lbl.to(device)

                out = optimizer_model(high, low)
                loss = loss_fn(out, buy_lbl, sell_lbl, margin_lbl)
                val_loss += loss.item()

                buy_correct += (out['buy_logits'].argmax(-1) == buy_lbl).sum().item()
                sell_correct += (out['sell_logits'].argmax(-1) == sell_lbl).sum().item()
                total += len(buy_lbl)

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = optimizer_model.state_dict()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Buy Acc: {buy_correct/total:.1%} | Sell Acc: {sell_correct/total:.1%}"
            )

    optimizer_model.load_state_dict(best_state)
    torch.save(best_state, args.output)
    logger.info(f"\nSaved trade optimizer to {args.output}")

    # Demo
    logger.info("\n" + "="*60)
    logger.info("DEMO: Sample 100")
    logger.info("="*60)

    sample_idx = 100
    high_np = high_preds[sample_idx].numpy()
    low_np = low_preds[sample_idx].numpy()

    timing = compute_sequential_timing(high_np, low_np)
    logger.info(f"\nPredicted optimal timing:")
    logger.info(f"  Place BUY offer, expect fill within: {timing['buy_horizon_hours']}h")
    logger.info(f"  Then place SELL offer, expect fill by: {timing['sell_horizon_hours']}h total")
    logger.info(f"  Expected margin: {timing['expected_margin']*100:.1f}%")

    actual_buy = labels['buy_labels'][sample_idx]
    actual_sell = labels['sell_labels'][sample_idx]
    actual_margin = labels['margin_labels'][sample_idx]
    logger.info(f"\nActual optimal (hindsight):")
    logger.info(f"  Buy filled within: {HORIZONS[actual_buy]}h")
    logger.info(f"  Sell filled by: {HORIZONS[actual_sell]}h total")
    logger.info(f"  Actual margin: {actual_margin*100:.1f}%")

    # Pretty output
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  CANNONBALL TRADE RECOMMENDATION                             ║
╠══════════════════════════════════════════════════════════════╣
║  Current Price: ~180 GP                                      ║
╠══════════════════════════════════════════════════════════════╣
║  STEP 1: Place BUY offer                                     ║
║  → Expect fill within {timing['buy_horizon_hours']:>2}h                                   ║
║  → Target: {180*(1+timing['buy_price_pct']):>3.0f} GP ({timing['buy_price_pct']*100:>+5.1f}% from current)           ║
╠══════════════════════════════════════════════════════════════╣
║  STEP 2: After buy fills, place SELL offer                   ║
║  → Expect fill by {timing['sell_horizon_hours']:>2}h from now                            ║
║  → Target: {180*(1+timing['sell_price_pct']):>3.0f} GP ({timing['sell_price_pct']*100:>+5.1f}% from current)           ║
╠══════════════════════════════════════════════════════════════╣
║  EXPECTED PROFIT: {timing['expected_margin']*100:>5.1f}%                                    ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
