"""Fill probability modeling utilities (V3).

This package is intentionally small and pragmatic:
- Train a sell-fill probability model from existing TimescaleDB price history.
- Run inference on Ampere (CPU) using PyTorch.

The engine will consume these outputs as `P_SELL_FILL(price, remaining_hours)`.
"""

