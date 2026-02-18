"""
Sell-fill probability modeling utilities (V3).

This package supports:
- Building a dataset from existing TimescaleDB price history (5m candles)
- Training a lightweight model to approximate:
    P_SELL_FILL(price, remaining_hours)
- Exporting artifacts for inference:
    - TorchScript (for Ampere Linux)
    - CoreML (for Apple runtime)
"""

