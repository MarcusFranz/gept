#!/usr/bin/env python3
"""
Inference Runner - 5-minute refresh cycle

Generates predictions for all items and writes to TimescaleDB.
Uses optimized inference order:
  1. Most volatile items first (fresher data for fast-moving items)
  2. Short-term predictions first (hours 1-4 before hours 17-24)

Model Type:
  - PatchTST quantile regression (7 horizons x 5 quantiles per item)

Optimizations:
  - Batched GPU/CPU inference across all items
  - Connection pooling for DB connections
  - COPY protocol for fast bulk writes

Run via cron every 5 minutes:
    */5 * * * * cd /path/to/gept && python run_inference.py >> logs/inference.log 2>&1
"""

import sys
import os

# Add pipeline source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'pipeline'))

from pipeline.run_patchtst_inference import main as patchtst_main


def main():
    return patchtst_main()


if __name__ == "__main__":
    sys.exit(main())
