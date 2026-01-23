"""
PatchTST Training Pipeline
==========================

A clean, four-stage pipeline for training PatchTST models:

1. stage1_extract - Extract and clean data from PostgreSQL
2. stage2_sample - Generate valid sample index
3. stage3_precompute - Precompute features into chunks
4. stage4_train - Train the model

Usage:
    python -m pipeline.stage1_extract --items items.txt --output data/cleaned/
    python -m pipeline.stage2_sample --input data/cleaned/ --output data/samples/
    python -m pipeline.stage3_precompute --samples data/samples/samples.csv --output data/features/
    python -m pipeline.stage4_train --config configs/research.yaml
"""

__version__ = "1.0.0"
