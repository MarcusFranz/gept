#!/usr/bin/env python3
"""
Download trained models from GCS to local directory.

Usage:
    python download_models.py --bucket osrs-models --run-id 20260107_150000
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage


def download_models(bucket_name: str, run_id: str, output_dir: str, workers: int = 8):
    """Download all models from a training run."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # List all blobs in the models directory
    prefix = f'runs/{run_id}/models/'
    blobs = list(bucket.list_blobs(prefix=prefix))

    print(f"Found {len(blobs)} files to download")

    def download_blob(blob):
        # Extract relative path: runs/{run_id}/models/{item_id}/{file}
        relative_path = blob.name.replace(prefix, '')
        local_path = output_path / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_path))
        return relative_path

    start_time = time.time()
    downloaded = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(download_blob, blobs):
            downloaded += 1
            if downloaded % 100 == 0:
                print(f"  Downloaded {downloaded}/{len(blobs)} files...")

    elapsed = time.time() - start_time
    print(f"\nDownloaded {downloaded} files in {elapsed:.1f}s")

    # Count items
    item_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    print(f"Items: {len(item_dirs)}")

    # Count models
    model_count = sum(1 for f in output_path.rglob('*_model.pkl'))
    print(f"Models: {model_count}")

    return output_path


def convert_to_onnx(models_dir: str):
    """Optionally convert downloaded models to ONNX format."""
    print("\nConverting to ONNX format...")

    # Import the converter
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from convert_to_onnx import convert_all_models

    convert_all_models(models_dir)


def main():
    parser = argparse.ArgumentParser(description='Download models from GCS')
    parser.add_argument('--bucket', type=str, required=True,
                        help='GCS bucket name')
    parser.add_argument('--run-id', type=str, required=True,
                        help='Training run ID')
    parser.add_argument('--output', type=str, default='models_expanded',
                        help='Output directory (default: models_expanded)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Download parallelism')
    parser.add_argument('--convert-onnx', action='store_true',
                        help='Convert to ONNX after download')
    args = parser.parse_args()

    print(f"Downloading models from gs://{args.bucket}/runs/{args.run_id}/")
    print(f"Output directory: {args.output}")
    print("=" * 60)

    output_path = download_models(
        args.bucket,
        args.run_id,
        args.output,
        args.workers
    )

    if args.convert_onnx:
        convert_to_onnx(str(output_path))

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
