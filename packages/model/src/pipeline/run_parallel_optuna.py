#!/usr/bin/env python3
"""
Parallel Optuna Tuning Across Multiple GPUs
============================================

Spawns multiple Optuna workers, each on a different GPU.
All workers share the same study via SQLite database.

Usage:
    python -m src.pipeline.run_parallel_optuna \
        --config configs/rtx3090_1000items.yaml \
        --n_gpus 8 \
        --trials_per_gpu 4 \
        --study_name patchtst_tuning

Volume-aware tuning (adds volume prediction head):
    python -m src.pipeline.run_parallel_optuna \
        --config configs/rtx3090_1000items.yaml \
        --n_gpus 8 \
        --trials_per_gpu 4 \
        --study_name patchtst_volume_tuning \
        --enable-volume

This will run 8 workers (one per GPU), each doing 4 trials = 32 total trials.
"""

import argparse
import subprocess
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - GPU%(gpu)s - %(message)s'
)


def run_worker(gpu_id: int, config_path: str, n_trials: int, study_name: str,
               enable_volume: bool, output_queue: Queue):
    """Run a single Optuna worker on a specific GPU."""

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log_file = f'/workspace/optuna_gpu{gpu_id}.log'

    # Select module based on volume flag
    module = 'src.pipeline.optuna_tune_volume' if enable_volume else 'src.pipeline.optuna_tune'

    cmd = [
        sys.executable, '-m', module,
        '--config', config_path,
        '--n_trials', str(n_trials),
        '--study_name', study_name,
    ]

    mode_str = " [VOLUME]" if enable_volume else ""
    print(f"[GPU {gpu_id}]{mode_str} Starting worker: {' '.join(cmd)}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd='/workspace'
        )

    process.wait()
    output_queue.put((gpu_id, process.returncode))
    print(f"[GPU {gpu_id}] Worker finished with code {process.returncode}")


def main():
    parser = argparse.ArgumentParser(description='Run parallel Optuna tuning')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--n_gpus', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--trials_per_gpu', type=int, default=4,
                        help='Trials per GPU worker')
    parser.add_argument('--study_name', default='patchtst_tuning',
                        help='Optuna study name')
    parser.add_argument('--enable-volume', action='store_true',
                        help='Enable volume prediction head (uses optuna_tune_volume)')
    args = parser.parse_args()

    # Verify GPUs available
    import torch
    available_gpus = torch.cuda.device_count()
    if available_gpus < args.n_gpus:
        print(f"Warning: Requested {args.n_gpus} GPUs but only {available_gpus} available")
        args.n_gpus = available_gpus

    volume_str = "YES (two-stage calibration)" if args.enable_volume else "NO"
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  PARALLEL OPTUNA TUNING                                      ║
╠══════════════════════════════════════════════════════════════╣
║  GPUs:          {args.n_gpus:3d}                                         ║
║  Trials/GPU:    {args.trials_per_gpu:3d}                                         ║
║  Total trials:  {args.n_gpus * args.trials_per_gpu:3d}                                         ║
║  Study:         {args.study_name:<43} ║
║  Config:        {args.config:<43} ║
║  Volume Head:   {volume_str:<43} ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Queue for collecting results
    result_queue = Queue()

    # Start workers
    processes = []
    for gpu_id in range(args.n_gpus):
        p = Process(
            target=run_worker,
            args=(gpu_id, args.config, args.trials_per_gpu, args.study_name,
                  args.enable_volume, result_queue)
        )
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger starts to avoid DB contention

    print(f"\nStarted {len(processes)} workers. Logs at /workspace/optuna_gpu*.log")
    print("Monitor with: tail -f /workspace/optuna_gpu*.log")
    print("\nWaiting for all workers to complete...\n")

    # Wait for all workers
    for p in processes:
        p.join()

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Print summary
    print("\n" + "="*60)
    print("PARALLEL TUNING COMPLETE")
    print("="*60)

    success = sum(1 for _, code in results if code == 0)
    failed = sum(1 for _, code in results if code != 0)

    print(f"Successful workers: {success}/{args.n_gpus}")
    if failed > 0:
        print(f"Failed workers: {failed}")
        for gpu_id, code in results:
            if code != 0:
                print(f"  GPU {gpu_id}: exit code {code}")

    # Load and print best result
    try:
        import optuna
        storage = f'sqlite:////workspace/optuna_{args.study_name}.db'
        study = optuna.load_study(study_name=args.study_name, storage=storage)

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best val_loss: {study.best_trial.value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Could not load study results: {e}")


if __name__ == '__main__':
    main()
