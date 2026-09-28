#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练包装脚本
接收 --data_dir, --output_dir, --epochs 等参数，调用异构图欺诈检测模型的训练流程
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Training wrapper')
    parser.add_argument('--data_dir', required=True, help='Directory containing graph.pkl, features.npy, labels.npy')
    parser.add_argument('--output_dir', required=True, help='Output directory for model, logs, results')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--dataset', type=str, default='amazon', choices=['yelp', 'amazon'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Create output subdirectories
    checkpoints_dir = output_dir / 'checkpoints'
    logs_dir = output_dir / 'logs'
    results_dir = output_dir / 'results'
    for d in [checkpoints_dir, logs_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Verify data files exist
    required_files = ['graph.pkl', 'features.npy', 'labels.npy']
    for f in required_files:
        if not (data_dir / f).exists():
            print(f"Error: Required file not found: {data_dir / f}", file=sys.stderr)
            sys.exit(1)

    # Locate the training module
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent  # scripts/ -> 系统/ -> project root
    train_module_dir = project_root / '异构图欺诈检测模型'

    if not train_module_dir.exists():
        print(f"Error: Training module directory not found: {train_module_dir}", file=sys.stderr)
        sys.exit(1)

    # Add to sys.path
    sys.path.insert(0, str(train_module_dir))

    # Import and patch config
    import config

    # Patch dataset paths to point to our data_dir
    config.DATASET_PATHS[args.dataset] = data_dir

    # Patch output directories
    config.CHECKPOINTS_DIR = checkpoints_dir
    config.LOGS_DIR = logs_dir
    config.RESULTS_DIR = results_dir

    # Recreate dirs (config.py creates them on import)
    for d in [checkpoints_dir, logs_dir, results_dir]:
        d.mkdir(exist_ok=True)

    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    sys.stdout.flush()

    # Build sys.argv for the training script's argparse
    train_args = [
        'train.py',
        '--dataset', args.dataset,
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--device', args.device,
        '--save_checkpoints',
        '--eval_test',
        '--save_plots',
    ]
    sys.argv = train_args

    # Import and run training
    from train import main as train_main
    train_main()

    # After training, collect results into a predictions.json for the web system
    print("Training completed. Collecting results...")
    sys.stdout.flush()

    # Find the test results file
    result_files = sorted(logs_dir.glob('test_results_*.json'))
    if result_files:
        latest_result = result_files[-1]
        # Copy to results dir as well
        import shutil
        dest = results_dir / latest_result.name
        if not dest.exists():
            shutil.copy2(latest_result, dest)
        print(f"Test results: {latest_result}")

    # Find best model
    model_files = sorted(checkpoints_dir.glob('best_model_*.pth'))
    if model_files:
        print(f"Best model: {model_files[-1]}")

    print("Training pipeline completed successfully.")


if __name__ == '__main__':
    main()
