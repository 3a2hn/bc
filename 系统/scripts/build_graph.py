#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建图包装脚本
接收 --input_csv 和 --output_dir 参数，调用 新建图方法_Amazon 的建图流程
"""

import sys
import os
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Graph building wrapper')
    parser.add_argument('--input_csv', required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', required=True, help='Output directory for graph data')
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        print(f"Error: Input CSV not found: {input_csv}", file=sys.stderr)
        sys.exit(1)

    # Locate the graph building module
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent  # scripts/ -> 系统/ -> project root
    graph_module_dir = project_root / '新建图方法_Amazon'

    if not graph_module_dir.exists():
        print(f"Error: Graph module directory not found: {graph_module_dir}", file=sys.stderr)
        sys.exit(1)

    # Add to sys.path
    sys.path.insert(0, str(graph_module_dir))

    # Import and monkey-patch config_amazon
    import config_amazon as config

    # Patch paths to use our input/output
    config.INPUT_CSV = input_csv
    config.OUTPUT_DIR = output_dir
    config.GRAPH_OUTPUT = output_dir / 'graph.pkl'
    config.EDGES_OUTPUT_DIR = output_dir / 'edges'
    config.EDGES_OUTPUT_DIR.mkdir(exist_ok=True)
    config.FEATURES_OUTPUT = output_dir / 'features.npy'
    config.FEATURES_STATS_OUTPUT = output_dir / 'feature_statistics.json'
    config.LABELS_OUTPUT = output_dir / 'labels.npy'
    config.SPLITS_OUTPUT = output_dir / 'splits.pkl'
    config.MAPPINGS_OUTPUT = output_dir / 'id_mappings.pkl'
    config.STATISTICS_OUTPUT = output_dir / 'graph_statistics.json'

    # Patch log directory
    config.LOGS_DIR = output_dir / 'logs'
    config.LOGS_DIR.mkdir(exist_ok=True)
    config.LOG_FILE = config.LOGS_DIR / 'build_graph.log'

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Graph module: {graph_module_dir}")
    sys.stdout.flush()

    # Import and run main
    from main_amazon import main as build_main
    build_main()

    print("Graph building completed successfully.")


if __name__ == '__main__':
    main()
