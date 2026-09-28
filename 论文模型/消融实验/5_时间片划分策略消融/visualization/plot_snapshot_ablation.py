# -*- coding: utf-8 -*-
"""
实验五：时间片划分策略消融 — 可视化

生成折线图：X轴为时间片数量(4/8/12/16)，Y轴为分类性能指标
双Y轴设计：左轴 AUC/AP/F1，右轴 Recall

两种数据来源：
  1. 从 results/ 目录自动读取已有的实验结果JSON
  2. 手动传入结果字典（用于 run_ablation.py 跑完后直接调用）

用法：
  python plot_snapshot_ablation.py                          # 从results/自动读取
  python plot_snapshot_ablation.py --results_dir ./results  # 指定目录
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 样式配置
# ============================================================================

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

METRIC_STYLES = {
    'auc':       {'color': '#e74c3c', 'marker': 'o', 'label': 'AUC'},
    'ap':        {'color': '#3498db', 'marker': 's', 'label': 'AP'},
    'f1':        {'color': '#2ecc71', 'marker': '^', 'label': 'F1'},
    'recall':    {'color': '#f39c12', 'marker': 'D', 'label': 'Recall'},
}


def load_results_from_dir(results_dir: Path) -> dict:
    """
    从results目录读取各变体的最新结果JSON

    Returns
    -------
    dict: {num_snapshots: {metric: value, ...}, ...}
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        logger.error(f"结果目录不存在: {results_dir}")
        return {}

    # 变体名 → num_snapshots 映射
    variant_to_snapshots = {
        'snapshot_4': 4,
        'snapshot_8': 8,
        'snapshot_12': 12,
        'snapshot_16': 16,
    }

    results = {}

    for variant_name, num_snap in variant_to_snapshots.items():
        # 找该变体的所有结果文件，取最新的
        pattern = f'{variant_name}_results_*.json'
        files = sorted(results_dir.glob(pattern))

        if not files:
            logger.warning(f"未找到 {variant_name} 的结果文件")
            continue

        latest_file = files[-1]  # 按时间戳排序，取最新
        logger.info(f"读取 {variant_name}: {latest_file.name}")

        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metrics = data.get('final_metrics', {})
        results[num_snap] = {
            'auc': metrics.get('auc', 0),
            'ap': metrics.get('ap', 0),
            'f1': metrics.get('f1', 0),
            'recall': metrics.get('recall', 0),
            'precision': metrics.get('precision', 0),
            'accuracy': metrics.get('accuracy', 0),
        }

    return results


def plot_snapshot_performance(results: dict, output_path: Path, dataset: str = ''):
    """
    绘制时间片数量 vs 分类性能的折线图

    Parameters
    ----------
    results : dict
        {num_snapshots: {metric: value, ...}, ...}
    output_path : Path
        输出图片路径
    dataset : str
        数据集名称（用于标题）
    """
    if not results:
        logger.error("无结果数据，无法绘图")
        return

    snapshots = sorted(results.keys())
    x = np.array(snapshots)

    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    # 左Y轴：AUC / AP / F1
    left_metrics = ['auc', 'ap', 'f1']
    for metric in left_metrics:
        style = METRIC_STYLES[metric]
        y = [results[s][metric] for s in snapshots]
        ax1.plot(x, y,
                 color=style['color'], marker=style['marker'],
                 linewidth=2, markersize=8, label=style['label'])
        # 标注数值
        for xi, yi in zip(x, y):
            ax1.annotate(f'{yi:.4f}', (xi, yi),
                         textcoords='offset points', xytext=(0, 10),
                         ha='center', fontsize=8, color=style['color'])

    ax1.set_xlabel('Number of Snapshots', fontsize=13)
    ax1.set_ylabel('AUC / AP / F1', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in snapshots], fontsize=11)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.grid(True, alpha=0.3)

    # 右Y轴：Recall
    ax2 = ax1.twinx()
    style = METRIC_STYLES['recall']
    y_recall = [results[s]['recall'] for s in snapshots]
    ax2.plot(x, y_recall,
             color=style['color'], marker=style['marker'],
             linewidth=2, markersize=8, linestyle='--', label=style['label'])
    for xi, yi in zip(x, y_recall):
        ax2.annotate(f'{yi:.4f}', (xi, yi),
                     textcoords='offset points', xytext=(0, -14),
                     ha='center', fontsize=8, color=style['color'])

    ax2.set_ylabel('Recall', fontsize=13, color=style['color'])
    ax2.tick_params(axis='y', labelcolor=style['color'], labelsize=11)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower right', fontsize=11, framealpha=0.9)

    # 标记默认配置
    if 8 in snapshots:
        ax1.axvline(x=8, color='gray', linestyle=':', alpha=0.5)
        ax1.text(8, ax1.get_ylim()[1], ' default', va='top', ha='left',
                 fontsize=9, color='gray', style='italic')

    title = 'Snapshot Granularity vs Classification Performance'
    if dataset:
        title += f' ({dataset.upper()})'
    fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"图表已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='时间片划分策略消融 — 可视化')
    parser.add_argument('--results_dir', type=str,
                        default=str(Path(__file__).parent.parent / 'results'),
                        help='结果JSON所在目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图片路径')
    parser.add_argument('--dataset', type=str, default='yelp',
                        help='数据集名称（用于标题）')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results_from_dir(results_dir)

    if not results:
        logger.error("未读取到任何结果，请先运行消融实验")
        return

    # 打印汇总
    print(f"\n{'Snapshots':<12} {'AUC':<10} {'AP':<10} {'F1':<10} {'Recall':<10}")
    print("-" * 52)
    for s in sorted(results.keys()):
        r = results[s]
        print(f"{s:<12} {r['auc']:.4f}    {r['ap']:.4f}    {r['f1']:.4f}    {r['recall']:.4f}")

    output_path = args.output or str(
        Path(__file__).parent / f'snapshot_ablation_{args.dataset}.png'
    )
    plot_snapshot_performance(results, output_path, dataset=args.dataset)


if __name__ == '__main__':
    main()
