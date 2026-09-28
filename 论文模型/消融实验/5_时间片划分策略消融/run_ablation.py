# -*- coding: utf-8 -*-
"""
实验五：时间片划分策略消融实验
分析时间片数量对模型性能的影响
"""

import sys
import gc
from pathlib import Path
import argparse
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ablation_config import ABLATION_SNAPSHOT
from ablation_base import StandardAblationExperiment


class SnapshotAblation(StandardAblationExperiment):
    """时间片划分策略消融实验"""
    
    def __init__(self, variant_key: str, **kwargs):
        self.variant_key = variant_key
        ablation_info = ABLATION_SNAPSHOT[variant_key]
        
        output_dir = Path(__file__).parent / 'results'
        
        super().__init__(
            experiment_name='5_时间片划分策略消融',
            variant_name=variant_key,
            config=ablation_info['config'],
            output_dir=output_dir,
            **kwargs
        )
        
        self.logger.info(f"变体描述: {ablation_info['description']}")
        self.logger.info(f"时间片数量: {ablation_info['config']['num_snapshots']}")


def run_single_experiment(variant_key: str, dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行单个消融实验"""
    experiment = SnapshotAblation(
        variant_key=variant_key,
        dataset=dataset,
        seed=seed,
        gpu=gpu
    )
    return experiment.run(num_epochs=epochs)


def run_all_experiments(dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行所有时间片划分策略消融实验"""
    results = {}

    for variant_key in ABLATION_SNAPSHOT.keys():
        print(f"\n{'='*80}")
        print(f"运行实验: {variant_key}")
        print('='*80)

        try:
            result = run_single_experiment(variant_key, dataset, seed, epochs, gpu)
            results[variant_key] = result
        except Exception as e:
            print(f"实验 {variant_key} 失败: {e}")
            import traceback
            traceback.print_exc()
            results[variant_key] = {'error': str(e)}
        finally:
            # 每个实验结束后彻底清理GPU显存，防止残留影响下一个实验
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 打印汇总结果
    print("\n" + "="*80)
    print("时间片划分策略消融实验结果汇总")
    print("="*80)
    print(f"{'变体':<25} {'时间片数':<10} {'AUC':<10} {'F1':<10} {'Recall':<10}")
    print("-"*65)

    for variant_key, result in results.items():
        num_snapshots = ABLATION_SNAPSHOT[variant_key]['config']['num_snapshots']
        if 'error' in result:
            print(f"{variant_key:<25} {num_snapshots:<10} ERROR")
        else:
            print(f"{variant_key:<25} {num_snapshots:<10} {result['auc']:.4f}    {result['f1']:.4f}    {result['recall']:.4f}")

    # 自动生成折线图
    _plot_results(results, dataset)

    return results


def _plot_results(results: dict, dataset: str):
    """实验跑完后自动调用可视化"""
    try:
        from visualization.plot_snapshot_ablation import plot_snapshot_performance

        # 将 {variant_key: metrics} 转换为 {num_snapshots: metrics}
        plot_data = {}
        for variant_key, metrics in results.items():
            if 'error' in metrics:
                continue
            num_snap = ABLATION_SNAPSHOT[variant_key]['config']['num_snapshots']
            plot_data[num_snap] = metrics

        if not plot_data:
            print("无有效结果，跳过绘图")
            return

        output_path = Path(__file__).parent / 'visualization' / f'snapshot_ablation_{dataset}.png'
        plot_snapshot_performance(plot_data, output_path, dataset=dataset)
        print(f"折线图已保存: {output_path}")
    except Exception as e:
        print(f"绘图失败（不影响实验结果）: {e}")


def main():
    parser = argparse.ArgumentParser(description='时间片划分策略消融实验')
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(ABLATION_SNAPSHOT.keys()),
                        help='指定运行的变体')
    parser.add_argument('--all', action='store_true',
                        help='运行所有变体')
    parser.add_argument('--dataset', type=str, default='yelp',
                        choices=['yelp', 'amazon'],
                        help='数据集')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID (默认: 0)')
    
    args = parser.parse_args()
    
    print(f"使用GPU: cuda:{args.gpu}")
    
    if args.all:
        run_all_experiments(args.dataset, args.seed, args.epochs, args.gpu)
    elif args.variant:
        run_single_experiment(args.variant, args.dataset, args.seed, args.epochs, args.gpu)
    else:
        print("请指定 --variant 或 --all")
        print(f"可用变体: {list(ABLATION_SNAPSHOT.keys())}")


if __name__ == '__main__':
    main()
