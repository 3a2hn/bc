# -*- coding: utf-8 -*-
"""
实验二：HGT层组件消融实验
分析HGT层内部各组件的贡献
"""

import sys
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ablation_config import ABLATION_HGT_LAYER
from ablation_base import StandardAblationExperiment
from config import get_model_config_for_dataset


class HGTLayerAblation(StandardAblationExperiment):
    """HGT层组件消融实验"""
    
    def __init__(self, variant_key: str, **kwargs):
        self.variant_key = variant_key
        ablation_info = ABLATION_HGT_LAYER[variant_key]
        
        # 设置输出目录
        output_dir = Path(__file__).parent / 'results'
        
        super().__init__(
            experiment_name='2_HGT层组件消融',
            variant_name=variant_key,
            config=ablation_info['config'],
            output_dir=output_dir,
            **kwargs
        )
        
        self.logger.info(f"变体描述: {ablation_info['description']}")


def run_single_experiment(variant_key: str, dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行单个消融实验"""
    experiment = HGTLayerAblation(
        variant_key=variant_key,
        dataset=dataset,
        seed=seed,
        gpu=gpu
    )
    return experiment.run(num_epochs=epochs)


def run_all_experiments(dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行所有HGT层消融实验"""
    results = {}
    
    for variant_key in ABLATION_HGT_LAYER.keys():
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
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("HGT层组件消融实验结果汇总")
    print("="*80)
    print(f"{'变体':<25} {'AUC':<10} {'F1':<10} {'Recall':<10}")
    print("-"*55)
    
    for variant_key, result in results.items():
        if 'error' in result:
            print(f"{variant_key:<25} ERROR")
        else:
            print(f"{variant_key:<25} {result['auc']:.4f}    {result['f1']:.4f}    {result['recall']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='HGT层组件消融实验')
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(ABLATION_HGT_LAYER.keys()),
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
        print(f"可用变体: {list(ABLATION_HGT_LAYER.keys())}")


if __name__ == '__main__':
    main()
