# -*- coding: utf-8 -*-
"""
实验一：核心模块消融实验
验证HGT、TGN记忆、时序注意力三大核心模块的贡献
"""

import sys
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ablation_config import ABLATION_CORE_MODULES
from ablation_base import AblationExperimentBase
from models.ablation_models import (
    HTGATFraudNoMemory,
    HTGATFraudNoTemporalAttention,
    HTGATFraudWithGCN,
    HTGATFraudStatic
)
from config import get_model_config_for_dataset


class CoreModuleAblation(AblationExperimentBase):
    """核心模块消融实验"""
    
    def __init__(self, variant_key: str, **kwargs):
        self.variant_key = variant_key
        ablation_info = ABLATION_CORE_MODULES[variant_key]
        
        super().__init__(
            experiment_name='1_核心模块消融',
            variant_name=variant_key,
            config=ablation_info['config'],
            **kwargs
        )
        
        self.logger.info(f"变体描述: {ablation_info['description']}")
    
    def create_model(self):
        """根据变体创建对应的模型"""
        # 合并数据集特定配置
        model_config = get_model_config_for_dataset(self.dataset, self.config)
        
        if self.variant_key == 'baseline':
            from src.models.ht_gat_fraud import HTGATFraud
            return HTGATFraud(model_config)
        
        elif self.variant_key == 'wo_tgn_memory':
            return HTGATFraudNoMemory(model_config)
        
        elif self.variant_key == 'wo_temporal_attention':
            return HTGATFraudNoTemporalAttention(model_config)
        
        elif self.variant_key == 'wo_hgt_use_gcn':
            return HTGATFraudWithGCN(model_config)
        
        elif self.variant_key == 'static_baseline':
            model_config['num_snapshots'] = 1
            return HTGATFraudStatic(model_config)
        
        else:
            raise ValueError(f"未知的变体: {self.variant_key}")


def run_single_experiment(variant_key: str, dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行单个消融实验"""
    output_dir = Path(__file__).parent / 'results'
    
    experiment = CoreModuleAblation(
        variant_key=variant_key,
        dataset=dataset,
        seed=seed,
        gpu=gpu,
        output_dir=output_dir
    )
    
    return experiment.run(num_epochs=epochs)


def run_all_experiments(dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行所有核心模块消融实验"""
    results = {}
    
    for variant_key in ABLATION_CORE_MODULES.keys():
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
    
    # 打印汇总结果（包含 AP 指标）
    print("\n" + "="*80)
    print("核心模块消融实验结果汇总")
    print("="*80)
    print(f"{'变体':<25} {'AUC':<10} {'AP':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*80)
    
    for variant_key, result in results.items():
        if 'error' in result:
            print(f"{variant_key:<25} ERROR")
        else:
            auc = result['auc']
            ap = result.get('ap', 0)
            f1 = result['f1']
            precision = result['precision']
            recall = result['recall']
            print(f"{variant_key:<25} {auc:.4f}    {ap:.4f}    {f1:.4f}    {precision:.4f}      {recall:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='核心模块消融实验')
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(ABLATION_CORE_MODULES.keys()),
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
        print(f"可用变体: {list(ABLATION_CORE_MODULES.keys())}")


if __name__ == '__main__':
    main()
