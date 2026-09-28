# -*- coding: utf-8 -*-
"""
记忆权重搜索实验
搜索最优的固定记忆权重值
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ablation_base import AblationExperimentBase
from config import get_model_config_for_dataset, MODEL_CONFIG


class MemoryWeightExperiment(AblationExperimentBase):
    """记忆权重搜索实验"""
    
    def __init__(self, weight: float, **kwargs):
        self.weight = weight
        
        # 构建配置
        config = {
            **MODEL_CONFIG,
            'model_name': f'HTGATFraud-MemWeight-{weight}',
            'use_rl_memory_gate': False,
            'fixed_memory_weight': weight,
        }
        
        super().__init__(
            experiment_name='1_核心模块消融',
            variant_name=f'memory_weight_{weight}',
            config=config,
            **kwargs
        )
        
        self.logger.info(f"固定记忆权重: {weight}")
    
    def create_model(self):
        """创建完整模型（使用指定的记忆权重）"""
        from src.models.ht_gat_fraud import HTGATFraud
        model_config = get_model_config_for_dataset(self.dataset, self.config)
        return HTGATFraud(model_config)


def run_weight_search(weights: list, dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行记忆权重搜索实验"""
    output_dir = Path(__file__).parent / 'results' / 'weight_search'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for weight in weights:
        print(f"\n{'='*80}")
        print(f"运行实验: 固定记忆权重 = {weight}")
        print('='*80)
        
        try:
            experiment = MemoryWeightExperiment(
                weight=weight,
                dataset=dataset,
                seed=seed,
                gpu=gpu,
                output_dir=output_dir
            )
            result = experiment.run(num_epochs=epochs)
            results[weight] = result
        except Exception as e:
            print(f"实验 weight={weight} 失败: {e}")
            import traceback
            traceback.print_exc()
            results[weight] = {'error': str(e)}
    
    # 打印汇总结果（包含 AP）
    print("\n" + "="*80)
    print("记忆权重搜索实验结果汇总")
    print("="*80)
    print(f"{'权重':<10} {'AUC':<10} {'AP':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*70)
    
    best_weight = None
    best_auc = 0
    
    for weight, result in sorted(results.items()):
        if 'error' in result:
            print(f"{weight:<10} ERROR")
        else:
            auc = result['auc']
            ap = result.get('ap', 0)
            f1 = result['f1']
            precision = result['precision']
            recall = result['recall']
            print(f"{weight:<10} {auc:.4f}    {ap:.4f}    {f1:.4f}    {precision:.4f}      {recall:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_weight = weight
    
    print("-"*70)
    print(f"最佳权重: {best_weight} (AUC: {best_auc:.4f})")
    
    # 保存结果到 JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'weight_search_results_{timestamp}.json'
    
    # 转换结果（处理不可序列化的值）
    serializable_results = {}
    for weight, result in results.items():
        serializable_results[str(weight)] = result
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'weights': weights,
            'results': serializable_results,
            'best_weight': best_weight,
            'best_auc': best_auc,
            'dataset': dataset,
            'seed': seed,
            'epochs': epochs,
            'timestamp': timestamp,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {results_file}")
    
    return results, best_weight


def main():
    parser = argparse.ArgumentParser(description='记忆权重搜索实验')
    parser.add_argument('--weights', type=float, nargs='+', 
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        help='要搜索的权重值列表')
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
    print(f"搜索权重: {args.weights}")
    
    run_weight_search(args.weights, args.dataset, args.seed, args.epochs, args.gpu)


if __name__ == '__main__':
    main()
