# -*- coding: utf-8 -*-
"""
实验四：时序注意力层消融实验
分析时序注意力层内部各组件的贡献

变体：
  wo_position_embedding  — 移除位置编码
  w_causal_mask          — 启用因果mask（只看过去）
  temporal_2heads        — 注意力头数减少到2
  last_step_only         — 用取最后活跃时间步替代时序注意力聚合
"""

import sys
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ablation_config import ABLATION_TEMPORAL_ATTENTION
from ablation_base import AblationExperimentBase, StandardAblationExperiment
from config import get_model_config_for_dataset

from models.ablation_models import HTGATFraudLastStepOnly

# 需要自定义模型的变体
VARIANT_MODEL_MAP = {
    'last_step_only': HTGATFraudLastStepOnly,
}


class TemporalAttentionAblation(AblationExperimentBase):
    """时序注意力层消融实验（统一入口，自动选择标准模型或自定义模型）"""

    def __init__(self, variant_key: str, **kwargs):
        self.variant_key = variant_key
        ablation_info = ABLATION_TEMPORAL_ATTENTION[variant_key]

        output_dir = Path(__file__).parent / 'results'

        super().__init__(
            experiment_name='4_时序注意力层消融',
            variant_name=variant_key,
            config=ablation_info['config'],
            output_dir=output_dir,
            **kwargs
        )

        self.logger.info(f"变体描述: {ablation_info['description']}")

    def create_model(self):
        model_config = get_model_config_for_dataset(self.dataset, self.config)

        if self.variant_key in VARIANT_MODEL_MAP:
            # 使用自定义模型类
            model_cls = VARIANT_MODEL_MAP[self.variant_key]
            return model_cls(model_config)
        else:
            # 标准模型，通过config开关控制消融
            from src.models.ht_gat_fraud import HTGATFraud
            return HTGATFraud(model_config)


def run_single_experiment(variant_key: str, dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行单个消融实验"""
    experiment = TemporalAttentionAblation(
        variant_key=variant_key,
        dataset=dataset,
        seed=seed,
        gpu=gpu
    )
    return experiment.run(num_epochs=epochs)


def run_all_experiments(dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行所有时序注意力层消融实验"""
    results = {}

    for variant_key in ABLATION_TEMPORAL_ATTENTION.keys():
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
    print("时序注意力层消融实验结果汇总")
    print("="*80)
    print(f"{'变体':<25} {'AUC':<10} {'AP':<10} {'F1':<10} {'Recall':<10}")
    print("-"*65)

    for variant_key, result in results.items():
        if 'error' in result:
            print(f"{variant_key:<25} ERROR")
        else:
            print(f"{variant_key:<25} "
                  f"{result['auc']:.4f}    "
                  f"{result['ap']:.4f}    "
                  f"{result['f1']:.4f}    "
                  f"{result['recall']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='时序注意力层消融实验')
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(ABLATION_TEMPORAL_ATTENTION.keys()),
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
        print(f"可用变体: {list(ABLATION_TEMPORAL_ATTENTION.keys())}")
        print("\n示例:")
        print(f"  python run_ablation.py --variant wo_position_embedding --dataset yelp --epochs 200")
        print(f"  python run_ablation.py --variant last_step_only --dataset yelp --epochs 200")
        print(f"  python run_ablation.py --all --dataset yelp --epochs 200")


if __name__ == '__main__':
    main()
