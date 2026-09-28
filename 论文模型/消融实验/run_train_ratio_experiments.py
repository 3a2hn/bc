# -*- coding: utf-8 -*-
"""
训练集比例实验
使用消融实验框架运行不同训练集比例的baseline模型
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_config import ABLATION_CORE_MODULES
from ablation_base import AblationExperimentBase
from config import get_model_config_for_dataset, SPLIT_CONFIG


class TrainRatioExperiment(AblationExperimentBase):
    """训练集比例实验"""

    def __init__(self, train_ratio: float, val_ratio: float, **kwargs):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio

        # 使用baseline配置
        ablation_info = ABLATION_CORE_MODULES['baseline']

        # 创建变体名称
        variant_name = f'baseline_train{int(train_ratio*100)}'

        super().__init__(
            experiment_name='训练集比例实验',
            variant_name=variant_name,
            config=ablation_info['config'],
            **kwargs
        )

        self.logger.info(f"训练集比例: {train_ratio*100:.0f}%")
        self.logger.info(f"验证集比例: {val_ratio*100:.0f}%")
        self.logger.info(f"测试集比例: {self.test_ratio*100:.0f}%")

    def _load_data(self):
        """重写数据加载方法，使用自定义的数据划分比例"""
        from config import get_data_dir
        from src.utils.data_loader import load_fraud_detection_data
        from src.utils.node_split import create_node_split

        data_dir = get_data_dir(self.dataset)

        self.logger.info("=" * 80)
        self.logger.info(f"实验: {self.experiment_name}")
        self.logger.info(f"变体: {self.variant_name}")
        self.logger.info(f"数据集: {self.dataset.upper()}")
        self.logger.info(f"数据目录: {data_dir}")
        self.logger.info("=" * 80)

        # 加载数据
        self.data, self.class_weights = load_fraud_detection_data(
            data_dir, dataset=self.dataset
        )
        self.data = self.data.to(self.device)
        self.class_weights = self.class_weights.to(self.device)

        # 使用自定义的划分比例
        splits = create_node_split(
            labels=self.data.y.cpu(),
            timestamps=self.data.timestamps.cpu() if hasattr(self.data, 'timestamps') else None,
            method=SPLIT_CONFIG['split_method'],
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )

        self.data.train_mask = splits['train_mask'].to(self.device)
        self.data.val_mask = splits['val_mask'].to(self.device)
        self.data.test_mask = splits['test_mask'].to(self.device)

        # 打印划分统计
        self.logger.info(f"\n数据划分统计:")
        self.logger.info(f"  训练集: {self.data.train_mask.sum().item():,} ({self.train_ratio*100:.0f}%)")
        self.logger.info(f"  验证集: {self.data.val_mask.sum().item():,} ({self.val_ratio*100:.0f}%)")
        self.logger.info(f"  测试集: {self.data.test_mask.sum().item():,} ({self.test_ratio*100:.0f}%)")

    def create_model(self):
        """创建baseline模型"""
        from src.models.ht_gat_fraud import HTGATFraud

        # 合并数据集特定配置
        model_config = get_model_config_for_dataset(self.dataset, self.config)

        return HTGATFraud(model_config)


def run_single_experiment(train_ratio: float, val_ratio: float, dataset: str,
                         seed: int, epochs: int, gpu: int = 0):
    """运行单个训练集比例实验"""
    output_dir = Path(__file__).parent / 'train_ratio_results'
    output_dir.mkdir(exist_ok=True)

    experiment = TrainRatioExperiment(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        dataset=dataset,
        seed=seed,
        gpu=gpu,
        output_dir=output_dir
    )

    return experiment.run(num_epochs=epochs)


def run_all_experiments(train_ratios, val_ratio, dataset: str, seed: int,
                       epochs: int, gpu: int = 0):
    """运行所有训练集比例实验"""
    results = {}

    for train_ratio in train_ratios:
        ratio_key = f'{int(train_ratio*100)}%'
        print(f"\n{'='*80}")
        print(f"运行实验: 训练集={ratio_key}, 验证集={int(val_ratio*100)}%")
        print('='*80)

        try:
            result = run_single_experiment(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                dataset=dataset,
                seed=seed,
                epochs=epochs,
                gpu=gpu
            )
            results[ratio_key] = result
        except Exception as e:
            print(f"实验 {ratio_key} 失败: {e}")
            import traceback
            traceback.print_exc()
            results[ratio_key] = {'error': str(e)}

    # 打印汇总结果
    print("\n" + "="*80)
    print("训练集比例实验结果汇总")
    print("="*80)
    print(f"{'训练集比例':<12} {'AUC':<10} {'AP':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*80)

    for ratio_key, result in results.items():
        if 'error' in result:
            print(f"{ratio_key:<12} ERROR")
        else:
            auc = result['auc']
            ap = result.get('ap', 0)
            f1 = result['f1']
            precision = result['precision']
            recall = result['recall']
            print(f"{ratio_key:<12} {auc:.4f}    {ap:.4f}    {f1:.4f}    "
                  f"{precision:.4f}      {recall:.4f}")

    # 保存汇总结果
    output_dir = Path(__file__).parent / 'train_ratio_results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = output_dir / f'summary_{timestamp}.json'

    summary = {
        'experiment': 'train_ratio_comparison',
        'timestamp': timestamp,
        'config': {
            'train_ratios': train_ratios,
            'val_ratio': val_ratio,
            'dataset': dataset,
            'seed': seed,
            'epochs': epochs,
            'gpu': gpu
        },
        'results': results
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n汇总结果已保存: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='训练集比例实验')
    parser.add_argument('--train_ratios', type=float, nargs='+',
                       default=[0.2, 0.4, 0.6],
                       help='训练集比例列表')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--dataset', type=str, default='yelp',
                       choices=['yelp', 'amazon'],
                       help='数据集')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU设备ID')

    args = parser.parse_args()

    print(f"使用GPU: cuda:{args.gpu}")
    print(f"训练集比例: {[f'{r*100:.0f}%' for r in args.train_ratios]}")
    print(f"验证集比例: {args.val_ratio*100:.0f}%")

    run_all_experiments(
        train_ratios=args.train_ratios,
        val_ratio=args.val_ratio,
        dataset=args.dataset,
        seed=args.seed,
        epochs=args.epochs,
        gpu=args.gpu
    )


if __name__ == '__main__':
    main()
