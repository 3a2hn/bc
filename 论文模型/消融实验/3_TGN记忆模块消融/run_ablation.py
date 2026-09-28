# -*- coding: utf-8 -*-
"""
实验三：TGN记忆模块消融实验
分析TGN记忆模块流水线中4个关键架构决策的贡献

变体：
  wo_gated_fusion      — 门控融合 → 简单加法融合
  wo_memory_filling    — 不用记忆填充不活跃时间片
  message_from_memory  — 消息基于记忆而非HGT嵌入
  wo_time_encoding     — 消息函数去掉时间编码

数据集划分：train:val:test = 5:1:4
不包含baseline（已有结果，无需重跑）
"""

import sys
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ablation_config import ABLATION_TGN_MEMORY
from ablation_base import AblationExperimentBase
from config import get_model_config_for_dataset

from models.ablation_models import (
    HTGATFraudSimpleFusion,
    HTGATFraudNoMemoryFilling,
    HTGATFraudMessageFromMemory,
    HTGATFraudNoTimeEncoding,
)

# 变体 → 模型类的映射
VARIANT_MODEL_MAP = {
    'wo_gated_fusion':     HTGATFraudSimpleFusion,
    'wo_memory_filling':   HTGATFraudNoMemoryFilling,
    'message_from_memory': HTGATFraudMessageFromMemory,
    'wo_time_encoding':    HTGATFraudNoTimeEncoding,
}


class TGNMemoryAblation(AblationExperimentBase):
    """TGN记忆模块消融实验（使用自定义模型类）"""

    # 覆盖划分比例：5:1:4
    SPLIT_OVERRIDE = {
        'train_ratio': 0.5,
        'val_ratio':   0.1,
        'test_ratio':  0.4,
    }

    def __init__(self, variant_key: str, **kwargs):
        self.variant_key = variant_key
        ablation_info = ABLATION_TGN_MEMORY[variant_key]

        output_dir = Path(__file__).parent / 'results'

        super().__init__(
            experiment_name='3_TGN记忆模块消融',
            variant_name=variant_key,
            config=ablation_info['config'],
            output_dir=output_dir,
            **kwargs
        )

        self.logger.info(f"变体描述: {ablation_info['description']}")

    def _load_data(self):
        """覆盖父类的数据加载，强制使用 5:1:4 划分"""
        from config import get_data_dir, SPLIT_CONFIG
        from src.utils.data_loader import load_fraud_detection_data
        from src.utils.node_split import create_node_split

        data_dir = get_data_dir(self.dataset)

        self.logger.info("=" * 80)
        self.logger.info(f"消融实验: {self.experiment_name}")
        self.logger.info(f"变体: {self.variant_name}")
        self.logger.info(f"数据集: {self.dataset.upper()}")
        self.logger.info(f"数据目录: {data_dir}")
        self.logger.info(f"划分比例: train={self.SPLIT_OVERRIDE['train_ratio']}, "
                         f"val={self.SPLIT_OVERRIDE['val_ratio']}, "
                         f"test={self.SPLIT_OVERRIDE['test_ratio']}")
        self.logger.info("=" * 80)

        self.data, self.class_weights = load_fraud_detection_data(
            data_dir, dataset=self.dataset
        )
        self.data = self.data.to(self.device)
        self.class_weights = self.class_weights.to(self.device)

        splits = create_node_split(
            labels=self.data.y.cpu(),
            timestamps=self.data.timestamps.cpu() if hasattr(self.data, 'timestamps') else None,
            method=SPLIT_CONFIG['split_method'],
            train_ratio=self.SPLIT_OVERRIDE['train_ratio'],
            val_ratio=self.SPLIT_OVERRIDE['val_ratio'],
            test_ratio=self.SPLIT_OVERRIDE['test_ratio'],
            seed=self.seed,
        )

        self.data.train_mask = splits['train_mask'].to(self.device)
        self.data.val_mask = splits['val_mask'].to(self.device)
        self.data.test_mask = splits['test_mask'].to(self.device)

        self.logger.info(f"训练集: {self.data.train_mask.sum().item():,}")
        self.logger.info(f"验证集: {self.data.val_mask.sum().item():,}")
        self.logger.info(f"测试集: {self.data.test_mask.sum().item():,}")

    def create_model(self):
        """根据variant_key创建对应的消融模型"""
        model_cls = VARIANT_MODEL_MAP[self.variant_key]
        model_config = get_model_config_for_dataset(self.dataset, self.config)
        return model_cls(model_config)


def run_single_experiment(variant_key: str, dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行单个消融实验"""
    experiment = TGNMemoryAblation(
        variant_key=variant_key,
        dataset=dataset,
        seed=seed,
        gpu=gpu,
    )
    return experiment.run(num_epochs=epochs)


def run_all_experiments(dataset: str, seed: int, epochs: int, gpu: int = 0):
    """运行所有TGN记忆模块消融实验"""
    results = {}

    for variant_key in ABLATION_TGN_MEMORY.keys():
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
    print("TGN记忆模块消融实验结果汇总")
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
    parser = argparse.ArgumentParser(description='TGN记忆模块消融实验')
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(ABLATION_TGN_MEMORY.keys()),
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
    print(f"数据集划分: train:val:test = 5:1:4")

    if args.all:
        run_all_experiments(args.dataset, args.seed, args.epochs, args.gpu)
    elif args.variant:
        run_single_experiment(args.variant, args.dataset, args.seed, args.epochs, args.gpu)
    else:
        print("请指定 --variant 或 --all")
        print(f"可用变体: {list(ABLATION_TGN_MEMORY.keys())}")
        print("\n示例:")
        print(f"  python run_ablation.py --variant wo_gated_fusion --dataset yelp --epochs 200")
        print(f"  python run_ablation.py --all --dataset yelp --epochs 200")


if __name__ == '__main__':
    main()
