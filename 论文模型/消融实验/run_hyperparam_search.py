# -*- coding: utf-8 -*-
"""
超参数搜索实验
使用40%训练比例快速搜索最优超参数组合，然后用最优配置跑完整实验

搜索策略：分阶段搜索，避免组合爆炸
- Phase 1: 搜索损失函数相关参数 (focal_gamma, label_smoothing)
- Phase 2: 搜索正则化参数 (classifier_dropout, weight_decay)
- Phase 3: 搜索模型容量参数 (post_hgt_dim, classifier_hidden_dim)
- Phase 4: 搜索学习率参数 (learning_rate, warmup_epochs)
每阶段固定其他参数，只搜索当前阶段的参数
"""

import sys
from pathlib import Path
import argparse
import json
import itertools
from datetime import datetime
from copy import deepcopy

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_config import ABLATION_CORE_MODULES
from ablation_base import AblationExperimentBase
from config import get_model_config_for_dataset, SPLIT_CONFIG, TRAIN_CONFIG


class HyperparamSearchExperiment(AblationExperimentBase):
    """超参数搜索实验"""

    def __init__(self, train_ratio: float, val_ratio: float,
                 config_overrides: dict = None, variant_name: str = 'hp_search',
                 **kwargs):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio

        # 使用baseline配置作为基础
        base_config = deepcopy(ABLATION_CORE_MODULES['baseline']['config'])

        # 应用超参数覆盖
        if config_overrides:
            base_config.update(config_overrides)

        super().__init__(
            experiment_name='超参数搜索',
            variant_name=variant_name,
            config=base_config,
            **kwargs
        )

    def _load_data(self):
        """重写数据加载方法"""
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

        self.data, self.class_weights = load_fraud_detection_data(
            data_dir, dataset=self.dataset
        )
        self.data = self.data.to(self.device)
        self.class_weights = self.class_weights.to(self.device)

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

        self.logger.info(f"\n数据划分统计:")
        self.logger.info(f"  训练集: {self.data.train_mask.sum().item():,} ({self.train_ratio*100:.0f}%)")
        self.logger.info(f"  验证集: {self.data.val_mask.sum().item():,} ({self.val_ratio*100:.0f}%)")
        self.logger.info(f"  测试集: {self.data.test_mask.sum().item():,} ({self.test_ratio*100:.0f}%)")

    def create_model(self):
        """创建模型"""
        from src.models.ht_gat_fraud import HTGATFraud
        model_config = get_model_config_for_dataset(self.dataset, self.config)
        return HTGATFraud(model_config)


# ============================================================================
# 搜索空间定义
# ============================================================================

# Phase 1: 损失函数参数
PHASE1_SEARCH_SPACE = {
    'focal_gamma': [0.5, 1.0, 1.5],
    'label_smoothing': [0.0, 0.05, 0.1],
}

# Phase 2: 正则化参数
PHASE2_SEARCH_SPACE = {
    'classifier_dropout': [0.2, 0.3, 0.4],
    'weight_decay': [5e-4, 1e-3],
}

# Phase 3: 模型容量参数
PHASE3_SEARCH_SPACE = {
    'post_hgt_dim': [96, 128, 160],
    'classifier_hidden_dim': [48, 64, 96],
}

# Phase 4: 学习率参数
PHASE4_SEARCH_SPACE = {
    'learning_rate': [5e-4, 1e-3, 2e-3],
    'warmup_epochs': [10, 20, 30],
}

ALL_PHASES = {
    1: ('损失函数参数', PHASE1_SEARCH_SPACE),
    2: ('正则化参数', PHASE2_SEARCH_SPACE),
    3: ('模型容量参数', PHASE3_SEARCH_SPACE),
    4: ('学习率参数', PHASE4_SEARCH_SPACE),
}


def generate_combinations(search_space: dict):
    """生成参数组合"""
    keys = list(search_space.keys())
    values = list(search_space.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# TRAIN_CONFIG中由ablation_base.py直接读取的参数列表
# 这些参数不在self.config中，需要临时patch TRAIN_CONFIG
TRAIN_CONFIG_KEYS = {
    'focal_gamma', 'label_smoothing', 'learning_rate', 'weight_decay',
    'warmup_epochs', 'early_stopping_patience',
}


def run_single_search(config_overrides: dict, variant_name: str,
                      train_ratio: float, val_ratio: float,
                      dataset: str, seed: int, epochs: int, gpu: int,
                      output_dir: Path):
    """运行单个超参数配置"""
    # 临时patch TRAIN_CONFIG中的训练参数
    saved_values = {}
    for key in TRAIN_CONFIG_KEYS:
        if key in config_overrides:
            saved_values[key] = TRAIN_CONFIG.get(key)
            TRAIN_CONFIG[key] = config_overrides[key]

    try:
        experiment = HyperparamSearchExperiment(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            config_overrides=config_overrides,
            variant_name=variant_name,
            dataset=dataset,
            seed=seed,
            gpu=gpu,
            output_dir=output_dir
        )
        return experiment.run(num_epochs=epochs)
    finally:
        # 恢复TRAIN_CONFIG原始值
        for key, val in saved_values.items():
            if val is None:
                TRAIN_CONFIG.pop(key, None)
            else:
                TRAIN_CONFIG[key] = val


def run_phase(phase_num: int, search_space: dict, best_params: dict,
              train_ratio: float, val_ratio: float,
              dataset: str, seed: int, epochs: int, gpu: int,
              output_dir: Path):
    """运行一个搜索阶段"""
    phase_name = ALL_PHASES[phase_num][0]
    print(f"\n{'='*80}")
    print(f"Phase {phase_num}: {phase_name}")
    print(f"搜索空间: {search_space}")
    print(f"已确定参数: {best_params}")
    print(f"{'='*80}")

    results = []
    combinations = list(generate_combinations(search_space))
    total = len(combinations)

    for i, combo in enumerate(combinations):
        # 合并已确定的最优参数和当前搜索参数
        config_overrides = {**best_params, **combo}

        # 生成变体名称
        param_str = '_'.join(f'{k}={v}' for k, v in combo.items())
        variant_name = f'phase{phase_num}_{param_str}'

        print(f"\n[{i+1}/{total}] {combo}")

        try:
            result = run_single_search(
                config_overrides=config_overrides,
                variant_name=variant_name,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                dataset=dataset,
                seed=seed,
                epochs=epochs,
                gpu=gpu,
                output_dir=output_dir
            )
            result['config'] = combo
            results.append(result)
            print(f"  -> AUC: {result['auc']:.4f}, AP: {result.get('ap', 0):.4f}, "
                  f"F1: {result['f1']:.4f}")
        except Exception as e:
            print(f"  -> 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({'config': combo, 'auc': 0, 'error': str(e)})

    # 按AUC排序
    results.sort(key=lambda x: x.get('auc', 0), reverse=True)

    # 打印阶段结果
    print(f"\n{'='*80}")
    print(f"Phase {phase_num} 结果排名 ({phase_name})")
    print(f"{'='*80}")
    print(f"{'排名':<6} {'AUC':<10} {'AP':<10} {'F1':<10} {'参数'}")
    print('-' * 80)
    for rank, r in enumerate(results, 1):
        if 'error' in r:
            print(f"{rank:<6} ERROR    {r['config']}")
        else:
            print(f"{rank:<6} {r['auc']:.4f}    {r.get('ap', 0):.4f}    "
                  f"{r['f1']:.4f}    {r['config']}")

    # 返回最优参数
    best = results[0]
    return best['config'], results


def run_full_search(dataset: str, seed: int, epochs: int, gpu: int,
                    train_ratio: float = 0.4, val_ratio: float = 0.1,
                    phases: list = None):
    """运行完整的分阶段超参数搜索"""
    output_dir = Path(__file__).parent / 'hyperparam_search_results'
    output_dir.mkdir(exist_ok=True)

    if phases is None:
        phases = [1, 2, 3, 4]

    best_params = {}
    all_phase_results = {}

    for phase_num in phases:
        phase_name, search_space = ALL_PHASES[phase_num]
        best_combo, phase_results = run_phase(
            phase_num=phase_num,
            search_space=search_space,
            best_params=best_params,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            dataset=dataset,
            seed=seed,
            epochs=epochs,
            gpu=gpu,
            output_dir=output_dir
        )

        # 将本阶段最优参数加入已确定参数
        best_params.update(best_combo)
        all_phase_results[f'phase{phase_num}_{phase_name}'] = {
            'best_params': best_combo,
            'all_results': [
                {
                    'config': r['config'],
                    'auc': r.get('auc', 0),
                    'ap': r.get('ap', 0),
                    'f1': r.get('f1', 0),
                    'precision': r.get('precision', 0),
                    'recall': r.get('recall', 0),
                }
                for r in phase_results
            ]
        }

        print(f"\nPhase {phase_num} 最优参数: {best_combo}")
        print(f"累计最优参数: {best_params}")

    # 保存最终结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        'experiment': 'hyperparameter_search',
        'timestamp': timestamp,
        'dataset': dataset,
        'train_ratio': train_ratio,
        'seed': seed,
        'epochs': epochs,
        'final_best_params': best_params,
        'phase_results': all_phase_results,
    }

    summary_file = output_dir / f'hp_search_summary_{timestamp}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # 打印最终汇总
    print(f"\n{'='*80}")
    print("超参数搜索完成")
    print(f"{'='*80}")
    print(f"最优超参数组合: {best_params}")
    print(f"结果已保存: {summary_file}")
    print(f"\n建议: 使用以上最优参数更新 config.py，然后运行完整的 20%/40%/60% 实验验证")

    return best_params, summary


def main():
    parser = argparse.ArgumentParser(description='超参数搜索实验')
    parser.add_argument('--dataset', type=str, default='amazon',
                       choices=['yelp', 'amazon'], help='数据集')
    parser.add_argument('--train_ratio', type=float, default=0.4,
                       help='搜索用训练比例 (默认40%%，平衡速度和代表性)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--epochs', type=int, default=400, help='训练轮数')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--phases', type=int, nargs='+', default=None,
                       help='要运行的阶段 (默认全部: 1 2 3 4)')

    args = parser.parse_args()

    print(f"超参数搜索配置:")
    print(f"  数据集: {args.dataset}")
    print(f"  训练比例: {args.train_ratio*100:.0f}%%")
    print(f"  GPU: cuda:{args.gpu}")
    print(f"  Epochs: {args.epochs}")
    print(f"  搜索阶段: {args.phases or [1,2,3,4]}")

    run_full_search(
        dataset=args.dataset,
        seed=args.seed,
        epochs=args.epochs,
        gpu=args.gpu,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        phases=args.phases,
    )


if __name__ == '__main__':
    main()
