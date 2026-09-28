# -*- coding: utf-8 -*-
"""
消融实验统一运行器
支持运行所有消融实验或指定类别的实验
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ablation_config import ALL_ABLATIONS, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW


def run_experiment_category(category: str, dataset: str, seed: int, epochs: int):
    """运行指定类别的消融实验"""
    print(f"\n{'#'*80}")
    print(f"# 运行实验类别: {category}")
    print('#'*80)
    
    # 根据类别导入对应的运行脚本
    if category == '1_核心模块消融':
        from 核心模块消融 import run_ablation as exp_module
        exp_dir = Path(__file__).parent / '1_核心模块消融'
    elif category == '2_HGT层组件消融':
        from HGT层组件消融 import run_ablation as exp_module
        exp_dir = Path(__file__).parent / '2_HGT层组件消融'
    elif category == '3_TGN记忆模块消融':
        from TGN记忆模块消融 import run_ablation as exp_module
        exp_dir = Path(__file__).parent / '3_TGN记忆模块消融'
    elif category == '4_时序注意力层消融':
        from 时序注意力层消融 import run_ablation as exp_module
        exp_dir = Path(__file__).parent / '4_时序注意力层消融'
    elif category == '5_时间片划分策略消融':
        from 时间片划分策略消融 import run_ablation as exp_module
        exp_dir = Path(__file__).parent / '5_时间片划分策略消融'
    elif category == '6_分类器结构消融':
        from 分类器结构消融 import run_ablation as exp_module
        exp_dir = Path(__file__).parent / '6_分类器结构消融'
    else:
        # 通用方式：直接执行对应目录的脚本
        exp_dir = Path(__file__).parent / category
        if not exp_dir.exists():
            print(f"实验目录不存在: {exp_dir}")
            return {}
        
        import subprocess
        result = subprocess.run(
            [sys.executable, str(exp_dir / 'run_ablation.py'), '--all',
             '--dataset', dataset, '--seed', str(seed), '--epochs', str(epochs)],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"错误: {result.stderr}")
        return {}
    
    return exp_module.run_all_experiments(dataset, seed, epochs)


def run_priority_experiments(priority: str, dataset: str, seed: int, epochs: int):
    """按优先级运行实验"""
    if priority == 'high':
        variants = PRIORITY_HIGH
    elif priority == 'medium':
        variants = PRIORITY_MEDIUM
    elif priority == 'low':
        variants = PRIORITY_LOW
    else:
        raise ValueError(f"未知的优先级: {priority}")
    
    print(f"\n{'#'*80}")
    print(f"# 运行{priority.upper()}优先级实验")
    print(f"# 变体: {variants}")
    print('#'*80)
    
    results = {}
    
    # 找出每个变体所属的类别并运行
    for category, ablations in ALL_ABLATIONS.items():
        for variant_key in ablations.keys():
            if variant_key in variants:
                exp_dir = Path(__file__).parent / category
                
                try:
                    # 动态导入运行函数
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "run_ablation", exp_dir / "run_ablation.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    result = module.run_single_experiment(variant_key, dataset, seed, epochs)
                    results[f"{category}/{variant_key}"] = result
                except Exception as e:
                    print(f"实验 {category}/{variant_key} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    results[f"{category}/{variant_key}"] = {'error': str(e)}
    
    return results


def run_all_experiments(dataset: str, seed: int, epochs: int, categories: list = None):
    """运行所有消融实验"""
    all_results = {}
    
    if categories is None:
        categories = list(ALL_ABLATIONS.keys())
    
    for category in categories:
        if category not in ALL_ABLATIONS:
            print(f"警告: 未知的实验类别 {category}")
            continue
        
        exp_dir = Path(__file__).parent / category
        
        try:
            # 动态导入运行函数
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_ablation", exp_dir / "run_ablation.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            results = module.run_all_experiments(dataset, seed, epochs)
            all_results[category] = results
        except Exception as e:
            print(f"实验类别 {category} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[category] = {'error': str(e)}
    
    return all_results


def save_summary(all_results: dict, output_dir: Path, dataset: str):
    """保存实验结果汇总"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细JSON
    json_file = output_dir / f'ablation_summary_{dataset}_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n详细结果已保存: {json_file}")
    
    # 生成汇总表格
    rows = []
    for category, results in all_results.items():
        if isinstance(results, dict) and 'error' not in results:
            for variant, metrics in results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    rows.append({
                        '实验类别': category,
                        '变体': variant,
                        'AUC': metrics.get('auc', 'N/A'),
                        'AP': metrics.get('ap', 'N/A'),
                        'F1': metrics.get('f1', 'N/A'),
                        'Precision': metrics.get('precision', 'N/A'),
                        'Recall': metrics.get('recall', 'N/A'),
                    })
    
    if rows:
        df = pd.DataFrame(rows)
        csv_file = output_dir / f'ablation_summary_{dataset}_{timestamp}.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"汇总表格已保存: {csv_file}")
        
        # 打印汇总
        print("\n" + "="*100)
        print("消融实验结果汇总")
        print("="*100)
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='消融实验统一运行器')
    
    # 运行模式
    parser.add_argument('--all', action='store_true',
                        help='运行所有消融实验')
    parser.add_argument('--category', type=str, nargs='+', default=None,
                        choices=list(ALL_ABLATIONS.keys()),
                        help='指定运行的实验类别')
    parser.add_argument('--priority', type=str, default=None,
                        choices=['high', 'medium', 'low'],
                        help='按优先级运行实验')
    
    # 通用参数
    parser.add_argument('--dataset', type=str, default='yelp',
                        choices=['yelp', 'amazon'],
                        help='数据集')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录')
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("HTGATFraud 消融实验运行器")
    print("="*80)
    print(f"数据集: {args.dataset}")
    print(f"随机种子: {args.seed}")
    print(f"训练轮数: {args.epochs}")
    print(f"输出目录: {output_dir}")
    print("="*80)
    
    # 运行实验
    if args.all:
        all_results = run_all_experiments(args.dataset, args.seed, args.epochs)
        save_summary(all_results, output_dir, args.dataset)
    elif args.category:
        all_results = run_all_experiments(args.dataset, args.seed, args.epochs, args.category)
        save_summary(all_results, output_dir, args.dataset)
    elif args.priority:
        results = run_priority_experiments(args.priority, args.dataset, args.seed, args.epochs)
        save_summary({'priority_' + args.priority: results}, output_dir, args.dataset)
    else:
        print("\n请指定运行模式:")
        print("  --all           运行所有实验")
        print("  --category X    运行指定类别的实验")
        print("  --priority X    按优先级运行实验 (high/medium/low)")
        print("\n可用的实验类别:")
        for category in ALL_ABLATIONS.keys():
            print(f"  - {category}")


if __name__ == '__main__':
    main()
