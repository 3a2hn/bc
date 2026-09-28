# -*- coding: utf-8 -*-
"""
消融实验结果汇总与可视化
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ablation_config import ALL_ABLATIONS


def collect_results(base_dir: Path = None):
    """
    收集所有消融实验的结果
    
    Parameters
    ----------
    base_dir : Path
        消融实验根目录
        
    Returns
    -------
    dict
        所有实验结果
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    all_results = {}
    
    for category in ALL_ABLATIONS.keys():
        category_dir = base_dir / category / 'results'
        if not category_dir.exists():
            continue
        
        category_results = {}
        
        # 查找所有结果文件
        for result_file in category_dir.glob('*_results_*.json'):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                variant_name = data.get('variant_name', result_file.stem)
                
                # 如果同一变体有多个结果，取最新的
                if variant_name not in category_results:
                    category_results[variant_name] = data
        
        if category_results:
            all_results[category] = category_results
    
    return all_results


def create_summary_table(all_results: dict) -> pd.DataFrame:
    """
    创建实验结果汇总表格
    
    Parameters
    ----------
    all_results : dict
        所有实验结果
        
    Returns
    -------
    pd.DataFrame
        汇总表格
    """
    rows = []
    
    for category, results in all_results.items():
        for variant_name, data in results.items():
            if 'final_metrics' in data:
                metrics = data['final_metrics']
                rows.append({
                    '实验类别': category,
                    '变体': variant_name,
                    '变体描述': ALL_ABLATIONS.get(category, {}).get(variant_name, {}).get('description', ''),
                    'AUC': metrics.get('auc', np.nan),
                    'AP': metrics.get('ap', np.nan),
                    'F1': metrics.get('f1', np.nan),
                    'Precision': metrics.get('precision', np.nan),
                    'Recall': metrics.get('recall', np.nan),
                    '最佳AUC': data.get('best_test_auc', np.nan),
                    '最佳Epoch': data.get('best_epoch', np.nan),
                    '参数量': data.get('num_parameters', np.nan),
                })
    
    df = pd.DataFrame(rows)
    return df


def plot_ablation_comparison(df: pd.DataFrame, output_dir: Path, metric: str = 'AUC'):
    """
    绘制消融实验对比图
    
    Parameters
    ----------
    df : pd.DataFrame
        结果表格
    output_dir : Path
        输出目录
    metric : str
        要绘制的指标
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 按实验类别分组绘制
    categories = df['实验类别'].unique()
    
    for category in categories:
        category_df = df[df['实验类别'] == category].copy()
        category_df = category_df.sort_values(metric, ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 找出baseline（如果存在）
        baseline_value = None
        if 'baseline' in category_df['变体'].values:
            baseline_value = category_df[category_df['变体'] == 'baseline'][metric].values[0]
        
        # 绘制条形图
        colors = []
        for _, row in category_df.iterrows():
            if row['变体'] == 'baseline':
                colors.append('#2ecc71')  # 绿色表示baseline
            elif baseline_value and row[metric] < baseline_value:
                colors.append('#e74c3c')  # 红色表示性能下降
            else:
                colors.append('#3498db')  # 蓝色表示正常
        
        bars = ax.barh(range(len(category_df)), category_df[metric], color=colors)
        ax.set_yticks(range(len(category_df)))
        ax.set_yticklabels(category_df['变体'])
        ax.set_xlabel(metric)
        ax.set_title(f'{category} - {metric} 对比')
        
        # 添加数值标签
        for bar, value in zip(bars, category_df[metric]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', va='center', fontsize=9)
        
        # 添加baseline参考线
        if baseline_value:
            ax.axvline(x=baseline_value, color='#2ecc71', linestyle='--', 
                      label=f'Baseline ({baseline_value:.4f})')
            ax.legend()
        
        plt.tight_layout()
        
        # 保存图片
        safe_category = category.replace('/', '_')
        plot_file = output_dir / f'ablation_{safe_category}_{metric.lower()}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {plot_file}")


def plot_component_contribution(df: pd.DataFrame, output_dir: Path):
    """
    绘制组件贡献度图
    """
    # 只针对核心模块消融实验
    core_df = df[df['实验类别'] == '1_核心模块消融'].copy()
    
    if core_df.empty:
        return
    
    # 获取baseline的AUC
    baseline_auc = core_df[core_df['变体'] == 'baseline']['AUC'].values
    if len(baseline_auc) == 0:
        return
    baseline_auc = baseline_auc[0]
    
    # 计算每个变体相对于baseline的性能下降
    contributions = []
    for _, row in core_df.iterrows():
        if row['变体'] != 'baseline':
            contribution = baseline_auc - row['AUC']
            contributions.append({
                '组件': row['变体'].replace('wo_', '').replace('_', ' ').title(),
                '贡献度': contribution * 100  # 转为百分比
            })
    
    if not contributions:
        return
    
    contrib_df = pd.DataFrame(contributions)
    contrib_df = contrib_df.sort_values('贡献度', ascending=True)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(contrib_df)))
    bars = ax.barh(range(len(contrib_df)), contrib_df['贡献度'], color=colors)
    ax.set_yticks(range(len(contrib_df)))
    ax.set_yticklabels(contrib_df['组件'])
    ax.set_xlabel('AUC 贡献度 (%)')
    ax.set_title('核心模块对AUC的贡献度')
    
    # 添加数值标签
    for bar, value in zip(bars, contrib_df['贡献度']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
               f'{value:.2f}%', va='center', fontsize=10)
    
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'component_contribution.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"组件贡献度图已保存: {plot_file}")


def generate_latex_table(df: pd.DataFrame, output_dir: Path):
    """
    生成LaTeX格式的结果表格
    """
    # 选择关键列
    latex_df = df[['实验类别', '变体', 'AUC', 'F1', 'Recall']].copy()
    latex_df = latex_df.round(4)
    
    # 生成LaTeX代码
    latex_code = latex_df.to_latex(index=False, escape=False)
    
    latex_file = output_dir / 'ablation_results.tex'
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"LaTeX表格已保存: {latex_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='消融实验结果汇总')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='消融实验根目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("消融实验结果汇总")
    print("="*80)
    
    # 收集结果
    all_results = collect_results(base_dir)
    
    if not all_results:
        print("未找到任何实验结果！")
        print("请先运行消融实验:")
        print("  python run_all_ablations.py --all")
        return
    
    # 创建汇总表格
    df = create_summary_table(all_results)
    
    if df.empty:
        print("结果表格为空！")
        return
    
    # 保存CSV
    csv_file = output_dir / 'ablation_summary.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\nCSV汇总表已保存: {csv_file}")
    
    # 打印表格
    print("\n" + "="*100)
    print("消融实验结果汇总表")
    print("="*100)
    print(df[['实验类别', '变体', 'AUC', 'F1', 'Recall']].to_string(index=False))
    
    # 绘制对比图
    print("\n生成可视化图表...")
    for metric in ['AUC', 'F1', 'Recall']:
        plot_ablation_comparison(df, output_dir, metric)
    
    # 绘制组件贡献度图
    plot_component_contribution(df, output_dir)
    
    # 生成LaTeX表格
    generate_latex_table(df, output_dir)
    
    print("\n" + "="*80)
    print("结果汇总完成！")
    print("="*80)


if __name__ == '__main__':
    main()
