# -*- coding: utf-8 -*-
"""
主启动文件
执行完整的图构建和特征工程流程

设计原则（Transductive 设定）：
- 建图阶段只负责构建完整的时间感知异构图
- 不进行 train/val/test 划分，划分逻辑放到训练阶段
- 所有时间信息显式保存，供后续模块直接消费

输出文件：
- graph.pkl: 完整图数据（含节点/边时间戳、标签，不含划分）
- features.npy: 节点特征矩阵
- labels.npy: 节点标签（独立保存，方便访问）
"""

import sys
import time
import numpy as np
from pathlib import Path

# 导入配置
import config

# 导入工具
from utils.logger import setup_logger, get_logger
from utils.data_loader import save_pickle, save_numpy
from utils.validators import validate_graph_quality, validate_features

# 导入模块
from preprocessor import run_preprocessing
from edges.graph_builder import run_graph_building
from features.feature_engineer import run_feature_engineering


def extract_and_save_labels(df):
    """
    提取并保存标签（独立文件，方便访问）
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据
        
    Returns
    -------
    labels : numpy.ndarray
        标签数组
    """
    logger = get_logger()
    logger.info("="*80)
    logger.info("提取标签")
    logger.info("="*80)

    labels = df['label'].to_numpy().astype(np.int32)
    
    # 统计标签分布
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "正常评论" if label == 1 else "欺诈评论"
        logger.info(f"  {label_name} (label={label}): {count:,} 条 ({count/len(labels)*100:.2f}%)")
    
    # 保存标签
    save_numpy(labels, config.LABELS_OUTPUT)
    logger.info(f"标签已保存到: {config.LABELS_OUTPUT}")
    logger.info("="*80)
    
    return labels


def print_summary(df, graph_data, features, labels):
    """
    打印完整流程的总结报告
    
    Parameters
    ----------
    df : pandas.DataFrame
        数据
    graph_data : dict
        图数据
    features : numpy.ndarray
        特征矩阵
    labels : numpy.ndarray
        标签
    """
    logger = get_logger()
    
    print("\n" + "="*80)
    print("完整流程总结报告（Transductive 设定）".center(80))
    print("="*80)
    
    # 数据统计
    print("\n【数据统计】")
    print(f"  总评论数: {len(df):,}")
    print(f"  用户数: {df['json_user_id'].nunique():,}")
    print(f"  商家数: {df['json_business_id'].nunique():,}")
    print(f"  时间跨度: {df['timestamp'].min().date()} 到 {df['timestamp'].max().date()}")
    
    # 图统计
    print("\n【图统计】")
    stats = graph_data['statistics']
    print(f"  节点数: {stats['num_nodes']:,}")
    print(f"  总边数: {stats['total_edges']:,}")
    print(f"  R-U-R边: {stats['rur_edges']:,} ({stats['rur_ratio']:.2f}%)")
    print(f"  R-T-R边: {stats['rtr_edges']:,} ({stats['rtr_ratio']:.2f}%)")
    print(f"  R-S-R边: {stats['rsr_edges']:,} ({stats['rsr_ratio']:.2f}%)")
    print(f"  边/节点比: {stats['edge_node_ratio']:.2f}")
    print(f"  平均度数: {stats['avg_degree']:.2f}")
    
    # 时间戳统计
    print("\n【时间戳统计】")
    ts_range = graph_data['timestamp_range']
    print(f"  全局时间范围: ts_min={ts_range['ts_min']:.2f}, ts_max={ts_range['ts_max']:.2f}")
    print(f"  节点时间戳: 归一化到 [0, 100]")
    print(f"  边时间戳: 归一化到 [0, 100]，定义为 max(t_raw(i), t_raw(j))")
    
    # 特征统计
    print("\n【特征统计】")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  样本数: {features.shape[0]}")
    print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")
    
    # 标签统计
    print("\n【标签统计】")
    fraud_count = (labels == 0).sum()
    real_count = (labels == 1).sum()
    print(f"  欺诈评论: {fraud_count:,} ({fraud_count/len(labels)*100:.2f}%)")
    print(f"  正常评论: {real_count:,} ({real_count/len(labels)*100:.2f}%)")
    
    # 输出文件
    print("\n【输出文件】")
    print(f"  图数据: {config.GRAPH_OUTPUT}")
    print(f"  特征矩阵: {config.FEATURES_OUTPUT}")
    print(f"  标签: {config.LABELS_OUTPUT}")
    print(f"  ID映射: {config.MAPPINGS_OUTPUT}")
    print(f"  统计信息: {config.STATISTICS_OUTPUT}")
    print(f"  日志文件: {config.LOG_FILE}")
    
    # 说明
    print("\n【说明】")
    print("  本流程采用 Transductive 设定：")
    print("  - 图数据包含完整的节点和边，不进行 train/val/test 划分")
    print("  - 划分逻辑由训练阶段统一处理")
    print("  - 训练时可利用整张图作为结构上下文，但只在训练集节点上使用标签监督")
    
    print("\n" + "="*80)
    print("流程完成！所有数据已保存。".center(80))
    print("="*80)


def main():
    """
    主函数
    执行完整的图构建流程（Transductive 设定）
    
    流程：
    1. 预处理：加载数据、清洗、ID映射
    2. 图构建：构建三类边（RUR/RTR/RSR）
    3. 特征工程：提取节点特征
    4. 标签提取：提取并保存节点标签
    
    注意：不进行 train/val/test 划分，划分逻辑放到训练阶段
    """
    start_time = time.time()
    
    # 设置日志
    setup_logger(log_file=config.LOG_FILE, level='INFO')
    logger = get_logger()
    
    print("\n" + "="*80)
    print("时间感知异构图构建流程（Transductive 设定）".center(80))
    print("="*80)
    print(f"\n输入文件: {config.INPUT_CSV}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print("\n说明：")
    print("  - 本流程构建完整的时间感知异构图")
    print("  - 不进行 train/val/test 划分")
    print("  - 划分逻辑由训练阶段统一处理")
    print("="*80)
    
    # ========================================================================
    # Step 1/4: 预处理
    # ========================================================================
    print("\n" + "-"*80)
    print("Step 1/4: 数据预处理")
    print("-"*80)
    
    df, mappings = run_preprocessing(config.INPUT_CSV)
    
    # ========================================================================
    # Step 2/4: 图构建
    # ========================================================================
    print("\n" + "-"*80)
    print("Step 2/4: 图构建")
    print("-"*80)
    
    graph_data = run_graph_building(df)
    
    # 验证图质量
    is_valid, issues = validate_graph_quality(graph_data)
    if not is_valid:
        logger.warning(f"图质量检查发现问题: {issues}")
    
    # ========================================================================
    # Step 3/4: 特征工程
    # ========================================================================
    print("\n" + "-"*80)
    print("Step 3/4: 特征工程")
    print("-"*80)
    
    features = run_feature_engineering(df, graph_data)
    
    # 验证特征
    is_valid, issues = validate_features(features)
    if not is_valid:
        logger.warning(f"特征验证发现问题: {issues}")
    
    # ========================================================================
    # Step 4/4: 标签提取
    # ========================================================================
    print("\n" + "-"*80)
    print("Step 4/4: 标签提取")
    print("-"*80)
    
    labels = extract_and_save_labels(df)
    
    # ========================================================================
    # 完成
    # ========================================================================
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"流程完成！总耗时: {elapsed_time:.2f} 秒")
    print("="*80)
    
    # 打印总结报告
    print_summary(df, graph_data, features, labels)
    
    return graph_data, features, labels


if __name__ == '__main__':
    main()