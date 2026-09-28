# -*- coding: utf-8 -*-
"""
主启动文件 - Amazon电子产品数据集
以评论为节点的时间感知异构图构建（与Yelp结构一致）

边类型对比：
| Amazon | Yelp   | 含义 |
|--------|--------|------|
| U-P-U  | R-U-R  | 由同一用户发布的评论之间的连接 |
| U-S-U  | R-S-R  | 一周内至少共享一次相同星级评分的用户之间的连接 |
| U-V-U  | -      | 评论文本相似度(TF-IDF)位于所有用户前5%的用户之间的连接 |

时间戳存储（与Yelp完全一致）：
- 节点时间戳：评论发布时间
  - node_timestamps_raw: 原始时间戳（毫秒）
  - node_timestamps_norm: 归一化时间戳 [0, 100]
- 边时间戳：max(t_raw(i), t_raw(j))
  - edge_timestamps_raw: 原始边时间戳（毫秒）
  - edge_timestamps_norm: 归一化边时间戳 [0, 100]
  - edge_time_diffs: 边两端节点的时间差（天）
  - edge_weights: 边权重

输出文件：
- graph.pkl: 完整图数据（评论节点、U-P-U/U-S-U/U-V-U边）
- features.npy: 评论特征矩阵
- labels.npy: 评论标签
"""

import sys
import time
import logging
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 导入配置
import config_amazon as config

# 导入预处理器
from preprocessor_amazon import preprocess_amazon_data

# 导入图构建器
from edges_amazon.graph_builder_amazon import GraphBuilderAmazon

# 导入特征工程
from features_amazon.feature_engineer_amazon import run_feature_engineering


def setup_logging():
    """设置日志系统"""
    config.LOGS_DIR.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_CONFIG['level']),
        format=config.LOGGING_CONFIG['format'],
        datefmt=config.LOGGING_CONFIG['date_format'],
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_and_save_labels(df, config):
    """提取并保存标签"""
    logger = logging.getLogger(__name__)
    logger.info("提取评论标签...")
    
    labels = df['label'].to_numpy().astype(np.int32)
    
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "欺诈评论" if label == 0 else "正常评论"
        logger.info(f"  {label_name} (label={label}): {count:,} 条 ({count/len(labels)*100:.2f}%)")
    
    np.save(config.LABELS_OUTPUT, labels)
    logger.info(f"标签已保存到: {config.LABELS_OUTPUT}")
    
    return labels


def validate_quality(graph_stats, config):
    """质量验证"""
    warnings = []
    thresholds = config.QUALITY_THRESHOLDS
    
    total_edges = graph_stats['total_edges']
    if total_edges < thresholds['total_edges_min']:
        warnings.append(f"总边数过少: {total_edges}")
    elif total_edges > thresholds['total_edges_max']:
        warnings.append(f"总边数过多: {total_edges}")
    
    for edge_type, key_prefix in [('U-P-U', 'upu'), ('U-S-U', 'usu'), ('U-V-U', 'uvu')]:
        edge_count = graph_stats['edge_counts'].get(edge_type, 0)
        min_key = f"{key_prefix}_edges_min"
        max_key = f"{key_prefix}_edges_max"
        
        if edge_count < thresholds.get(min_key, 0):
            warnings.append(f"{edge_type}边数过少: {edge_count}")
        elif edge_count > thresholds.get(max_key, float('inf')):
            warnings.append(f"{edge_type}边数过多: {edge_count}")
    
    isolated_ratio = graph_stats.get('isolated_nodes', 0) / max(graph_stats['num_nodes'], 1)
    if isolated_ratio > thresholds['isolated_nodes_max_ratio']:
        warnings.append(f"孤立节点比例过高: {isolated_ratio:.2%}")
    
    passed = len(warnings) == 0
    return passed, warnings


def print_summary(df, graph_data, features, labels):
    """打印总结报告"""
    print("\n" + "=" * 80)
    print("Amazon评论图构建完成 - 总结报告".center(80))
    print("=" * 80)
    
    # 数据统计
    print("\n【数据统计】")
    print(f"  总评论数（节点数）: {len(df):,}")
    print(f"  用户数: {df['user_idx'].nunique():,}")
    print(f"  产品数: {df['product_idx'].nunique():,}")
    print(f"  时间跨度: {df['timestamp'].min().date()} 到 {df['timestamp'].max().date()}")
    
    # 图统计
    print("\n【图统计】")
    stats = graph_data['statistics']
    print(f"  节点数（评论）: {stats['num_nodes']:,}")
    print(f"  总边数: {stats['total_edges']:,}")
    print(f"  U-P-U边: {stats['upu_edges']:,} ({stats['upu_ratio']:.2f}%)")
    print(f"  U-S-U边: {stats['usu_edges']:,} ({stats['usu_ratio']:.2f}%)")
    print(f"  U-V-U边: {stats['uvu_edges']:,} ({stats['uvu_ratio']:.2f}%)")
    print(f"  边/节点比: {stats['edge_node_ratio']:.2f}")
    print(f"  平均度数: {stats['avg_degree']:.2f}")
    print(f"  孤立节点: {stats['isolated_nodes']}")
    
    # 时间戳统计
    print("\n【时间戳存储（与Yelp一致）】")
    ts_range = graph_data['timestamp_range']
    print(f"  全局时间范围: ts_min={ts_range['ts_min']:.0f}, ts_max={ts_range['ts_max']:.0f} (毫秒)")
    print(f"  节点时间戳: node_timestamps_raw (毫秒), node_timestamps_norm [0, 100]")
    print(f"  边时间戳: edge_timestamps_raw (毫秒), edge_timestamps_norm [0, 100]")
    print(f"  边时间差: edge_time_diffs (天)")
    print(f"  边权重: edge_weights")
    
    # 特征统计
    print("\n【特征统计】")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  样本数: {features.shape[0]}")
    print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")
    
    # 标签统计
    print("\n【标签统计】")
    fraud_count = (labels == 0).sum()
    normal_count = (labels == 1).sum()
    print(f"  欺诈评论: {fraud_count:,} ({fraud_count/len(labels)*100:.2f}%)")
    print(f"  正常评论: {normal_count:,} ({normal_count/len(labels)*100:.2f}%)")
    
    # 与Yelp对比
    print("\n【与Yelp边类型对比】")
    print("  | Amazon | Yelp   | 含义")
    print("  |--------|--------|------")
    print("  | U-P-U  | R-U-R  | 由同一用户发布的评论之间的连接")
    print("  | U-S-U  | R-S-R  | 一周内至少共享一次相同星级评分的用户之间的连接")
    print("  | U-V-U  | -      | 评论文本相似度(TF-IDF)位于所有用户前5%的用户之间的连接")
    
    # 输出文件
    print("\n【输出文件】")
    print(f"  图数据: {config.GRAPH_OUTPUT}")
    print(f"  特征矩阵: {config.FEATURES_OUTPUT}")
    print(f"  标签: {config.LABELS_OUTPUT}")
    print(f"  ID映射: {config.MAPPINGS_OUTPUT}")
    print(f"  统计信息: {config.STATISTICS_OUTPUT}")
    
    print("\n" + "=" * 80)
    print("流程完成！所有数据已保存。".center(80))
    print("=" * 80)


def main():
    """主函数"""
    start_time = time.time()
    
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("Amazon电子产品评论欺诈检测 - 评论图构建")
    logger.info("边类型: U-P-U / U-S-U / U-V-U")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config.print_config()
    
    try:
        # Step 1: 数据预处理
        logger.info("\n" + "=" * 60)
        logger.info("第1步: 数据预处理")
        logger.info("=" * 60)
        
        df, mappings = preprocess_amazon_data(config)
        
        # Step 2: 图构建
        logger.info("\n" + "=" * 60)
        logger.info("第2步: 图构建 (U-P-U / U-S-U / U-V-U)")
        logger.info("=" * 60)
        
        graph_builder = GraphBuilderAmazon(df, config)
        graph_data = graph_builder.build_complete_graph()
        graph_builder.save_graph()
        
        # Step 3: 特征工程
        logger.info("\n" + "=" * 60)
        logger.info("第3步: 特征工程")
        logger.info("=" * 60)
        
        features = run_feature_engineering(df, include_text=True, normalize=True)
        
        # Step 4: 标签提取
        logger.info("\n" + "=" * 60)
        logger.info("第4步: 标签提取")
        logger.info("=" * 60)
        
        labels = extract_and_save_labels(df, config)
        
        # Step 5: 质量验证
        logger.info("\n" + "=" * 60)
        logger.info("第5步: 质量验证")
        logger.info("=" * 60)
        
        passed, warnings = validate_quality(graph_data['statistics'], config)
        
        if passed:
            logger.info("✓ 质量检查通过!")
        else:
            logger.warning("⚠ 质量检查发现以下问题:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        # 完成
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info(f"流程完成！总耗时: {elapsed_time:.2f} 秒")
        logger.info("=" * 60)
        
        print_summary(df, graph_data, features, labels)
        
        return graph_data, features, labels
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
