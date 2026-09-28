# -*- coding: utf-8 -*-
"""
R-S-R边构建器（Review-Star-Review）
同一商家具有相同评分的评论对

边时间戳定义：
- t_raw(e) = max(t_raw(i), t_raw(j))  # 边事件时间：两端节点中较晚的时间戳
- time_diff = |t_raw(i) - t_raw(j)|   # 时间差：两端节点的时间间隔
"""

import numpy as np
from itertools import combinations
from tqdm import tqdm
from utils.logger import get_logger
import config

logger = get_logger()


def build_rsr_edges(df):
    """
    构建R-S-R边
    
    对同一商家具有相同星级评分的所有评论建立完全子图
    边权重基于评分极端性
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据，必须包含列:
        - business_idx: 商家索引
        - review_idx: 评论索引
        - json_stars: 星级评分（1-5）
        - timestamp: 时间戳
    
    Returns
    -------
    edges : list of tuple
        边列表 [(review_idx1, review_idx2), ...]
    edge_attrs : list of dict
        边属性列表，每个字典包含:
        - edge_type: 边类型 ('R-S-R')
        - weight: 边权重（评分极端性）
        - rating: 评分 (1-5)
        - extremity: 极端性 [0, 1]
        - time_diff: 时间差（天数），|t_raw(i) - t_raw(j)|
        - timestamp: 边事件时间（秒），max(t_raw(i), t_raw(j))
        - business_idx: 商家索引
    """
    logger.info("="*80)
    logger.info("开始构建 R-S-R 边（Review-Star-Review）")
    logger.info("="*80)
    logger.info(f"边事件时间: max(t_raw(i), t_raw(j)) - 较晚一端")
    
    edges = []
    edge_attrs = []
    
    # 按商家和评分双重分组
    star_groups = df.groupby(['business_idx', 'json_stars'])
    total_groups = len(star_groups)
    
    logger.info(f"总评分组数: {total_groups}")
    
    # 统计各星级的边数
    star_edge_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # 遍历每个评分组
    for (business_idx, rating), group in tqdm(star_groups, 
                                              desc="构建R-S-R边", 
                                              unit="组"):
        # 获取该组的所有评论
        review_indices = group['review_idx'].tolist()
        timestamps = group['timestamp'].tolist()
        
        # 至少需要2条评论才能建边
        if len(review_indices) < 2:
            continue
        
        # 计算评分极端性
        extremity = abs(rating - 3.0) / 2.0  # [0, 1]
        
        # 计算权重
        if config.RSR_EXTREMITY_WEIGHT:
            weight = 1.0 + extremity  # [1.0, 2.0]
        else:
            weight = 1.0
        
        # 生成所有评论对（完全子图）
        edge_count_before = len(edges)
        
        for (r1, t1), (r2, t2) in combinations(zip(review_indices, timestamps), 2):
            # 确保无自环
            if r1 == r2:
                continue
            
            # 保证边的方向一致
            if r1 > r2:
                r1, r2 = r2, r1
                t1, t2 = t2, t1
            
            # 计算时间差（天数）：|t_raw(i) - t_raw(j)|
            time_diff_days = abs((t2 - t1).days)
            
            # 边事件时间：max(t_raw(i), t_raw(j))
            # 语义：只有当两条评论都发表后，这条边代表的关系才完整存在
            edge_timestamp = max(t1.timestamp(), t2.timestamp())
            
            # 添加边
            edges.append((r1, r2))
            
            # 添加边属性
            edge_attrs.append({
                'edge_type': 'R-S-R',
                'weight': weight,
                'rating': int(rating),
                'extremity': extremity,
                'time_diff': time_diff_days,      # 时间差（天）
                'timestamp': edge_timestamp,       # 边事件时间（秒）
                'business_idx': business_idx,
            })
        
        # 统计该星级产生的边数
        edge_count_after = len(edges)
        edges_added = edge_count_after - edge_count_before
        star_edge_counts[int(rating)] += edges_added
    
    # 打印统计信息
    logger.info("各星级边数分布:")
    for star in [1, 2, 3, 4, 5]:
        count = star_edge_counts[star]
        percentage = (count / len(edges) * 100) if len(edges) > 0 else 0
        logger.info(f"  {star}星: {count:,} 条边 ({percentage:.2f}%)")
    
    logger.info(f"✓ R-S-R边构建完成: {len(edges):,} 条边")
    logger.info("="*80)
    
    return edges, edge_attrs


if __name__ == '__main__':
    # 测试代码
    import pandas as pd
    from datetime import datetime
    
    # 创建测试数据
    test_data = {
        'review_idx': [0, 1, 2, 3, 4, 5],
        'business_idx': [0, 0, 0, 0, 1, 1],
        'json_stars': [5, 5, 4, 3, 5, 5],
        'timestamp': [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
            datetime(2024, 1, 5),
            datetime(2024, 1, 6),
        ],
    }
    df_test = pd.DataFrame(test_data)
    
    edges, attrs = build_rsr_edges(df_test)
    
    print(f"\n生成的边: {edges}")
    print(f"边数: {len(edges)}")
    print(f"边属性示例: {attrs[0] if attrs else 'None'}")

