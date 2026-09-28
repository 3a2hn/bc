# -*- coding: utf-8 -*-
"""
R-U-R边构建器（Review-User-Review）
通过用户连接的评论对

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


def build_rur_edges(df):
    """
    构建R-U-R边
    
    对每个用户的所有评论建立完全子图（Clique）
    边权重基于时间衰减: weight = exp(-Δt / τ)
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据，必须包含列:
        - user_idx: 用户索引
        - review_idx: 评论索引
        - timestamp: 时间戳
    
    Returns
    -------
    edges : list of tuple
        边列表 [(review_idx1, review_idx2), ...]
    edge_attrs : list of dict
        边属性列表，每个字典包含:
        - edge_type: 边类型 ('R-U-R')
        - weight: 边权重（时间衰减）
        - time_diff: 时间差（天数），|t_raw(i) - t_raw(j)|
        - timestamp: 边事件时间（秒），max(t_raw(i), t_raw(j))
        - user_idx: 用户索引
    """
    logger.info("="*80)
    logger.info("开始构建 R-U-R 边（Review-User-Review）")
    logger.info("="*80)
    
    edges = []
    edge_attrs = []
    
    # 按用户分组
    user_groups = df.groupby('user_idx')
    total_users = len(user_groups)
    
    logger.info(f"总用户数: {total_users}")
    
    # 遍历每个用户
    for user_idx, group in tqdm(user_groups, desc="构建R-U-R边", unit="用户"):
        # 获取该用户的所有评论
        review_indices = group['review_idx'].tolist()
        timestamps = group['timestamp'].tolist()
        
        # 至少需要2条评论才能建边
        if len(review_indices) < 2:
            continue
        
        # 生成所有评论对（完全子图）
        for (r1, t1), (r2, t2) in combinations(zip(review_indices, timestamps), 2):
            # 确保无自环
            if r1 == r2:
                continue
            
            # 保证边的方向一致（r1 < r2），避免重复边
            if r1 > r2:
                r1, r2 = r2, r1
                t1, t2 = t2, t1
            
            # 计算时间差（天数）：|t_raw(i) - t_raw(j)|
            time_diff_days = abs((t2 - t1).days)
            
            # 计算权重：时间衰减
            weight = np.exp(-time_diff_days / config.RUR_TIME_DECAY_TAU)
            
            # 边事件时间：max(t_raw(i), t_raw(j))
            # 语义：只有当两条评论都发表后，这条边代表的关系才完整存在
            edge_timestamp = max(t1.timestamp(), t2.timestamp())
            
            # 添加边
            edges.append((r1, r2))
            
            # 添加边属性
            edge_attrs.append({
                'edge_type': 'R-U-R',
                'weight': weight,
                'time_diff': time_diff_days,      # 时间差（天）
                'timestamp': edge_timestamp,       # 边事件时间（秒）
                'user_idx': user_idx,
            })
    
    logger.info(f"✓ R-U-R边构建完成: {len(edges):,} 条边")
    logger.info("="*80)
    
    return edges, edge_attrs


if __name__ == '__main__':
    # 测试代码
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 创建测试数据
    test_data = {
        'review_idx': [0, 1, 2, 3, 4],
        'user_idx': [0, 0, 0, 1, 1],
        'timestamp': [
            datetime(2024, 1, 1),
            datetime(2024, 1, 15),
            datetime(2024, 2, 1),
            datetime(2024, 1, 10),
            datetime(2024, 1, 20),
        ]
    }
    df_test = pd.DataFrame(test_data)
    
    edges, attrs = build_rur_edges(df_test)
    
    print(f"\n生成的边: {edges}")
    print(f"边属性: {attrs}")