# -*- coding: utf-8 -*-
"""
R-T-R边构建器（Review-Temporal-Review）
同一商家在同一时间窗口内的评论对

边时间戳定义：
- t_raw(e) = max(t_raw(i), t_raw(j))  # 边事件时间：两端节点中较晚的时间戳
- time_diff = |t_raw(i) - t_raw(j)|   # 时间差：两端节点的时间间隔

注意：
- "时间窗口条件"（year_month）仅用于判断是否建边
- "边事件时间"使用较晚一端的时间戳，不使用月份桶
"""

import numpy as np
from itertools import combinations
from tqdm import tqdm
from utils.logger import get_logger
import config

logger = get_logger()


def build_rtr_edges(df):
    """
    构建R-T-R边
    
    对同一商家在同一时间窗口内的所有评论建立完全子图
    时间窗口默认为月粒度（仅用于判断是否建边）
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据，必须包含列:
        - business_idx: 商家索引
        - review_idx: 评论索引
        - timestamp: 时间戳
        - year_month: 年月（Period类型，仅用于分组条件）
    
    Returns
    -------
    edges : list of tuple
        边列表 [(review_idx1, review_idx2), ...]
    edge_attrs : list of dict
        边属性列表，每个字典包含:
        - edge_type: 边类型 ('R-T-R')
        - weight: 边权重（时间集中度）
        - time_diff: 时间差（天数），|t_raw(i) - t_raw(j)|
        - timestamp: 边事件时间（秒），max(t_raw(i), t_raw(j))
        - year_month: 年月（仅记录，不作为边时间戳）
        - business_idx: 商家索引
    """
    logger.info("="*80)
    logger.info("开始构建 R-T-R 边（Review-Temporal-Review）")
    logger.info("="*80)
    logger.info(f"时间窗口粒度: {config.RTR_TIME_WINDOW} (月) - 仅用于建边条件")
    logger.info(f"边事件时间: max(t_raw(i), t_raw(j)) - 较晚一端")
    
    edges = []
    edge_attrs = []
    
    # 按商家和年月双重分组（年月仅用于判断是否建边）
    temporal_groups = df.groupby(['business_idx', 'year_month'])
    total_groups = len(temporal_groups)
    
    logger.info(f"总时间窗口数: {total_groups}")
    
    # 统计窗口大小分布
    window_sizes = []
    
    # 遍历每个时间窗口
    for (business_idx, year_month), group in tqdm(temporal_groups, 
                                                   desc="构建R-T-R边", 
                                                   unit="窗口"):
        # 获取该窗口内的所有评论
        review_indices = group['review_idx'].tolist()
        timestamps = group['timestamp'].tolist()
        
        window_sizes.append(len(review_indices))
        
        # 至少需要2条评论才能建边
        if len(review_indices) < 2:
            continue
        
        # 生成所有评论对（完全子图）
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
            
            # 计算权重：时间集中度（月内越集中，权重越大）
            weight = 1.0 / (1.0 + time_diff_days / config.RTR_TIME_CONCENTRATION_SCALE)
            
            # 边事件时间：max(t_raw(i), t_raw(j))
            # 语义：只有当两条评论都发表后，这条边代表的关系才完整存在
            # 注意：不使用月份桶作为边时间戳，月份仅用于"是否连边"的判定条件
            edge_timestamp = max(t1.timestamp(), t2.timestamp())
            
            # 添加边
            edges.append((r1, r2))
            
            # 添加边属性
            edge_attrs.append({
                'edge_type': 'R-T-R',
                'weight': weight,
                'time_diff': time_diff_days,      # 时间差（天）
                'timestamp': edge_timestamp,       # 边事件时间（秒）
                'year_month': str(year_month),     # 年月（仅记录，不作为边时间戳）
                'business_idx': business_idx,
            })
    
    # 统计信息
    if window_sizes:
        avg_window_size = np.mean(window_sizes)
        max_window_size = np.max(window_sizes)
        logger.info(f"  平均窗口大小: {avg_window_size:.2f} 条评论")
        logger.info(f"  最大窗口大小: {max_window_size} 条评论")
    
    logger.info(f"✓ R-T-R边构建完成: {len(edges):,} 条边")
    logger.info("="*80)
    
    return edges, edge_attrs


if __name__ == '__main__':
    # 测试代码
    import pandas as pd
    from datetime import datetime
    
    # 创建测试数据
    test_data = {
        'review_idx': [0, 1, 2, 3, 4],
        'business_idx': [0, 0, 0, 0, 1],
        'timestamp': [
            datetime(2024, 1, 5),
            datetime(2024, 1, 15),
            datetime(2024, 1, 25),
            datetime(2024, 2, 5),
            datetime(2024, 1, 10),
        ],
    }
    df_test = pd.DataFrame(test_data)
    df_test['year_month'] = pd.to_datetime(df_test['timestamp']).dt.to_period('M')
    
    edges, attrs = build_rtr_edges(df_test)
    
    print(f"\n生成的边: {edges}")
    print(f"边属性: {attrs}")

