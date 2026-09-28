# -*- coding: utf-8 -*-
"""
U-S-U边构建器（User-Star-User）
在一周内至少共享一次相同星级评分的用户之间的评论连接

边定义：
- 节点：Review（评论）
- 边：如果两个用户在一周内至少有一次给出相同星级评分，
      则这两个用户的所有评论之间建边

边时间戳定义：
- t_raw(e) = max(t_raw(i), t_raw(j))  # 边事件时间：两端节点中较晚的时间戳
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def build_rsr_edges(df, config):
    """
    构建U-S-U边（User-Star-User）

    步骤：
    1. 找出在一周内至少共享一次相同星级评分的用户对
    2. 对这些用户对的所有评论建边
    """
    logger.info("=" * 80)
    logger.info("开始构建 U-S-U 边（User-Star-User）")
    logger.info("说明：一周内至少共享一次相同星级评分的用户之间的评论连接")
    logger.info("=" * 80)
    
    # 参数
    time_window_days = getattr(config, 'USU_TIME_WINDOW_DAYS', 7)  # 一周
    use_extremity_weight = getattr(config, 'USU_EXTREMITY_WEIGHT', True)
    
    logger.info(f"时间窗口: {time_window_days} 天")
    logger.info(f"使用极端性权重: {use_extremity_weight}")
    
    # 一周的毫秒数
    week_ms = time_window_days * 86400 * 1000
    
    # Step 1: 找出在一周内共享相同评分的用户对
    logger.info("Step 1: 查找一周内共享相同评分的用户对...")
    
    # 按评分分组
    rating_groups = df.groupby('overall')
    
    # 记录用户对是否满足条件
    valid_user_pairs = set()
    
    for rating, group in tqdm(rating_groups, desc="分析评分组", unit="评分"):
        # 获取该评分下所有评论，按时间排序
        reviews = group.sort_values('detailed_timestamp')[['user_idx', 'detailed_timestamp']].values
        
        # 滑动窗口找一周内的用户对
        n = len(reviews)
        for i in range(n):
            user_i, ts_i = reviews[i]
            for j in range(i + 1, n):
                user_j, ts_j = reviews[j]
                
                # 如果超出一周窗口，后面的也都会超出
                if ts_j - ts_i > week_ms:
                    break
                
                # 同一用户跳过
                if user_i == user_j:
                    continue
                
                # 记录这对用户（保证顺序一致）
                pair = (min(int(user_i), int(user_j)), max(int(user_i), int(user_j)))
                valid_user_pairs.add(pair)
    
    logger.info(f"找到 {len(valid_user_pairs)} 对满足条件的用户")
    
    if len(valid_user_pairs) == 0:
        logger.warning("没有找到满足条件的用户对，U-S-U边数为0")
        return [], []
    
    # Step 2: 为满足条件的用户对的所有评论建边
    logger.info("Step 2: 为用户对的评论建边...")
    
    # 获取每个用户的所有评论
    user_reviews = df.groupby('user_idx').apply(
        lambda x: list(zip(x['review_idx'].tolist(), x['detailed_timestamp'].tolist()))
    ).to_dict()
    
    edges = []
    edge_attrs = []
    
    for user_i, user_j in tqdm(valid_user_pairs, desc="构建U-S-U边", unit="用户对"):
        reviews_i = user_reviews.get(user_i, [])
        reviews_j = user_reviews.get(user_j, [])
        
        if not reviews_i or not reviews_j:
            continue
        
        # 为两个用户的所有评论建边
        for (r1, t1) in reviews_i:
            for (r2, t2) in reviews_j:
                # 保证边的方向一致（使用局部变量，避免覆盖外层循环变量）
                src, dst = (r1, r2) if r1 < r2 else (r2, r1)
                
                # 计算时间差（天）—— abs 保证顺序无关
                time_diff_days = abs(t2 - t1) / (1000 * 86400)
                
                # 边事件时间 —— max 保证顺序无关
                edge_timestamp = max(t1, t2)
                
                # 权重（可以基于时间接近性）
                weight = 1.0 / (1.0 + time_diff_days / 7.0)  # 一周内权重更高
                
                edges.append((src, dst))
                edge_attrs.append({
                    'edge_type': 'U-S-U',
                    'weight': weight,
                    'time_diff': time_diff_days,
                    'timestamp': edge_timestamp,
                    'user_pair': (user_i, user_j),
                })
    
    logger.info(f"✓ U-S-U边构建完成: {len(edges):,} 条边")
    
    # 统计信息
    if edges:
        weights = [attr['weight'] for attr in edge_attrs]
        logger.info(f"  权重范围: [{min(weights):.4f}, {max(weights):.4f}]")
    
    logger.info("=" * 80)
    
    return edges, edge_attrs


if __name__ == '__main__':
    import pandas as pd
    
    test_data = {
        'review_idx': [0, 1, 2, 3, 4, 5],
        'user_idx': [0, 0, 1, 1, 2, 2],
        'overall': [5, 4, 5, 3, 5, 4],
        'detailed_timestamp': [
            1000000000,
            1000000000 + 86400000 * 2,
            1000000000 + 86400000 * 3,
            1000000000 + 86400000 * 10,
            1000000000 + 86400000 * 5,
            1000000000 + 86400000 * 20,
        ]
    }
    df_test = pd.DataFrame(test_data)
    
    class MockConfig:
        USU_TIME_WINDOW_DAYS = 7
        USU_EXTREMITY_WEIGHT = True
    
    logging.basicConfig(level=logging.INFO)
    edges, attrs = build_rsr_edges(df_test, MockConfig())
    
    print(f"\n生成的边数: {len(edges)}")
    print(f"前几条边: {edges[:5]}")
