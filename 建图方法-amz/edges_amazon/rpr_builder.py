# -*- coding: utf-8 -*-
"""
U-P-U边构建器（User-Product-User）
通过同一产品关联的用户评论之间的连接

边定义：
- 节点：Review（评论）
- 边：同一产品下的不同评论之间建边（完全子图/Clique）

欺诈检测意义：
- 刷单攻击通常针对特定产品，多个水军账号集中给同一产品刷好评/差评
- U-P-U 边捕获这种产品级协同攻击信号

边时间戳定义：
- t_raw(e) = max(t_raw(i), t_raw(j))  # 边事件时间：两端节点中较晚的时间戳
- time_diff = |t_raw(i) - t_raw(j)|   # 时间差：两端节点的时间间隔
"""

import numpy as np
from itertools import combinations
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def build_rpr_edges(df, config):
    """
    构建U-P-U边（User-Product-User）

    对每个产品的所有评论建立完全子图（Clique）
    边权重基于时间衰减: weight = exp(-Δt / τ)

    Parameters
    ----------
    df : pandas.DataFrame
        评论数据，必须包含列:
        - product_idx: 产品索引
        - review_idx: 评论索引
        - detailed_timestamp: 详细时间戳（毫秒）
    config : module
        配置模块

    Returns
    -------
    edges : list of tuple
        边列表 [(review_idx1, review_idx2), ...]
    edge_attrs : list of dict
        边属性列表
    """
    logger.info("=" * 80)
    logger.info("开始构建 U-P-U 边（User-Product-User）")
    logger.info("说明：通过同一产品关联的用户评论之间的连接")
    logger.info("=" * 80)

    edges = []
    edge_attrs = []

    # 时间衰减参数
    time_decay_tau = getattr(config, 'UPU_TIME_DECAY_TAU', 45.0)
    logger.info(f"时间衰减参数 τ: {time_decay_tau} 天")

    # 按产品分组
    product_groups = df.groupby('product_idx')
    total_products = len(product_groups)
    multi_review_products = sum(1 for _, g in product_groups if len(g) >= 2)

    logger.info(f"总产品数: {total_products}")
    logger.info(f"有2+评论的产品数: {multi_review_products}")

    # 遍历每个产品
    for product_idx, group in tqdm(product_groups, desc="构建U-P-U边", unit="产品"):
        # 获取该产品的所有评论
        review_indices = group['review_idx'].tolist()
        timestamps = group['detailed_timestamp'].tolist()  # 毫秒
        
        # 至少需要2条评论才能建边
        if len(review_indices) < 2:
            continue
        
        # 生成所有评论对（完全子图）
        for (r1, t1), (r2, t2) in combinations(zip(review_indices, timestamps), 2):
            # 确保无自环
            if r1 == r2:
                continue
            
            # 保证边的方向一致（r1 < r2），避免重复边
            src, dst = (r1, r2) if r1 < r2 else (r2, r1)
            
            # 计算时间差（天数）：毫秒转天
            time_diff_days = abs(t2 - t1) / (1000 * 86400)
            
            # 计算权重：时间衰减
            weight = np.exp(-time_diff_days / time_decay_tau)
            
            # 边事件时间：max(t_raw(i), t_raw(j))
            edge_timestamp = max(t1, t2)  # 毫秒
            
            # 添加边
            edges.append((src, dst))
            
            # 添加边属性
            edge_attrs.append({
                'edge_type': 'U-P-U',
                'weight': weight,
                'time_diff': time_diff_days,      # 时间差（天）
                'timestamp': edge_timestamp,       # 边事件时间（毫秒）
                'product_idx': product_idx,
            })
    
    logger.info(f"✓ U-P-U边构建完成: {len(edges):,} 条边")
    
    # 统计信息
    if edges:
        weights = [attr['weight'] for attr in edge_attrs]
        time_diffs = [attr['time_diff'] for attr in edge_attrs]
        logger.info(f"  权重范围: [{min(weights):.4f}, {max(weights):.4f}]")
        logger.info(f"  时间差范围: [{min(time_diffs):.2f}, {max(time_diffs):.2f}] 天")
    
    logger.info("=" * 80)
    
    return edges, edge_attrs


if __name__ == '__main__':
    # 测试代码
    import pandas as pd
    
    test_data = {
        'review_idx': [0, 1, 2, 3, 4],
        'product_idx': [0, 0, 0, 1, 1],
        'detailed_timestamp': [
            1000000000, 1000000000 + 86400000,  # 相差1天
            1000000000 + 86400000 * 15,         # 相差15天
            2000000000, 2000000000 + 86400000 * 10
        ]
    }
    df_test = pd.DataFrame(test_data)
    
    class MockConfig:
        UPU_TIME_DECAY_TAU = 45.0
    
    logging.basicConfig(level=logging.INFO)
    edges, attrs = build_rpr_edges(df_test, MockConfig())
    
    print(f"\n生成的边: {edges}")
    print(f"边属性: {attrs}")
