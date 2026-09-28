# -*- coding: utf-8 -*-
"""
U-V-U边构建器（User-Vocabulary-User）
评论文本相似度（基于TF-IDF计算）位于所有用户前5%的用户之间的评论连接

边定义：
- 节点：Review（评论）
- 边：计算用户级别的文本相似度（基于该用户所有评论的TF-IDF向量），
      选取相似度前5%的用户对，然后为这些用户对的所有评论建边

边时间戳定义：
- t_raw(e) = max(t_raw(i), t_raw(j))  # 边事件时间：两端节点中较晚的时间戳
"""

import numpy as np
import heapq
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def build_rvr_edges(df, config):
    """
    构建U-V-U边（User-Vocabulary-User）

    步骤：
    1. 为每个用户计算聚合的TF-IDF向量（合并该用户所有评论文本）
    2. 分批计算用户间的文本相似度
    3. 选取相似度前5%的用户对（带最低相似度阈值）
    4. 为这些用户对的所有评论建边
    """
    logger.info("=" * 80)
    logger.info("开始构建 U-V-U 边（User-Vocabulary-User）")
    logger.info("说明：评论文本相似度（TF-IDF）位于所有用户前5%的用户之间的评论连接")
    logger.info("=" * 80)
    
    # 参数
    top_percent = getattr(config, 'UVU_TOP_PERCENT', 0.05)
    min_similarity = getattr(config, 'UVU_MIN_SIMILARITY', 0.10)
    tfidf_config = getattr(config, 'TFIDF_CONFIG', {
        'max_features': 5000,
        'min_df': 2,
        'max_df': 0.95,
        'ngram_range': (1, 2),
    })
    
    logger.info(f"选取相似度前 {top_percent * 100:.2f}% 的用户对")
    logger.info(f"最低相似度阈值: {min_similarity}")
    
    # ========== Step 1: 聚合每个用户的评论文本 ==========
    logger.info("Step 1: 聚合用户评论文本...")
    
    text_col = 'reviewText' if 'reviewText' in df.columns else 'text'
    
    user_texts = df.groupby('user_idx')[text_col].apply(
        lambda x: ' '.join(str(t) for t in x if pd.notna(t))
    )
    
    user_indices = np.array(user_texts.index.tolist())
    texts = user_texts.values.tolist()
    n_users = len(user_indices)
    
    logger.info(f"用户数: {n_users}")
    
    if n_users < 2:
        logger.warning("用户数不足，无法构建U-V-U边")
        return [], []
    
    # ========== Step 2: 计算TF-IDF向量 ==========
    logger.info("Step 2: 计算TF-IDF向量...")
    
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config.get('max_features', 5000),
        min_df=tfidf_config.get('min_df', 2),
        max_df=tfidf_config.get('max_df', 0.95),
        ngram_range=tfidf_config.get('ngram_range', (1, 2)),
        stop_words='english'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF矩阵形状: {tfidf_matrix.shape}")
    except Exception as e:
        logger.error(f"TF-IDF计算失败: {e}")
        return [], []
    
    # ========== Step 3: 分批计算相似度，找出Top-K用户对 ==========
    logger.info("Step 3: 分批计算用户间文本相似度...")
    
    # 计算需要选取的用户对数量
    total_pairs = n_users * (n_users - 1) // 2
    top_k = max(1, int(total_pairs * top_percent))
    logger.info(f"总用户对数: {total_pairs:,}，目标选取: {top_k:,} 对")
    
    # 分批计算，避免生成完整 N×N 矩阵
    batch_size = min(2000, n_users)
    
    # 使用最小堆维护 top_k 个最大相似度对
    top_heap = []
    valid_pairs_above_threshold = 0
    
    for batch_start in tqdm(range(0, n_users, batch_size), desc="分批计算相似度", unit="批"):
        batch_end = min(batch_start + batch_size, n_users)
        
        # 计算当前批次与所有用户的相似度
        sim_chunk = cosine_similarity(tfidf_matrix[batch_start:batch_end], tfidf_matrix)
        
        # 只处理上三角部分（j > i），避免重复
        for local_i in range(batch_end - batch_start):
            global_i = batch_start + local_i
            j_start = global_i + 1
            if j_start >= n_users:
                continue
            
            row_sims = sim_chunk[local_i, j_start:]
            
            # 按阈值过滤
            mask = row_sims >= min_similarity
            valid_pairs_above_threshold += mask.sum()
            
            if not mask.any():
                continue
            
            filtered_js = np.where(mask)[0] + j_start
            filtered_sims = row_sims[mask]
            
            # 加入堆
            for j_idx, sim_val in zip(filtered_js, filtered_sims):
                sim_val = float(sim_val)
                if len(top_heap) < top_k:
                    heapq.heappush(top_heap, (sim_val, int(global_i), int(j_idx)))
                elif sim_val > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (sim_val, int(global_i), int(j_idx)))
        
        del sim_chunk
    
    logger.info(f"满足阈值 (>={min_similarity}) 的用户对数: {valid_pairs_above_threshold:,}")
    logger.info(f"堆中保留的 Top-K 用户对数: {len(top_heap):,}")
    
    if len(top_heap) == 0:
        logger.warning("没有找到满足条件的用户对，U-V-U边数为0")
        return [], []
    
    # 提取结果
    top_pairs = [(user_indices[i], user_indices[j], sim) for sim, i, j in top_heap]
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"选取的用户对数: {len(top_pairs):,}")
    logger.info(f"相似度范围: [{top_pairs[-1][2]:.4f}, {top_pairs[0][2]:.4f}]")
    
    # 创建有效用户对字典
    valid_user_pairs = {(min(p[0], p[1]), max(p[0], p[1])): p[2] for p in top_pairs}
    
    # ========== Step 4: 为用户对的评论建边 ==========
    logger.info("Step 4: 为用户对的评论建边...")
    
    # 获取每个用户的所有评论
    user_reviews = df.groupby('user_idx').apply(
        lambda x: list(zip(x['review_idx'].tolist(), x['detailed_timestamp'].tolist()))
    ).to_dict()
    
    edges = []
    edge_attrs = []
    
    for (user_i, user_j), sim in tqdm(valid_user_pairs.items(), desc="构建U-V-U边", unit="用户对"):
        reviews_i = user_reviews.get(user_i, [])
        reviews_j = user_reviews.get(user_j, [])
        
        if not reviews_i or not reviews_j:
            continue
        
        # 为两个用户的所有评论建边
        for (r1, t1) in reviews_i:
            for (r2, t2) in reviews_j:
                # 保证边的方向一致（使用局部变量，避免覆盖外层循环变量）
                src, dst = (r1, r2) if r1 < r2 else (r2, r1)
                
                # 计算时间差（天）
                time_diff_days = abs(t2 - t1) / (1000 * 86400)
                
                # 边事件时间
                edge_timestamp = max(t1, t2)
                
                # 权重基于文本相似度
                weight = sim
                
                edges.append((src, dst))
                edge_attrs.append({
                    'edge_type': 'U-V-U',
                    'weight': weight,
                    'similarity': sim,
                    'time_diff': time_diff_days,
                    'timestamp': edge_timestamp,
                    'user_pair': (user_i, user_j),
                })
    
    logger.info(f"✓ U-V-U边构建完成: {len(edges):,} 条边")
    
    if edges:
        weights = [attr['weight'] for attr in edge_attrs]
        logger.info(f"  权重范围: [{min(weights):.4f}, {max(weights):.4f}]")
    
    logger.info("=" * 80)
    
    return edges, edge_attrs


if __name__ == '__main__':
    test_data = {
        'review_idx': [0, 1, 2, 3, 4, 5],
        'user_idx': [0, 0, 1, 1, 2, 2],
        'reviewText': [
            "This product is excellent, very good quality",
            "Great item, highly recommended",
            "Excellent quality, amazing product",
            "Good value for money",
            "This is amazing, excellent quality product",
            "Not bad, decent product",
        ],
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
        UVU_TOP_PERCENT = 0.5
        UVU_MIN_SIMILARITY = 0.0
        TFIDF_CONFIG = {
            'max_features': 1000,
            'min_df': 1,
            'max_df': 0.95,
            'ngram_range': (1, 1),
        }
    
    logging.basicConfig(level=logging.INFO)
    edges, attrs = build_rvr_edges(df_test, MockConfig())
    
    print(f"\n生成的边数: {len(edges)}")
    if edges:
        print(f"前几条边: {edges[:5]}")
        print(f"边属性示例: {attrs[0]}")
