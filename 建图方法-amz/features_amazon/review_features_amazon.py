# -*- coding: utf-8 -*-
"""
评论特征计算模块 - Amazon版本
计算10个评论级别特征：
基础7维: REX, ARD_review, RL, Hour, DayOfWeek, SRI, RE
文本3维: Mean_Sim, Max_Sim, Sim_STD
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_amazon as config

logger = logging.getLogger(__name__)

# 列名配置
USER_COL = 'reviewerID'
PRODUCT_COL = 'asin'
RATING_COL = 'overall'
TEXT_COL = 'reviewText'


def compute_review_features(df, include_text=True):
    """
    计算所有评论特征
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据
    include_text : bool
        是否包含文本特征
        
    Returns
    -------
    review_features : numpy.ndarray
        评论特征矩阵，形状 (n_reviews, 7) 或 (n_reviews, 10)
    """
    logger.info("="*80)
    if include_text:
        logger.info("开始计算评论特征 (10维: 7基础 + 3文本)")
    else:
        logger.info("开始计算评论特征 (7维: 基础特征)")
    logger.info("="*80)
    
    n_reviews = len(df)
    n_features = 10 if include_text else 7
    review_features = np.zeros((n_reviews, n_features), dtype=np.float32)
    
    # 预计算产品平均评分
    product_avg_ratings = df.groupby(PRODUCT_COL)[RATING_COL].mean().to_dict()
    
    # 预计算用户评论数
    user_review_counts = df.groupby(USER_COL).size().to_dict()
    
    # 对每条评论计算特征
    for idx, row in tqdm(df.iterrows(), total=n_reviews, desc="计算评论特征"):
        # 特征0: REX - 评分极端性
        rex = abs(row[RATING_COL] - 3.0) / 2.0
        review_features[idx, 0] = rex
        
        # 特征1: ARD_review - 绝对评分偏差
        product_id = row[PRODUCT_COL]
        review_idx = row['review_idx']
        
        # 计算该产品的平均评分（排除当前评论）
        product_reviews = df[
            (df[PRODUCT_COL] == product_id) &
            (df['review_idx'] != review_idx)
        ]
        if len(product_reviews) > 0:
            product_avg = product_reviews[RATING_COL].mean()
            ard_review = abs(row[RATING_COL] - product_avg)
        else:
            ard_review = 0
        review_features[idx, 1] = ard_review
        
        # 特征2: RL - 评论长度（字数）
        text = row.get(TEXT_COL, '')
        if pd.notna(text):
            if config.REVIEW_LENGTH_METRIC == 'words':
                review_length = len(str(text).split())
            else:  # chars
                review_length = len(str(text))
        else:
            review_length = 0
        review_features[idx, 2] = review_length
        
        # 特征3: Hour - 提交小时
        review_features[idx, 3] = row['hour']
        
        # 特征4: DayOfWeek - 提交星期
        review_features[idx, 4] = row['dayofweek']
        
        # 特征5: SRI - 单例评论指示器
        user_id = row[USER_COL]
        user_count = user_review_counts.get(user_id, 1)
        sri = 1 if user_count == 1 else 0
        review_features[idx, 5] = sri
        
        # 特征6: RE - 评论排名（时间顺序）
        product_reviews_sorted = df[df[PRODUCT_COL] == product_id].sort_values('timestamp')
        if len(product_reviews_sorted) > 0:
            rank_position = (product_reviews_sorted['review_idx'] == review_idx).to_numpy().nonzero()[0]
            if len(rank_position) > 0:
                re = (rank_position[0] + 1) / len(product_reviews_sorted)
            else:
                re = 0.5
        else:
            re = 0.5
        review_features[idx, 6] = re
    
    # 文本特征（如果需要）
    if include_text:
        logger.info("计算文本相似度特征...")
        text_features = _compute_text_features(df)
        review_features[:, 7:10] = text_features
    
    logger.info("✓ 评论特征计算完成")
    logger.info(f"  特征形状: {review_features.shape}")
    logger.info("="*80)
    
    return review_features


def _compute_text_features(df):
    """
    计算文本相似度特征
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据
        
    Returns
    -------
    text_features : numpy.ndarray
        文本特征矩阵，形状 (n_reviews, 3)
        特征: Mean_Sim, Max_Sim, Sim_STD
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    n_reviews = len(df)
    text_features = np.zeros((n_reviews, 3), dtype=np.float32)
    
    # 按产品分组计算相似度
    product_groups = df.groupby(PRODUCT_COL)
    
    for product_id, group in tqdm(product_groups, desc="计算文本相似度"):
        if len(group) < 2:
            # 只有1条评论，相似度为0
            continue
        
        # 提取文本
        texts = group[TEXT_COL].fillna('').tolist()
        indices = group.index.tolist()
        
        try:
            # TF-IDF向量化
            vectorizer = TfidfVectorizer(
                max_features=config.TFIDF_CONFIG['max_features'],
                min_df=1,  # 对于小数据集，min_df设为1
                max_df=0.95,
                ngram_range=config.TFIDF_CONFIG['ngram_range'],
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 计算余弦相似度矩阵
            sim_matrix = cosine_similarity(tfidf_matrix)
            
            # 对每条评论计算相似度统计
            for i, idx in enumerate(indices):
                # 获取当前评论与其他评论的相似度
                similarities = sim_matrix[i]
                # 排除自己（对角线元素）
                other_sims = np.concatenate([similarities[:i], similarities[i+1:]])
                
                if len(other_sims) > 0:
                    text_features[idx, 0] = other_sims.mean()  # Mean_Sim
                    text_features[idx, 1] = other_sims.max()   # Max_Sim
                    text_features[idx, 2] = other_sims.std()   # Sim_STD
                else:
                    text_features[idx, :] = 0
        
        except Exception as e:
            logger.warning(f"产品 {product_id} 文本特征计算失败: {e}")
            for idx in indices:
                text_features[idx, :] = 0
    
    logger.info("  ✓ 文本特征计算完成")
    
    return text_features


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    from preprocessor_amazon import preprocess_amazon_data
    
    # 加载测试数据
    df, mappings = preprocess_amazon_data(config)
    
    # 计算评论特征
    review_feats = compute_review_features(df[:1000], include_text=False)
    
    print(f"\n评论特征统计:")
    print(f"  形状: {review_feats.shape}")
    print(f"  均值: {review_feats.mean(axis=0)}")
    print(f"  标准差: {review_feats.std(axis=0)}")
