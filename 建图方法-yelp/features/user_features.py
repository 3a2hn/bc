# -*- coding: utf-8 -*-
"""
用户特征计算模块
计算6个用户级别特征：MRD, ARI, RIE, PRR, NRR, ARD
"""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from utils.logger import get_logger
import config

logger = get_logger()


def compute_user_features(df):
    """
    计算所有用户特征
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据
        
    Returns
    -------
    user_features : numpy.ndarray
        用户特征矩阵，形状 (n_reviews, 6)
        特征顺序: MRD, ARI, RIE, PRR, NRR, ARD
    """
    logger.info("="*80)
    logger.info("开始计算用户特征 (6维)")
    logger.info("="*80)
    
    n_reviews = len(df)
    user_features = np.zeros((n_reviews, 6), dtype=np.float32)
    
    # 预计算用户统计信息（加速）
    user_stats = _precompute_user_stats(df)
    
    # 对每条评论计算用户特征
    for idx, row in tqdm(df.iterrows(), total=n_reviews, desc="计算用户特征"):
        user_id = row['json_user_id']
        
        # 获取该用户的预计算统计
        stats_dict = user_stats.get(user_id, {})
        
        # 特征0: MRD - 单日最大评论数
        user_features[idx, 0] = stats_dict.get('mrd', 1)
        
        # 特征1: ARI - 平均评论间隔
        user_features[idx, 1] = stats_dict.get('ari', 0)
        
        # 特征2: RIE - 评论间隔熵
        user_features[idx, 2] = stats_dict.get('rie', 0)
        
        # 特征3: PRR - 正面评价比例
        user_features[idx, 3] = stats_dict.get('prr', 0.5)
        
        # 特征4: NRR - 负面评价比例
        user_features[idx, 4] = stats_dict.get('nrr', 0.5)
        
        # 特征5: ARD - 平均评分偏差
        user_features[idx, 5] = _compute_ard_for_review(df, row)
    
    logger.info("✓ 用户特征计算完成")
    logger.info(f"  特征形状: {user_features.shape}")
    logger.info("="*80)
    
    return user_features


def _precompute_user_stats(df):
    """
    预计算所有用户的统计信息
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据
        
    Returns
    -------
    user_stats : dict
        用户统计信息字典 {user_id: {stat_name: value}}
    """
    logger.info("预计算用户统计信息...")
    
    user_stats = {}
    user_groups = df.groupby('json_user_id')
    
    for user_id, group in user_groups:
        stats_dict = {}
        
        # 1. MRD - 单日最大评论数
        reviews_per_day = group.groupby('date').size()
        stats_dict['mrd'] = reviews_per_day.max() if len(reviews_per_day) > 0 else 1
        
        # 2 & 3. ARI, RIE - 评论间隔相关
        if len(group) > 1:
            sorted_times = group['timestamp'].sort_values()
            intervals = sorted_times.diff().dt.days.dropna().values
            
            # ARI - 平均间隔
            stats_dict['ari'] = intervals.mean() if len(intervals) > 0 else 0
            
            # RIE - 间隔熵
            if len(intervals) > 0:
                bins = config.INTERVAL_BINS
                hist, _ = np.histogram(intervals, bins=bins)
                probs = hist / hist.sum() if hist.sum() > 0 else hist
                probs = probs[probs > 0]  # 移除0概率
                rie = -np.sum(probs * np.log(probs + 1e-10))
                stats_dict['rie'] = rie
            else:
                stats_dict['rie'] = 0
        else:
            stats_dict['ari'] = 0
            stats_dict['rie'] = 0
        
        # 4. PRR - 正面评价比例
        total_reviews = len(group)
        positive_reviews = (group['json_stars'] >= 4).sum()
        stats_dict['prr'] = positive_reviews / total_reviews if total_reviews > 0 else 0.5
        
        # 5. NRR - 负面评价比例
        negative_reviews = (group['json_stars'] <= 2).sum()
        stats_dict['nrr'] = negative_reviews / total_reviews if total_reviews > 0 else 0.5
        
        user_stats[user_id] = stats_dict
    
    logger.info(f"  预计算完成: {len(user_stats)} 个用户")
    
    return user_stats


def _compute_ard_for_review(df, current_review):
    """
    计算单条评论的ARD（平均评分偏差）
    
    Parameters
    ----------
    df : pandas.DataFrame
        全部评论数据
    current_review : pandas.Series
        当前评论
        
    Returns
    -------
    ard : float
        平均评分偏差
    """
    user_id = current_review['json_user_id']
    current_review_idx = current_review['review_idx']
    
    # 获取该用户的所有评论
    user_reviews = df[df['json_user_id'] == user_id]
    
    deviations = []
    
    for _, review in user_reviews.iterrows():
        business_id = review['json_business_id']
        user_rating = review['json_stars']
        review_idx = review['review_idx']
        
        # 计算该商家的平均评分（排除当前评论）
        business_reviews = df[
            (df['json_business_id'] == business_id) &
            (df['review_idx'] != review_idx)
        ]
        
        if len(business_reviews) > 0:
            business_avg = business_reviews['json_stars'].mean()
            deviation = abs(user_rating - business_avg)
            deviations.append(deviation)
    
    # 返回平均偏差
    if len(deviations) > 0:
        return np.mean(deviations)
    else:
        return 0.0


if __name__ == '__main__':
    # 测试代码
    from utils.logger import setup_logger
    from preprocessor import run_preprocessing
    
    setup_logger(level='INFO')
    
    # 加载测试数据
    df, mappings = run_preprocessing(config.INPUT_CSV)
    
    # 计算用户特征
    user_feats = compute_user_features(df[:1000])  # 测试前1000条
    
    print(f"\n用户特征统计:")
    print(f"  形状: {user_feats.shape}")
    print(f"  均值: {user_feats.mean(axis=0)}")
    print(f"  标准差: {user_feats.std(axis=0)}")

