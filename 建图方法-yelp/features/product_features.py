# -*- coding: utf-8 -*-
"""
产品特征计算模块
计算6个产品级别特征：PMRD, Product_PRR, Product_NRR, Rating_STD, Review_Count, PTIE
"""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from utils.logger import get_logger
import config

logger = get_logger()


def compute_product_features(df):
    """
    计算所有产品特征
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据
        
    Returns
    -------
    product_features : numpy.ndarray
        产品特征矩阵，形状 (n_reviews, 6)
        特征顺序: PMRD, Product_PRR, Product_NRR, Rating_STD, Review_Count, PTIE
    """
    logger.info("="*80)
    logger.info("开始计算产品特征 (6维)")
    logger.info("="*80)
    
    n_reviews = len(df)
    product_features = np.zeros((n_reviews, 6), dtype=np.float32)
    
    # 预计算产品统计信息
    product_stats = _precompute_product_stats(df)
    
    # 对每条评论计算产品特征
    for idx, row in tqdm(df.iterrows(), total=n_reviews, desc="计算产品特征"):
        business_id = row['json_business_id']
        
        # 获取该商家的预计算统计
        stats_dict = product_stats.get(business_id, {})
        
        # 特征0: PMRD - 产品单日最大评论数
        product_features[idx, 0] = stats_dict.get('pmrd', 1)
        
        # 特征1: Product_PRR - 产品正面评价比例
        product_features[idx, 1] = stats_dict.get('prr', 0.5)
        
        # 特征2: Product_NRR - 产品负面评价比例
        product_features[idx, 2] = stats_dict.get('nrr', 0.5)
        
        # 特征3: Rating_STD - 评分标准差
        product_features[idx, 3] = stats_dict.get('rating_std', 0)
        
        # 特征4: Review_Count - 评论总数
        product_features[idx, 4] = stats_dict.get('review_count', 1)
        
        # 特征5: PTIE - 产品时间间隔熵
        product_features[idx, 5] = stats_dict.get('ptie', 0)
    
    logger.info("✓ 产品特征计算完成")
    logger.info(f"  特征形状: {product_features.shape}")
    logger.info("="*80)
    
    return product_features


def _precompute_product_stats(df):
    """
    预计算所有产品的统计信息
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据
        
    Returns
    -------
    product_stats : dict
        产品统计信息字典 {business_id: {stat_name: value}}
    """
    logger.info("预计算产品统计信息...")
    
    product_stats = {}
    business_groups = df.groupby('json_business_id')
    
    for business_id, group in business_groups:
        stats_dict = {}
        
        # 1. PMRD - 产品单日最大评论数
        reviews_per_day = group.groupby('date').size()
        stats_dict['pmrd'] = reviews_per_day.max() if len(reviews_per_day) > 0 else 1
        
        # 2. Product_PRR - 正面评价比例
        total_reviews = len(group)
        positive_reviews = (group['json_stars'] >= 4).sum()
        stats_dict['prr'] = positive_reviews / total_reviews if total_reviews > 0 else 0.5
        
        # 3. Product_NRR - 负面评价比例
        negative_reviews = (group['json_stars'] <= 2).sum()
        stats_dict['nrr'] = negative_reviews / total_reviews if total_reviews > 0 else 0.5
        
        # 4. Rating_STD - 评分标准差
        stats_dict['rating_std'] = group['json_stars'].std() if len(group) > 1 else 0
        
        # 5. Review_Count - 评论总数
        stats_dict['review_count'] = total_reviews
        
        # 6. PTIE - 产品时间间隔熵
        if len(group) > 1:
            sorted_times = group['timestamp'].sort_values()
            intervals = sorted_times.diff().dt.days.dropna().values
            
            if len(intervals) > 0:
                bins = config.INTERVAL_BINS
                hist, _ = np.histogram(intervals, bins=bins)
                probs = hist / hist.sum() if hist.sum() > 0 else hist
                probs = probs[probs > 0]  # 移除0概率
                ptie = -np.sum(probs * np.log(probs + 1e-10))
                stats_dict['ptie'] = ptie
            else:
                stats_dict['ptie'] = 0
        else:
            stats_dict['ptie'] = 0
        
        product_stats[business_id] = stats_dict
    
    logger.info(f"  预计算完成: {len(product_stats)} 个商家")
    
    return product_stats


if __name__ == '__main__':
    # 测试代码
    from utils.logger import setup_logger
    from preprocessor import run_preprocessing
    
    setup_logger(level='INFO')
    
    # 加载测试数据
    df, mappings = run_preprocessing(config.INPUT_CSV)
    
    # 计算产品特征
    product_feats = compute_product_features(df[:1000])
    
    print(f"\n产品特征统计:")
    print(f"  形状: {product_feats.shape}")
    print(f"  均值: {product_feats.mean(axis=0)}")
    print(f"  标准差: {product_feats.std(axis=0)}")

