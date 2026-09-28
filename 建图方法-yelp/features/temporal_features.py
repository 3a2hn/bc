# -*- coding: utf-8 -*-
"""
时间特征计算模块
计算时间特征：TSLR + Time2Vec编码
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.logger import get_logger
from .time2vec import Time2VecEncoder, initialize_time2vec_with_frequencies

logger = get_logger()


def compute_temporal_features(df, use_time2vec=True, time2vec_dim=64):
    """
    计算所有时间特征
    
    Parameters
    ----------
    df : pandas.DataFrame
        评论数据
    use_time2vec : bool
        是否使用Time2Vec编码（默认True）
    time2vec_dim : int
        Time2Vec输出维度（默认64：1线性 + 63周期）
        
    Returns
    -------
    temporal_features : numpy.ndarray
        时间特征矩阵
        - 如果use_time2vec=True: 形状 (n_reviews, 1+time2vec_dim)
          特征顺序: TSLR, Time2Vec[0], Time2Vec[1], ..., Time2Vec[k-1]
        - 如果use_time2vec=False: 形状 (n_reviews, 2)
          特征顺序: TSLR, Timestamp_norm
    """
    if use_time2vec:
        logger.info("="*80)
        logger.info(f"开始计算时间特征 (1+{time2vec_dim}维: TSLR + Time2Vec)")
        logger.info(f"  Time2Vec维度分解: 1线性 + {time2vec_dim-1}周期 = {time2vec_dim}维")
        logger.info("="*80)
    else:
        logger.info("="*80)
        logger.info("开始计算时间特征 (2维: TSLR + Timestamp_norm)")
        logger.info("="*80)
    
    n_reviews = len(df)
    
    # 计算全局时间范围
    min_timestamp = df['timestamp'].min()
    max_timestamp = df['timestamp'].max()
    time_range = (max_timestamp - min_timestamp).total_seconds()
    
    if time_range == 0:
        time_range = 1  # 防止除零
    
    logger.info(f"时间范围: {min_timestamp.date()} 到 {max_timestamp.date()}")
    
    # 预先计算归一化时间戳（用于Time2Vec或直接使用）
    timestamps_norm = ((df['timestamp'] - min_timestamp).dt.total_seconds() / time_range).values
    
    # 按用户分组，计算每个用户的评论时间序列
    user_groups = df.groupby('json_user_id')
    user_timelines = {}
    
    for user_id, group in user_groups:
        sorted_reviews = group.sort_values('timestamp')
        user_timelines[user_id] = {
            'review_indices': sorted_reviews.index.tolist(),
            'timestamps': sorted_reviews['timestamp'].tolist(),
        }
    
    # 计算TSLR特征
    logger.info("计算TSLR特征（自上次评论的时间）...")
    tslr_features = np.zeros(n_reviews, dtype=np.float32)
    
    for idx, row in tqdm(df.iterrows(), total=n_reviews, desc="计算TSLR"):
        user_id = row['json_user_id']
        current_time = row['timestamp']
        
        # TSLR - 自上次评论的时间（天数）
        timeline = user_timelines.get(user_id, {})
        review_indices = timeline.get('review_indices', [])
        timestamps = timeline.get('timestamps', [])
        
        if idx in review_indices:
            position = review_indices.index(idx)
            if position > 0:
                # 有前一条评论
                prev_time = timestamps[position - 1]
                tslr = (current_time - prev_time).days
            else:
                # 第一条评论
                tslr = 0
        else:
            tslr = 0
        
        tslr_features[idx] = tslr
    
    # 根据配置选择Time2Vec或简单归一化
    if use_time2vec:
        # 使用Time2Vec编码时间戳
        logger.info(f"使用Time2Vec编码时间戳（维度={time2vec_dim}）...")
        encoder = Time2VecEncoder(out_features=time2vec_dim, use_cos=False)
        
        # 可选：使用预定义频率初始化（捕捉日/周/月周期）
        initialize_time2vec_with_frequencies(encoder, frequencies=[365, 52, 12, 4, 1])
        
        # 编码
        time_vectors = encoder.encode(timestamps_norm)  # (n_reviews, time2vec_dim)
        
        # 合并TSLR和Time2Vec
        temporal_features = np.column_stack([
            tslr_features.reshape(-1, 1),  # TSLR (1维)
            time_vectors                    # Time2Vec (time2vec_dim维)
        ])
        
        logger.info("✓ 时间特征计算完成（含Time2Vec）")
        logger.info(f"  特征形状: {temporal_features.shape}")
        logger.info(f"  TSLR 统计: 均值={tslr_features.mean():.2f}天, "
                   f"最大={tslr_features.max():.2f}天")
        logger.info(f"  Time2Vec 范围: [{time_vectors.min():.4f}, {time_vectors.max():.4f}]")
    else:
        # 使用简单的归一化时间戳
        temporal_features = np.column_stack([
            tslr_features.reshape(-1, 1),      # TSLR (1维)
            timestamps_norm.reshape(-1, 1)     # Timestamp_norm (1维)
        ])
        
        logger.info("✓ 时间特征计算完成（传统方法）")
        logger.info(f"  特征形状: {temporal_features.shape}")
        logger.info(f"  TSLR 统计: 均值={tslr_features.mean():.2f}天, "
                   f"最大={tslr_features.max():.2f}天")
        logger.info(f"  Timestamp_norm 范围: [{timestamps_norm.min():.4f}, "
                   f"{timestamps_norm.max():.4f}]")
    
    logger.info("="*80)
    
    return temporal_features


if __name__ == '__main__':
    # 测试代码
    from utils.logger import setup_logger
    from preprocessor import run_preprocessing
    import config
    
    setup_logger(level='INFO')
    
    # 加载测试数据
    df, mappings = run_preprocessing(config.INPUT_CSV)
    
    # 计算时间特征
    temporal_feats = compute_temporal_features(df[:1000])
    
    print(f"\n时间特征统计:")
    print(f"  形状: {temporal_feats.shape}")
    print(f"  均值: {temporal_feats.mean(axis=0)}")
    print(f"  标准差: {temporal_feats.std(axis=0)}")

