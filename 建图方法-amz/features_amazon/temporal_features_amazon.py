# -*- coding: utf-8 -*-
"""
时间特征计算模块 - Amazon版本
计算时间特征：TSLR + Time2Vec编码
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


class Time2VecEncoder:
    """
    Time2Vec编码器
    
    将标量时间值转换为向量表示
    t → [w0*t + b0, sin(w1*t + b1), ..., sin(wk*t + bk)]
    """
    
    def __init__(self, out_features=64, use_cos=False):
        """
        Parameters
        ----------
        out_features : int
            输出维度（包含1个线性维度和k-1个周期维度）
        use_cos : bool
            是否使用cos替代sin（默认False）
        """
        self.out_features = out_features
        self.use_cos = use_cos
        
        # 初始化权重（可学习，但这里用固定值）
        np.random.seed(42)
        self.w = np.random.randn(out_features) * 0.1
        self.b = np.random.randn(out_features) * 0.1
        
        # 第一个维度使用线性变换
        self.w[0] = 1.0
        self.b[0] = 0.0
    
    def encode(self, t):
        """
        编码时间戳
        
        Parameters
        ----------
        t : numpy.ndarray
            归一化时间戳，形状 (n,)
            
        Returns
        -------
        time_vectors : numpy.ndarray
            时间向量，形状 (n, out_features)
        """
        t = np.asarray(t).reshape(-1, 1)  # (n, 1)
        
        # 计算 w*t + b
        x = t @ self.w.reshape(1, -1) + self.b  # (n, out_features)
        
        # 第一维线性，其余周期
        output = np.zeros_like(x)
        output[:, 0] = x[:, 0]  # 线性
        
        if self.use_cos:
            output[:, 1:] = np.cos(x[:, 1:])
        else:
            output[:, 1:] = np.sin(x[:, 1:])
        
        return output


def initialize_time2vec_with_frequencies(encoder, frequencies=None):
    """
    用预定义频率初始化Time2Vec编码器
    
    Parameters
    ----------
    encoder : Time2VecEncoder
        编码器实例
    frequencies : list
        频率列表（天数单位），如 [365, 52, 12, 4, 1] 代表年/周数/月/季度/周
    """
    if frequencies is None:
        frequencies = [365, 52, 12, 4, 1]
    
    # 前几个维度使用预定义频率
    n_predefined = min(len(frequencies), encoder.out_features - 1)
    
    for i, freq in enumerate(frequencies[:n_predefined]):
        # 将频率转换为弧度频率
        encoder.w[i + 1] = 2 * np.pi / freq
        encoder.b[i + 1] = 0


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
    user_groups = df.groupby(USER_COL)
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
        user_id = row[USER_COL]
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
    logging.basicConfig(level=logging.INFO)
    
    from preprocessor_amazon import preprocess_amazon_data
    
    # 加载测试数据
    df, mappings = preprocess_amazon_data(config)
    
    # 计算时间特征
    temporal_feats = compute_temporal_features(df[:1000])
    
    print(f"\n时间特征统计:")
    print(f"  形状: {temporal_feats.shape}")
    print(f"  均值: {temporal_feats.mean(axis=0)[:5]}...")  # 只显示前5个
    print(f"  标准差: {temporal_feats.std(axis=0)[:5]}...")
