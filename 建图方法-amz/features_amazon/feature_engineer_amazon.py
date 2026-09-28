# -*- coding: utf-8 -*-
"""
特征工程主模块 - Amazon版本
整合所有特征计算，并进行归一化
"""

import numpy as np
import json
import logging
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_amazon as config

from .user_features_amazon import compute_user_features
from .product_features_amazon import compute_product_features
from .review_features_amazon import compute_review_features
from .temporal_features_amazon import compute_temporal_features

logger = logging.getLogger(__name__)


class FeatureEngineerAmazon:
    """特征工程器 - Amazon版本"""
    
    def __init__(self, df):
        """
        初始化
        
        Parameters
        ----------
        df : pandas.DataFrame
            预处理后的评论数据
        """
        self.df = df
        self.features = None
        self.feature_names = None
        
    def build_features(self, include_text=None, normalize=True):
        """
        构建完整特征矩阵
        
        Parameters
        ----------
        include_text : bool, optional
            是否包含文本特征，默认从config读取
        normalize : bool
            是否归一化特征
            
        Returns
        -------
        features : numpy.ndarray
            特征矩阵，形状 (n_reviews, n_features)
        """
        logger.info("="*80)
        logger.info("开始构建完整特征矩阵 (Amazon)")
        logger.info("="*80)
        
        if include_text is None:
            include_text = config.FEATURE_CONFIG['include_text_features']
        
        # 1. 计算各类特征
        user_feats = compute_user_features(self.df)
        product_feats = compute_product_features(self.df)
        review_feats = compute_review_features(self.df, include_text=include_text)
        
        # 时间特征（支持Time2Vec）
        use_time2vec = config.FEATURE_CONFIG.get('use_time2vec', False)
        time2vec_dim = config.FEATURE_CONFIG.get('time2vec_dim', 8)
        temporal_feats = compute_temporal_features(
            self.df, 
            use_time2vec=use_time2vec,
            time2vec_dim=time2vec_dim
        )
        
        # 2. 合并特征
        all_features = np.concatenate([
            user_feats,      # 6维
            product_feats,   # 6维
            review_feats,    # 7维或10维
            temporal_feats,  # 2维或1+time2vec_dim维
        ], axis=1)
        
        # 设置特征名称（动态生成）
        self.feature_names = config.get_feature_names(
            include_text=include_text,
            use_time2vec=use_time2vec,
            time2vec_dim=time2vec_dim
        )
        
        logger.info(f"\n合并后特征形状: {all_features.shape}")
        logger.info(f"特征名称: {self.feature_names}")
        
        # 3. 检查和处理异常值
        all_features = self._handle_invalid_values(all_features)
        
        # 4. 归一化
        if normalize:
            all_features = self._normalize_features(all_features)
        
        self.features = all_features
        
        logger.info("="*80)
        logger.info("特征构建完成！")
        logger.info(f"最终特征形状: {self.features.shape}")
        logger.info("="*80)
        
        return self.features
    
    def _handle_invalid_values(self, features):
        """
        处理NaN和Inf值
        
        Parameters
        ----------
        features : numpy.ndarray
            特征矩阵
            
        Returns
        -------
        features : numpy.ndarray
            处理后的特征矩阵
        """
        logger.info("\n检查异常值...")
        
        # 检查NaN
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            logger.warning(f"发现 {nan_count} 个NaN值，替换为0")
            features = np.nan_to_num(features, nan=0.0)
        else:
            logger.info("  ✓ 无NaN值")
        
        # 检查Inf
        inf_count = np.isinf(features).sum()
        if inf_count > 0:
            logger.warning(f"发现 {inf_count} 个Inf值，替换为最大/最小有限值")
            features = np.nan_to_num(features, posinf=1e6, neginf=-1e6)
        else:
            logger.info("  ✓ 无Inf值")
        
        return features
    
    def _normalize_features(self, features):
        """
        归一化特征（CDF变换）
        
        Parameters
        ----------
        features : numpy.ndarray
            原始特征矩阵
            
        Returns
        -------
        normalized_features : numpy.ndarray
            归一化后的特征矩阵
        """
        logger.info("\n开始归一化特征 (CDF变换)...")
        
        method = config.FEATURE_CONFIG['normalize_method']
        
        if method == 'cdf':
            normalized = self._cdf_normalize(features)
        elif method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized = scaler.fit_transform(features)
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(features)
        else:
            logger.warning(f"未知归一化方法: {method}，跳过归一化")
            normalized = features
        
        logger.info(f"  ✓ 归一化完成 (方法: {method})")
        logger.info(f"  归一化后范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
        
        return normalized
    
    def _cdf_normalize(self, features):
        """
        CDF归一化
        
        将每个特征值转换为其在经验分布中的累积概率
        根据特征的可疑性方向调整
        
        Parameters
        ----------
        features : numpy.ndarray
            原始特征矩阵
            
        Returns
        -------
        normalized : numpy.ndarray
            CDF归一化后的特征矩阵
        """
        normalized = np.zeros_like(features, dtype=np.float32)
        
        for i, feature_name in enumerate(self.feature_names):
            feature_col = features[:, i]
            
            # 获取该特征的可疑性方向
            direction = config.FEATURE_DIRECTIONS.get(feature_name, 'none')
            
            if direction == 'none':
                # 不归一化（如Hour, DayOfWeek, Timestamp_norm）
                normalized[:, i] = feature_col
                continue
            
            # 计算经验CDF
            sorted_values = np.sort(feature_col)
            n = len(sorted_values)
            
            # 对每个值计算其CDF值
            cdf_values = np.searchsorted(sorted_values, feature_col, side='right') / n
            
            # 根据方向调整
            if direction == 'high':
                # 值越大越可疑：CDF直接使用
                normalized[:, i] = cdf_values
            elif direction == 'low':
                # 值越小越可疑：使用1-CDF
                normalized[:, i] = 1.0 - cdf_values
            elif direction == 'extreme':
                # 极端值可疑：距离0.5越远越可疑
                normalized[:, i] = np.abs(cdf_values - 0.5) * 2.0
            
        return normalized
    
    def save_features(self, output_path=None):
        """
        保存特征矩阵
        
        Parameters
        ----------
        output_path : str or Path, optional
            保存路径，默认使用config中的路径
        """
        if output_path is None:
            output_path = config.FEATURES_OUTPUT
        
        # 保存特征矩阵
        np.save(output_path, self.features)
        logger.info(f"特征矩阵已保存到: {output_path}")
        
        # 保存特征统计信息
        stats = self._compute_feature_stats()
        stats_path = config.FEATURES_STATS_OUTPUT
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"特征统计信息已保存到: {stats_path}")
    
    def _compute_feature_stats(self):
        """
        计算特征统计信息
        
        Returns
        -------
        stats : dict
            特征统计字典
        """
        stats = {
            'feature_dim': self.features.shape[1],
            'num_samples': self.features.shape[0],
            'feature_names': self.feature_names,
            'feature_stats': {}
        }
        
        for i, name in enumerate(self.feature_names):
            feature_col = self.features[:, i]
            stats['feature_stats'][name] = {
                'mean': float(feature_col.mean()),
                'std': float(feature_col.std()),
                'min': float(feature_col.min()),
                'max': float(feature_col.max()),
                'median': float(np.median(feature_col)),
            }
        
        return stats


def run_feature_engineering(df, include_text=None, normalize=True):
    """
    运行特征工程流程
    
    Parameters
    ----------
    df : pandas.DataFrame
        预处理后的数据
    include_text : bool, optional
        是否包含文本特征
    normalize : bool
        是否归一化
        
    Returns
    -------
    features : numpy.ndarray
        特征矩阵
    """
    engineer = FeatureEngineerAmazon(df)
    features = engineer.build_features(include_text=include_text, normalize=normalize)
    engineer.save_features()
    
    return features


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from preprocessor_amazon import preprocess_amazon_data
    
    # 运行预处理
    df, mappings = preprocess_amazon_data(config)
    
    # 构建特征
    features = run_feature_engineering(df, include_text=True, normalize=True)
    
    print("\n" + "="*80)
    print("特征工程完成！")
    print(f"特征形状: {features.shape}")
    print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
    print("="*80)
