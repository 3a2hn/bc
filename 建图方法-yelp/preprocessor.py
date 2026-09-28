# -*- coding: utf-8 -*-
"""
Yelp数据预处理器
负责加载和预处理Yelp评论数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import get_logger
from utils.data_loader import save_pickle
import config


def load_data(csv_path):
    """
    加载CSV数据
    
    Parameters
    ----------
    csv_path : str or Path
        CSV文件路径
        
    Returns
    -------
    df : pandas.DataFrame
        加载的数据
    """
    logger = get_logger()
    logger.info(f"正在加载数据: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"成功加载 {len(df):,} 条评论")
    
    # 打印基本统计
    logger.info(f"用户数: {df['json_user_id'].nunique():,}")
    logger.info(f"商家数: {df['json_business_id'].nunique():,}")
    
    if 'label' in df.columns:
        logger.info(f"标签分布:\n{df['label'].value_counts()}")
    
    return df


def clean_data(df):
    """
    清洗数据
    
    Parameters
    ----------
    df : pandas.DataFrame
        原始数据
        
    Returns
    -------
    df : pandas.DataFrame
        清洗后的数据
    """
    logger = get_logger()
    logger.info("开始数据清洗...")
    
    initial_count = len(df)
    
    # 重命名列：只重命名json_text为text，保留其他json_前缀的列名
    # 因为edges和features模块都使用json_stars, json_user_id, json_business_id
    if 'json_text' in df.columns:
        df = df.rename(columns={'json_text': 'text'})
    
    # 1. 删除缺失值
    required_columns = ['json_user_id', 'json_business_id', 'text', 
                       'json_stars', 'timestamp', 'label']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")
    
    df = df.dropna(subset=required_columns)
    logger.info(f"删除缺失值后: {len(df):,} 条 (删除 {initial_count - len(df):,} 条)")
    
    # 2. 删除重复评论
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=['json_user_id', 'json_business_id', 'text'],
        keep='first'
    )
    logger.info(f"删除重复评论后: {len(df):,} 条 (删除 {before_dedup - len(df):,} 条)")
    
    # 3. 验证评分范围
    df = df[df['json_stars'].isin([1, 2, 3, 4, 5])]
    
    # 4. 处理标签（可能是1.0/-1.0，需要转换为1/0）
    # 假设 1.0 = 正常评论, -1.0 = 欺诈评论
    if df['label'].dtype == float:
        df['label'] = df['label'].apply(lambda x: 1 if x == 1.0 else 0)
    
    # 验证标签
    df = df[df['label'].isin([0, 1])]
    
    logger.info(f"数据清洗完成，保留 {len(df):,} 条评论")
    
    return df


def parse_timestamps(df):
    """
    解析和标准化时间戳
    
    Parameters
    ----------
    df : pandas.DataFrame
        数据
        
    Returns
    -------
    df : pandas.DataFrame
        添加时间特征后的数据
    """
    logger = get_logger()
    logger.info("解析时间戳...")
    
    # 确保timestamp是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 提取时间特征
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # 创建日期列（不含时间，用于用户特征计算）
    df['date'] = df['timestamp'].dt.date
    
    # 创建年月列（用于R-T-R边构建）
    df['year_month'] = df['timestamp'].dt.to_period('M')
    
    # 计算Unix时间戳（秒）
    df['date_unix'] = df['timestamp'].astype(np.int64) // 10**9
    
    # 计算归一化时间戳 [0, 1]
    min_ts = df['date_unix'].min()
    max_ts = df['date_unix'].max()
    df['timestamp_norm'] = (df['date_unix'] - min_ts) / (max_ts - min_ts)
    
    logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    logger.info(f"时间跨度: {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
    
    return df


def create_mappings(df):
    """
    创建ID映射
    
    Parameters
    ----------
    df : pandas.DataFrame
        数据
        
    Returns
    -------
    df : pandas.DataFrame
        添加索引列后的数据
    mappings : dict
        ID映射字典
    """
    logger = get_logger()
    logger.info("创建ID映射...")
    
    # 用户ID映射
    unique_users = sorted(df['json_user_id'].unique())
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    df['user_idx'] = df['json_user_id'].map(user_to_idx)
    
    # 商家ID映射
    unique_businesses = sorted(df['json_business_id'].unique())
    business_to_idx = {biz: idx for idx, biz in enumerate(unique_businesses)}
    df['business_idx'] = df['json_business_id'].map(business_to_idx)
    
    # 评论索引（按时间排序后的索引）
    df = df.sort_values('date_unix').reset_index(drop=True)
    df['review_idx'] = df.index
    
    logger.info(f"映射创建完成:")
    logger.info(f"  - {len(user_to_idx):,} 个用户")
    logger.info(f"  - {len(business_to_idx):,} 个商家")
    logger.info(f"  - {len(df):,} 条评论")
    
    mappings = {
        'user_to_idx': user_to_idx,
        'business_to_idx': business_to_idx,
        'idx_to_user': {idx: user for user, idx in user_to_idx.items()},
        'idx_to_business': {idx: biz for biz, idx in business_to_idx.items()},
    }
    
    return df, mappings


def print_statistics(df):
    """
    打印数据统计信息
    
    Parameters
    ----------
    df : pandas.DataFrame
        数据
    """
    logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("数据统计")
    logger.info("=" * 80)
    logger.info(f"总评论数: {len(df):,}")
    logger.info(f"用户数: {df['user_idx'].nunique():,}")
    logger.info(f"商家数: {df['business_idx'].nunique():,}")
    logger.info(f"平均每用户评论数: {len(df) / df['user_idx'].nunique():.2f}")
    logger.info(f"平均每商家评论数: {len(df) / df['business_idx'].nunique():.2f}")
    
    # 标签分布
    fraud_count = (df['label'] == 0).sum()
    real_count = (df['label'] == 1).sum()
    logger.info(f"\n标签分布:")
    logger.info(f"  欺诈评论 (label=0): {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    logger.info(f"  正常评论 (label=1): {real_count:,} ({real_count/len(df)*100:.2f}%)")
    
    # 评分分布
    logger.info(f"\n评分分布:")
    for star in sorted(df['json_stars'].unique()):
        count = (df['json_stars'] == star).sum()
        logger.info(f"  {star}星: {count:,} ({count/len(df)*100:.2f}%)")
    
    logger.info("=" * 80)


def run_preprocessing(csv_path):
    """
    执行完整的预处理流程
    
    Parameters
    ----------
    csv_path : str or Path
        输入CSV文件路径
        
    Returns
    -------
    df : pandas.DataFrame
        处理后的数据
    mappings : dict
        ID映射字典
    """
    logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("开始数据预处理")
    logger.info("=" * 80)
    
    # 1. 加载数据
    df = load_data(csv_path)
    
    # 2. 清洗数据
    df = clean_data(df)
    
    # 3. 解析时间戳
    df = parse_timestamps(df)
    
    # 4. 创建ID映射
    df, mappings = create_mappings(df)
    
    # 5. 打印统计信息
    print_statistics(df)
    
    # 6. 保存映射
    save_pickle(mappings, config.MAPPINGS_OUTPUT)
    logger.info(f"ID映射已保存到: {config.MAPPINGS_OUTPUT}")
    
    logger.info("=" * 80)
    logger.info("预处理完成！")
    logger.info("=" * 80)
    
    return df, mappings


if __name__ == '__main__':
    # 测试预处理器
    from utils.logger import setup_logger
    
    # 设置日志
    setup_logger(log_file=config.LOG_FILE, level='INFO')
    
    # 运行预处理
    df, mappings = run_preprocessing(config.INPUT_CSV)
    
    print("\n数据预览:")
    print(df.head())
    print("\n数据列:")
    print(df.columns.tolist())
