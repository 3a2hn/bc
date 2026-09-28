# -*- coding: utf-8 -*-
"""
配置文件
包含所有路径、参数和超参数配置
"""

import os
from pathlib import Path

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_CSV = DATA_DIR / 'matched_reviews_40k_cleaned.csv'  # 清洗后的数据

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# 图数据输出
GRAPH_OUTPUT = OUTPUT_DIR / 'graph.pkl'
EDGES_OUTPUT_DIR = OUTPUT_DIR / 'edges'
EDGES_OUTPUT_DIR.mkdir(exist_ok=True)

# 特征输出
FEATURES_OUTPUT = OUTPUT_DIR / 'features.npy'
FEATURES_STATS_OUTPUT = OUTPUT_DIR / 'feature_statistics.json'

# 其他输出
LABELS_OUTPUT = OUTPUT_DIR / 'labels.npy'
SPLITS_OUTPUT = OUTPUT_DIR / 'splits.pkl'
MAPPINGS_OUTPUT = OUTPUT_DIR / 'id_mappings.pkl'
STATISTICS_OUTPUT = OUTPUT_DIR / 'graph_statistics.json'

# 日志路径
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / 'build_graph.log'

# ============================================================================
# 图构建参数
# ============================================================================

# R-U-R 边参数（Review-User-Review）
RUR_TIME_DECAY_TAU = 45.0  # 时间衰减参数（天）

# R-T-R 边参数（Review-Temporal-Review）
RTR_TIME_WINDOW = 'M'  # 时间窗口：'M'=月, 'W'=周, 'D'=日
RTR_TIME_CONCENTRATION_SCALE = 7.0  # 时间集中度权重参数（天）

# R-S-R 边参数（Review-Star-Review）
RSR_EXTREMITY_WEIGHT = True  # 是否使用极端性权重

# ============================================================================
# 特征工程参数
# ============================================================================

# 特征配置
FEATURE_CONFIG = {
    'include_text_features': True,  # 是否包含文本特征（Mean_Sim, Max_Sim, Sim_STD）
    'text_vectorization': 'tfidf',  # 'tfidf' or 'bert'
    'normalize_method': 'cdf',      # 'cdf', 'standard', 'minmax'
    'use_time2vec': True,           # 是否使用Time2Vec编码时间戳
    'time2vec_dim': 64,             # Time2Vec输出维度（1线性 + 63周期 = 64维）
}

# 用户特征：间隔熵的时间桶（天）
INTERVAL_BINS = [0, 1, 7, 30, float('inf')]

# 评论长度归一化方式
REVIEW_LENGTH_METRIC = 'words'  # 'words' or 'chars'

# ============================================================================
# 特征归一化方向配置
# ============================================================================

# 定义每个特征的可疑性方向
# 'high': 值越大越可疑
# 'low': 值越小越可疑
# 'extreme': 极端值（接近0或1）可疑
# 'none': 不需要归一化或作为原始特征
FEATURE_DIRECTIONS = {
    # 用户特征 (6维)
    'MRD': 'high',          # 单日最大评论数
    'ARI': 'low',           # 平均评论间隔
    'RIE': 'low',           # 评论间隔熵
    'PRR': 'extreme',       # 正面评价比例
    'NRR': 'extreme',       # 负面评价比例
    'ARD': 'high',          # 平均评分偏差
    
    # 产品特征 (6维)
    'PMRD': 'high',         # 产品单日最大评论数
    'Product_PRR': 'extreme',
    'Product_NRR': 'extreme',
    'Rating_STD': 'extreme',
    'Review_Count': 'none',
    'PTIE': 'low',          # 产品时间间隔熵
    
    # 评论特征 (7维基础 + 3维文本)
    'REX': 'high',          # 评分极端性
    'ARD_review': 'high',   # 评论评分偏差
    'RL': 'extreme',        # 评论长度
    'Hour': 'none',         # 提交小时（需要特殊处理）
    'DayOfWeek': 'none',    # 提交星期
    'SRI': 'high',          # 单例评论指示器
    'RE': 'low',            # 评论排名
    'Mean_Sim': 'high',     # 平均文本相似度
    'Max_Sim': 'high',      # 最大文本相似度
    'Sim_STD': 'low',       # 文本相似度标准差
    
    # 时间特征 (2维)
    'TSLR': 'low',          # 自上次评论时间
    'Timestamp_norm': 'none',
}

# ============================================================================
# 数据划分参数（已移至训练阶段）
# ============================================================================

# 注意：Transductive 设定下，建图阶段不进行 train/val/test 划分
# 以下配置仅供训练阶段参考，建图阶段不使用
# SPLIT_CONFIG = {
#     'method': 'temporal',    # 'temporal' or 'random'
#     'train_ratio': 0.6,
#     'val_ratio': 0.2,
#     'test_ratio': 0.2,
# }

# ============================================================================
# 质量检查阈值
# ============================================================================

QUALITY_THRESHOLDS = {
    # 边数量范围
    'total_edges_min': 2_500_000,
    'total_edges_max': 3_500_000,
    'rur_edges_min': 15_000,
    'rur_edges_max': 40_000,
    'rtr_edges_min': 300_000,
    'rtr_edges_max': 550_000,
    'rsr_edges_min': 2_000_000,
    'rsr_edges_max': 3_000_000,
    
    # 边类型占比范围（百分比）
    'rur_ratio_min': 0.5,
    'rur_ratio_max': 2.5,
    'rtr_ratio_min': 10.0,
    'rtr_ratio_max': 20.0,
    'rsr_ratio_min': 80.0,
    'rsr_ratio_max': 92.0,
    
    # 图统计
    'edge_node_ratio_min': 60,
    'edge_node_ratio_max': 90,
    'isolated_nodes_max_ratio': 0.01,  # 1%
    'avg_degree_min': 100,
    'avg_degree_max': 180,
}

# ============================================================================
# 性能配置
# ============================================================================

PERFORMANCE_CONFIG = {
    'use_multiprocessing': False,  # 是否使用多进程（Windows下可能有问题）
    'n_jobs': 4,                   # 进程数
    'chunk_size': 1000,            # 分块处理大小
    'cache_results': True,         # 是否缓存中间结果
}

# ============================================================================
# 日志配置
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
}

# ============================================================================
# 预期数据集统计（用于验证）
# ============================================================================

EXPECTED_STATS = {
    'num_reviews': 39012,
    'num_users': 20219,
    'num_businesses': 601,
    'avg_reviews_per_user': 1.93,
    'avg_reviews_per_business': 64.91,
    'fraud_rate': 0.1119,  # 11.19%
}

# ============================================================================
# 特征名称列表（按顺序）
# ============================================================================

def get_feature_names(include_text=True, use_time2vec=True, time2vec_dim=8):
    """
    动态生成特征名称列表
    
    Parameters
    ----------
    include_text : bool
        是否包含文本特征
    use_time2vec : bool
        是否使用Time2Vec
    time2vec_dim : int
        Time2Vec维度
        
    Returns
    -------
    feature_names : list
        特征名称列表
    """
    # 用户特征 (0-5)
    user_features = ['MRD', 'ARI', 'RIE', 'PRR', 'NRR', 'ARD']
    
    # 产品特征 (6-11)
    product_features = ['PMRD', 'Product_PRR', 'Product_NRR', 'Rating_STD', 
                       'Review_Count', 'PTIE']
    
    # 评论特征 - 基础 (12-18)
    review_basic_features = ['REX', 'ARD_review', 'RL', 'Hour', 'DayOfWeek', 
                            'SRI', 'RE']
    
    # 评论特征 - 文本 (可选)
    review_text_features = ['Mean_Sim', 'Max_Sim', 'Sim_STD'] if include_text else []
    
    # 时间特征
    if use_time2vec:
        # TSLR + Time2Vec向量
        time_features = ['TSLR'] + [f'Time2Vec_{i}' for i in range(time2vec_dim)]
    else:
        # 传统方法：TSLR + Timestamp_norm
        time_features = ['TSLR', 'Timestamp_norm']
    
    # 合并所有特征
    feature_names = (user_features + product_features + 
                    review_basic_features + review_text_features + 
                    time_features)
    
    return feature_names


# 预定义常用配置的特征名称
FEATURE_NAMES_BASIC = get_feature_names(include_text=False, use_time2vec=False)  # 24维
FEATURE_NAMES_FULL = get_feature_names(include_text=True, use_time2vec=False)   # 27维
FEATURE_NAMES_TIME2VEC = get_feature_names(include_text=False, use_time2vec=True, time2vec_dim=64)  # 87维 (6+6+7+1+64)
FEATURE_NAMES_FULL_TIME2VEC = get_feature_names(include_text=True, use_time2vec=True, time2vec_dim=64)  # 90维 (6+6+10+1+64)

# 保持向后兼容
FEATURE_NAMES = FEATURE_NAMES_FULL  # 默认使用完整特征（传统方法）
BASIC_FEATURE_NAMES = FEATURE_NAMES_BASIC
FULL_FEATURE_NAMES = FEATURE_NAMES_FULL

# ============================================================================
# TF-IDF 配置（用于文本特征）
# ============================================================================

TFIDF_CONFIG = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.95,
    'ngram_range': (1, 2),  # unigram + bigram
    'use_idf': True,
}

# ============================================================================
# 辅助函数
# ============================================================================

def get_output_path(filename):
    """获取输出文件的完整路径"""
    return OUTPUT_DIR / filename

def get_feature_dim():
    """
    获取特征维度
    
    根据当前配置动态计算特征维度
    """
    include_text = FEATURE_CONFIG['include_text_features']
    use_time2vec = FEATURE_CONFIG.get('use_time2vec', False)
    time2vec_dim = FEATURE_CONFIG.get('time2vec_dim', 8)
    
    feature_names = get_feature_names(
        include_text=include_text,
        use_time2vec=use_time2vec,
        time2vec_dim=time2vec_dim
    )
    
    return len(feature_names)

def print_config():
    """打印当前配置"""
    print("=" * 80)
    print("当前配置")
    print("=" * 80)
    print(f"数据文件: {INPUT_CSV}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"特征维度: {get_feature_dim()}")
    print(f"  - 用户特征: 6维")
    print(f"  - 产品特征: 6维")
    print(f"  - 评论特征: {7 if not FEATURE_CONFIG['include_text_features'] else 10}维")
    if FEATURE_CONFIG.get('use_time2vec', False):
        time2vec_dim = FEATURE_CONFIG.get('time2vec_dim', 64)
        print(f"  - 时间特征: {1+time2vec_dim}维 (TSLR + Time2Vec[{time2vec_dim}])")
        print(f"    └─ Time2Vec: 1线性 + {time2vec_dim-1}周期 = {time2vec_dim}维")
    else:
        print(f"  - 时间特征: 2维 (TSLR + Timestamp_norm)")
    print(f"包含文本特征: {FEATURE_CONFIG['include_text_features']}")
    print(f"使用Time2Vec: {FEATURE_CONFIG.get('use_time2vec', False)}")
    print(f"时间窗口粒度: {RTR_TIME_WINDOW} (月)")
    print(f"归一化方法: {FEATURE_CONFIG['normalize_method']}")
    print("=" * 80)

if __name__ == '__main__':
    print_config()

