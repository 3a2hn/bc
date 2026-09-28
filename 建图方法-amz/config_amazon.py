# -*- coding: utf-8 -*-
"""
配置文件 - Amazon电子产品数据集
以评论为节点的异构图构建配置（与Yelp结构一致）

边类型对比：
| Amazon | Yelp   | 含义 |
|--------|--------|------|
| U-P-U  | R-U-R  | 由同一用户发布的评论之间的连接 |
| U-S-U  | R-S-R  | 一周内至少共享一次相同星级评分的用户之间的连接 |
| U-V-U  | -      | 评论文本相似度(TF-IDF)位于所有用户前5%的用户之间的连接 |
"""

import os
from pathlib import Path

# ============================================================================
# 路径配置
# ============================================================================

PROJECT_ROOT = Path(__file__).parent

# 数据路径
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_CSV = DATA_DIR / 'final_labeled_fake_reviews_unix.csv'

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
LOGS_DIR = PROJECT_ROOT / 'logs_amazon'
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / 'build_graph_amazon.log'

# ============================================================================
# 图构建参数
# ============================================================================

# U-P-U 边参数（User-Product-User，由同一用户发布的评论之间的连接）
UPU_TIME_DECAY_TAU = 45.0  # 时间衰减参数（天）

# U-S-U 边参数（User-Star-User，一周内至少共享一次相同星级评分的用户之间的连接）
USU_TIME_WINDOW_DAYS = 7  # 时间窗口（一周）
USU_EXTREMITY_WEIGHT = True  # 是否使用评分极端性权重

# U-V-U 边参数（User-Vocabulary-User，评论文本相似度(TF-IDF)位于所有用户前5%的用户之间的连接）
UVU_TOP_PERCENT = 0.005    # 前0.5%
UVU_MIN_SIMILARITY = 0.20  # 相似度下限

# ============================================================================
# 特征工程参数
# ============================================================================

FEATURE_CONFIG = {
    'include_text_features': True,
    'text_vectorization': 'tfidf',
    'normalize_method': 'cdf',
    'use_time2vec': True,
    'time2vec_dim': 64,
}

# 用户特征：间隔熵的时间桶（天）
INTERVAL_BINS = [0, 1, 7, 30, float('inf')]

# 评论长度归一化方式
REVIEW_LENGTH_METRIC = 'words'

# ============================================================================
# 特征归一化方向配置（评论级别特征，与Yelp一致）
# ============================================================================

FEATURE_DIRECTIONS = {
    # 用户特征 (6维)
    'MRD': 'high',          # 单日最大评论数
    'ARI': 'low',           # 平均评论间隔
    'RIE': 'low',           # 评论间隔熵
    'PRR': 'extreme',       # 正面评价比例
    'NRR': 'extreme',       # 负面评价比例
    'ARD': 'high',          # 平均评分偏差
    
    # 产品特征 (6维)
    'PMRD': 'high',
    'Product_PRR': 'extreme',
    'Product_NRR': 'extreme',
    'Rating_STD': 'extreme',
    'Review_Count': 'none',
    'PTIE': 'low',
    
    # 评论特征 (7维基础 + 3维文本)
    'REX': 'high',
    'ARD_review': 'high',
    'RL': 'extreme',
    'Hour': 'none',
    'DayOfWeek': 'none',
    'SRI': 'high',
    'RE': 'low',
    'Mean_Sim': 'high',
    'Max_Sim': 'high',
    'Sim_STD': 'low',
    
    # 时间特征
    'TSLR': 'low',
    'Timestamp_norm': 'none',
}

# ============================================================================
# 质量检查阈值
# ============================================================================

QUALITY_THRESHOLDS = {
    # 边数量范围（30k节点数据集）
    'total_edges_min': 10_000,
    'total_edges_max': 10_000_000,
    'upu_edges_min': 100,
    'upu_edges_max': 500_000,
    'usu_edges_min': 1_000,
    'usu_edges_max': 5_000_000,
    'uvu_edges_min': 1_000,
    'uvu_edges_max': 5_000_000,

    # 边类型占比范围（百分比）
    'upu_ratio_min': 0.1,
    'upu_ratio_max': 30.0,
    'usu_ratio_min': 5.0,
    'usu_ratio_max': 70.0,
    'uvu_ratio_min': 5.0,
    'uvu_ratio_max': 70.0,
    
    # 图统计
    'edge_node_ratio_min': 1,
    'edge_node_ratio_max': 500,
    'isolated_nodes_max_ratio': 0.3,
    'avg_degree_min': 2,
    'avg_degree_max': 1000,
}

# ============================================================================
# 性能配置
# ============================================================================

PERFORMANCE_CONFIG = {
    'use_multiprocessing': False,
    'n_jobs': 4,
    'chunk_size': 500,
    'cache_results': True,
}

# ============================================================================
# 日志配置
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
}

# ============================================================================
# 预期数据集统计
# ============================================================================

EXPECTED_STATS = {
    'num_reviews': 30000,
    'num_users': 29641,
    'num_products': 18686,
    'fraud_rate': 0.15,  # 15.00%
}

# ============================================================================
# 特征名称列表（评论级别，与Yelp一致）
# ============================================================================

def get_feature_names(include_text=True, use_time2vec=True, time2vec_dim=64):
    """动态生成特征名称列表"""
    # 用户特征 (6维)
    user_features = ['MRD', 'ARI', 'RIE', 'PRR', 'NRR', 'ARD']
    
    # 产品特征 (6维)
    product_features = ['PMRD', 'Product_PRR', 'Product_NRR', 'Rating_STD', 
                       'Review_Count', 'PTIE']
    
    # 评论特征 (7维基础)
    review_basic_features = ['REX', 'ARD_review', 'RL', 'Hour', 'DayOfWeek', 
                            'SRI', 'RE']
    
    # 文本特征 (可选)
    review_text_features = ['Mean_Sim', 'Max_Sim', 'Sim_STD'] if include_text else []
    
    # 时间特征
    if use_time2vec:
        time_features = ['TSLR'] + [f'Time2Vec_{i}' for i in range(time2vec_dim)]
    else:
        time_features = ['TSLR', 'Timestamp_norm']
    
    feature_names = (user_features + product_features + 
                    review_basic_features + review_text_features + 
                    time_features)
    
    return feature_names


FEATURE_NAMES_BASIC = get_feature_names(include_text=False, use_time2vec=False)
FEATURE_NAMES_FULL = get_feature_names(include_text=True, use_time2vec=False)
FEATURE_NAMES_TIME2VEC = get_feature_names(include_text=False, use_time2vec=True, time2vec_dim=64)
FEATURE_NAMES_FULL_TIME2VEC = get_feature_names(include_text=True, use_time2vec=True, time2vec_dim=64)

FEATURE_NAMES = FEATURE_NAMES_FULL
BASIC_FEATURE_NAMES = FEATURE_NAMES_BASIC
FULL_FEATURE_NAMES = FEATURE_NAMES_FULL

# ============================================================================
# TF-IDF 配置
# ============================================================================

TFIDF_CONFIG = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.95,
    'ngram_range': (1, 2),
    'use_idf': True,
}

# ============================================================================
# 辅助函数
# ============================================================================

def get_output_path(filename):
    return OUTPUT_DIR / filename

def get_feature_dim():
    include_text = FEATURE_CONFIG['include_text_features']
    use_time2vec = FEATURE_CONFIG.get('use_time2vec', False)
    time2vec_dim = FEATURE_CONFIG.get('time2vec_dim', 64)
    
    feature_names = get_feature_names(
        include_text=include_text,
        use_time2vec=use_time2vec,
        time2vec_dim=time2vec_dim
    )
    
    return len(feature_names)

def print_config():
    print("=" * 80)
    print("Amazon评论图建图配置（与Yelp结构一致）")
    print("=" * 80)
    print(f"数据文件: {INPUT_CSV}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"节点类型: 评论 (Review)")
    print()
    print("边类型对比:")
    print("  | Amazon | Yelp   | 含义")
    print("  |--------|--------|------")
    print("  | U-P-U  | R-U-R  | 由同一用户发布的评论之间的连接")
    print("  | U-S-U  | R-S-R  | 一周内至少共享一次相同星级评分的用户之间的连接")
    print("  | U-V-U  | -      | 评论文本相似度(TF-IDF)位于所有用户前5%的用户之间的连接")
    print()
    print(f"特征维度: {get_feature_dim()}")
    print(f"  - 用户特征: 6维")
    print(f"  - 产品特征: 6维")
    print(f"  - 评论特征: {7 if not FEATURE_CONFIG['include_text_features'] else 10}维")
    if FEATURE_CONFIG.get('use_time2vec', False):
        time2vec_dim = FEATURE_CONFIG.get('time2vec_dim', 64)
        print(f"  - 时间特征: {1+time2vec_dim}维 (TSLR + Time2Vec[{time2vec_dim}])")
    else:
        print(f"  - 时间特征: 2维 (TSLR + Timestamp_norm)")
    print("=" * 80)

if __name__ == '__main__':
    print_config()
