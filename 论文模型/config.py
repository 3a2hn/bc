# -*- coding: utf-8 -*-
"""
HTGATFraud模型配置文件
包含所有超参数和路径配置
支持Yelp和Amazon两个数据集切换
"""

from pathlib import Path
import torch

# ============================================================================
# 数据集选择配置
# ============================================================================

# 支持的数据集: 'yelp' 或 'amazon'
# 可通过命令行参数 --dataset 覆盖
DEFAULT_DATASET = 'yelp'

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据集路径映射
DATASET_PATHS = {
    'yelp': PROJECT_ROOT.parent / '建图方法-amz' / 'output',
    'amazon': PROJECT_ROOT.parent / '建图方法-yelp' / 'output',
}

# 数据集边类型映射
DATASET_EDGE_TYPES = {
    'yelp': {
        'edge_type_names': ['R-U-R', 'R-T-R', 'R-S-R'],
        'edge_type_mapping': {'R-U-R': 0, 'R-T-R': 1, 'R-S-R': 2},
        'num_edge_types': 3,
    },
    'amazon': {
        'edge_type_names': ['U-P-U', 'U-S-U', 'U-V-U'],
        'edge_type_mapping': {'U-P-U': 0, 'U-S-U': 1, 'U-V-U': 2},
        'num_edge_types': 3,
    },
}

def get_data_dir(dataset: str = None) -> Path:
    """获取数据集目录"""
    dataset = dataset or DEFAULT_DATASET
    if dataset not in DATASET_PATHS:
        raise ValueError(f"未知数据集: {dataset}，支持的数据集: {list(DATASET_PATHS.keys())}")
    return DATASET_PATHS[dataset]

def get_edge_type_config(dataset: str = None) -> dict:
    """获取数据集边类型配置"""
    dataset = dataset or DEFAULT_DATASET
    if dataset not in DATASET_EDGE_TYPES:
        raise ValueError(f"未知数据集: {dataset}，支持的数据集: {list(DATASET_EDGE_TYPES.keys())}")
    return DATASET_EDGE_TYPES[dataset]

# 默认数据路径（兼容旧代码）
DATA_DIR = DATASET_PATHS[DEFAULT_DATASET]
GRAPH_FILE = DATA_DIR / 'graph.pkl'
FEATURES_FILE = DATA_DIR / 'features.npy'
LABELS_FILE = DATA_DIR / 'labels.npy'
# SPLITS_FILE = DATA_DIR / 'splits.pkl'  # Transductive设定：划分由训练阶段负责
ID_MAPPINGS_FILE = DATA_DIR / 'id_mappings.pkl'

# 输出路径
SRC_DIR = PROJECT_ROOT / 'src'
CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = PROJECT_ROOT / 'logs'

# 创建必要的目录
for dir_path in [SRC_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# 数据配置
# ============================================================================

# Yelp数据集配置
YELP_DATA_CONFIG = {
    'num_nodes': 38743,  # 总节点数
    'num_features': 87,  # 特征维度 (6用户 + 6产品 + 10评论 + 65时间)
    'num_classes': 2,    # 二分类：0=欺诈, 1=正常
    'num_node_types': 1, # 只有评论节点
    'num_edge_types': 3, # R-U-R=0, R-T-R=1, R-S-R=2
    'edge_type_names': ['R-U-R', 'R-T-R', 'R-S-R'],
    # 标签映射：原始标签-1(欺诈)→0, 1(正常)→1
    'label_mapping': {-1: 0, 1: 1},
}

# Amazon数据集配置
# 注意：num_features会在重新运行main_amazon.py后更新
# 当前配置：6用户+6产品+10评论(含文本)+65时间(1+64 Time2Vec) = 87维
AMAZON_DATA_CONFIG = {
    'num_nodes': 49940,  # 总节点数（扩充后的数据集）
    'num_features': 87,  # 特征维度（需重新运行main_amazon.py生成）
    'num_classes': 2,    # 二分类：0=欺诈, 1=正常
    'num_node_types': 1, # 只有评论节点
    'num_edge_types': 3, # U-P-U=0, U-S-U=1, U-V-U=2
    'edge_type_names': ['U-P-U', 'U-S-U', 'U-V-U'],
    # 标签映射：原始标签-1(欺诈)→0, 1(正常)→1
    'label_mapping': {-1: 0, 1: 1},
}

# 数据集配置映射
DATASET_DATA_CONFIGS = {
    'yelp': YELP_DATA_CONFIG,
    'amazon': AMAZON_DATA_CONFIG,
}

def get_data_config(dataset: str = None) -> dict:
    """获取数据集配置"""
    dataset = dataset or DEFAULT_DATASET
    if dataset not in DATASET_DATA_CONFIGS:
        raise ValueError(f"未知数据集: {dataset}，支持的数据集: {list(DATASET_DATA_CONFIGS.keys())}")
    return DATASET_DATA_CONFIGS[dataset].copy()

# 默认数据配置（兼容旧代码）
DATA_CONFIG = YELP_DATA_CONFIG.copy()

# ============================================================================
# 模型超参数配置
# ============================================================================

MODEL_CONFIG = {
    # 基础配置
    'model_name': 'HTGATFraud',
    'num_classes': 2,
    
    # 输入特征维度
    'input_dim': 87,  # 来自特征工程的87维特征
    
    # 时间片配置
    'num_snapshots': 24,  # 时间片数量（超参数搜索+消融实验确认24片最优）
    'snapshot_strategy': 'edge_based',  # 'edge_based': 按边划分(DySAT标准，推荐), 'uniform': 按节点时间戳分割(旧方法)
    
    # HGT层配置
    'hgt_num_layers': 2,      # 保持2层（3层OOM）
    'hgt_hidden_dim': 96,     # 保持96（显存受限，128会OOM）
    'hgt_num_heads': 4,       # HGT注意力头数
    'hgt_dropout': 0.3,       # 恢复0.3，0.2正则化不足导致AUC下降
    'hgt_use_norm': True,     # 是否使用LayerNorm
    'hgt_use_rte': True,      # 是否使用相对时间编码

    # HGT后投影配置：节点级别扩展维度，不增加边级别显存
    # HGT在96维运行（边级别显存瓶颈），投影到160维增加表征容量
    'post_hgt_dim': 160,      # 128→160，超参数搜索Phase3最优

    # 时序注意力配置
    'temporal_num_heads': 4,         # 确保能整除hidden_dim
    'temporal_dropout': 0.3,         # 恢复0.3，0.2正则化不足导致AUC下降
    'use_position_embedding': True,  # 是否使用位置编码
    'use_causal_mask': True,         # False → True 启用因果mask（提升性能）
    
    # DySAT配置
    'dysat_hidden_dim': 96,      # 与HGT保持一致
    'use_residual': True,        # False → True 启用残差连接（提升性能）
    'use_position_ffn': True,    # 是否使用位置前馈网络
    
    # 因果注意力配置（优先级最低，最后添加）
    'use_causal_attention': False,  # 是否使用因果注意力
    'causal_threshold': 0.3,
    'use_mixup': False,
    'mixup_alpha': 0.2,
    
    # 分类器配置
    'classifier_hidden_dim': 48,   # 64→48，超参数搜索Phase3最优
    'classifier_dropout': 0.2,     # 0.3→0.2，超参数搜索Phase2最优
    'use_batch_norm': True,        # 是否使用BatchNorm
    
    # 内存优化配置
    'use_gradient_checkpointing': True,  # 使用梯度检查点节省显存
    
    # TGN记忆模块配置
    'use_memory': True,           # 是否使用TGN记忆模块
    'num_nodes': 38743,            # 总节点数（从新数据集自动更新）
    'memory_dim': 96,              # 记忆向量维度（与hgt_hidden_dim一致）
    'message_dim': 96,             # 消息向量维度
    'time_dim': 96,                # 时间编码维度
    'message_function': 'mlp',     # 消息函数类型：'mlp' or 'identity'
    'message_aggregator': 'last',  # 消息聚合方式：'last', 'mean', 'max'
    'memory_updater': 'gru',       # 记忆更新器：'gru', 'rnn', 'mlp'
    'edge_feature_dim': 0,         # 边特征维度（当前未使用）
    'message_dropout': 0.2,        # 0.1 → 0.2 增强消息函数dropout
    'fusion_dropout': 0.2,         # 0.1 → 0.2 增强融合层dropout
    
    # 记忆门控配置
    'use_rl_memory_gate': False,   # 禁用RL记忆门控，使用固定权重
    'fixed_memory_weight': 0.3,    # 固定记忆权重（用于填充不活跃时间片）
}

# ============================================================================
# 节点划分配置（Transductive 设定）
# ============================================================================

SPLIT_CONFIG = {
    # 划分方法
    'split_method': 'stratified',  # 'stratified'（随机分层）或 'temporal'（时序）
    
    # 划分比例
    'train_ratio': 0.5,   # 训练集比例（50%）
    'val_ratio': 0.1,     # 验证集比例（10% - 用于早停监控）
    'test_ratio': 0.4,    # 测试集比例（40%）
    
    # 随机种子（保证可复现）
    'split_seed': 42,
}

# ============================================================================
# 训练配置
# ============================================================================

TRAIN_CONFIG = {
    # 基础训练参数
    'num_epochs': 200,
    'learning_rate': 2e-3,          # 1e-3→2e-3，超参数搜索Phase4最优
    'weight_decay': 1e-3,           # 5e-4→1e-3，超参数搜索Phase2最优
    'optimizer': 'adamw',           # adam → adamw, 更好的权重衰减解耦
    
    # 梯度裁剪
    'max_gradient_norm': 1.0,
    
    # 学习率调度
    'use_scheduler': True,
    'scheduler_type': 'cosine',
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
    'warmup_epochs': 10,            # 20→10，超参数搜索Phase4最优
    
    # 早停配置（基于验证集）
    'use_early_stopping': True,                # 启用早停
    'early_stopping_patience': 40,             # 30→40，gamma=1.0下收敛曲线更平缓需要更多耐心
    'early_stopping_metric': 'auc',            # 监控验证集AUC
    'early_stopping_min_delta': 0.0001,        # 最小变化阈值
    
    # 类别不平衡处理
    'use_class_weights': True,  # 自动计算类别权重
    'focal_gamma': 0.5,        # 1.0→0.5，超参数搜索Phase1最优
    'label_smoothing': 0.05,   # 标签平滑，超参数搜索Phase1确认0.05最优
    
    # 设备配置
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 检查点保存
    'save_checkpoints': True,
    'save_best_only': True,
    'save_frequency': 10,
    
    # 内存优化
    'use_amp': True,
}

# ============================================================================
# 评估配置
# ============================================================================

EVAL_CONFIG = {
    # 评估指标
    'metrics': ['auc', 'f1', 'precision', 'recall', 'accuracy'],
    
    # Recall@K配置
    'recall_at_k': [50, 100, 200, 500],
    
    # 可视化配置
    'visualize_embeddings': True,
    'visualize_attention': True,
    'embedding_method': 'tsne',  # 'tsne' or 'umap'
    'num_visualization_samples': 2000,
    
    # 案例分析
    'num_case_studies': 10,
    
    # 输出配置
    'save_predictions': True,
    'save_attention_weights': True,
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
# 辅助函数
# ============================================================================

def get_model_config_for_dataset(dataset: str = None, base_config: dict = None) -> dict:
    """
    获取特定数据集的模型配置
    
    Parameters
    ----------
    dataset : str
        数据集名称 ('yelp' 或 'amazon')
    base_config : dict
        基础模型配置（可选，默认使用MODEL_CONFIG）
        
    Returns
    -------
    dict
        更新后的模型配置
    """
    dataset = dataset or DEFAULT_DATASET
    config = (base_config or MODEL_CONFIG).copy()
    data_config = get_data_config(dataset)
    
    # 更新数据集相关参数
    config['num_nodes'] = data_config['num_nodes']
    config['input_dim'] = data_config['num_features']
    config['num_classes'] = data_config['num_classes']
    
    return config


def print_config(dataset: str = None):
    """打印当前配置"""
    dataset = dataset or DEFAULT_DATASET
    data_dir = get_data_dir(dataset)
    data_config = get_data_config(dataset)
    edge_config = get_edge_type_config(dataset)
    
    print("=" * 80)
    print(f"HTGATFraud模型配置 - 数据集: {dataset.upper()}")
    print("=" * 80)
    print(f"数据路径: {data_dir}")
    print(f"节点数: {data_config['num_nodes']:,}")
    print(f"特征维度: {data_config['num_features']}")
    print(f"边类型数: {data_config['num_edge_types']}")
    print(f"边类型: {', '.join(edge_config['edge_type_names'])}")
    print()
    print(f"节点划分方法: {SPLIT_CONFIG['split_method']}")
    print(f"划分比例: Train={SPLIT_CONFIG['train_ratio']:.0%}, "
          f"Val={SPLIT_CONFIG['val_ratio']:.0%}, Test={SPLIT_CONFIG['test_ratio']:.0%}")
    print(f"随机种子: {SPLIT_CONFIG['split_seed']}")
    print()
    print(f"模型: {MODEL_CONFIG['model_name']}")
    print(f"时间片数量: {MODEL_CONFIG['num_snapshots']}")
    print(f"HGT层数: {MODEL_CONFIG['hgt_num_layers']}")
    print(f"HGT隐藏维度: {MODEL_CONFIG['hgt_hidden_dim']}")
    print(f"时序注意力头数: {MODEL_CONFIG['temporal_num_heads']}")
    print()
    print(f"训练轮数: {TRAIN_CONFIG['num_epochs']}")
    print(f"学习率: {TRAIN_CONFIG['learning_rate']}")
    print(f"设备: {TRAIN_CONFIG['device']}")
    print(f"使用记忆模块: {MODEL_CONFIG['use_memory']}")
    print(f"使用因果注意力: {MODEL_CONFIG['use_causal_attention']}")
    print("=" * 80)

if __name__ == '__main__':
    print("="*80)
    print("支持的数据集配置:")
    print("="*80)
    for ds in ['yelp', 'amazon']:
        print(f"\n[{ds.upper()}]")
        print_config(ds)
