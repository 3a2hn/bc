# -*- coding: utf-8 -*-
"""
HTGATFraud模型改进配置文件
"""

from pathlib import Path
import torch

# ============================================================================
# 路径配置（与原config.py相同）
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT.parent / '新建图方法' / 'output'
GRAPH_FILE = DATA_DIR / 'graph.pkl'
FEATURES_FILE = DATA_DIR / 'features.npy'
LABELS_FILE = DATA_DIR / 'labels.npy'
SPLITS_FILE = DATA_DIR / 'splits.pkl'
ID_MAPPINGS_FILE = DATA_DIR / 'id_mappings.pkl'

SRC_DIR = PROJECT_ROOT / 'src'
CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = PROJECT_ROOT / 'logs'

for dir_path in [SRC_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# 数据配置
# ============================================================================

DATA_CONFIG = {
    'num_nodes': 38266,
    'num_features': 87,
    'num_classes': 2,
    'num_node_types': 1,
    'num_edge_types': 3,
    'label_mapping': {-1: 0, 1: 1},
}

# ============================================================================
# 模型超参数配置（改进版 v1.1）
# ============================================================================

MODEL_CONFIG = {
    # 基础配置
    'model_name': 'HTGATFraud_v1.1',
    'num_classes': 2,
    'input_dim': 87,
    
    # ========== 时间片配置 ==========
    'num_snapshots': 12,  # ✨ 10 → 12 (更细粒度)
    'snapshot_strategy': 'uniform',
    
    # ========== HGT层配置 ==========
    'hgt_num_layers': 3,      # ✨ 2 → 3 (更深的图建模)
    'hgt_hidden_dim': 128,    # 保持不变
    'hgt_num_heads': 8,       # 保持不变
    'hgt_dropout': 0.3,       # ✨ 0.2 → 0.3 (缓解过拟合)
    'hgt_use_norm': True,
    'hgt_use_rte': True,      # ✨ False → True (启用相对时间编码)
    
    # ========== 时序注意力配置 ==========
    'temporal_num_heads': 4,         # 保持不变
    'temporal_dropout': 0.4,         # ✨ 0.3 → 0.4 (缓解过拟合)
    'use_position_embedding': True,
    'use_causal_mask': True,         # ✨ False → True (严格时序约束)
    
    # ========== DySAT配置 ==========
    'dysat_hidden_dim': 128,
    'use_residual': True,            # ✨ False → True (残差连接)
    'use_position_ffn': True,
    
    # ========== 记忆模块（暂不启用）==========
    'use_memory': False,
    'memory_dim': 128,
    'message_dim': 128,
    
    # ========== 因果注意力（暂不启用）==========
    'use_causal_attention': False,
    'causal_threshold': 0.3,
    'use_mixup': False,
    'mixup_alpha': 0.2,
    
    # ========== 分类器配置 ==========
    'classifier_hidden_dim': 64,
    'classifier_dropout': 0.5,       # ✨ 0.3 → 0.5 (缓解过拟合)
    'use_batch_norm': True,
}

# ============================================================================
# 训练配置（改进版）
# ============================================================================

TRAIN_CONFIG = {
    # 基础训练参数
    'num_epochs': 300,               # ✨ 200 → 300 (更充分训练)
    'learning_rate': 0.0005,         # ✨ 0.001 → 0.0005 (降低学习率)
    'weight_decay': 1e-5,
    'optimizer': 'adam',
    
    # 梯度裁剪
    'max_gradient_norm': 1.0,
    
    # 学习率调度
    'use_scheduler': True,
    'scheduler_type': 'reduce_on_plateau',
    'scheduler_patience': 15,        # ✨ 10 → 15 (更耐心)
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
    
    # 早停
    'use_early_stopping': True,
    'early_stopping_patience': 30,   # ✨ 20 → 30 (更耐心)
    'early_stopping_metric': 'auc',
    
    # 类别不平衡处理
    'use_class_weights': True,
    
    # 设备配置
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 检查点保存
    'save_checkpoints': True,
    'save_best_only': True,
    'save_frequency': 10,
}

# ============================================================================
# 评估配置
# ============================================================================

EVAL_CONFIG = {
    'metrics': ['auc', 'f1', 'precision', 'recall', 'accuracy'],
    'recall_at_k': [50, 100, 200, 500],
    'visualize_embeddings': True,
    'visualize_attention': True,
    'embedding_method': 'tsne',
    'num_visualization_samples': 2000,
    'num_case_studies': 10,
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

def print_config():
    """打印当前配置"""
    print("=" * 80)
    print("HTGATFraud模型配置 (改进版 v1.1)")
    print("=" * 80)
    print(f"数据路径: {DATA_DIR}")
    print(f"节点数: {DATA_CONFIG['num_nodes']:,}")
    print(f"特征维度: {DATA_CONFIG['num_features']}")
    print()
    print(f"模型: {MODEL_CONFIG['model_name']}")
    print(f"时间片数量: {MODEL_CONFIG['num_snapshots']}")
    print(f"HGT层数: {MODEL_CONFIG['hgt_num_layers']}")
    print(f"HGT隐藏维度: {MODEL_CONFIG['hgt_hidden_dim']}")
    print()
    print("✨ 改进点:")
    print(f"  - 时间片数量: 10 → {MODEL_CONFIG['num_snapshots']}")
    print(f"  - HGT层数: 2 → {MODEL_CONFIG['hgt_num_layers']}")
    print(f"  - HGT Dropout: 0.2 → {MODEL_CONFIG['hgt_dropout']}")
    print(f"  - 时序 Dropout: 0.3 → {MODEL_CONFIG['temporal_dropout']}")
    print(f"  - 分类器 Dropout: 0.3 → {MODEL_CONFIG['classifier_dropout']}")
    print(f"  - 相对时间编码: False → {MODEL_CONFIG['hgt_use_rte']}")
    print(f"  - 因果mask: False → {MODEL_CONFIG['use_causal_mask']}")
    print(f"  - 残差连接: False → {MODEL_CONFIG['use_residual']}")
    print()
    print(f"训练轮数: {TRAIN_CONFIG['num_epochs']}")
    print(f"学习率: {TRAIN_CONFIG['learning_rate']}")
    print(f"早停耐心值: {TRAIN_CONFIG['early_stopping_patience']}")
    print(f"设备: {TRAIN_CONFIG['device']}")
    print("=" * 80)
    print("\n预期提升: AUC 0.75 → 0.82-0.85 (+7-10%)")
    print("=" * 80)

if __name__ == '__main__':
    print_config()

