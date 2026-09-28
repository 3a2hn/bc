# -*- coding: utf-8 -*-
"""
消融实验配置管理
定义所有消融实验的配置和变体
"""

from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_CONFIG, TRAIN_CONFIG

# ============================================================================
# 基准模型配置（完整模型）
# ============================================================================

BASELINE_CONFIG = {
    **MODEL_CONFIG,
    'model_name': 'HTGATFraud-Full',
}

# ============================================================================
# 实验一：核心模块消融
# ============================================================================

ABLATION_CORE_MODULES = {
    'baseline': {
        'name': '完整模型 (Baseline)',
        'description': '完整HTGATFraud模型，包含HGT、TGN记忆、时序注意力（固定记忆权重）',
        'config': {
            **BASELINE_CONFIG,
            # ===== 记忆门控配置 =====
            'use_rl_memory_gate': False,          # 禁用RL记忆门控
            'fixed_memory_weight': 0.3,           # 恢复最优记忆权重，0.5过高引入噪声导致AUC下降
        },
    },
    'wo_tgn_memory': {
        'name': 'w/o TGN Memory',
        'description': '移除TGN记忆模块，不活跃时间片用可学习默认嵌入填充（mask=0.1）',
        'config': {
            **BASELINE_CONFIG,
            'use_memory': False,
        },
    },
    'wo_temporal_attention': {
        'name': 'w/o Temporal Attention',
        'description': '用简单平均池化替代时序自注意力',
        'config': {
            **BASELINE_CONFIG,
            'use_temporal_attention': False,
            'temporal_aggregation': 'mean',  # 使用平均聚合替代
        },
    },
    'wo_hgt_use_gcn': {
        'name': 'w/o HGT (Use GCN)',
        'description': '用朴素GCN替代HGT（无残差/LayerNorm），移除TGN记忆，仅消融空间聚合',
        'config': {
            **BASELINE_CONFIG,
            'use_hgt': False,
            'use_gcn': True,
            'use_memory': False,
        },
    },
    'static_baseline': {
        'name': 'Static Baseline',
        'description': '纯静态模型，单时间片，朴素GCN（无残差/LayerNorm），禁用TGN和时序注意力',
        'config': {
            **BASELINE_CONFIG,
            'num_snapshots': 1,
            'use_memory': False,
            'use_temporal_attention': False,
        },
    },
}

# ============================================================================
# 实验二：HGT层组件消融
# ============================================================================

ABLATION_HGT_LAYER = {
    'hgt_1layer': {
        'name': 'HGT-1Layer',
        'description': '减少HGT层数到1层',
        'config': {
            **BASELINE_CONFIG,
            'hgt_num_layers': 1,
        },
    },
    'hgt_3layer': {
        'name': 'HGT-3Layer',
        'description': '增加HGT层数到3层',
        'config': {
            **BASELINE_CONFIG,
            'hgt_num_layers': 3,
        },
    },
    'wo_relation_specific': {
        'name': 'w/o Relation-Specific',
        'description': '所有边类型共享相同的注意力参数',
        'config': {
            **BASELINE_CONFIG,
            'hgt_use_relation_specific': False,
        },
    },
    'hgt_2heads': {
        'name': 'HGT-2Heads',
        'description': '减少HGT注意力头数到2',
        'config': {
            **BASELINE_CONFIG,
            'hgt_num_heads': 2,
        },
    },
    'wo_rte': {
        'name': 'w/o RTE',
        'description': '移除相对时间编码',
        'config': {
            **BASELINE_CONFIG,
            'hgt_use_rte': False,
        },
    },
}

# ============================================================================
# 实验三：TGN记忆模块消融
# ============================================================================

ABLATION_TGN_MEMORY = {
    'wo_gated_fusion': {
        'name': 'w/o Gated Fusion',
        'description': '跳过门控融合，活跃时间片只用纯HGT嵌入，记忆仅用于填充不活跃时间片',
        'config': {
            **BASELINE_CONFIG,
            'use_memory': True,
        },
    },
    'wo_memory_filling': {
        'name': 'w/o Memory Filling',
        'description': '彻底移除记忆影响：活跃时间片跳过融合+不活跃时间片不填充，记忆零贡献',
        'config': {
            **BASELINE_CONFIG,
            'use_memory': True,
        },
    },
    'message_from_memory': {
        'name': 'Message from Memory',
        'description': '消息基于记忆而非HGT嵌入计算，还原冷启动问题，验证消息来源设计',
        'config': {
            **BASELINE_CONFIG,
            'use_memory': True,
        },
    },
    'wo_time_encoding': {
        'name': 'w/o Time Encoding',
        'description': '消息函数不传入时间编码，验证时间感知消息传递的必要性',
        'config': {
            **BASELINE_CONFIG,
            'use_memory': True,
        },
    },
}

# ============================================================================
# 实验四：时序注意力层消融
# ============================================================================

ABLATION_TEMPORAL_ATTENTION = {
    'wo_position_embedding': {
        'name': 'w/o Position Embedding',
        'description': '移除位置编码，时序注意力不区分时间片顺序',
        'config': {
            **BASELINE_CONFIG,
            'use_position_embedding': False,
        },
    },
    'w_causal_mask': {
        'name': 'w/ Causal Mask',
        'description': '启用因果mask，只看当前及过去的时间片',
        'config': {
            **BASELINE_CONFIG,
            'use_causal_mask': True,
        },
    },
    'temporal_2heads': {
        'name': 'Temporal-2Heads',
        'description': '减少时序注意力头数到2',
        'config': {
            **BASELINE_CONFIG,
            'temporal_num_heads': 2,
        },
    },
    'last_step_only': {
        'name': 'Last Step Only',
        'description': '用取最后活跃时间步替代时序注意力聚合，验证多时间步聚合的必要性',
        'config': {
            **BASELINE_CONFIG,
        },
        'custom_model': True,
    },
}

# ============================================================================
# 实验五：时间片划分策略消融
# ============================================================================

ABLATION_SNAPSHOT = {
    'snapshot_8': {
        'name': 'Snapshot-8',
        'description': '粗粒度划分，8个时间片',
        'config': {
            **BASELINE_CONFIG,
            'num_snapshots': 8,
        },
    },
    'snapshot_24': {
        'name': 'Snapshot-24 (Default)',
        'description': '默认设置，24个时间片',
        'config': {
            **BASELINE_CONFIG,
            'num_snapshots': 24,
        },
    },
    'snapshot_48': {
        'name': 'Snapshot-48',
        'description': '细粒度划分，48个时间片',
        'config': {
            **BASELINE_CONFIG,
            'num_snapshots': 48,
        },
    },
    'snapshot_96': {
        'name': 'Snapshot-96',
        'description': '极细粒度划分，96个时间片',
        'config': {
            **BASELINE_CONFIG,
            'num_snapshots': 96,
        },
    },
}

# ============================================================================
# 实验六：分类器结构消融
# ============================================================================

ABLATION_CLASSIFIER = {
    'classifier_32': {
        'name': 'Classifier-32',
        'description': '分类器隐藏维度减少到32',
        'config': {
            **BASELINE_CONFIG,
            'classifier_hidden_dim': 32,
        },
    },
    'classifier_96': {
        'name': 'Classifier-96',
        'description': '分类器隐藏维度增加到96',
        'config': {
            **BASELINE_CONFIG,
            'classifier_hidden_dim': 96,
        },
    },
    'wo_batchnorm': {
        'name': 'w/o BatchNorm',
        'description': '移除分类器中的BatchNorm层',
        'config': {
            **BASELINE_CONFIG,
            'use_batch_norm': False,
        },
    },
}

# ============================================================================
# 所有消融实验汇总
# ============================================================================

ALL_ABLATIONS = {
    '1_核心模块消融': ABLATION_CORE_MODULES,
    '2_HGT层组件消融': ABLATION_HGT_LAYER,
    '3_TGN记忆模块消融': ABLATION_TGN_MEMORY,
    '4_时序注意力层消融': ABLATION_TEMPORAL_ATTENTION,
    '5_时间片划分策略消融': ABLATION_SNAPSHOT,
    '6_分类器结构消融': ABLATION_CLASSIFIER,
}

# 实验优先级
PRIORITY_HIGH = ['wo_tgn_memory', 'wo_temporal_attention', 'wo_hgt_use_gcn', 'static_baseline',
                 'wo_relation_specific', 'wo_position_embedding', 'last_step_only',
                 'wo_gated_fusion', 'wo_memory_filling', 'message_from_memory']
PRIORITY_MEDIUM = ['hgt_1layer', 'hgt_3layer',
                   'wo_time_encoding', 'w_causal_mask', 'snapshot_2', 'snapshot_24', 'snapshot_32']
PRIORITY_LOW = ['hgt_2heads', 'wo_rte', 'temporal_2heads',
                'classifier_32', 'classifier_96', 'wo_batchnorm']


def get_ablation_config(experiment_category: str, variant_key: str) -> dict:
    """
    获取指定消融实验的配置
    
    Parameters
    ----------
    experiment_category : str
        实验类别，如 '1_核心模块消融'
    variant_key : str
        变体键，如 'wo_tgn_memory'
        
    Returns
    -------
    dict
        实验配置
    """
    if experiment_category not in ALL_ABLATIONS:
        raise ValueError(f"未知的实验类别: {experiment_category}")
    
    ablations = ALL_ABLATIONS[experiment_category]
    if variant_key not in ablations:
        raise ValueError(f"未知的变体: {variant_key}")
    
    return ablations[variant_key]


def list_all_ablations():
    """列出所有消融实验"""
    print("=" * 80)
    print("HTGATFraud 消融实验配置列表")
    print("=" * 80)
    
    for category, ablations in ALL_ABLATIONS.items():
        print(f"\n【{category}】")
        for key, ablation in ablations.items():
            priority = '⭐⭐⭐' if key in PRIORITY_HIGH else ('⭐⭐' if key in PRIORITY_MEDIUM else '⭐')
            print(f"  {key}: {ablation['name']} {priority}")
            print(f"      {ablation['description']}")


if __name__ == '__main__':
    list_all_ablations()
