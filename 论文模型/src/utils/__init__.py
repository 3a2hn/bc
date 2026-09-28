# -*- coding: utf-8 -*-
"""
工具模块（Transductive 设定）

- data_loader: 加载完整图数据（不含划分）
- node_split: 训练阶段的节点划分
- temporal_split: 时间片构造（服务于时序模块）
"""

from .temporal_split import TemporalGraphSplitter, create_temporal_snapshots
from .data_loader import FraudDataLoader, load_fraud_detection_data
from .node_split import create_node_split, create_stratified_split, create_temporal_split

__all__ = [
    # 数据加载
    'FraudDataLoader',
    'load_fraud_detection_data',
    # 节点划分（训练阶段）
    'create_node_split',
    'create_stratified_split',
    'create_temporal_split',
    # 时间片构造（时序模块）
    'TemporalGraphSplitter',
    'create_temporal_snapshots',
]
