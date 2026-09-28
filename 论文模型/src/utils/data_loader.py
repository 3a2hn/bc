# -*- coding: utf-8 -*-
"""
数据加载器（Transductive 设定）

职责：
- 加载完整图数据（graph.pkl, features.npy, labels.npy）
- 不加载 splits.pkl（划分由训练阶段负责）
- 提供新的时间戳字段访问
- 支持多数据集（Yelp, Amazon）

设计原则：
- 建图阶段不进行 train/val/test 划分
- 划分逻辑完全放到训练阶段
- 所有时间信息从 graph.pkl 直接读取
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# 边类型映射配置（支持多数据集）
EDGE_TYPE_MAPPINGS = {
    'yelp': {'R-U-R': 0, 'R-T-R': 1, 'R-S-R': 2},
    'amazon': {'U-P-U': 0, 'U-S-U': 1, 'U-V-U': 2},
}

def detect_dataset_type(edge_type_names: list) -> str:
    """
    根据边类型名称自动检测数据集类型
    
    Parameters
    ----------
    edge_type_names : list
        边类型名称列表
        
    Returns
    -------
    str
        数据集类型 ('yelp' 或 'amazon')
    """
    # Yelp边类型
    yelp_types = {'R-U-R', 'R-T-R', 'R-S-R'}
    # Amazon边类型
    amazon_types = {'U-P-U', 'U-S-U', 'U-V-U'}
    
    edge_set = set(edge_type_names)
    
    if edge_set & yelp_types:
        return 'yelp'
    elif edge_set & amazon_types:
        return 'amazon'
    else:
        logger.warning(f"无法识别边类型: {edge_type_names}，默认使用yelp配置")
        return 'yelp'


class FraudDataLoader:
    """
    欺诈检测数据加载器（Transductive 设定）
    
    从新建图方法生成的文件加载数据，不加载划分信息
    支持多数据集（Yelp, Amazon）
    """
    
    def __init__(self, data_dir: Path, dataset: str = None):
        """
        初始化数据加载器
        
        Parameters
        ----------
        data_dir : Path
            数据目录路径（新建图方法/output 或 新建图方法_Amazon/output）
        dataset : str, optional
            数据集类型 ('yelp' 或 'amazon')，如果不指定则自动检测
        """
        self.data_dir = Path(data_dir)
        self.dataset = dataset  # 可以为None，后续自动检测
        
        # 检查必要文件（不再需要 splits.pkl）
        required_files = ['graph.pkl', 'features.npy', 'labels.npy']
        for f in required_files:
            if not (self.data_dir / f).exists():
                raise FileNotFoundError(f"缺少必要文件: {f}")
        
        logger.info(f"数据目录: {self.data_dir}")
        if dataset:
            logger.info(f"数据集类型: {dataset.upper()}")
        else:
            logger.info("数据集类型: 自动检测")
        logger.info("注意：Transductive 设定，划分由训练阶段负责")
    
    def load_data(self) -> Data:
        """
        加载完整图数据（不含划分）
        
        Returns
        -------
        data : Data
            PyTorch Geometric Data对象，包含完整图
        """
        # 1. 加载图数据
        logger.info("加载图数据...")
        with open(self.data_dir / 'graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)
        
        # 构建edge_index和edge_type
        # 优先使用 edges 字段（与 edge_timestamps 匹配）
        if 'edges' in graph_data and 'edge_attributes' in graph_data:
            # 使用统一的 edges 和 edge_attributes 字段
            logger.info("✓ 使用 'edges' 和 'edge_attributes' 字段")
            all_edges = graph_data['edges']
            edge_attrs = graph_data['edge_attributes']
            
            # 检测边类型名称以确定数据集类型
            detected_edge_types = set()
            for attr in edge_attrs:
                if isinstance(attr, dict):
                    if 'edge_types' in attr:
                        detected_edge_types.update(attr['edge_types'])
                    elif 'edge_type' in attr:
                        detected_edge_types.add(attr['edge_type'])
            
            # 自动检测或使用指定的数据集类型
            if self.dataset is None:
                self.dataset = detect_dataset_type(list(detected_edge_types))
                logger.info(f"✓ 自动检测数据集类型: {self.dataset.upper()}")
            
            # 获取对应数据集的边类型映射
            edge_type_mapping = EDGE_TYPE_MAPPINGS.get(self.dataset, EDGE_TYPE_MAPPINGS['yelp'])
            edge_type_names = list(edge_type_mapping.keys())
            
            logger.info(f"  边类型映射: {edge_type_mapping}")
            
            # 从 edge_attributes 中提取边类型
            all_edge_types = []
            
            for attr in edge_attrs:
                if isinstance(attr, dict):
                    if 'edge_types' in attr:
                        # 取第一个边类型作为主要类型
                        primary_type = attr['edge_types'][0]
                        all_edge_types.append(edge_type_mapping.get(primary_type, 0))
                    elif 'edge_type' in attr:
                        # 单一边类型
                        primary_type = attr['edge_type']
                        all_edge_types.append(edge_type_mapping.get(primary_type, 0))
                    else:
                        all_edge_types.append(1)  # 默认类型
                elif isinstance(attr, (int, np.integer)):
                    # 如果已经是整数，直接使用
                    all_edge_types.append(int(attr))
                else:
                    all_edge_types.append(1)  # 默认类型
            
            edge_index = np.array(all_edges, dtype=np.int64).T
            edge_type = np.array(all_edge_types, dtype=np.int64)
            
            logger.info(f"  边数: {edge_index.shape[1]:,}")
            # 打印边类型统计
            edge_stats = [f"{name}={np.sum(edge_type==idx):,}" for name, idx in edge_type_mapping.items()]
            logger.info(f"  边类型统计: {', '.join(edge_stats)}")
        else:
            # 回退到 edges_by_type（可能与边时间戳不匹配）
            logger.warning("使用 'edges_by_type' 字段（可能与边时间戳不匹配）")
            all_edges = []
            all_edge_types = []
            
            # 检测数据集类型
            if self.dataset is None:
                edge_type_names_in_data = list(graph_data['edges_by_type'].keys())
                self.dataset = detect_dataset_type(edge_type_names_in_data)
                logger.info(f"✓ 自动检测数据集类型: {self.dataset.upper()}")
            
            edge_type_mapping = EDGE_TYPE_MAPPINGS.get(self.dataset, EDGE_TYPE_MAPPINGS['yelp'])
            
            for edge_type_name, edges in graph_data['edges_by_type'].items():
                edge_type_id = edge_type_mapping.get(edge_type_name, 0)
                for edge in edges:
                    all_edges.append(edge)
                    all_edge_types.append(edge_type_id)
            
            edge_index = np.array(all_edges, dtype=np.int64).T
            edge_type = np.array(all_edge_types, dtype=np.int64)
        
        # 2. 提取时间戳（优先使用新字段）
        logger.info("提取时间戳...")
        
        # 节点时间戳
        if 'node_timestamps_norm' in graph_data:
            node_timestamps = graph_data['node_timestamps_norm']
            logger.info("✓ 使用 node_timestamps_norm")
        elif 'timestamps' in graph_data:
            node_timestamps = graph_data['timestamps']
            logger.info("✓ 使用 timestamps（兼容旧字段）")
        else:
            raise ValueError("graph.pkl 中缺少节点时间戳字段")
        
        # 节点原始时间戳（可选）
        node_timestamps_raw = graph_data.get('node_timestamps_raw', None)
        
        # 边时间戳
        # 优先从 edge_attributes 中提取
        if 'edges' in graph_data and 'edge_attributes' in graph_data:
            edge_attrs = graph_data['edge_attributes']
            edge_timestamps = []
            
            for attr in edge_attrs:
                if isinstance(attr, dict):
                    # 优先使用 timestamp 字段（单一时间戳）
                    if 'timestamp' in attr:
                        edge_timestamps.append(attr['timestamp'])
                    elif 'timestamps' in attr and len(attr['timestamps']) > 0:
                        # 如果有多个时间戳，取第一个
                        edge_timestamps.append(attr['timestamps'][0])
                    else:
                        edge_timestamps.append(0.0)
                else:
                    edge_timestamps.append(0.0)
            
            edge_timestamps = np.array(edge_timestamps)
            logger.info(f"✓ 从 edge_attributes 提取边时间戳，共 {len(edge_timestamps)} 条")
        elif 'edge_timestamps_norm' in graph_data:
            edge_timestamps = graph_data['edge_timestamps_norm']
            logger.info(f"✓ 使用 edge_timestamps_norm，共 {len(edge_timestamps)} 条边")
        elif 'edge_timestamps' in graph_data:
            edge_timestamps = graph_data['edge_timestamps']
            logger.info(f"✓ 使用 edge_timestamps，共 {len(edge_timestamps)} 条边")
        else:
            edge_timestamps = None
            logger.warning("graph.pkl 中缺少边时间戳字段")
        
        # 验证边时间戳与边数量是否匹配
        if edge_timestamps is not None:
            if len(edge_timestamps) != edge_index.shape[1]:
                logger.error(f"✗ 边时间戳数量 ({len(edge_timestamps)}) 与边数量 ({edge_index.shape[1]}) 不匹配！")
                logger.error(f"  边时间戳: {len(edge_timestamps):,}, 边数量: {edge_index.shape[1]:,}")
                raise ValueError(f"边时间戳数量与边数量不匹配")
            else:
                logger.info(f"✓ 边时间戳与边数量匹配")
                logger.info(f"  时间戳范围: [{np.min(edge_timestamps):.2f}, {np.max(edge_timestamps):.2f}]")
        
        # 边时间差和权重
        # 优先从 edge_attributes 中提取
        if 'edges' in graph_data and 'edge_attributes' in graph_data:
            edge_attrs = graph_data['edge_attributes']
            edge_time_diffs = []
            edge_weights = []
            
            for attr in edge_attrs:
                if isinstance(attr, dict):
                    # 提取时间差（取第一个值或平均值）
                    if 'time_diffs' in attr and len(attr['time_diffs']) > 0:
                        edge_time_diffs.append(attr['time_diffs'][0])
                    else:
                        edge_time_diffs.append(0)
                    
                    # 提取权重（取第一个值或平均值）
                    if 'weights' in attr and len(attr['weights']) > 0:
                        edge_weights.append(attr['weights'][0])
                    else:
                        edge_weights.append(1.0)
                else:
                    edge_time_diffs.append(0)
                    edge_weights.append(1.0)
            
            edge_time_diffs = np.array(edge_time_diffs)
            edge_weights = np.array(edge_weights)
            logger.info(f"✓ 从 edge_attributes 提取边权重和时间差")
        else:
            edge_time_diffs = graph_data.get('edge_time_diffs', None)
            edge_weights = graph_data.get('edge_weights', None)
        
        # 全局时间边界
        timestamp_range = graph_data.get('timestamp_range', None)
        
        # 3. 加载特征
        logger.info("加载特征...")
        features = np.load(self.data_dir / 'features.npy')
        features = torch.FloatTensor(features)
        
        # 4. 加载标签
        logger.info("加载标签...")
        labels = np.load(self.data_dir / 'labels.npy')
        labels = torch.LongTensor(labels)
        
        # 标签转换：确保 0=欺诈, 1=正常
        # 如果原始标签是 -1(欺诈), 1(正常)，需要转换
        if labels.min().item() == -1:
            labels = (labels + 1) // 2
            logger.info("标签转换: -1→0(欺诈), 1→1(正常)")
        
        # 5. 创建Data对象（不含划分掩码）
        num_nodes = features.shape[0]
        
        data = Data(
            x=features,
            edge_index=torch.LongTensor(edge_index),
            edge_type=torch.LongTensor(edge_type),
            y=labels,
            # 节点时间戳
            timestamps=torch.FloatTensor(node_timestamps),
            node_timestamps_norm=torch.FloatTensor(node_timestamps),
        )
        
        # 添加可选字段
        if node_timestamps_raw is not None:
            data.node_timestamps_raw = torch.FloatTensor(node_timestamps_raw)
        
        if edge_timestamps is not None:
            data.edge_timestamps = torch.FloatTensor(edge_timestamps)
            data.edge_timestamps_norm = torch.FloatTensor(edge_timestamps)
        
        if edge_time_diffs is not None:
            data.edge_time_diffs = torch.FloatTensor(edge_time_diffs)
        
        if edge_weights is not None:
            data.edge_weights = torch.FloatTensor(edge_weights)
        
        if timestamp_range is not None:
            data.ts_min = timestamp_range['ts_min']
            data.ts_max = timestamp_range['ts_max']
        
        # 打印统计信息
        logger.info(f"数据加载完成:")
        logger.info(f"  节点数: {num_nodes:,}")
        logger.info(f"  边数: {data.edge_index.size(1):,}")
        logger.info(f"  特征维度: {data.x.size(1)}")
        logger.info(f"  时间戳范围: [{node_timestamps.min():.2f}, {node_timestamps.max():.2f}]")
        logger.info(f"  欺诈率: {(labels == 0).sum().item()/num_nodes*100:.2f}%")
        logger.info(f"  注意：未加载划分信息，由训练阶段负责划分")
        
        return data
    
    def get_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        计算类别权重（用于处理类别不平衡）
        
        Parameters
        ----------
        labels : torch.Tensor
            标签张量
            
        Returns
        -------
        class_weights : torch.Tensor
            类别权重 [num_classes]
        """
        num_fraud = (labels == 0).sum().item()
        num_normal = (labels == 1).sum().item()
        
        # 权重 = 1 / 类别频率
        weight_fraud = num_normal / num_fraud if num_fraud > 0 else 1.0
        weight_normal = 1.0
        
        class_weights = torch.tensor([weight_fraud, weight_normal], dtype=torch.float)
        
        logger.info(f"类别权重: 欺诈={weight_fraud:.2f}, 正常={weight_normal:.2f}")
        
        return class_weights


def load_fraud_detection_data(data_dir: Path, dataset: str = None) -> Tuple[Data, torch.Tensor]:
    """
    便捷函数：加载欺诈检测数据（Transductive 设定）
    
    注意：
    - 本函数只负责加载完整图数据
    - 不进行 train/val/test 划分
    - 节点划分由训练阶段的 node_split 模块负责
    - 支持多数据集（Yelp, Amazon）
    
    Parameters
    ----------
    data_dir : Path
        数据目录路径
    dataset : str, optional
        数据集类型 ('yelp' 或 'amazon')，如果不指定则自动检测
        
    Returns
    -------
    data : Data
        完整图数据（不含划分掩码）
    class_weights : torch.Tensor
        类别权重（用于处理类别不平衡）
    """
    loader = FraudDataLoader(data_dir, dataset=dataset)
    data = loader.load_data()
    class_weights = loader.get_class_weights(data.y)
    
    # 将数据集类型附加到data对象
    data.dataset_type = loader.dataset
    
    return data, class_weights


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试Yelp数据集
    print("\n" + "="*80)
    print("测试 Yelp 数据集")
    print("="*80)
    data_dir_yelp = Path(__file__).parent.parent.parent.parent / '新建图方法' / 'output'
    if data_dir_yelp.exists():
        data, class_weights = load_fraud_detection_data(data_dir_yelp, dataset='yelp')
        print(f"\n数据形状:")
        print(f"  特征: {data.x.shape}")
        print(f"  边索引: {data.edge_index.shape}")
        print(f"  标签: {data.y.shape}")
        print(f"  时间戳: {data.timestamps.shape}")
        print(f"  数据集类型: {data.dataset_type}")
        print(f"\n类别权重: {class_weights}")
        print("\n✓ Yelp数据加载器测试通过！")
    else:
        print(f"Yelp数据目录不存在: {data_dir_yelp}")
    
    # 测试Amazon数据集
    print("\n" + "="*80)
    print("测试 Amazon 数据集")
    print("="*80)
    data_dir_amazon = Path(__file__).parent.parent.parent.parent / '新建图方法_Amazon' / 'output'
    if data_dir_amazon.exists():
        data, class_weights = load_fraud_detection_data(data_dir_amazon, dataset='amazon')
        print(f"\n数据形状:")
        print(f"  特征: {data.x.shape}")
        print(f"  边索引: {data.edge_index.shape}")
        print(f"  标签: {data.y.shape}")
        print(f"  时间戳: {data.timestamps.shape}")
        print(f"  数据集类型: {data.dataset_type}")
        print(f"\n类别权重: {class_weights}")
        print("\n✓ Amazon数据加载器测试通过！")
    else:
        print(f"Amazon数据目录不存在: {data_dir_amazon}")
