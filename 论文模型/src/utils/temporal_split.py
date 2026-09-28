# -*- coding: utf-8 -*-
"""
时间片划分工具（Transductive 设定）

职责：
- 将完整图按时间戳划分为多个时间片快照
- 仅服务于时序表示学习，不与 train/val/test 划分对齐

设计原则：
- 时间片划分与节点划分解耦
- 时间片用于 DySAT/时序注意力模块
- 不强制因果顺序，允许整个时间演化过程为节点提供上下文

划分策略：
- 'edge_based': 按边时间划分（DySAT标准做法，推荐）
  所有节点在所有时间片都存在，边按时间分配到不同时间片
- 'uniform': 按节点时间戳均匀分割（旧方法，有边损失问题）
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TemporalGraphSplitter:
    """
    时间图划分器（Transductive 设定）
    
    将完整的动态图按时间戳划分为多个快照子图，
    仅服务于时序表示学习，不做 train/val/test 划分
    """
    
    def __init__(self, num_snapshots: int = 10, strategy: str = 'edge_based'):
        """
        初始化时间图划分器
        
        Parameters
        ----------
        num_snapshots : int
            时间片数量（默认10）
        strategy : str
            划分策略：
            - 'edge_based': 按边时间划分，所有节点保留（DySAT标准，推荐）
            - 'uniform': 按节点时间戳均匀分割（旧方法）
            - 'equal_nodes': 每个时间片节点数相等（旧方法）
        """
        self.num_snapshots = num_snapshots
        self.strategy = strategy
        
    def split(self, data: Data, timestamps: torch.Tensor = None) -> List[Data]:
        """
        将图数据划分为时间片快照
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            完整图数据
        timestamps : torch.Tensor
            节点时间戳 [N]（可选，默认从 data.timestamps 获取）
            
        Returns
        -------
        snapshots : List[Data]
            时间片快照列表
        """
        # 获取时间戳
        if timestamps is None:
            if hasattr(data, 'timestamps'):
                timestamps = data.timestamps
            elif hasattr(data, 'node_timestamps_norm'):
                timestamps = data.node_timestamps_norm
            else:
                raise ValueError("必须提供timestamps或在data中包含timestamps属性")
        
        # 确保timestamps是1D张量
        if timestamps.dim() > 1:
            timestamps = timestamps.squeeze()
        
        # 关键：确保timestamps长度与节点数一致
        num_nodes = data.x.size(0)
        if len(timestamps) != num_nodes:
            logger.error(f"时间戳长度 ({len(timestamps)}) 与节点数 ({num_nodes}) 不匹配！")
            raise ValueError(f"timestamps长度必须与节点数一致: {len(timestamps)} vs {num_nodes}")
        
        # 按边划分策略（DySAT标准做法）
        if self.strategy == 'edge_based':
            return self._split_edge_based(data, timestamps)
            
        # 以下是旧的按节点划分策略
        # 根据策略获取时间片边界
        if self.strategy == 'uniform':
            time_boundaries = self._get_uniform_boundaries(timestamps)
        elif self.strategy == 'equal_nodes':
            time_boundaries = self._get_equal_nodes_boundaries(timestamps)
        else:
            raise ValueError(f"未知的划分策略: {self.strategy}")
        
        # 为每个时间片创建子图
        snapshots = []
        for i in range(self.num_snapshots):
            t_start, t_end = time_boundaries[i], time_boundaries[i + 1]
            
            # 选择时间窗口内的节点
            node_mask = (timestamps >= t_start) & (timestamps < t_end)
            num_nodes_in_snapshot = node_mask.sum().item()
            
            # 跳过空的时间片
            if num_nodes_in_snapshot == 0:
                logger.warning(f"时间片 {i} 为空，跳过")
                continue
            
            # 创建子图
            snapshot = self._create_snapshot(
                data, node_mask, i, t_start, t_end
            )
            snapshots.append(snapshot)
        
        logger.info(f"成功创建 {len(snapshots)}/{self.num_snapshots} 个时间片快照")
        return snapshots
    
    def _split_edge_based(self, data: Data, timestamps: torch.Tensor) -> List[Data]:
        """
        按边时间划分（DySAT标准做法）
        
        所有节点在每个时间片都存在，边按时间分配到不同时间片。
        边的时间 = max(两端节点的时间戳)，表示交互发生在较晚节点出现时。
        
        优势：
        - 0% 边损失（所有边都被保留）
        - 每个节点在每个时间片都有表示
        - 不需要记忆填充
        - 不同时间片反映不同时期的邻居关系变化
        
        Parameters
        ----------
        data : Data
            完整图数据
        timestamps : torch.Tensor
            节点时间戳 [N]
            
        Returns
        -------
        snapshots : List[Data]
            时间片快照列表
        """
        num_nodes = data.x.size(0)
        device = data.x.device
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # 1. 计算边时间戳：取两端节点时间戳的最大值
        src_times = timestamps[edge_index[0]]  # [E]
        dst_times = timestamps[edge_index[1]]  # [E]
        edge_times = torch.max(src_times, dst_times)  # [E]
        
        # 2. 计算时间边界（使用边时间的等分位数，确保每个时间片边数均匀）
        sorted_edge_times = torch.sort(edge_times)[0]
        boundaries = [sorted_edge_times[0].item()]
        for i in range(1, self.num_snapshots):
            idx = int(i * num_edges / self.num_snapshots)
            boundaries.append(sorted_edge_times[idx].item())
        boundaries.append(sorted_edge_times[-1].item() + 1e-6)
        boundaries = np.array(boundaries)
        
        logger.info(f"[edge_based] 边时间范围: [{boundaries[0]:.2f}, {boundaries[-1]:.2f}]")
        
        # 3. 创建每个时间片的快照
        snapshots = []
        all_indices = torch.arange(num_nodes, device=device)
        
        for i in range(self.num_snapshots):
            t_start, t_end = boundaries[i], boundaries[i + 1]
            
            # 选择当前时间窗口内的边
            edge_mask = (edge_times >= t_start) & (edge_times < t_end)
            snapshot_edge_index = edge_index[:, edge_mask]
            snapshot_num_edges = snapshot_edge_index.size(1)
            
            # 提取边类型
            edge_type = None
            if hasattr(data, 'edge_type') and data.edge_type is not None:
                edge_type = data.edge_type[edge_mask]
            
            # 创建快照：所有节点 + 当前时间片的边
            snapshot = Data(
                x=data.x,                       # 所有节点特征
                edge_index=snapshot_edge_index,  # 当前时间片的边（全局索引）
                y=data.y if hasattr(data, 'y') else None,
                node_type=torch.zeros(num_nodes, dtype=torch.long, device=device),
                edge_type=edge_type,
                timestamps=timestamps,            # 所有节点的时间戳
                original_indices=all_indices,     # 所有节点的索引
                snapshot_id=i,
                time_range=(t_start, t_end),
                num_nodes=num_nodes
            )
            
            snapshots.append(snapshot)
            logger.info(f"  时间片 {i}: 节点={num_nodes:,}  边={snapshot_num_edges:,}  "
                       f"时间=[{t_start:.2f}, {t_end:.2f})")
        
        logger.info(f"[edge_based] 成功创建 {len(snapshots)} 个时间片（所有节点保留，边均匀分配）")
        return snapshots
    
    def _get_uniform_boundaries(self, timestamps: torch.Tensor) -> np.ndarray:
        """
        获取均匀分割的时间边界
        
        Parameters
        ----------
        timestamps : torch.Tensor
            节点时间戳
            
        Returns
        -------
        boundaries : np.ndarray
            时间边界数组，长度为num_snapshots+1
        """
        t_min = timestamps.min().item()
        t_max = timestamps.max().item()
        
        # 均匀分割时间轴
        boundaries = np.linspace(t_min, t_max, self.num_snapshots + 1)
        
        # 确保最后一个边界稍微大一点，包含最大值
        boundaries[-1] = t_max + 1e-6
        
        logger.info(f"时间范围: [{t_min:.2f}, {t_max:.2f}]")
        logger.info(f"每个时间片宽度: {(t_max - t_min) / self.num_snapshots:.2f}")
        
        return boundaries
    
    def _get_equal_nodes_boundaries(self, timestamps: torch.Tensor) -> np.ndarray:
        """
        获取等节点数分割的时间边界
        
        Parameters
        ----------
        timestamps : torch.Tensor
            节点时间戳
            
        Returns
        -------
        boundaries : np.ndarray
            时间边界数组
        """
        sorted_ts = torch.sort(timestamps)[0]
        n = len(sorted_ts)
        
        boundaries = [sorted_ts[0].item()]
        for i in range(1, self.num_snapshots):
            idx = int(i * n / self.num_snapshots)
            boundaries.append(sorted_ts[idx].item())
        boundaries.append(sorted_ts[-1].item() + 1e-6)
        
        return np.array(boundaries)
    
    def _create_snapshot(
        self,
        data: Data,
        node_mask: torch.Tensor,
        snapshot_id: int,
        t_start: float,
        t_end: float
    ) -> Data:
        """
        创建单个时间片快照
        
        Parameters
        ----------
        data : Data
            完整图数据
        node_mask : torch.Tensor
            节点选择掩码 [N]
        snapshot_id : int
            快照ID
        t_start : float
            时间窗口起始
        t_end : float
            时间窗口结束
            
        Returns
        -------
        snapshot : Data
            时间片快照
        """
        # 获取节点的原始索引
        original_indices = torch.where(node_mask)[0]
        num_nodes_snapshot = original_indices.size(0)
        
        # 获取设备
        device = data.x.device
        
        # 安全检查：确保original_indices不超出data.x的范围
        num_nodes_total = data.x.size(0)
        if len(original_indices) > 0 and original_indices.max().item() >= num_nodes_total:
            logger.error(f"快照 {snapshot_id}: original_indices 最大值 ({original_indices.max().item()}) >= 节点总数 ({num_nodes_total})")
            logger.error(f"这通常说明 timestamps 长度与 data.x.size(0) 不匹配")
            raise ValueError(f"节点索引超出范围")
        
        # 创建全局索引到局部索引的映射
        old_to_new = torch.full((num_nodes_total,), -1, dtype=torch.long, device=device)
        old_to_new[original_indices] = torch.arange(num_nodes_snapshot, device=device)
        
        # 提取节点特征
        x = data.x[node_mask]
        
        # 提取节点标签
        y = data.y[node_mask] if hasattr(data, 'y') and data.y is not None else None
        
        # 提取节点类型（评论节点全部为0）
        node_type = torch.zeros(num_nodes_snapshot, dtype=torch.long, device=device)
        
        # 提取时间戳
        timestamps = data.timestamps[node_mask] if hasattr(data, 'timestamps') else None
        
        # 选择边：源节点和目标节点都在时间窗口内
        edge_index = data.edge_index
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        
        # 提取并重新映射边索引
        snapshot_edge_index = edge_index[:, edge_mask]
        snapshot_edge_index = old_to_new[snapshot_edge_index]
        
        # 提取边类型
        edge_type = data.edge_type[edge_mask] if hasattr(data, 'edge_type') else None
        
        # 提取边时间戳（如果有）
        edge_timestamps = None
        if hasattr(data, 'edge_timestamps') and data.edge_timestamps is not None:
            # 检查边时间戳的长度是否与边掩码匹配
            if len(data.edge_timestamps) == len(edge_mask):
                edge_timestamps = data.edge_timestamps[edge_mask]
            else:
                logger.warning(f"边时间戳长度 ({len(data.edge_timestamps)}) 与边掩码长度 ({len(edge_mask)}) 不匹配，跳过边时间戳")
        
        # 提取边权重（如果有）
        edge_weights = None
        if hasattr(data, 'edge_weights') and data.edge_weights is not None:
            # 检查边权重的长度是否与边掩码匹配
            if len(data.edge_weights) == len(edge_mask):
                edge_weights = data.edge_weights[edge_mask]
            else:
                logger.warning(f"边权重长度 ({len(data.edge_weights)}) 与边掩码长度 ({len(edge_mask)}) 不匹配，跳过边权重")
        
        # 创建快照Data对象
        snapshot = Data(
            x=x,
            edge_index=snapshot_edge_index,
            y=y,
            node_type=node_type,
            edge_type=edge_type,
            timestamps=timestamps,
            edge_timestamps=edge_timestamps,
            edge_weights=edge_weights,
            original_indices=original_indices,  # 关键：保存原始索引映射
            snapshot_id=snapshot_id,
            time_range=(t_start, t_end),
            num_nodes=num_nodes_snapshot
        )
        
        return snapshot


def create_temporal_snapshots(
    data: Data,
    num_snapshots: int = 10,
    strategy: str = 'uniform'
) -> List[Data]:
    """
    便捷函数：将图划分为时间片快照
    
    Parameters
    ----------
    data : Data
        完整图数据
    num_snapshots : int
        时间片数量
    strategy : str
        划分策略 ('uniform', 'equal_nodes')
        
    Returns
    -------
    snapshots : List[Data]
        时间片快照列表
    """
    splitter = TemporalGraphSplitter(num_snapshots=num_snapshots, strategy=strategy)
    return splitter.split(data)


def get_snapshot_info(snapshots: List[Data]) -> Dict:
    """
    获取时间片快照的统计信息
    
    Parameters
    ----------
    snapshots : List[Data]
        时间片快照列表
        
    Returns
    -------
    info : Dict
        统计信息
    """
    info = {
        'num_snapshots': len(snapshots),
        'snapshots': []
    }
    
    for i, snap in enumerate(snapshots):
        snap_info = {
            'id': i,
            'num_nodes': snap.num_nodes,
            'num_edges': snap.edge_index.size(1) if snap.edge_index is not None else 0,
            'time_range': snap.time_range if hasattr(snap, 'time_range') else None,
        }
        info['snapshots'].append(snap_info)
    
    return info


def split_graph_into_snapshots(
    data: Data,
    num_snapshots: int = 10,
    strategy: str = 'uniform',
    timestamps: Optional[torch.Tensor] = None
) -> List[Data]:
    """
    便捷函数：将完整图划分为时间片快照
    
    这是 create_temporal_snapshots 的别名，为了保持向后兼容性
    
    Parameters
    ----------
    data : Data
        完整图数据
    num_snapshots : int
        时间片数量
    strategy : str
        划分策略 ('uniform', 'equal_nodes')
    timestamps : torch.Tensor, optional
        节点时间戳（可选，默认从data中获取）
        
    Returns
    -------
    snapshots : List[Data]
        时间片快照列表
    """
    splitter = TemporalGraphSplitter(num_snapshots=num_snapshots, strategy=strategy)
    return splitter.split(data, timestamps=timestamps)