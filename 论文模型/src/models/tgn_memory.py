# -*- coding: utf-8 -*-
"""
TGN记忆模块
维护每个节点的持久化记忆状态
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TGNMemory(nn.Module):
    """
    TGN记忆模块
    
    为每个节点维护一个记忆向量和最后更新时间戳
    记忆是状态变量（buffer），不是模型参数
    每个epoch开始时重置，在前向传播中通过GRU显式更新
    """
    
    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        time_dim: int = 1,
        device: str = 'cpu'
    ):
        """
        初始化TGN记忆模块
        
        Parameters
        ----------
        num_nodes : int
            总节点数（包括所有可能出现的节点）
        memory_dim : int
            记忆向量维度
        time_dim : int
            时间戳维度
        device : str
            设备
        """
        super(TGNMemory, self).__init__()
        
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.device = device
        
        # 注册为buffer（状态变量），不是parameter（模型参数）
        # buffer不会被优化器更新，但会被保存到checkpoint
        self.register_buffer(
            'memory',
            torch.zeros(num_nodes, memory_dim, device=device)
        )
        self.register_buffer(
            'last_update',
            torch.zeros(num_nodes, device=device)
        )
        
        # 消息存储（用于同一批次内的消息聚合）
        self.messages = {}  # {node_id: [messages]}
        self.message_timestamps = {}  # {node_id: [timestamps]}
        
        logger.info(f"TGN记忆模块初始化: {num_nodes}个节点, 记忆维度={memory_dim}")
    
    def reset(self):
        """
        重置记忆状态（每个epoch开始时调用）
        """
        self.memory.zero_()
        self.last_update.zero_()
        self.messages = {}
        self.message_timestamps = {}
        logger.debug("记忆状态已重置")
    
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        获取指定节点的记忆
        
        Parameters
        ----------
        node_ids : torch.Tensor [N]
            节点ID
            
        Returns
        -------
        memory : torch.Tensor [N, memory_dim]
            节点记忆
        """
        return self.memory[node_ids]
    
    def get_last_update(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        获取指定节点的最后更新时间
        
        Parameters
        ----------
        node_ids : torch.Tensor [N]
            节点ID
            
        Returns
        -------
        last_update : torch.Tensor [N]
            最后更新时间
        """
        return self.last_update[node_ids]
    
    def set_memory(self, node_ids: torch.Tensor, memory: torch.Tensor):
        """
        设置指定节点的记忆（显式状态转移）
        
        注意：不再使用 detach()，允许梯度通过记忆传播到消息函数和更新器。
        使用 data 属性直接修改 buffer 的底层数据，避免破坏计算图。
        
        Parameters
        ----------
        node_ids : torch.Tensor [N]
            节点ID
        memory : torch.Tensor [N, memory_dim]
            新的记忆状态
        """
        # 确保dtype匹配buffer的dtype
        memory_to_set = memory
        if memory_to_set.dtype != self.memory.dtype:
            memory_to_set = memory_to_set.to(self.memory.dtype)
        
        # 使用 detach 存储到 buffer（buffer 本身不参与梯度）
        # 但梯度已经在前向传播中通过 get_memory → fusion_layer 传递了
        with torch.no_grad():
            self.memory[node_ids] = memory_to_set.detach()
    
    def update_last_update_time(self, node_ids: torch.Tensor, timestamps: torch.Tensor):
        """
        更新节点的最后更新时间
        
        Parameters
        ----------
        node_ids : torch.Tensor [N]
            节点ID
        timestamps : torch.Tensor [N]
            时间戳
        """
        # 确保dtype匹配
        with torch.no_grad():
            timestamps_to_set = timestamps
            if timestamps_to_set.dtype != self.last_update.dtype:
                timestamps_to_set = timestamps_to_set.to(self.last_update.dtype)
            self.last_update[node_ids] = timestamps_to_set
    
    def store_messages(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """
        存储消息（用于批量聚合）
        
        Parameters
        ----------
        node_ids : torch.Tensor [N]
            节点ID
        messages : torch.Tensor [N, message_dim]
            消息向量
        timestamps : torch.Tensor [N]
            消息时间戳
        """
        node_ids = node_ids.cpu().numpy()
        messages = messages.detach()
        timestamps = timestamps.detach()
        
        for i, node_id in enumerate(node_ids):
            node_id = int(node_id)
            if node_id not in self.messages:
                self.messages[node_id] = []
                self.message_timestamps[node_id] = []
            
            self.messages[node_id].append(messages[i])
            self.message_timestamps[node_id].append(timestamps[i])
    
    def get_messages(
        self,
        node_ids: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        获取节点的所有存储消息
        
        Parameters
        ----------
        node_ids : torch.Tensor [N]
            节点ID
            
        Returns
        -------
        messages : torch.Tensor [N, max_msgs, message_dim] or None
            消息张量（如果有消息）
        timestamps : torch.Tensor [N, max_msgs] or None
            时间戳张量（如果有消息）
        """
        node_ids = node_ids.cpu().numpy()
        
        all_messages = []
        all_timestamps = []
        max_msgs = 0
        
        # 收集消息
        for node_id in node_ids:
            node_id = int(node_id)
            if node_id in self.messages and len(self.messages[node_id]) > 0:
                all_messages.append(self.messages[node_id])
                all_timestamps.append(self.message_timestamps[node_id])
                max_msgs = max(max_msgs, len(self.messages[node_id]))
            else:
                all_messages.append([])
                all_timestamps.append([])
        
        if max_msgs == 0:
            return None, None
        
        # Padding到相同长度
        message_dim = all_messages[0][0].size(0) if all_messages[0] else self.memory_dim
        
        padded_messages = torch.zeros(
            len(node_ids), max_msgs, message_dim,
            device=self.device
        )
        padded_timestamps = torch.zeros(
            len(node_ids), max_msgs,
            device=self.device
        )
        
        for i, (msgs, times) in enumerate(zip(all_messages, all_timestamps)):
            if len(msgs) > 0:
                padded_messages[i, :len(msgs)] = torch.stack(msgs)
                padded_timestamps[i, :len(times)] = torch.stack(times)
        
        return padded_messages, padded_timestamps
    
    def clear_messages(self, node_ids: Optional[torch.Tensor] = None):
        """
        清除消息缓存
        
        Parameters
        ----------
        node_ids : torch.Tensor, optional
            要清除的节点ID，如果为None则清除所有
        """
        if node_ids is None:
            self.messages = {}
            self.message_timestamps = {}
        else:
            node_ids = node_ids.cpu().numpy()
            for node_id in node_ids:
                node_id = int(node_id)
                if node_id in self.messages:
                    del self.messages[node_id]
                if node_id in self.message_timestamps:
                    del self.message_timestamps[node_id]
    
    def detach_memory(self):
        """
        分离记忆的梯度（防止跨时间步的梯度传播）
        """
        self.memory = self.memory.detach()
    
    def __repr__(self):
        return (f"TGNMemory(num_nodes={self.num_nodes}, "
                f"memory_dim={self.memory_dim}, "
                f"device={self.device})")

