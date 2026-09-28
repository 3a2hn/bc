# -*- coding: utf-8 -*-
"""
TGN记忆更新器
使用GRU/RNN更新节点记忆
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MemoryUpdater(nn.Module):
    """
    记忆更新器基类
    """
    
    def __init__(self, memory_dim: int):
        super(MemoryUpdater, self).__init__()
        self.memory_dim = memory_dim
    
    def forward(
        self,
        messages: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        更新记忆
        
        Parameters
        ----------
        messages : torch.Tensor [N, message_dim]
            聚合后的消息
        memory : torch.Tensor [N, memory_dim]
            当前记忆
            
        Returns
        -------
        new_memory : torch.Tensor [N, memory_dim]
            更新后的记忆
        """
        raise NotImplementedError


class GRUMemoryUpdater(MemoryUpdater):
    """
    基于GRU的记忆更新器
    
    new_memory = GRU(message, old_memory)
    """
    
    def __init__(self, memory_dim: int, message_dim: int):
        """
        初始化GRU记忆更新器
        
        Parameters
        ----------
        memory_dim : int
            记忆维度
        message_dim : int
            消息维度
        """
        super(GRUMemoryUpdater, self).__init__(memory_dim)
        
        self.message_dim = message_dim
        
        # GRU单元
        self.gru_cell = nn.GRUCell(
            input_size=message_dim,
            hidden_size=memory_dim
        )
        
        logger.info(f"GRU记忆更新器: message_dim={message_dim}, memory_dim={memory_dim}")
    
    def forward(
        self,
        messages: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        使用GRU更新记忆
        
        Parameters
        ----------
        messages : torch.Tensor [N, message_dim]
            聚合后的消息
        memory : torch.Tensor [N, memory_dim]
            当前记忆
            
        Returns
        -------
        new_memory : torch.Tensor [N, memory_dim]
            更新后的记忆
        """
        # GRU更新
        new_memory = self.gru_cell(messages, memory)
        
        return new_memory


class RNNMemoryUpdater(MemoryUpdater):
    """
    基于RNN的记忆更新器
    
    new_memory = RNN(message, old_memory)
    """
    
    def __init__(self, memory_dim: int, message_dim: int):
        """
        初始化RNN记忆更新器
        
        Parameters
        ----------
        memory_dim : int
            记忆维度
        message_dim : int
            消息维度
        """
        super(RNNMemoryUpdater, self).__init__(memory_dim)
        
        self.message_dim = message_dim
        
        # RNN单元
        self.rnn_cell = nn.RNNCell(
            input_size=message_dim,
            hidden_size=memory_dim
        )
        
        logger.info(f"RNN记忆更新器: message_dim={message_dim}, memory_dim={memory_dim}")
    
    def forward(
        self,
        messages: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        使用RNN更新记忆
        
        Parameters
        ----------
        messages : torch.Tensor [N, message_dim]
            聚合后的消息
        memory : torch.Tensor [N, memory_dim]
            当前记忆
            
        Returns
        -------
        new_memory : torch.Tensor [N, memory_dim]
            更新后的记忆
        """
        # RNN更新
        new_memory = self.rnn_cell(messages, memory)
        
        return new_memory


class MLPMemoryUpdater(MemoryUpdater):
    """
    基于MLP的记忆更新器（简化版）
    
    new_memory = MLP([message, old_memory])
    """
    
    def __init__(self, memory_dim: int, message_dim: int, dropout: float = 0.1):
        """
        初始化MLP记忆更新器
        
        Parameters
        ----------
        memory_dim : int
            记忆维度
        message_dim : int
            消息维度
        dropout : float
            Dropout率
        """
        super(MLPMemoryUpdater, self).__init__(memory_dim)
        
        self.message_dim = message_dim
        
        # MLP
        input_dim = message_dim + memory_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim)
        )
        
        logger.info(f"MLP记忆更新器: {input_dim} -> {memory_dim}")
    
    def forward(
        self,
        messages: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        使用MLP更新记忆
        
        Parameters
        ----------
        messages : torch.Tensor [N, message_dim]
            聚合后的消息
        memory : torch.Tensor [N, memory_dim]
            当前记忆
            
        Returns
        -------
        new_memory : torch.Tensor [N, memory_dim]
            更新后的记忆
        """
        # 拼接并通过MLP
        x = torch.cat([messages, memory], dim=-1)
        new_memory = self.mlp(x)
        
        return new_memory


def get_memory_updater(
    updater_type: str,
    memory_dim: int,
    message_dim: int
) -> MemoryUpdater:
    """
    获取记忆更新器
    
    Parameters
    ----------
    updater_type : str
        更新器类型：'gru', 'rnn', 'mlp'
    memory_dim : int
        记忆维度
    message_dim : int
        消息维度
        
    Returns
    -------
    updater : MemoryUpdater
        记忆更新器实例
    """
    updater_type = updater_type.lower()
    
    if updater_type == 'gru':
        return GRUMemoryUpdater(memory_dim, message_dim)
    elif updater_type == 'rnn':
        return RNNMemoryUpdater(memory_dim, message_dim)
    elif updater_type == 'mlp':
        return MLPMemoryUpdater(memory_dim, message_dim)
    else:
        raise ValueError(f"未知的更新器类型: {updater_type}")

