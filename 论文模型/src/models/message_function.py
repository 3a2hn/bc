# -*- coding: utf-8 -*-
"""
TGN消息函数
将原始信息转换为消息向量
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MessageFunction(nn.Module):
    """
    消息函数基类
    """
    
    def __init__(self, message_dim: int):
        super(MessageFunction, self).__init__()
        self.message_dim = message_dim
    
    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        time_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算消息
        
        Parameters
        ----------
        src_memory : torch.Tensor [N, memory_dim]
            源节点记忆
        dst_memory : torch.Tensor [N, memory_dim]
            目标节点记忆
        edge_features : torch.Tensor [N, edge_dim], optional
            边特征
        time_encoding : torch.Tensor [N, time_dim], optional
            时间编码
            
        Returns
        -------
        messages : torch.Tensor [N, message_dim]
            消息向量
        """
        raise NotImplementedError


class MLPMessageFunction(MessageFunction):
    """
    基于MLP的消息函数
    
    message = MLP([src_memory, dst_memory, edge_features, time_encoding])
    """
    
    def __init__(
        self,
        memory_dim: int,
        edge_dim: int,
        time_dim: int,
        message_dim: int,
        dropout: float = 0.1
    ):
        """
        初始化MLP消息函数
        
        Parameters
        ----------
        memory_dim : int
            记忆维度
        edge_dim : int
            边特征维度
        time_dim : int
            时间编码维度
        message_dim : int
            消息输出维度
        dropout : float
            Dropout率
        """
        super(MLPMessageFunction, self).__init__(message_dim)
        
        self.memory_dim = memory_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        
        # 输入维度：src_memory + dst_memory + edge_features + time_encoding
        input_dim = 2 * memory_dim + edge_dim + time_dim
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(message_dim, message_dim)
        )
        
        logger.info(f"MLP消息函数: {input_dim} -> {message_dim}")
    
    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        time_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算消息
        
        Parameters
        ----------
        src_memory : torch.Tensor [N, memory_dim]
            源节点记忆
        dst_memory : torch.Tensor [N, memory_dim]
            目标节点记忆
        edge_features : torch.Tensor [N, edge_dim], optional
            边特征
        time_encoding : torch.Tensor [N, time_dim], optional
            时间编码
            
        Returns
        -------
        messages : torch.Tensor [N, message_dim]
            消息向量
        """
        # 拼接所有输入
        inputs = [src_memory, dst_memory]
        
        # 添加边特征
        if edge_features is not None:
            if edge_features.size(0) != src_memory.size(0):
                raise ValueError(
                    f"边特征数量({edge_features.size(0)})与节点数量({src_memory.size(0)})不匹配"
                )
            inputs.append(edge_features)
        else:
            # 用零填充
            inputs.append(torch.zeros(
                src_memory.size(0), self.edge_dim,
                device=src_memory.device, dtype=src_memory.dtype
            ))
        
        # 添加时间编码
        if time_encoding is not None:
            if time_encoding.size(0) != src_memory.size(0):
                raise ValueError(
                    f"时间编码数量({time_encoding.size(0)})与节点数量({src_memory.size(0)})不匹配"
                )
            inputs.append(time_encoding)
        else:
            # 用零填充
            inputs.append(torch.zeros(
                src_memory.size(0), self.time_dim,
                device=src_memory.device, dtype=src_memory.dtype
            ))
        
        # 拼接并通过MLP
        x = torch.cat(inputs, dim=-1)
        messages = self.mlp(x)
        
        return messages


class IdentityMessageFunction(MessageFunction):
    """
    恒等消息函数（简化版）
    
    message = concat([src_memory, dst_memory])
    """
    
    def __init__(self, memory_dim: int):
        super(IdentityMessageFunction, self).__init__(2 * memory_dim)
        self.memory_dim = memory_dim
    
    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        time_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算消息（简单拼接）
        """
        return torch.cat([src_memory, dst_memory], dim=-1)

