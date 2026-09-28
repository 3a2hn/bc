# -*- coding: utf-8 -*-
"""
TGN消息聚合器
聚合同一节点的多个消息
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MessageAggregator(nn.Module):
    """
    消息聚合器基类
    """
    
    def __init__(self):
        super(MessageAggregator, self).__init__()
    
    def forward(
        self,
        messages: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        聚合消息
        
        Parameters
        ----------
        messages : torch.Tensor [N, num_messages, message_dim]
            消息张量
        timestamps : torch.Tensor [N, num_messages], optional
            消息时间戳
            
        Returns
        -------
        aggregated : torch.Tensor [N, message_dim]
            聚合后的消息
        """
        raise NotImplementedError


class LastMessageAggregator(MessageAggregator):
    """
    最后消息聚合器
    
    只使用最后一条消息（按时间戳排序）
    """
    
    def __init__(self):
        super(LastMessageAggregator, self).__init__()
        logger.info("消息聚合策略: Last")
    
    def forward(
        self,
        messages: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        返回最后一条消息
        
        Parameters
        ----------
        messages : torch.Tensor [N, num_messages, message_dim]
            消息张量
        timestamps : torch.Tensor [N, num_messages], optional
            消息时间戳
            
        Returns
        -------
        last_message : torch.Tensor [N, message_dim]
            最后一条消息
        """
        if timestamps is not None:
            # 按时间戳排序，取最后一条
            _, indices = timestamps.sort(dim=-1, descending=False)
            last_indices = indices[:, -1].unsqueeze(-1).unsqueeze(-1)
            last_indices = last_indices.expand(-1, -1, messages.size(-1))
            last_message = messages.gather(1, last_indices).squeeze(1)
        else:
            # 直接取最后一条
            last_message = messages[:, -1, :]
        
        return last_message


class MeanMessageAggregator(MessageAggregator):
    """
    平均消息聚合器
    
    对所有消息取平均
    """
    
    def __init__(self):
        super(MeanMessageAggregator, self).__init__()
        logger.info("消息聚合策略: Mean")
    
    def forward(
        self,
        messages: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        返回消息的平均值
        
        Parameters
        ----------
        messages : torch.Tensor [N, num_messages, message_dim]
            消息张量
        timestamps : torch.Tensor [N, num_messages], optional
            消息时间戳（未使用）
            
        Returns
        -------
        mean_message : torch.Tensor [N, message_dim]
            平均消息
        """
        # 计算平均（处理padding的零向量）
        # 假设零向量是padding
        if timestamps is not None:
            # 使用时间戳来确定有效消息（非零时间戳）
            mask = (timestamps > 0).float().unsqueeze(-1)  # [N, num_messages, 1]
            sum_messages = (messages * mask).sum(dim=1)  # [N, message_dim]
            count = mask.sum(dim=1).clamp(min=1)  # [N, 1]
            mean_message = sum_messages / count
        else:
            mean_message = messages.mean(dim=1)
        
        return mean_message


class MaxMessageAggregator(MessageAggregator):
    """
    最大池化消息聚合器
    
    对所有消息进行max pooling
    """
    
    def __init__(self):
        super(MaxMessageAggregator, self).__init__()
        logger.info("消息聚合策略: Max")
    
    def forward(
        self,
        messages: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        返回消息的最大值
        
        Parameters
        ----------
        messages : torch.Tensor [N, num_messages, message_dim]
            消息张量
        timestamps : torch.Tensor [N, num_messages], optional
            消息时间戳（未使用）
            
        Returns
        -------
        max_message : torch.Tensor [N, message_dim]
            最大消息
        """
        max_message, _ = messages.max(dim=1)
        return max_message


def get_message_aggregator(aggregator_type: str) -> MessageAggregator:
    """
    获取消息聚合器
    
    Parameters
    ----------
    aggregator_type : str
        聚合器类型：'last', 'mean', 'max'
        
    Returns
    -------
    aggregator : MessageAggregator
        消息聚合器实例
    """
    aggregator_type = aggregator_type.lower()
    
    if aggregator_type == 'last':
        return LastMessageAggregator()
    elif aggregator_type == 'mean':
        return MeanMessageAggregator()
    elif aggregator_type == 'max':
        return MaxMessageAggregator()
    else:
        raise ValueError(f"未知的聚合器类型: {aggregator_type}")

