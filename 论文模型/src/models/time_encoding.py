# -*- coding: utf-8 -*-
"""
时间编码模块
将时间差编码为向量
"""

import torch
import torch.nn as nn
import numpy as np
import math


class TimeEncoding(nn.Module):
    """
    时间编码器（类似于TGAT的设计）
    
    将时间差Δt编码为固定维度的向量
    使用可学习的基函数（basis functions）
    """
    
    def __init__(self, dimension: int):
        """
        初始化时间编码器
        
        Parameters
        ----------
        dimension : int
            输出维度
        """
        super(TimeEncoding, self).__init__()
        
        self.dimension = dimension
        
        # 可学习的频率和相位参数
        self.w = nn.Linear(1, dimension)
        
        # 初始化为正弦位置编码的频率
        with torch.no_grad():
            # 使用Transformer的位置编码初始化
            position = torch.arange(0, dimension, dtype=torch.float).unsqueeze(0)
            div_term = torch.exp(
                torch.arange(0, dimension, 2).float() * 
                (-math.log(10000.0) / dimension)
            )
            
            # 初始化权重
            if dimension % 2 == 0:
                self.w.weight[0::2, 0] = div_term
                self.w.weight[1::2, 0] = div_term
            else:
                self.w.weight[0::2, 0] = div_term
                self.w.weight[1::2, 0] = div_term[:-1]
            
            self.w.bias.zero_()
    
    def forward(self, time_delta: torch.Tensor) -> torch.Tensor:
        """
        编码时间差
        
        Parameters
        ----------
        time_delta : torch.Tensor [..., 1] or [...]
            时间差
            
        Returns
        -------
        encoding : torch.Tensor [..., dimension]
            时间编码
        """
        # 确保time_delta是正确的形状
        if time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(-1)
        elif time_delta.size(-1) != 1:
            time_delta = time_delta.unsqueeze(-1)
        
        # 线性变换
        output = self.w(time_delta)
        
        # 应用周期激活（cos + sin交替）
        # 偶数维度用cos，奇数维度用sin
        output_cos = torch.cos(output[..., 0::2])
        output_sin = torch.sin(output[..., 1::2])
        
        # 交错合并
        if self.dimension % 2 == 0:
            encoding = torch.stack([output_cos, output_sin], dim=-1)
            encoding = encoding.view(*time_delta.shape[:-1], self.dimension)
        else:
            # 奇数维度：最后一维只用cos
            encoding = torch.zeros(
                *time_delta.shape[:-1], self.dimension,
                device=time_delta.device,
                dtype=time_delta.dtype
            )
            encoding[..., 0::2] = output_cos
            encoding[..., 1::2] = output_sin[..., :-1] if output_sin.size(-1) > output_cos.size(-1) else output_sin
        
        return encoding


class FixedTimeEncoding(nn.Module):
    """
    固定的时间编码（不可学习，类似Transformer位置编码）
    """
    
    def __init__(self, dimension: int):
        super(FixedTimeEncoding, self).__init__()
        self.dimension = dimension
    
    def forward(self, time_delta: torch.Tensor) -> torch.Tensor:
        """
        编码时间差
        
        Parameters
        ----------
        time_delta : torch.Tensor [...]
            时间差
            
        Returns
        -------
        encoding : torch.Tensor [..., dimension]
            时间编码
        """
        batch_size = time_delta.shape[0] if time_delta.dim() > 0 else 1
        device = time_delta.device
        
        # 确保time_delta是2D
        if time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(-1)
        
        # 位置编码的频率
        div_term = torch.exp(
            torch.arange(0, self.dimension, 2, device=device).float() * 
            (-math.log(10000.0) / self.dimension)
        )
        
        # 计算编码
        encoding = torch.zeros(*time_delta.shape[:-1], self.dimension, device=device)
        
        # 偶数维度用sin
        encoding[..., 0::2] = torch.sin(time_delta * div_term)
        
        # 奇数维度用cos
        if self.dimension % 2 == 0:
            encoding[..., 1::2] = torch.cos(time_delta * div_term)
        else:
            encoding[..., 1::2] = torch.cos(time_delta * div_term[:-1])
        
        return encoding

