# -*- coding: utf-8 -*-
"""
时序注意力层（PyTorch版）
改编自DySAT的TemporalAttentionLayer
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
import logging

logger = logging.getLogger(__name__)


class TemporalAttentionLayer(nn.Module):
    """
    时序注意力层
    
    对节点的时间序列表示应用自注意力机制，聚合跨时间片的信息
    
    输入: [N, T, F] (节点数, 时间步数, 特征维度)
    输出: [N, F] (聚合后的节点嵌入)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_time_steps: int,
        n_heads: int = 4,
        attn_drop: float = 0.3,
        residual: bool = False,
        use_position_embedding: bool = True,
        use_causal_mask: bool = False,
        use_position_ffn: bool = True
    ):
        """
        初始化时序注意力层
        
        Parameters
        ----------
        input_dim : int
            输入特征维度
        num_time_steps : int
            时间步数
        n_heads : int
            注意力头数
        attn_drop : float
            注意力Dropout率
        residual : bool
            是否使用残差连接
        use_position_embedding : bool
            是否使用位置编码
        use_causal_mask : bool
            是否使用因果mask（只看过去）
        use_position_ffn : bool
            是否使用位置前馈网络
        """
        super(TemporalAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.num_time_steps = num_time_steps
        self.n_heads = n_heads
        self.attn_drop = attn_drop
        self.residual = residual
        self.use_position_embedding = use_position_embedding
        self.use_causal_mask = use_causal_mask
        self.use_position_ffn = use_position_ffn
        
        # 确保input_dim可被n_heads整除
        assert input_dim % n_heads == 0, \
            f"input_dim ({input_dim}) 必须能被 n_heads ({n_heads}) 整除"
        
        self.d_k = input_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        
        # 位置编码
        if use_position_embedding:
            self.position_embeddings = nn.Parameter(
                torch.Tensor(num_time_steps, input_dim)
            )
            nn.init.xavier_uniform_(self.position_embeddings)
        
        # Q, K, V变换矩阵
        self.W_q = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.W_k = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.W_v = nn.Parameter(torch.Tensor(input_dim, input_dim))
        
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        
        # Dropout层
        self.dropout = nn.Dropout(attn_drop)

        # LayerNorm（Transformer标准做法：自注意力后 + FFN后各一个）
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim) if use_position_ffn else None

        # 位置前馈网络
        if use_position_ffn:
            self.ffn = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(input_dim, input_dim, kernel_size=1)
            )
        
        # 可学习的时间聚合器 - 使用attention机制计算每个时间步的重要性
        # 这个模块的权重参与梯度计算，可以学习如何聚合时间信息
        self.time_aggregator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # 用于存储注意力权重（可视化用，不参与梯度）
        self.attn_weights = None
        # 用于存储时间聚合权重（可视化用，不参与梯度）
        self.time_agg_weights = None
        
    def forward(self, temporal_inputs: torch.Tensor, temporal_mask: torch.Tensor = None):
        """
        前向传播
        
        Parameters
        ----------
        temporal_inputs : torch.Tensor
            时序输入 [N, T, F]
        temporal_mask : torch.Tensor
            时序掩码 [N, T]，标记哪些时间片有效（1=有效，0=无效）
            
        Returns
        -------
        output : torch.Tensor
            聚合后的节点嵌入 [N, F]
        """
        N, T, F = temporal_inputs.shape
        
        # 1. 添加位置编码
        if self.use_position_embedding:
            # position_inputs: [N, T]
            position_inputs = torch.arange(T, device=temporal_inputs.device).unsqueeze(0).repeat(N, 1)
            # 添加位置编码: [N, T, F]
            temporal_inputs = temporal_inputs + self.position_embeddings[position_inputs]
        
        # 2. Q, K, V变换
        # [N, T, F] @ [F, F] -> [N, T, F]
        q = torch.matmul(temporal_inputs, self.W_q)
        k = torch.matmul(temporal_inputs, self.W_k)
        v = torch.matmul(temporal_inputs, self.W_v)
        
        # 3. 分割为多头: [N, T, F] -> [N, n_heads, T, d_k]
        q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # 4. 计算注意力分数: [N, n_heads, T, T]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_dk
        
        # 5. 应用mask
        if self.use_causal_mask:
            # 因果mask：只看当前及过去的时间步
            # 创建下三角矩阵
            causal_mask = torch.tril(torch.ones(T, T, device=attn_scores.device))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        if temporal_mask is not None:
            # temporal_mask: [N, T] -> [N, 1, 1, T]
            mask = temporal_mask.unsqueeze(1).unsqueeze(2)
            # 将mask为0的位置设为-inf（softmax后为0）
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 6. Softmax得到注意力权重
        # 处理全为-inf的情况（某些节点在所有时间片都不出现）
        # 检查是否有全为-inf的行
        all_inf = torch.isinf(attn_scores).all(dim=-1, keepdim=True)  # [N, n_heads, T, 1]
        # 对于全为-inf的行，用0替换，避免NaN
        attn_scores = torch.where(all_inf.expand_as(attn_scores), 
                                   torch.zeros_like(attn_scores), 
                                   attn_scores)
        
        attn_weights = functional.softmax(attn_scores, dim=-1)  # [N, n_heads, T, T]
        
        # 保存注意力权重（用于可视化）
        # 对多头求平均: [N, T, T]
        self.attn_weights = attn_weights.mean(dim=1).detach()
        
        # 7. Dropout
        attn_weights = self.dropout(attn_weights)
        
        # 8. 加权聚合V: [N, n_heads, T, T] @ [N, n_heads, T, d_k] -> [N, n_heads, T, d_k]
        output = torch.matmul(attn_weights, v)
        
        # 9. 合并多头: [N, n_heads, T, d_k] -> [N, T, F]
        output = output.transpose(1, 2).contiguous().view(N, T, F)

        # 9.5 Post-attention: 残差连接 + LayerNorm（Pre-LN Transformer标准做法）
        output = output + temporal_inputs  # 自注意力残差
        output = self.norm1(output)

        # 10. 位置前馈网络
        if self.use_position_ffn:
            # Conv1d需要 [N, F, T]
            residual = output
            output_ffn = output.transpose(1, 2)  # [N, F, T]
            output_ffn = self.ffn(output_ffn)
            output = residual + output_ffn.transpose(1, 2)  # FFN残差连接
            output = self.norm2(output)

        # 11. 残差连接已在步骤9.5中完成，此处不再重复添加
        
        # 12. 时间维度聚合：使用可学习的时间聚合器（权重参与梯度！）
        # 
        # 之前的问题：使用 self.attn_weights（已detach）做聚合，梯度被切断
        # 修复方案：使用可学习的 time_aggregator 计算每个时间步的重要性
        
        # 计算每个时间步的重要性分数: [N, T, F] -> [N, T, 1]
        time_scores = self.time_aggregator(output)  # [N, T, 1]
        
        # 应用时序mask：无效时间步的分数设为-inf
        if temporal_mask is not None:
            # temporal_mask: [N, T] -> [N, T, 1]
            mask = temporal_mask.unsqueeze(-1)
            time_scores = time_scores.masked_fill(mask == 0, float('-inf'))
        
        # 处理全为-inf的情况（某些节点在所有时间片都不出现）
        all_inf = torch.isinf(time_scores).all(dim=1, keepdim=True)  # [N, 1, 1]
        time_scores = torch.where(
            all_inf.expand_as(time_scores),
            torch.zeros_like(time_scores),  # 全部设为0，softmax后变成均匀分布
            time_scores
        )
        
        # Softmax得到归一化的时间聚合权重: [N, T, 1]
        time_weights = functional.softmax(time_scores, dim=1)
        
        # 保存时间聚合权重（用于可视化，不参与梯度）
        self.time_agg_weights = time_weights.squeeze(-1).detach()  # [N, T]
        
        # 加权聚合: [N, T, F] * [N, T, 1] -> [N, T, F] -> sum -> [N, F]
        output = (output * time_weights).sum(dim=1)
        
        return output
    
    def get_attention_weights(self):
        """
        获取注意力权重（用于可视化）
        
        Returns
        -------
        attn_weights : torch.Tensor
            时间步之间的注意力权重 [N, T, T]
        """
        return self.attn_weights
    
    def get_time_aggregation_weights(self):
        """
        获取时间聚合权重（用于可视化）
        
        Returns
        -------
        time_agg_weights : torch.Tensor
            每个时间步的聚合权重 [N, T]
        """
        return self.time_agg_weights


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 参数
    N = 100  # 节点数
    T = 10   # 时间步数
    F = 128  # 特征维度
    n_heads = 4
    
    # 创建模型
    model = TemporalAttentionLayer(
        input_dim=F,
        num_time_steps=T,
        n_heads=n_heads,
        attn_drop=0.3,
        use_position_embedding=True,
        use_causal_mask=False,
        use_position_ffn=True
    )
    
    # 创建测试数据
    temporal_inputs = torch.randn(N, T, F)
    
    # 创建时序mask（模拟稀疏时间序列）
    temporal_mask = torch.rand(N, T) > 0.7  # 30%的时间片有效
    temporal_mask = temporal_mask.float()
    
    print(f"输入形状: {temporal_inputs.shape}")
    print(f"Mask形状: {temporal_mask.shape}")
    print(f"有效时间片比例: {temporal_mask.mean().item():.2%}")
    
    # 前向传播
    output = model(temporal_inputs, temporal_mask)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {model.get_attention_weights().shape}")
    
    print("\n✓ 时序注意力层测试通过！")
