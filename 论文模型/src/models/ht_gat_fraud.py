# -*- coding: utf-8 -*-
"""
HTGATFraud主模型
整合Time2Vec、HGT、DySAT时序注意力、TGN记忆等模块
新增：强化学习自适应记忆门控机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, NamedTuple
import logging

from .hgt_layer import HGTLayer
from .temporal_attention import TemporalAttentionLayer
from .tgn_memory import TGNMemory
from .time_encoding import TimeEncoding
from .message_function import MLPMessageFunction, IdentityMessageFunction
from .message_aggregator import get_message_aggregator
from .memory_updater import get_memory_updater
from ..utils.temporal_split import split_graph_into_snapshots

logger = logging.getLogger(__name__)


class RLPolicyOutput(NamedTuple):
    """强化学习策略输出"""
    weights: torch.Tensor           # 记忆权重 [N]
    log_probs: torch.Tensor         # 动作的对数概率 [N]
    entropy: torch.Tensor           # 策略熵（标量）
    values: torch.Tensor            # 状态价值估计 [N]


class MemoryGatingPolicy(nn.Module):
    """
    强化学习记忆门控策略网络 (Actor-Critic)
    
    根据节点状态（活跃模式、记忆质量等）动态决定每个节点的记忆使用权重。
    
    设计思想：
    1. Actor（策略网络）：输出记忆权重的分布参数，使用 Beta 分布建模 [0,1] 区间
    2. Critic（价值网络）：估计当前状态的价值，用于计算优势函数减小方差
    3. 使用 REINFORCE with baseline 进行策略梯度更新
    
    状态特征：
    - 节点的记忆向量：历史信息的压缩表示
    - 活跃度统计：活跃时间片数量、最近活跃距离、活跃模式
    
    奖励设计：
    - 基于分类性能（AUC/F1）的改进作为奖励信号
    - 使用 Critic 估计的价值作为基线，减小方差
    """
    
    def __init__(
        self, 
        memory_dim: int, 
        hidden_dim: int = 64,
        num_snapshots: int = 10,
        dropout: float = 0.1
    ):
        """
        初始化记忆门控策略网络
        
        Parameters
        ----------
        memory_dim : int
            记忆向量维度
        hidden_dim : int
            隐藏层维度
        num_snapshots : int
            时间片数量（用于计算活跃度特征）
        dropout : float
            Dropout率
        """
        super(MemoryGatingPolicy, self).__init__()
        
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.num_snapshots = num_snapshots
        
        # 状态特征维度：记忆 + 活跃度统计
        # 活跃度特征：[活跃时间片比例, 最近活跃距离, 活跃时间片标准差, 首次活跃时间]
        self.state_dim = memory_dim + 4
        
        # 共享的状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Actor（策略网络）：输出 Beta 分布的参数 (alpha, beta)
        # Beta 分布自然地建模 [0,1] 区间，且可以表达多种形状
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 输出 (alpha_logit, beta_logit)
        )
        
        # Critic（价值网络）：估计状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化：让初始输出的权重接近 0.3（偏保守）
        # Beta(2, 5) 的期望是 2/(2+5) ≈ 0.29
        self._init_weights()
        
        logger.info(f"MemoryGatingPolicy 初始化: memory_dim={memory_dim}, "
                   f"hidden_dim={hidden_dim}, state_dim={self.state_dim}")
    
    def _init_weights(self):
        """初始化权重，使初始策略输出偏保守的记忆权重"""
        # 初始化 actor 输出层的偏置
        # 让 alpha ≈ 2, beta ≈ 5，这样期望权重 ≈ 0.29
        with torch.no_grad():
            self.actor[-1].bias[0] = 0.7   # softplus(0.7) + 1 ≈ 2.0
            self.actor[-1].bias[1] = 1.6   # softplus(1.6) + 1 ≈ 5.0
    
    def compute_state_features(
        self, 
        memory: torch.Tensor,           # [N, memory_dim]
        temporal_masks: torch.Tensor    # [N, T]
    ) -> torch.Tensor:
        """
        计算节点状态特征
        
        Parameters
        ----------
        memory : torch.Tensor [N, memory_dim]
            节点记忆向量
        temporal_masks : torch.Tensor [N, T]
            时间掩码，1 表示活跃，0 表示不活跃
            
        Returns
        -------
        state : torch.Tensor [N, state_dim]
            节点状态特征
        """
        N, T = temporal_masks.shape
        device = memory.device
        
        # 1. 活跃时间片比例
        active_ratio = temporal_masks.sum(dim=1, keepdim=True) / T  # [N, 1]
        
        # 2. 最近活跃距离（归一化）
        time_indices = torch.arange(T, device=device).float()  # [T]
        # 计算最后一个活跃时间片
        reversed_masks = temporal_masks.flip(dims=[1])  # 反转时间维度
        reversed_indices = torch.arange(T, device=device).float()
        # 找到第一个为 1 的位置（即原序列中最后一个活跃位置）
        last_active = T - 1 - (reversed_masks * reversed_indices).argmax(dim=1)  # [N]
        # 处理全零情况
        all_inactive = (temporal_masks.sum(dim=1) == 0)
        last_active = torch.where(all_inactive, torch.zeros_like(last_active), last_active)
        recent_distance = (T - 1 - last_active.float()) / T  # [N]，越小表示最近越活跃
        recent_distance = recent_distance.unsqueeze(1)  # [N, 1]
        
        # 3. 活跃时间片的标准差（衡量活跃分布的分散程度）
        # 加权平均活跃时间
        weighted_sum = (temporal_masks * time_indices).sum(dim=1)  # [N]
        count = temporal_masks.sum(dim=1).clamp(min=1)  # [N]
        mean_active_time = weighted_sum / count  # [N]
        # 计算方差
        squared_diff = temporal_masks * (time_indices - mean_active_time.unsqueeze(1)) ** 2
        variance = squared_diff.sum(dim=1) / count.clamp(min=1)
        std_active = (variance.sqrt() / T).unsqueeze(1)  # [N, 1]，归一化
        
        # 4. 首次活跃时间（归一化）
        # 找到第一个为 1 的位置
        first_active_idx = (temporal_masks * time_indices).argmax(dim=1)  # [N]
        # 处理全零情况
        first_active_idx = torch.where(all_inactive, 
                                       torch.full_like(first_active_idx, T-1), 
                                       first_active_idx)
        first_active = (first_active_idx.float() / T).unsqueeze(1)  # [N, 1]
        
        # 拼接所有特征
        state = torch.cat([
            memory,           # [N, memory_dim]
            active_ratio,     # [N, 1]
            recent_distance,  # [N, 1]
            std_active,       # [N, 1]
            first_active      # [N, 1]
        ], dim=-1)  # [N, state_dim]
        
        return state
    
    def forward(
        self, 
        memory: torch.Tensor,           # [N, memory_dim]
        temporal_masks: torch.Tensor,   # [N, T]
        deterministic: bool = False
    ) -> RLPolicyOutput:
        """
        前向传播：计算每个节点的记忆权重
        
        Parameters
        ----------
        memory : torch.Tensor [N, memory_dim]
            节点记忆向量
        temporal_masks : torch.Tensor [N, T]
            时间掩码
        deterministic : bool
            是否使用确定性策略（测试时使用）
            
        Returns
        -------
        output : RLPolicyOutput
            包含权重、对数概率、熵和价值估计
        """
        # 1. 计算状态特征
        state = self.compute_state_features(memory, temporal_masks)  # [N, state_dim]
        
        # 2. 状态编码
        encoded = self.state_encoder(state)  # [N, hidden_dim]
        
        # 3. Actor：输出 Beta 分布参数
        dist_params = self.actor(encoded)  # [N, 2]
        
        # 使用 softplus + 1 确保 alpha, beta > 1（单峰分布）
        alpha = F.softplus(dist_params[:, 0]) + 1.0  # [N]
        beta = F.softplus(dist_params[:, 1]) + 1.0   # [N]
        
        # 4. 创建 Beta 分布
        dist = torch.distributions.Beta(alpha, beta)
        
        # 5. 采样或使用期望
        if deterministic:
            # 测试时使用期望值
            weights = alpha / (alpha + beta)  # [N]
            log_probs = dist.log_prob(weights.clamp(1e-6, 1 - 1e-6))  # [N]
        else:
            # 训练时采样（使用重参数化采样以支持梯度传播）
            weights = dist.rsample()  # [N]
            log_probs = dist.log_prob(weights)  # [N]
        
        # 6. 计算策略熵（用于熵正则化）
        entropy = dist.entropy().mean()  # 标量
        
        # 7. Critic：估计状态价值
        values = self.critic(encoded).squeeze(-1)  # [N]
        
        return RLPolicyOutput(
            weights=weights,
            log_probs=log_probs,
            entropy=entropy,
            values=values
        )
    
    def get_value(
        self, 
        memory: torch.Tensor, 
        temporal_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        仅计算状态价值（用于计算 TD 目标等）
        """
        state = self.compute_state_features(memory, temporal_masks)
        encoded = self.state_encoder(state)
        return self.critic(encoded).squeeze(-1)


class GatedFusion(nn.Module):
    """
    门控融合模块
    
    使用GRU风格的门控机制融合HGT嵌入和TGN记忆，解决以下问题：
    1. 简单拼接可能破坏HGT学到的好特征
    2. 训练早期记忆为零向量时，会稀释HGT特征
    3. 模型可以学习何时依赖空间信息(HGT)，何时依赖历史信息(记忆)
    
    门控机制：
    - 重置门 r: 决定保留多少记忆信息用于候选状态计算
    - 更新门 z: 决定最终输出中HGT嵌入和变换后记忆的混合比例
    
    公式：
        r = sigmoid(W_r @ [h, m] + b_r)
        z = sigmoid(W_z @ [h, m] + b_z)
        m_transformed = tanh(W_m @ [h, r * m] + b_m)
        output = z * h + (1 - z) * m_transformed
    """
    
    def __init__(self, hgt_dim: int, memory_dim: int, dropout: float = 0.1):
        """
        初始化门控融合模块
        
        Parameters
        ----------
        hgt_dim : int
            HGT嵌入维度
        memory_dim : int
            记忆维度
        dropout : float
            Dropout率
        """
        super(GatedFusion, self).__init__()
        
        self.hgt_dim = hgt_dim
        self.memory_dim = memory_dim
        
        # 如果维度不同，先将记忆投影到HGT维度
        if memory_dim != hgt_dim:
            self.memory_proj = nn.Linear(memory_dim, hgt_dim)
        else:
            self.memory_proj = None
        
        # 重置门：决定保留多少记忆信息
        self.W_r = nn.Linear(hgt_dim * 2, hgt_dim)
        
        # 更新门：决定HGT和记忆的混合比例
        self.W_z = nn.Linear(hgt_dim * 2, hgt_dim)
        # 关键修复：初始化更新门偏置为正值，让 z 初始值接近 1
        # 这样训练早期更依赖 HGT 嵌入，避免记忆引入噪声
        nn.init.constant_(self.W_z.bias, 2.0)  # sigmoid(2) ≈ 0.88
        
        # 候选状态变换
        self.W_m = nn.Linear(hgt_dim * 2, hgt_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化（稳定训练）
        self.layer_norm = nn.LayerNorm(hgt_dim)
        
        logger.info(f"门控融合模块初始化: hgt_dim={hgt_dim}, memory_dim={memory_dim}")
    
    def forward(self, h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        门控融合HGT嵌入和记忆
        
        Parameters
        ----------
        h : torch.Tensor [N, hgt_dim]
            HGT空间聚合后的节点嵌入
        memory : torch.Tensor [N, memory_dim]
            节点的TGN记忆
            
        Returns
        -------
        output : torch.Tensor [N, hgt_dim]
            融合后的嵌入
        """
        # 1. 如果需要，将记忆投影到HGT维度
        if self.memory_proj is not None:
            m = self.memory_proj(memory)  # [N, hgt_dim]
        else:
            m = memory
        
        # 2. 拼接用于门控计算
        concat = torch.cat([h, m], dim=-1)  # [N, hgt_dim * 2]
        
        # 3. 计算重置门：决定保留多少记忆信息
        r = torch.sigmoid(self.W_r(concat))  # [N, hgt_dim]
        
        # 4. 计算更新门：决定最终混合比例
        z = torch.sigmoid(self.W_z(concat))  # [N, hgt_dim]
        
        # 5. 计算候选状态：使用重置门控制的记忆
        concat_reset = torch.cat([h, r * m], dim=-1)  # [N, hgt_dim * 2]
        m_transformed = torch.tanh(self.W_m(concat_reset))  # [N, hgt_dim]
        
        # 6. 门控融合：z控制HGT和变换后记忆的比例
        # z接近1时，输出更多依赖HGT嵌入
        # z接近0时，输出更多依赖记忆信息
        output = z * h + (1 - z) * m_transformed
        
        # 7. Dropout和层归一化
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output


class HTGATFraud(nn.Module):
    """
    异构时序图注意力欺诈检测模型（集成TGN记忆）
    
    整合了以下组件：
    1. Time2Vec时间编码器（已集成在特征中）
    2. HGT异构图层（处理空间邻居）
    3. TGN记忆模块（可选，编码历史演化状态）
    4. DySAT时序注意力层（聚合时间序列）
    5. 分类器（输出欺诈概率）
    
    数据流（启用TGN记忆）：
    完整图 -> 时间片划分 -> 
    [HGT空间聚合 -> 读取记忆 -> 融合 -> 更新记忆] (每个快照) -> 
    整合时间序列 -> 时序注意力 -> 分类器
    """
    
    def __init__(self, config: Dict):
        """
        初始化HTGATFraud模型
        
        Parameters
        ----------
        config : Dict
            模型配置字典，应包含：
            - input_dim: 输入特征维度
            - num_classes: 类别数
            - num_snapshots: 时间片数量
            - hgt_*: HGT相关配置
            - temporal_*: 时序注意力配置
            - classifier_*: 分类器配置
        """
        super(HTGATFraud, self).__init__()
        
        # 保存配置
        self.config = config
        self.input_dim = config['input_dim']
        self.num_classes = config['num_classes']
        self.num_snapshots = config['num_snapshots']
        self.hgt_hidden_dim = config['hgt_hidden_dim']
        self.dysat_hidden_dim = config.get('dysat_hidden_dim', config['hgt_hidden_dim'])
        self.use_memory = config.get('use_memory', False)
        self.num_nodes = config.get('num_nodes', 40000)  # 总节点数

        # HGT后投影维度：在节点级别扩展表征容量，不增加边级别显存
        # 如果 post_hgt_dim > hgt_hidden_dim，则在HGT输出后加投影层
        self.post_hgt_dim = config.get('post_hgt_dim', self.hgt_hidden_dim)
        # 实际用于时序注意力和分类器的工作维度
        self.working_dim = self.post_hgt_dim
        
        # 1. HGT层：处理异构图
        self.hgt = HGTLayer(
            in_dim=self.input_dim,
            hidden_dim=config['hgt_hidden_dim'],
            num_types=config.get('num_node_types', 1),
            num_relations=config.get('num_edge_types', 3),
            n_heads=config['hgt_num_heads'],
            n_layers=config['hgt_num_layers'],
            dropout=config['hgt_dropout'],
            use_norm=config['hgt_use_norm'],
            use_RTE=config['hgt_use_rte'],
            use_gradient_checkpointing=config.get('use_gradient_checkpointing', False)
        )

        # 1.5 HGT后投影层：节点级别扩展维度，不增加边级别显存
        if self.post_hgt_dim != self.hgt_hidden_dim:
            self.post_hgt_proj = nn.Sequential(
                nn.Linear(self.hgt_hidden_dim, self.post_hgt_dim),
                nn.GELU(),
                nn.LayerNorm(self.post_hgt_dim),
            )
            logger.info(f"HGT后投影层: {self.hgt_hidden_dim} -> {self.post_hgt_dim}")
        else:
            self.post_hgt_proj = None

        # 2. TGN记忆模块（可选）
        if self.use_memory:
            memory_dim = config.get('memory_dim', self.hgt_hidden_dim)
            message_dim = config.get('message_dim', self.hgt_hidden_dim)
            time_dim = config.get('time_dim', self.hgt_hidden_dim)
            self.memory_dim = memory_dim
            
            # 记忆存储
            self.memory = TGNMemory(
                num_nodes=self.num_nodes,
                memory_dim=memory_dim,
                time_dim=time_dim,
                device=config.get('device', 'cpu')
            )
            
            # 时间编码器
            self.time_encoder = TimeEncoding(dimension=time_dim)
            
            # 消息函数
            # 注意：消息函数的输入是HGT嵌入（hgt_hidden_dim），不是记忆（memory_dim）
            # 这样消息可以携带当前时间片的空间信息，而不是历史记忆
            message_function_type = config.get('message_function', 'mlp')
            if message_function_type == 'mlp':
                self.message_function = MLPMessageFunction(
                    memory_dim=self.hgt_hidden_dim,  # 改为使用HGT维度
                    edge_dim=config.get('edge_feature_dim', 0),
                    time_dim=time_dim,
                    message_dim=message_dim,
                    dropout=config.get('message_dropout', 0.1)
                )
            else:
                self.message_function = IdentityMessageFunction(self.hgt_hidden_dim)
            
            # 消息聚合器
            aggregator_type = config.get('message_aggregator', 'last')
            self.message_aggregator = get_message_aggregator(aggregator_type)
            
            # 记忆更新器
            updater_type = config.get('memory_updater', 'gru')
            self.memory_updater = get_memory_updater(
                updater_type=updater_type,
                memory_dim=memory_dim,
                message_dim=message_dim
            )
            
            # 融合层：使用门控机制融合HGT嵌入和记忆
            # 门控融合解决了简单拼接可能破坏HGT特征的问题
            # 模型可以学习何时依赖空间信息(HGT)，何时依赖历史信息(记忆)
            self.fusion_layer = GatedFusion(
                hgt_dim=self.hgt_hidden_dim,
                memory_dim=memory_dim,
                dropout=config.get('fusion_dropout', 0.1)
            )
            
            # ===== 记忆门控策略配置 =====
            self.use_rl_memory_gate = config.get('use_rl_memory_gate', False)  # 默认禁用RL
            self.fixed_memory_weight = config.get('fixed_memory_weight', 0.3)  # 固定记忆权重
            
            if self.use_rl_memory_gate:
                # 使用 Actor-Critic 架构动态学习每个节点的记忆使用权重
                self.memory_policy = MemoryGatingPolicy(
                    memory_dim=memory_dim,
                    hidden_dim=config.get('rl_policy_hidden_dim', 64),
                    num_snapshots=self.num_snapshots,
                    dropout=config.get('rl_policy_dropout', 0.1)
                )
                # 保存 RL 输出用于损失计算
                self.rl_output: Optional[RLPolicyOutput] = None
                # RL 损失权重配置
                self.rl_policy_weight = config.get('rl_policy_weight', 0.1)
                self.rl_value_weight = config.get('rl_value_weight', 0.1)
                self.rl_entropy_weight = config.get('rl_entropy_weight', 0.01)
                logger.info(f"RL记忆门控已启用: policy_weight={self.rl_policy_weight}, "
                           f"value_weight={self.rl_value_weight}, "
                           f"entropy_weight={self.rl_entropy_weight}")
            else:
                self.memory_policy = None
                self.rl_output = None
                logger.info(f"使用固定记忆权重: {self.fixed_memory_weight}")
            
            logger.info(f"TGN记忆模块已启用: memory_dim={memory_dim}")
        else:
            self.memory = None
            self.use_rl_memory_gate = False
            self.memory_policy = None
            self.rl_output = None
            logger.info("TGN记忆模块未启用")
        
        # 3. 时序注意力层：聚合跨时间片的信息
        self.temporal_attention = TemporalAttentionLayer(
            input_dim=self.working_dim,
            num_time_steps=self.num_snapshots,
            n_heads=config['temporal_num_heads'],
            attn_drop=config['temporal_dropout'],
            residual=config.get('use_residual', False),
            use_position_embedding=config.get('use_position_embedding', True),
            use_causal_mask=config.get('use_causal_mask', False),
            use_position_ffn=config.get('use_position_ffn', True)
        )

        # 4. 分类器：多层感知机
        classifier_hidden = config['classifier_hidden_dim']
        use_batch_norm = config.get('use_batch_norm', True)
        dropout = config['classifier_dropout']

        classifier_layers = []
        classifier_mid = classifier_hidden // 2  # 中间层维度（如 48->24 或 64->32）

        # 第一层：working_dim -> classifier_hidden_dim
        classifier_layers.append(nn.Linear(self.working_dim, classifier_hidden))
        classifier_layers.append(nn.ReLU())
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(classifier_hidden))
        classifier_layers.append(nn.Dropout(dropout))

        # 第二层：classifier_hidden_dim -> classifier_mid
        classifier_layers.append(nn.Linear(classifier_hidden, classifier_mid))
        classifier_layers.append(nn.ReLU())
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(classifier_mid))
        classifier_layers.append(nn.Dropout(dropout))

        # 第三层：classifier_mid -> num_classes
        classifier_layers.append(nn.Linear(classifier_mid, self.num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        logger.info(f"HTGATFraud模型初始化完成")
        logger.info(f"  输入维度: {self.input_dim}")
        logger.info(f"  HGT隐藏维度: {self.hgt_hidden_dim}")
        logger.info(f"  时间片数量: {self.num_snapshots}")
        logger.info(f"  使用TGN记忆: {self.use_memory}")
        logger.info(f"  输出类别数: {self.num_classes}")
    
    def forward(
        self,
        data,
        num_snapshots: Optional[int] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        前向传播
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            完整图数据，包含：
            - x: 节点特征 [N, F]
            - edge_index: 边索引 [2, E]
            - edge_type: 边类型 [E]
            - timestamps: 节点时间戳 [N]
        num_snapshots : int, optional
            时间片数量，默认使用配置中的值
        return_attention : bool
            是否返回注意力权重
            
        Returns
        -------
        logits : torch.Tensor
            分类logits [N, num_classes]
        attention_dict : Dict, optional
            注意力权重字典（如果return_attention=True）
        """
        if num_snapshots is None:
            num_snapshots = self.num_snapshots
        
        num_nodes = data.x.size(0)
        device = data.x.device
        
        # ========================================
        # Step 0: 重置TGN记忆（如果启用）
        # ========================================
        if self.use_memory:
            self.memory.reset()
        
        # ========================================
        # Step 1: 时间片划分
        # ========================================
        snapshot_strategy = self.config.get('snapshot_strategy', 'edge_based')
        snapshots = split_graph_into_snapshots(
            data,
            num_snapshots=num_snapshots,
            strategy=snapshot_strategy,
            timestamps=data.timestamps
        )
        
        logger.debug(f"划分为 {len(snapshots)} 个时间片")
        
        # ========================================
        # Step 2: 对每个时间片处理（HGT + TGN记忆）
        # ========================================
        hgt_outputs = []
        snapshot_times = []  # 记录每个快照的时间
        
        for i, snapshot in enumerate(snapshots):
            # 将snapshot移到正确的设备
            snapshot = snapshot.to(device)
            
            # 2.1 HGT空间聚合
            h = self.hgt(
                node_feature=snapshot.x,
                node_type=snapshot.node_type,
                edge_index=snapshot.edge_index,
                edge_type=snapshot.edge_type
            )  # [N_k, hgt_hidden_dim]

            # 2.2 如果启用TGN记忆，进行记忆融合和更新
            if self.use_memory:
                active_nodes = snapshot.original_indices
                local_edge_index = snapshot.edge_index  # [2, E_k] 局部索引
                num_edges = local_edge_index.size(1) if local_edge_index.numel() > 0 else 0
                num_nodes_in_snapshot = h.size(0)
                
                # 确保节点ID唯一（每个快照中每个节点应该只出现一次）
                if len(active_nodes.unique()) != len(active_nodes):
                    logger.warning(f"快照{i}中检测到重复节点，进行去重处理")
                    # 去重并保持顺序
                    unique_mask = torch.zeros(len(snapshot.original_indices), dtype=torch.bool, device=device)
                    seen = set()
                    for idx, node in enumerate(snapshot.original_indices.tolist()):
                        if node not in seen:
                            unique_mask[idx] = True
                            seen.add(node)
                    h = h[unique_mask]
                    active_nodes = active_nodes[unique_mask]
                    num_nodes_in_snapshot = h.size(0)
                    
                    # 需要重新映射边索引
                    old_to_new = torch.full((snapshot.original_indices.max().item() + 1,), -1, dtype=torch.long, device=device)
                    old_to_new[torch.where(unique_mask)[0]] = torch.arange(num_nodes_in_snapshot, device=device)
                    valid_edges = (old_to_new[local_edge_index[0]] >= 0) & (old_to_new[local_edge_index[1]] >= 0)
                    local_edge_index = local_edge_index[:, valid_edges]
                    local_edge_index = old_to_new[local_edge_index]
                    num_edges = local_edge_index.size(1)
                
                # 计算当前快照时间
                current_time = snapshot.timestamps.mean().item()
                snapshot_times.append(current_time)
                
                # 读取活跃节点的当前记忆
                node_memory = self.memory.get_memory(active_nodes)  # [N_k, memory_dim]
                
                # ========== 基于边的消息传递 ==========
                if num_edges > 0:
                    # 将局部索引转换为全局索引以访问记忆
                    local_src = local_edge_index[0]  # [E_k]
                    local_dst = local_edge_index[1]  # [E_k]
                    global_src = active_nodes[local_src]  # [E_k] 全局节点ID
                    global_dst = active_nodes[local_dst]  # [E_k] 全局节点ID
                    
                    # ===== 关键修复：使用HGT嵌入而不是记忆来计算消息 =====
                    # 原问题：使用记忆计算消息导致冷启动问题
                    # - 第一个时间片记忆全是零，消息也接近零
                    # - 形成恶性循环，记忆始终没有有用信息
                    # 
                    # 正确做法：消息应该基于当前的HGT嵌入
                    # - HGT嵌入包含当前快照的空间邻居信息
                    # - 消息传递将这些信息聚合并存入记忆供后续时间片使用
                    src_embedding = h[local_src]  # [E_k, hgt_hidden_dim] 使用HGT嵌入
                    dst_embedding = h[local_dst]  # [E_k, hgt_hidden_dim] 使用HGT嵌入
                    
                    # 计算每条边的时间编码
                    src_last_update = self.memory.get_last_update(global_src)  # [E_k]
                    time_delta = torch.full_like(src_last_update, current_time) - src_last_update
                    edge_time_encoding = self.time_encoder(time_delta)  # [E_k, time_dim]
                    
                    # 计算边消息：从src传递到dst（使用HGT嵌入）
                    edge_messages = self.message_function(
                        src_memory=src_embedding,  # 改为使用HGT嵌入
                        dst_memory=dst_embedding,  # 改为使用HGT嵌入
                        edge_features=None,
                        time_encoding=edge_time_encoding
                    )  # [E_k, message_dim]
                    
                    # 使用scatter_add聚合消息到目标节点
                    # aggregated_messages[j] = sum(edge_messages[k] for k where local_dst[k] == j)
                    aggregated_messages = torch.zeros(
                        num_nodes_in_snapshot, edge_messages.size(1),
                        device=device, dtype=edge_messages.dtype
                    )
                    aggregated_messages.scatter_add_(
                        0,
                        local_dst.unsqueeze(-1).expand_as(edge_messages),
                        edge_messages
                    )
                    
                    # 计算每个节点的入边数量（用于mean聚合）
                    edge_count = torch.zeros(num_nodes_in_snapshot, device=device)
                    edge_count.scatter_add_(0, local_dst, torch.ones(num_edges, device=device))
                    edge_count = edge_count.clamp(min=1).unsqueeze(-1)  # [N_k, 1]
                    
                    # 平均聚合
                    aggregated_messages = aggregated_messages / edge_count  # [N_k, message_dim]
                else:
                    # 没有边时，使用零消息
                    aggregated_messages = torch.zeros(
                        num_nodes_in_snapshot, self.config.get('message_dim', self.hgt_hidden_dim),
                        device=device
                    )
                
                # 更新记忆：使用聚合后的邻居消息
                new_memory = self.memory_updater(aggregated_messages, node_memory)
                self.memory.set_memory(active_nodes, new_memory)
                self.memory.update_last_update_time(
                    active_nodes,
                    torch.full((len(active_nodes),), current_time, device=device)
                )
                
                # ===== 每个时间片的记忆融合 =====
                # 直接使用 new_memory（保留梯度），而非从 buffer 重新读取（已detach）
                # 这样 fusion_layer、memory_updater、message_function 都能获得完整梯度
                if new_memory.dtype != h.dtype:
                    new_memory = new_memory.to(h.dtype)
                h = self.fusion_layer(h, new_memory)

            # 2.3 HGT后投影：节点级别扩展维度（不增加边级别显存）
            # 放在TGN融合之后，确保TGN管线在原始hgt_hidden_dim下运行
            if self.post_hgt_proj is not None:
                h = self.post_hgt_proj(h)  # [N_k, working_dim]

            hgt_outputs.append((snapshot.original_indices, h))
        
        # ========================================
        # Step 3: 整合为全局时间序列表示
        # ========================================
        # 创建时间序列张量: [N, T, F]
        temporal_sequences = torch.zeros(
            num_nodes,
            num_snapshots,
            self.working_dim,
            device=device
        )
        
        # 创建时序掩码: [N, T]
        temporal_masks = torch.zeros(num_nodes, num_snapshots, device=device)
        
        # 填充时间序列
        for t, (original_indices, h_t) in enumerate(hgt_outputs):
            # 确保dtype匹配（AMP可能产生float16，但temporal_sequences是float32）
            if h_t.dtype != temporal_sequences.dtype:
                h_t = h_t.to(temporal_sequences.dtype)
            temporal_sequences[original_indices, t, :] = h_t
            temporal_masks[original_indices, t] = 1.0
        
        # ===== 强化学习自适应记忆门控 =====
        # 核心改进：使用 RL 策略网络动态学习每个节点的最优记忆权重
        # - Actor 网络根据节点状态（记忆、活跃模式）输出权重分布
        # - Critic 网络估计状态价值，用于减小策略梯度方差
        # - 奖励信号来自分类性能的反馈
        if self.use_memory:
            # 找出不活跃的时间片
            inactive_mask = (temporal_masks == 0)  # [N, T]
            
            if inactive_mask.any():
                # 获取所有节点的最终记忆状态
                all_memory = self.memory.get_memory(
                    torch.arange(num_nodes, device=device)
                )  # [N, memory_dim]
                
                # 确保dtype匹配
                if all_memory.dtype != temporal_sequences.dtype:
                    all_memory = all_memory.to(temporal_sequences.dtype)
                
                # ===== 记忆权重策略 =====
                if self.use_rl_memory_gate and self.memory_policy is not None:
                    # 使用 RL 策略网络计算每个节点的记忆权重
                    self.rl_output = self.memory_policy(
                        memory=all_memory,
                        temporal_masks=temporal_masks,
                        deterministic=not self.training  # 测试时使用确定性策略
                    )
                    memory_weights = self.rl_output.weights  # [N]
                else:
                    # 使用固定记忆权重（从配置中读取）
                    memory_weights = torch.full(
                        (num_nodes,), self.fixed_memory_weight, 
                        device=device, dtype=temporal_sequences.dtype
                    )
                    self.rl_output = None
                
                # 记忆维度可能与工作维度不同，需要投影
                if all_memory.size(-1) != self.working_dim:
                    if self.post_hgt_proj is not None:
                        # 使用HGT后投影层将记忆从hgt_hidden_dim投影到working_dim
                        all_memory = self.post_hgt_proj(all_memory)
                    elif hasattr(self.fusion_layer, 'memory_proj') and self.fusion_layer.memory_proj is not None:
                        all_memory = self.fusion_layer.memory_proj(all_memory)
                
                # 扩展记忆到时间维度 [N, memory_dim] -> [N, T, F]
                memory_expanded = all_memory.unsqueeze(1).expand(-1, num_snapshots, -1)
                
                # 用记忆填充不活跃位置
                temporal_sequences = torch.where(
                    inactive_mask.unsqueeze(-1).expand_as(temporal_sequences),
                    memory_expanded,
                    temporal_sequences
                )
                
                # 使用 RL 学习的权重更新 temporal_masks
                # 每个节点有自己的记忆权重，扩展到时间维度
                weights_expanded = memory_weights.unsqueeze(1).expand(-1, num_snapshots)  # [N, T]
                temporal_masks = torch.where(
                    inactive_mask,
                    weights_expanded,  # RL 学习的记忆权重
                    temporal_masks     # 真实观测位置保持 1.0
                )
        
        logger.debug(f"时间序列形状: {temporal_sequences.shape}")
        logger.debug(f"有效时间片比例: {temporal_masks.mean().item():.2%}")
        
        # ========================================
        # Step 4: 时序注意力聚合
        # ========================================
        h_temporal = self.temporal_attention(temporal_sequences, temporal_masks)  # [N, hgt_hidden_dim]
        
        # ========================================
        # Step 5: 分类器
        # ========================================
        logits = self.classifier(h_temporal)  # [N, num_classes]
        
        # 返回结果
        if return_attention:
            attention_dict = {
                'temporal_attention': self.temporal_attention.get_attention_weights()
            }
            return logits, attention_dict
        else:
            return logits
    
    def compute_rl_loss(
        self, 
        reward: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算强化学习损失
        
        使用 Actor-Critic 方法：
        - Policy Loss: 策略梯度损失，使用优势函数 (reward - value)
        - Value Loss: Critic 网络的 MSE 损失
        - Entropy Loss: 熵正则化，鼓励探索
        
        Parameters
        ----------
        reward : torch.Tensor [N] 或标量
            奖励信号（通常基于分类性能）
            如果是标量，会广播到所有节点
        mask : torch.Tensor [N], optional
            节点掩码，用于只计算部分节点的损失
            
        Returns
        -------
        losses : Dict[str, torch.Tensor]
            包含各部分损失：
            - 'policy_loss': 策略梯度损失
            - 'value_loss': 价值网络损失
            - 'entropy_loss': 熵正则化项（负数，因为要最大化熵）
            - 'total_rl_loss': 加权后的总 RL 损失
        """
        if self.rl_output is None:
            # 没有 RL 输出时返回零损失
            zero = torch.tensor(0.0, device=reward.device if hasattr(reward, 'device') else 'cpu')
            return {
                'policy_loss': zero,
                'value_loss': zero,
                'entropy_loss': zero,
                'total_rl_loss': zero
            }
        
        # 获取 RL 输出
        log_probs = self.rl_output.log_probs    # [N]
        values = self.rl_output.values          # [N]
        entropy = self.rl_output.entropy        # 标量
        
        # 确保 reward 是正确的形状
        if reward.dim() == 0:
            # 标量 reward，广播到所有节点
            reward = reward.expand_as(values)
        
        # 应用 mask（如果提供）
        if mask is not None:
            log_probs = log_probs[mask]
            values = values[mask]
            reward = reward[mask] if reward.dim() > 0 else reward
        
        # 计算优势函数 A = r - V(s)
        # 使用 detach 阻止梯度流向 Critic（Actor 更新时不更新 Critic）
        advantage = reward - values.detach()
        
        # Policy Loss: -E[log π(a|s) * A]
        # 负号是因为我们要最大化期望奖励，但优化器做的是最小化
        policy_loss = -(log_probs * advantage).mean()
        
        # Value Loss: E[(r - V(s))^2]
        # Critic 试图准确预测奖励
        value_loss = F.mse_loss(values, reward.detach())
        
        # Entropy Loss: -H(π)
        # 熵越大越好（鼓励探索），但我们要最小化损失，所以取负
        entropy_loss = -entropy
        
        # 总 RL 损失
        total_rl_loss = (
            self.rl_policy_weight * policy_loss +
            self.rl_value_weight * value_loss +
            self.rl_entropy_weight * entropy_loss
        )
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_rl_loss': total_rl_loss
        }
    
    def get_rl_output(self) -> Optional[RLPolicyOutput]:
        """获取最近一次前向传播的 RL 输出"""
        return self.rl_output
    
    def get_memory_weights_stats(self) -> Optional[Dict[str, float]]:
        """
        获取记忆权重的统计信息（用于监控和调试）
        
        Returns
        -------
        stats : Dict[str, float] 或 None
            包含 mean, std, min, max 等统计量
        """
        if self.rl_output is None:
            return None
        
        weights = self.rl_output.weights
        return {
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'min': weights.min().item(),
            'max': weights.max().item(),
            'median': weights.median().item()
        }
    
    def get_embeddings(self, data, num_snapshots: Optional[int] = None) -> torch.Tensor:
        """
        获取节点嵌入（用于可视化等）
        
        Parameters
        ----------
        data : Data
            完整图数据
        num_snapshots : int, optional
            时间片数量
            
        Returns
        -------
        embeddings : torch.Tensor
            节点嵌入 [N, hgt_hidden_dim]
        """
        if num_snapshots is None:
            num_snapshots = self.num_snapshots
        
        num_nodes = data.x.size(0)
        device = data.x.device
        
        # 时间片划分
        snapshot_strategy = self.config.get('snapshot_strategy', 'edge_based')
        snapshots = split_graph_into_snapshots(
            data,
            num_snapshots=num_snapshots,
            strategy=snapshot_strategy,
            timestamps=data.timestamps
        )
        
        # HGT处理每个快照
        hgt_outputs = []
        for snapshot in snapshots:
            snapshot = snapshot.to(device)
            h = self.hgt(
                node_feature=snapshot.x,
                node_type=snapshot.node_type,
                edge_index=snapshot.edge_index,
                edge_type=snapshot.edge_type
            )
            # HGT后投影
            if self.post_hgt_proj is not None:
                h = self.post_hgt_proj(h)
            hgt_outputs.append((snapshot.original_indices, h))

        # 整合时间序列
        temporal_sequences = torch.zeros(num_nodes, num_snapshots, self.working_dim, device=device)
        temporal_masks = torch.zeros(num_nodes, num_snapshots, device=device)
        
        for t, (original_indices, h_t) in enumerate(hgt_outputs):
            # 确保dtype匹配（AMP可能产生float16，但temporal_sequences是float32）
            if h_t.dtype != temporal_sequences.dtype:
                h_t = h_t.to(temporal_sequences.dtype)
            temporal_sequences[original_indices, t, :] = h_t
            temporal_masks[original_indices, t] = 1.0
        
        # 时序注意力
        embeddings = self.temporal_attention(temporal_sequences, temporal_masks)
        
        return embeddings


if __name__ == '__main__':
    # 测试代码
    import sys
    sys.path.append('..')
    from pathlib import Path
    import pickle
    import numpy as np
    from torch_geometric.data import Data
    
    logging.basicConfig(level=logging.INFO)
    
    # 加载真实数据进行测试
    data_dir = Path(__file__).parent.parent.parent.parent / '新建图方法' / 'output'
    
    if data_dir.exists():
        print("加载真实数据测试...")
        
        # 加载图数据
        with open(data_dir / 'graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)
        
        # 加载特征
        features = np.load(data_dir / 'features.npy')
        features = torch.FloatTensor(features)
        
        # 加载标签
        labels = np.load(data_dir / 'labels.npy')
        labels = torch.LongTensor(labels)
        labels = (labels + 1) // 2  # -1 -> 0, 1 -> 1
        
        # 创建Data对象
        data = Data(
            x=features,
            edge_index=torch.LongTensor(graph_data['edge_index']),
            edge_type=torch.LongTensor(graph_data['edge_type']),
            y=labels,
            timestamps=torch.FloatTensor(graph_data['timestamps'])
        )
        
        print(f"数据加载完成:")
        print(f"  节点数: {data.x.size(0)}")
        print(f"  边数: {data.edge_index.size(1)}")
        print(f"  特征维度: {data.x.size(1)}")
    else:
        print("使用模拟数据测试...")
        # 使用模拟数据
        num_nodes = 1000
        num_edges = 5000
        input_dim = 87
        
        data = Data(
            x=torch.randn(num_nodes, input_dim),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_type=torch.randint(0, 3, (num_edges,)),
            y=torch.randint(0, 2, (num_nodes,)),
            timestamps=torch.rand(num_nodes) * 100
        )
    
    # 创建模型配置
    config = {
        'input_dim': data.x.size(1),
        'num_classes': 2,
        'num_snapshots': 10,
        'num_node_types': 1,
        'num_edge_types': 3,
        'hgt_num_layers': 2,
        'hgt_hidden_dim': 128,
        'hgt_num_heads': 8,
        'hgt_dropout': 0.2,
        'hgt_use_norm': True,
        'hgt_use_rte': False,
        'temporal_num_heads': 4,
        'temporal_dropout': 0.3,
        'use_position_embedding': True,
        'use_causal_mask': False,
        'use_position_ffn': True,
        'dysat_hidden_dim': 128,
        'use_residual': False,
        'classifier_hidden_dim': 64,
        'classifier_dropout': 0.3,
        'use_batch_norm': True,
    }
    
    # 创建模型
    model = HTGATFraud(config)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 前向传播
    print("\n开始前向传播...")
    logits, attention_dict = model(data, return_attention=True)
    
    print(f"\n输出:")
    print(f"  Logits形状: {logits.shape}")
    print(f"  时序注意力权重形状: {attention_dict['temporal_attention'].shape}")
    
    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, data.y)
    print(f"  损失: {loss.item():.4f}")
    
    print("\n✓ HTGATFraud模型测试通过！")
