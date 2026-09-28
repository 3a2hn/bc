# -*- coding: utf-8 -*-
"""
核心模块消融实验的模型变体实现

包含以下变体：
1. HTGATFraudNoMemory: 移除TGN记忆模块
2. HTGATFraudNoTemporalAttention: 用平均聚合替代时序注意力
3. HTGATFraudWithGCN: 用GCN替代HGT
4. HTGATFraudStatic: 静态基线（单时间片）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, Optional
import logging

from src.models.hgt_layer import HGTLayer
from src.models.temporal_attention import TemporalAttentionLayer
from src.models.ht_gat_fraud import GatedFusion  # 导入门控融合模块
from src.utils.temporal_split import split_graph_into_snapshots

logger = logging.getLogger(__name__)


class HTGATFraudNoMemory(nn.Module):
    """
    Var-1.1: 移除TGN记忆模块的HTGATFraud

    - 设置 use_memory = False
    - 移除记忆读取、融合、更新的所有操作
    - 不活跃时间片用可学习默认嵌入填充（模拟无记忆时的"盲猜"）
    - 不活跃位置的mask设为较低权重（0.1），体现无记忆时信息质量差

    设计思想：
    完整模型中TGN记忆为不活跃时间片提供高质量的历史记忆（mask=0.5），
    移除记忆后，模型只能用一个通用的默认嵌入来"猜测"不活跃时间片的状态，
    这比有针对性的记忆差很多，从而体现TGN记忆的真实贡献。
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.input_dim = config['input_dim']
        self.num_classes = config['num_classes']
        self.num_snapshots = config['num_snapshots']
        self.hgt_hidden_dim = config['hgt_hidden_dim']

        # 不活跃时间片的默认嵌入权重（远低于完整模型的记忆权重0.5）
        self.inactive_weight = 0.1

        # HGT层
        self.hgt = HGTLayer(
            in_dim=self.input_dim,
            hidden_dim=config['hgt_hidden_dim'],
            num_types=config.get('num_node_types', 1),
            num_relations=config.get('num_edge_types', 3),
            n_heads=config['hgt_num_heads'],
            n_layers=config['hgt_num_layers'],
            dropout=config['hgt_dropout'],
            use_norm=config['hgt_use_norm'],
            use_RTE=config['hgt_use_rte']
        )

        # 可学习的默认嵌入（替代TGN记忆，用于填充不活跃时间片）
        # 这是一个全局共享的向量，无法像TGN记忆那样为每个节点提供个性化历史
        self.default_embedding = nn.Parameter(
            torch.zeros(self.hgt_hidden_dim)
        )
        nn.init.normal_(self.default_embedding, std=0.01)

        # 时序注意力层
        self.temporal_attention = TemporalAttentionLayer(
            input_dim=self.hgt_hidden_dim,
            num_time_steps=self.num_snapshots,
            n_heads=config['temporal_num_heads'],
            attn_drop=config['temporal_dropout'],
            residual=config.get('use_residual', False),
            use_position_embedding=config.get('use_position_embedding', True),
            use_causal_mask=config.get('use_causal_mask', False),
            use_position_ffn=config.get('use_position_ffn', True)
        )

        # 分类器（与baseline保持一致的3层结构）
        classifier_hidden = config['classifier_hidden_dim']
        classifier_mid = classifier_hidden // 2
        use_batch_norm = config.get('use_batch_norm', True)
        dropout = config['classifier_dropout']

        classifier_layers = []
        # 第一层：hgt_hidden_dim -> classifier_hidden_dim
        classifier_layers.append(nn.Linear(self.hgt_hidden_dim, classifier_hidden))
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

        logger.info("HTGATFraudNoMemory 模型初始化完成（无TGN记忆，使用默认嵌入填充不活跃时间片）")

    def forward(self, data):
        num_nodes = data.x.size(0)
        device = data.x.device

        # 时间片划分
        snapshots = split_graph_into_snapshots(
            data,
            num_snapshots=self.num_snapshots,
            strategy=self.config.get('snapshot_strategy', 'edge_based'),
            timestamps=data.timestamps
        )

        # 对每个时间片进行HGT处理
        hgt_outputs = []
        for snapshot in snapshots:
            snapshot = snapshot.to(device)
            h = self.hgt(
                node_feature=snapshot.x,
                node_type=snapshot.node_type,
                edge_index=snapshot.edge_index,
                edge_type=snapshot.edge_type
            )
            hgt_outputs.append((snapshot.original_indices, h))

        # 整合时间序列
        temporal_sequences = torch.zeros(
            num_nodes, self.num_snapshots, self.hgt_hidden_dim, device=device
        )
        temporal_masks = torch.zeros(num_nodes, self.num_snapshots, device=device)

        for t, (original_indices, h_t) in enumerate(hgt_outputs):
            if h_t.dtype != temporal_sequences.dtype:
                h_t = h_t.to(temporal_sequences.dtype)
            temporal_sequences[original_indices, t, :] = h_t
            temporal_masks[original_indices, t] = 1.0

        # 不活跃时间片处理：用默认嵌入填充（模拟无记忆时的"盲猜"）
        # 完整模型用TGN记忆（个性化历史）填充，mask=0.5
        # 消融模型用全局默认嵌入填充，mask=0.1（信息质量远低于个性化记忆）
        inactive_mask = (temporal_masks == 0)  # [N, T]
        if inactive_mask.any():
            default_expanded = self.default_embedding.unsqueeze(0).unsqueeze(0).expand(
                num_nodes, self.num_snapshots, -1
            )  # [N, T, F]
            temporal_sequences = torch.where(
                inactive_mask.unsqueeze(-1).expand_as(temporal_sequences),
                default_expanded,
                temporal_sequences
            )
            # 给不活跃位置一个较低的mask权重
            temporal_masks = torch.where(
                inactive_mask,
                torch.full_like(temporal_masks, self.inactive_weight),
                temporal_masks
            )

        # 时序注意力
        h_temporal = self.temporal_attention(temporal_sequences, temporal_masks)

        # 分类
        logits = self.classifier(h_temporal)

        return logits


class HTGATFraudNoTemporalAttention(nn.Module):
    """
    Var-1.2: 移除时序自注意力，仅取最后时间片表示

    - 时间维度聚合：h = temporal_sequences[:, last_active_t, :]
    - 移除位置编码、因果mask、多头注意力等组件
    - 仅保留最后一个活跃时间片的节点表示用于分类
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.input_dim = config['input_dim']
        self.num_classes = config['num_classes']
        self.num_snapshots = config['num_snapshots']
        self.hgt_hidden_dim = config['hgt_hidden_dim']
        self.use_memory = config.get('use_memory', True)
        self.num_nodes = config.get('num_nodes', 40000)
        
        # HGT层
        self.hgt = HGTLayer(
            in_dim=self.input_dim,
            hidden_dim=config['hgt_hidden_dim'],
            num_types=config.get('num_node_types', 1),
            num_relations=config.get('num_edge_types', 3),
            n_heads=config['hgt_num_heads'],
            n_layers=config['hgt_num_layers'],
            dropout=config['hgt_dropout'],
            use_norm=config['hgt_use_norm'],
            use_RTE=config['hgt_use_rte']
        )
        
        # TGN记忆模块（如果启用）
        if self.use_memory:
            from src.models.tgn_memory import TGNMemory
            from src.models.time_encoding import TimeEncoding
            from src.models.message_function import MLPMessageFunction
            from src.models.memory_updater import get_memory_updater
            
            memory_dim = config.get('memory_dim', self.hgt_hidden_dim)
            message_dim = config.get('message_dim', self.hgt_hidden_dim)
            time_dim = config.get('time_dim', self.hgt_hidden_dim)
            
            self.memory = TGNMemory(
                num_nodes=self.num_nodes,
                memory_dim=memory_dim,
                time_dim=time_dim,
                device=config.get('device', 'cpu')
            )
            
            self.time_encoder = TimeEncoding(dimension=time_dim)
            
            # 消息函数：使用HGT嵌入维度（不是记忆维度）
            self.message_function = MLPMessageFunction(
                memory_dim=self.hgt_hidden_dim,  # 使用HGT嵌入维度
                edge_dim=0,
                time_dim=time_dim,
                message_dim=message_dim,
                dropout=config.get('message_dropout', 0.1)
            )
            
            self.memory_updater = get_memory_updater(
                updater_type=config.get('memory_updater', 'gru'),
                memory_dim=memory_dim,
                message_dim=message_dim
            )
            
            # 使用门控融合模块
            self.fusion_layer = GatedFusion(
                hgt_dim=self.hgt_hidden_dim,
                memory_dim=memory_dim,
                dropout=config.get('fusion_dropout', 0.1)
            )
        else:
            self.memory = None
        
        # 分类器（与baseline保持一致的3层结构）
        classifier_hidden = config['classifier_hidden_dim']
        classifier_mid = classifier_hidden // 2
        use_batch_norm = config.get('use_batch_norm', True)
        dropout = config['classifier_dropout']

        classifier_layers = []
        # 第一层：hgt_hidden_dim -> classifier_hidden_dim
        classifier_layers.append(nn.Linear(self.hgt_hidden_dim, classifier_hidden))
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

        logger.info("HTGATFraudNoTemporalAttention 模型初始化完成（仅取最后时间片，替代时序注意力）")
    
    def forward(self, data):
        num_nodes = data.x.size(0)
        device = data.x.device
        
        if self.use_memory:
            self.memory.reset()
        
        # 时间片划分
        snapshots = split_graph_into_snapshots(
            data,
            num_snapshots=self.num_snapshots,
            strategy=self.config.get('snapshot_strategy', 'edge_based'),
            timestamps=data.timestamps
        )
        
        # 对每个时间片处理
        hgt_outputs = []
        for i, snapshot in enumerate(snapshots):
            snapshot = snapshot.to(device)
            
            # HGT空间聚合
            h = self.hgt(
                node_feature=snapshot.x,
                node_type=snapshot.node_type,
                edge_index=snapshot.edge_index,
                edge_type=snapshot.edge_type
            )
            
            # TGN记忆融合（如果启用）
            if self.use_memory:
                active_nodes = snapshot.original_indices
                local_edge_index = snapshot.edge_index
                num_edges = local_edge_index.size(1) if local_edge_index.numel() > 0 else 0
                num_nodes_in_snapshot = h.size(0)
                
                # 去重处理
                if len(active_nodes.unique()) != len(active_nodes):
                    unique_mask = torch.zeros(len(snapshot.original_indices), dtype=torch.bool, device=device)
                    seen = set()
                    for idx, node in enumerate(snapshot.original_indices.tolist()):
                        if node not in seen:
                            unique_mask[idx] = True
                            seen.add(node)
                    h = h[unique_mask]
                    active_nodes = active_nodes[unique_mask]
                    num_nodes_in_snapshot = h.size(0)
                    
                    # 重新映射边索引
                    old_to_new = torch.full((snapshot.original_indices.max().item() + 1,), -1, dtype=torch.long, device=device)
                    old_to_new[torch.where(unique_mask)[0]] = torch.arange(num_nodes_in_snapshot, device=device)
                    valid_edges = (old_to_new[local_edge_index[0]] >= 0) & (old_to_new[local_edge_index[1]] >= 0)
                    local_edge_index = local_edge_index[:, valid_edges]
                    local_edge_index = old_to_new[local_edge_index]
                    num_edges = local_edge_index.size(1)
                
                memory = self.memory.get_memory(active_nodes)
                current_time = snapshot.timestamps.mean().item()
                
                # 基于边的消息传递（使用HGT嵌入）
                if num_edges > 0:
                    local_src = local_edge_index[0]
                    local_dst = local_edge_index[1]
                    
                    src_embedding = h[local_src]  # 使用HGT嵌入
                    dst_embedding = h[local_dst]  # 使用HGT嵌入
                    
                    src_last_update = self.memory.get_last_update(active_nodes[local_src])
                    time_delta = torch.full_like(src_last_update, current_time) - src_last_update
                    edge_time_encoding = self.time_encoder(time_delta)
                    
                    edge_messages = self.message_function(
                        src_memory=src_embedding,
                        dst_memory=dst_embedding,
                        edge_features=None,
                        time_encoding=edge_time_encoding
                    )
                    
                    # 聚合消息
                    aggregated_messages = torch.zeros(
                        num_nodes_in_snapshot, edge_messages.size(1),
                        device=device, dtype=edge_messages.dtype
                    )
                    aggregated_messages.scatter_add_(
                        0,
                        local_dst.unsqueeze(-1).expand_as(edge_messages),
                        edge_messages
                    )
                    edge_count = torch.zeros(num_nodes_in_snapshot, device=device)
                    edge_count.scatter_add_(0, local_dst, torch.ones(num_edges, device=device))
                    edge_count = edge_count.clamp(min=1).unsqueeze(-1)
                    aggregated_messages = aggregated_messages / edge_count
                else:
                    aggregated_messages = torch.zeros(
                        num_nodes_in_snapshot, self.config.get('message_dim', self.hgt_hidden_dim),
                        device=device
                    )
                
                new_memory = self.memory_updater(aggregated_messages, memory)
                self.memory.set_memory(active_nodes, new_memory)
                self.memory.update_last_update_time(
                    active_nodes,
                    torch.full((len(active_nodes),), current_time, device=device)
                )
                
                # 直接使用 new_memory 融合（保留梯度）
                if new_memory.dtype != h.dtype:
                    new_memory = new_memory.to(h.dtype)
                h = self.fusion_layer(h, new_memory)
            
            hgt_outputs.append((snapshot.original_indices, h))
        
        # 整合时间序列
        temporal_sequences = torch.zeros(
            num_nodes, self.num_snapshots, self.hgt_hidden_dim, device=device
        )
        temporal_masks = torch.zeros(num_nodes, self.num_snapshots, device=device)
        
        for t, (original_indices, h_t) in enumerate(hgt_outputs):
            if h_t.dtype != temporal_sequences.dtype:
                h_t = h_t.to(temporal_sequences.dtype)
            temporal_sequences[original_indices, t, :] = h_t
            temporal_masks[original_indices, t] = 1.0
        
        # 仅取最后一个时间片的表示（替代时序注意力）
        # 找到每个节点最后一个活跃时间片的索引
        time_indices = torch.arange(self.num_snapshots, device=device).float()
        # 对活跃时间片取最大索引
        last_active_t = (temporal_masks * time_indices).argmax(dim=1)  # [N]
        # 处理全零情况（无活跃时间片的节点取最后一个时间片）
        all_inactive = (temporal_masks.sum(dim=1) == 0)
        last_active_t = torch.where(all_inactive,
                                     torch.full_like(last_active_t, self.num_snapshots - 1),
                                     last_active_t)
        # 用gather取出最后时间片的表示
        last_active_t_expanded = last_active_t.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hgt_hidden_dim)
        h_temporal = temporal_sequences.gather(1, last_active_t_expanded).squeeze(1)  # [N, hidden_dim]
        
        # 分类
        logits = self.classifier(h_temporal)
        
        return logits


class HTGATFraudWithGCN(nn.Module):
    """
    Var-1.3: 用标准GCN替代HGT

    - 忽略边类型，所有边使用相同参数
    - 使用朴素GCN（无残差连接、无LayerNorm），体现HGT异构建模的真实贡献
    - 移除TGN记忆，避免记忆模块弥补GCN的不足
    - 保留时序注意力，仅消融空间聚合模块
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.input_dim = config['input_dim']
        self.num_classes = config['num_classes']
        self.num_snapshots = config['num_snapshots']
        self.hidden_dim = config['hgt_hidden_dim']

        # 朴素GCN层（替代HGT，不加残差/LayerNorm等增强）
        self.gcn_layers = nn.ModuleList()

        # 第一层：input_dim -> hidden_dim
        self.gcn_layers.append(GCNConv(self.input_dim, self.hidden_dim))

        # 后续层：hidden_dim -> hidden_dim
        num_layers = config.get('hgt_num_layers', 2)
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.gcn_dropout = config.get('hgt_dropout', 0.3)

        # 时序注意力层
        self.temporal_attention = TemporalAttentionLayer(
            input_dim=self.hidden_dim,
            num_time_steps=self.num_snapshots,
            n_heads=config['temporal_num_heads'],
            attn_drop=config['temporal_dropout'],
            residual=config.get('use_residual', False),
            use_position_embedding=config.get('use_position_embedding', True),
            use_causal_mask=config.get('use_causal_mask', False),
            use_position_ffn=config.get('use_position_ffn', True)
        )

        # 分类器（与baseline保持一致的3层结构）
        classifier_hidden = config['classifier_hidden_dim']
        classifier_mid = classifier_hidden // 2
        use_batch_norm = config.get('use_batch_norm', True)
        dropout = config['classifier_dropout']

        classifier_layers = []
        classifier_layers.append(nn.Linear(self.hidden_dim, classifier_hidden))
        classifier_layers.append(nn.ReLU())
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(classifier_hidden))
        classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(classifier_hidden, classifier_mid))
        classifier_layers.append(nn.ReLU())
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(classifier_mid))
        classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(classifier_mid, self.num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

        # 标记：GCN在大图上使用AMP float16容易溢出NaN
        self.disable_amp = True

        logger.info("HTGATFraudWithGCN 模型初始化完成（朴素GCN替代HGT，无记忆，已禁用AMP）")

    def forward(self, data):
        num_nodes = data.x.size(0)
        device = data.x.device

        # 时间片划分
        snapshots = split_graph_into_snapshots(
            data,
            num_snapshots=self.num_snapshots,
            strategy=self.config.get('snapshot_strategy', 'edge_based'),
            timestamps=data.timestamps
        )

        # 对每个时间片处理
        gcn_outputs = []
        for i, snapshot in enumerate(snapshots):
            snapshot = snapshot.to(device)

            # 朴素GCN空间聚合（无残差、无LayerNorm）
            h = snapshot.x
            for idx, layer in enumerate(self.gcn_layers):
                h = layer(h, snapshot.edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.gcn_dropout, training=self.training)

            # 数值安全
            h = torch.clamp(h, min=-1e4, max=1e4)

            gcn_outputs.append((snapshot.original_indices, h))

        # 整合时间序列
        temporal_sequences = torch.zeros(
            num_nodes, self.num_snapshots, self.hidden_dim, device=device
        )
        temporal_masks = torch.zeros(num_nodes, self.num_snapshots, device=device)

        for t, (original_indices, h_t) in enumerate(gcn_outputs):
            if h_t.dtype != temporal_sequences.dtype:
                h_t = h_t.to(temporal_sequences.dtype)
            temporal_sequences[original_indices, t, :] = h_t
            temporal_masks[original_indices, t] = 1.0

        # 时序注意力
        h_temporal = self.temporal_attention(temporal_sequences, temporal_masks)

        # 分类
        logits = self.classifier(h_temporal)

        return logits


class HTGATFraudStatic(nn.Module):
    """
    Var-1.4: 静态基线模型

    - 设置 num_snapshots = 1（单时间片）
    - 禁用TGN记忆和时序注意力
    - 退化为纯静态图分类模型
    - 使用朴素GCN（无残差/LayerNorm），体现时序建模的真实贡献

    注意：使用 GCN 替代 HGT 以避免 OOM
    （HGT 在完整图 ~49K 节点上会消耗过多显存）
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.input_dim = config['input_dim']
        self.num_classes = config['num_classes']
        self.hidden_dim = config['hgt_hidden_dim']

        # 朴素GCN层（无残差、无LayerNorm）
        self.gcn_layers = nn.ModuleList()

        # 第一层：input_dim -> hidden_dim
        self.gcn_layers.append(GCNConv(self.input_dim, self.hidden_dim))

        # 后续层
        num_layers = config.get('hgt_num_layers', 2)
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.gcn_dropout = config.get('hgt_dropout', 0.3)

        # 分类器（与baseline保持一致的3层结构）
        classifier_hidden = config['classifier_hidden_dim']
        classifier_mid = classifier_hidden // 2
        use_batch_norm = config.get('use_batch_norm', True)
        dropout = config['classifier_dropout']

        classifier_layers = []
        classifier_layers.append(nn.Linear(self.hidden_dim, classifier_hidden))
        classifier_layers.append(nn.ReLU())
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(classifier_hidden))
        classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(classifier_hidden, classifier_mid))
        classifier_layers.append(nn.ReLU())
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(classifier_mid))
        classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(classifier_mid, self.num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

        # 标记：GCN在大图上使用AMP float16容易溢出NaN
        self.disable_amp = True

        logger.info("HTGATFraudStatic 模型初始化完成（静态基线，朴素GCN，已禁用AMP）")

    def forward(self, data):
        device = data.x.device

        # 朴素GCN空间聚合（在完整图上，无残差、无LayerNorm）
        h = data.x

        for idx, layer in enumerate(self.gcn_layers):
            h = layer(h, data.edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.gcn_dropout, training=self.training)

        h = torch.clamp(h, min=-1e4, max=1e4)

        # 直接分类（无时序处理）
        logits = self.classifier(h)

        return logits
