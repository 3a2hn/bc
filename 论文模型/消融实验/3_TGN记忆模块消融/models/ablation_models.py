# -*- coding: utf-8 -*-
"""
TGN记忆模块消融实验 — 模型变体实现

消融目标：TGN记忆模块流水线中的4个关键架构决策
  HGT嵌入 → 消息计算 → 聚合 → 更新 → 门控融合 → 填充不活跃时间片
                ↑                        ↑              ↑
        message_from_memory       wo_gated_fusion  wo_memory_filling
        wo_time_encoding

包含以下变体：
1. HTGATFraudSimpleFusion      — 门控融合 → 简单加法融合
2. HTGATFraudNoMemoryFilling   — 不用记忆填充不活跃时间片
3. HTGATFraudMessageFromMemory — 消息基于记忆而非HGT嵌入（还原冷启动问题）
4. HTGATFraudNoTimeEncoding    — 消息函数去掉时间编码
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

from src.models.hgt_layer import HGTLayer
from src.models.temporal_attention import TemporalAttentionLayer
from src.models.tgn_memory import TGNMemory
from src.models.time_encoding import TimeEncoding
from src.models.message_function import MLPMessageFunction
from src.models.memory_updater import get_memory_updater
from src.models.ht_gat_fraud import GatedFusion
from src.utils.temporal_split import split_graph_into_snapshots

logger = logging.getLogger(__name__)


# ============================================================================
# 公共工具
# ============================================================================

def _build_classifier(input_dim: int, config: Dict) -> nn.Sequential:
    """构建与baseline一致的3层分类器"""
    hidden = config['classifier_hidden_dim']
    mid = hidden // 2
    use_bn = config.get('use_batch_norm', True)
    drop = config['classifier_dropout']

    layers = []
    for in_d, out_d in [(input_dim, hidden), (hidden, mid)]:
        layers.append(nn.Linear(in_d, out_d))
        layers.append(nn.ReLU())
        if use_bn:
            layers.append(nn.BatchNorm1d(out_d))
        layers.append(nn.Dropout(drop))
    layers.append(nn.Linear(mid, config['num_classes']))
    return nn.Sequential(*layers)


def _build_hgt(config: Dict) -> HGTLayer:
    """构建HGT层"""
    return HGTLayer(
        in_dim=config['input_dim'],
        hidden_dim=config['hgt_hidden_dim'],
        num_types=config.get('num_node_types', 1),
        num_relations=config.get('num_edge_types', 3),
        n_heads=config['hgt_num_heads'],
        n_layers=config['hgt_num_layers'],
        dropout=config['hgt_dropout'],
        use_norm=config['hgt_use_norm'],
        use_RTE=config['hgt_use_rte'],
    )


def _build_temporal_attention(config: Dict) -> TemporalAttentionLayer:
    """构建时序注意力层"""
    return TemporalAttentionLayer(
        input_dim=config['hgt_hidden_dim'],
        num_time_steps=config['num_snapshots'],
        n_heads=config['temporal_num_heads'],
        attn_drop=config['temporal_dropout'],
        residual=config.get('use_residual', False),
        use_position_embedding=config.get('use_position_embedding', True),
        use_causal_mask=config.get('use_causal_mask', False),
        use_position_ffn=config.get('use_position_ffn', True),
    )


def _build_tgn_components(config: Dict):
    """构建TGN记忆模块的全部子组件，返回元组"""
    hgt_dim = config['hgt_hidden_dim']
    memory_dim = config.get('memory_dim', hgt_dim)
    message_dim = config.get('message_dim', hgt_dim)
    time_dim = config.get('time_dim', hgt_dim)

    memory = TGNMemory(
        num_nodes=config.get('num_nodes', 40000),
        memory_dim=memory_dim,
        time_dim=time_dim,
        device=config.get('device', 'cpu'),
    )
    time_encoder = TimeEncoding(dimension=time_dim)
    message_function = MLPMessageFunction(
        memory_dim=hgt_dim,
        edge_dim=config.get('edge_feature_dim', 0),
        time_dim=time_dim,
        message_dim=message_dim,
        dropout=config.get('message_dropout', 0.1),
    )
    memory_updater = get_memory_updater(
        updater_type=config.get('memory_updater', 'gru'),
        memory_dim=memory_dim,
        message_dim=message_dim,
    )
    return memory, time_encoder, message_function, memory_updater, memory_dim, message_dim, time_dim


def _snapshot_loop_with_memory(
    model, data, snapshots,
    *,
    use_gated_fusion: bool = True,
    skip_fusion: bool = False,         # True → 完全不融合记忆，只用纯HGT嵌入
    message_source: str = 'hgt',       # 'hgt' | 'memory'
    use_time_encoding: bool = True,
):
    """
    通用的快照循环：HGT → 消息传递 → 记忆更新 → 融合

    通过参数控制消融点，避免每个变体重复大段相同代码。
    """
    num_nodes = data.x.size(0)
    device = data.x.device
    hgt_dim = model.hgt_hidden_dim
    num_snapshots = model.num_snapshots

    model.memory.reset()

    hgt_outputs = []
    for i, snapshot in enumerate(snapshots):
        snapshot = snapshot.to(device)

        # HGT空间聚合
        h = model.hgt(
            node_feature=snapshot.x,
            node_type=snapshot.node_type,
            edge_index=snapshot.edge_index,
            edge_type=snapshot.edge_type,
        )

        active_nodes = snapshot.original_indices
        local_edge_index = snapshot.edge_index
        num_edges = local_edge_index.size(1) if local_edge_index.numel() > 0 else 0
        num_nodes_in_snapshot = h.size(0)

        # 去重
        if len(active_nodes.unique()) != len(active_nodes):
            unique_mask = torch.zeros(len(active_nodes), dtype=torch.bool, device=device)
            seen = set()
            for idx, node in enumerate(active_nodes.tolist()):
                if node not in seen:
                    unique_mask[idx] = True
                    seen.add(node)
            h = h[unique_mask]
            active_nodes = active_nodes[unique_mask]
            num_nodes_in_snapshot = h.size(0)
            old_to_new = torch.full((snapshot.original_indices.max().item() + 1,), -1, dtype=torch.long, device=device)
            old_to_new[torch.where(unique_mask)[0]] = torch.arange(num_nodes_in_snapshot, device=device)
            valid_edges = (old_to_new[local_edge_index[0]] >= 0) & (old_to_new[local_edge_index[1]] >= 0)
            local_edge_index = local_edge_index[:, valid_edges]
            local_edge_index = old_to_new[local_edge_index]
            num_edges = local_edge_index.size(1)

        current_time = snapshot.timestamps.mean().item()
        node_memory = model.memory.get_memory(active_nodes)

        # ---------- 消息传递 ----------
        if num_edges > 0:
            local_src = local_edge_index[0]
            local_dst = local_edge_index[1]
            global_src = active_nodes[local_src]

            # 消融点：消息来源
            if message_source == 'memory':
                src_embedding = node_memory[local_src]   # 用记忆（冷启动问题）
                dst_embedding = node_memory[local_dst]
            else:
                src_embedding = h[local_src]             # baseline: 用HGT嵌入
                dst_embedding = h[local_dst]

            # 消融点：时间编码
            if use_time_encoding:
                src_last_update = model.memory.get_last_update(global_src)
                time_delta = torch.full_like(src_last_update, current_time) - src_last_update
                edge_time_encoding = model.time_encoder(time_delta)
            else:
                edge_time_encoding = None

            edge_messages = model.message_function(
                src_memory=src_embedding,
                dst_memory=dst_embedding,
                edge_features=None,
                time_encoding=edge_time_encoding,
            )

            aggregated_messages = torch.zeros(
                num_nodes_in_snapshot, edge_messages.size(1),
                device=device, dtype=edge_messages.dtype,
            )
            aggregated_messages.scatter_add_(
                0,
                local_dst.unsqueeze(-1).expand_as(edge_messages),
                edge_messages,
            )
            edge_count = torch.zeros(num_nodes_in_snapshot, device=device)
            edge_count.scatter_add_(0, local_dst, torch.ones(num_edges, device=device))
            edge_count = edge_count.clamp(min=1).unsqueeze(-1)
            aggregated_messages = aggregated_messages / edge_count
        else:
            msg_dim = model.config.get('message_dim', hgt_dim)
            aggregated_messages = torch.zeros(num_nodes_in_snapshot, msg_dim, device=device)

        # ---------- 记忆更新 ----------
        new_memory = model.memory_updater(aggregated_messages, node_memory)
        model.memory.set_memory(active_nodes, new_memory)
        model.memory.update_last_update_time(
            active_nodes,
            torch.full((len(active_nodes),), current_time, device=device),
        )

        # ---------- 融合 ----------
        if new_memory.dtype != h.dtype:
            new_memory = new_memory.to(h.dtype)

        if skip_fusion:
            # 完全不融合记忆，直接使用纯HGT嵌入
            # 记忆仍然被更新（供填充不活跃时间片使用），但不影响活跃时间片的表示
            pass
        elif use_gated_fusion:
            h = model.fusion_layer(h, new_memory)
        else:
            # 简单加法融合
            if hasattr(model, 'memory_proj') and model.memory_proj is not None:
                new_memory = model.memory_proj(new_memory)
            h = h + new_memory
            h = model.fusion_norm(h)

        hgt_outputs.append((active_nodes, h))

    return hgt_outputs


def _assemble_temporal(hgt_outputs, num_nodes, num_snapshots, hgt_dim, device):
    """将快照输出整合为 [N, T, F] 时间序列"""
    temporal_sequences = torch.zeros(num_nodes, num_snapshots, hgt_dim, device=device)
    temporal_masks = torch.zeros(num_nodes, num_snapshots, device=device)
    for t, (indices, h_t) in enumerate(hgt_outputs):
        if h_t.dtype != temporal_sequences.dtype:
            h_t = h_t.to(temporal_sequences.dtype)
        temporal_sequences[indices, t, :] = h_t
        temporal_masks[indices, t] = 1.0
    return temporal_sequences, temporal_masks


def _fill_inactive_with_memory(model, temporal_sequences, temporal_masks, num_nodes, num_snapshots, device):
    """用记忆填充不活跃时间片（baseline行为）"""
    inactive_mask = (temporal_masks == 0)
    if not inactive_mask.any():
        return temporal_sequences, temporal_masks

    all_memory = model.memory.get_memory(torch.arange(num_nodes, device=device))
    if all_memory.dtype != temporal_sequences.dtype:
        all_memory = all_memory.to(temporal_sequences.dtype)

    hgt_dim = temporal_sequences.size(-1)
    if all_memory.size(-1) != hgt_dim:
        if hasattr(model, 'fusion_layer') and hasattr(model.fusion_layer, 'memory_proj') and model.fusion_layer.memory_proj is not None:
            all_memory = model.fusion_layer.memory_proj(all_memory)

    memory_expanded = all_memory.unsqueeze(1).expand(-1, num_snapshots, -1)
    temporal_sequences = torch.where(
        inactive_mask.unsqueeze(-1).expand_as(temporal_sequences),
        memory_expanded,
        temporal_sequences,
    )

    weight = model.config.get('fixed_memory_weight', 0.3)
    weights_expanded = torch.full_like(temporal_masks, weight)
    temporal_masks = torch.where(inactive_mask, weights_expanded, temporal_masks)

    return temporal_sequences, temporal_masks


# ============================================================================
# 变体1: wo_gated_fusion — 门控融合 → 简单加法融合
# ============================================================================

class HTGATFraudSimpleFusion(nn.Module):
    """
    消融门控融合机制（GatedFusion → 完全不融合）

    baseline使用GRU风格的门控融合，包含重置门和更新门，
    能学习何时依赖空间信息、何时依赖历史记忆。

    本变体完全跳过融合步骤：活跃时间片只使用纯HGT嵌入，
    记忆仅用于填充不活跃时间片。验证门控融合对活跃时间片表示的增强作用。
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_snapshots = config['num_snapshots']
        self.hgt_hidden_dim = config['hgt_hidden_dim']

        self.hgt = _build_hgt(config)

        # TGN组件（记忆仍然更新，但不融合到HGT嵌入中）
        (self.memory, self.time_encoder, self.message_function,
         self.memory_updater, memory_dim, _, _) = _build_tgn_components(config)

        # 保留fusion_layer仅用于memory_proj（填充不活跃时间片时可能需要维度对齐）
        self.fusion_layer = GatedFusion(
            hgt_dim=self.hgt_hidden_dim,
            memory_dim=memory_dim,
            dropout=config.get('fusion_dropout', 0.1),
        )

        self.temporal_attention = _build_temporal_attention(config)
        self.classifier = _build_classifier(self.hgt_hidden_dim, config)

        logger.info("HTGATFraudSimpleFusion 初始化完成（跳过融合，记忆仅用于填充不活跃时间片）")

    def forward(self, data):
        device = data.x.device
        num_nodes = data.x.size(0)

        snapshots = split_graph_into_snapshots(
            data, num_snapshots=self.num_snapshots,
            strategy=self.config.get('snapshot_strategy', 'edge_based'),
            timestamps=data.timestamps,
        )

        hgt_outputs = _snapshot_loop_with_memory(
            self, data, snapshots,
            skip_fusion=True,  # 完全不融合记忆到HGT嵌入
        )

        temporal_sequences, temporal_masks = _assemble_temporal(
            hgt_outputs, num_nodes, self.num_snapshots, self.hgt_hidden_dim, device,
        )
        temporal_sequences, temporal_masks = _fill_inactive_with_memory(
            self, temporal_sequences, temporal_masks, num_nodes, self.num_snapshots, device,
        )

        h = self.temporal_attention(temporal_sequences, temporal_masks)
        return self.classifier(h)


# ============================================================================
# 变体2: wo_memory_filling — 不用记忆填充不活跃时间片
# ============================================================================

class HTGATFraudNoMemoryFilling(nn.Module):
    """
    消融不活跃时间片的记忆填充策略

    baseline中，节点在某个时间片不活跃时，用TGN记忆向量填充该位置，
    并赋予固定权重(0.3)，让时序注意力能利用历史信息。
    同时，活跃时间片通过门控融合也受到记忆的增强。

    本变体彻底移除记忆对最终表示的所有影响：
    1. 活跃时间片：跳过门控融合，只用纯HGT嵌入
    2. 不活跃时间片：不填充，保持零向量 + mask=0
    记忆模块仍然运行（保持计算图一致），但其输出不影响任何节点表示。
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_snapshots = config['num_snapshots']
        self.hgt_hidden_dim = config['hgt_hidden_dim']

        self.hgt = _build_hgt(config)

        (self.memory, self.time_encoder, self.message_function,
         self.memory_updater, memory_dim, _, _) = _build_tgn_components(config)

        self.fusion_layer = GatedFusion(
            hgt_dim=self.hgt_hidden_dim,
            memory_dim=memory_dim,
            dropout=config.get('fusion_dropout', 0.1),
        )

        self.temporal_attention = _build_temporal_attention(config)
        self.classifier = _build_classifier(self.hgt_hidden_dim, config)

        logger.info("HTGATFraudNoMemoryFilling 初始化完成（跳过融合 + 不活跃时间片不填充）")

    def forward(self, data):
        device = data.x.device
        num_nodes = data.x.size(0)

        snapshots = split_graph_into_snapshots(
            data, num_snapshots=self.num_snapshots,
            strategy=self.config.get('snapshot_strategy', 'edge_based'),
            timestamps=data.timestamps,
        )

        hgt_outputs = _snapshot_loop_with_memory(
            self, data, snapshots,
            skip_fusion=True,  # 活跃时间片不融合记忆
        )

        # 不活跃位置保持零向量 + mask=0，不调用 _fill_inactive_with_memory
        temporal_sequences, temporal_masks = _assemble_temporal(
            hgt_outputs, num_nodes, self.num_snapshots, self.hgt_hidden_dim, device,
        )

        h = self.temporal_attention(temporal_sequences, temporal_masks)
        return self.classifier(h)


# ============================================================================
# 变体3: message_from_memory — 消息基于记忆而非HGT嵌入
# ============================================================================

class HTGATFraudMessageFromMemory(nn.Module):
    """
    消融消息来源：用记忆向量替代HGT嵌入计算消息

    baseline的关键设计：消息基于当前时间片的HGT嵌入计算，
    这样消息携带的是"当前空间邻居信息"，避免了冷启动问题
    （第一个时间片记忆全零 → 消息全零 → 记忆永远无法积累有用信息）。

    本变体还原原始TGN的做法：消息基于记忆向量计算，
    验证这个冷启动修复的有效性。
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_snapshots = config['num_snapshots']
        self.hgt_hidden_dim = config['hgt_hidden_dim']

        self.hgt = _build_hgt(config)

        (self.memory, self.time_encoder, _,
         self.memory_updater, memory_dim, message_dim, time_dim) = _build_tgn_components(config)

        # 消息函数输入维度改为memory_dim（因为消息基于记忆计算）
        self.message_function = MLPMessageFunction(
            memory_dim=memory_dim,
            edge_dim=config.get('edge_feature_dim', 0),
            time_dim=time_dim,
            message_dim=message_dim,
            dropout=config.get('message_dropout', 0.1),
        )

        self.fusion_layer = GatedFusion(
            hgt_dim=self.hgt_hidden_dim,
            memory_dim=memory_dim,
            dropout=config.get('fusion_dropout', 0.1),
        )

        self.temporal_attention = _build_temporal_attention(config)
        self.classifier = _build_classifier(self.hgt_hidden_dim, config)

        logger.info("HTGATFraudMessageFromMemory 初始化完成（消息基于记忆计算）")

    def forward(self, data):
        device = data.x.device
        num_nodes = data.x.size(0)

        snapshots = split_graph_into_snapshots(
            data, num_snapshots=self.num_snapshots,
            strategy=self.config.get('snapshot_strategy', 'edge_based'),
            timestamps=data.timestamps,
        )

        hgt_outputs = _snapshot_loop_with_memory(
            self, data, snapshots,
            use_gated_fusion=True,
            message_source='memory',
        )

        temporal_sequences, temporal_masks = _assemble_temporal(
            hgt_outputs, num_nodes, self.num_snapshots, self.hgt_hidden_dim, device,
        )
        temporal_sequences, temporal_masks = _fill_inactive_with_memory(
            self, temporal_sequences, temporal_masks, num_nodes, self.num_snapshots, device,
        )

        h = self.temporal_attention(temporal_sequences, temporal_masks)
        return self.classifier(h)


# ============================================================================
# 变体4: wo_time_encoding — 消息函数去掉时间编码
# ============================================================================

class HTGATFraudNoTimeEncoding(nn.Module):
    """
    消融消息函数中的时间编码

    baseline的消息函数输入为 [src_embedding, dst_embedding, time_encoding]，
    时间编码让消息感知"距上次更新过了多久"，帮助模型区分近期活跃和长期沉寂的节点。

    本变体传入 time_encoding=None，消息函数内部会用零向量填充时间编码位置，
    验证时间感知的消息传递是否必要。
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_snapshots = config['num_snapshots']
        self.hgt_hidden_dim = config['hgt_hidden_dim']

        self.hgt = _build_hgt(config)

        (self.memory, self.time_encoder, self.message_function,
         self.memory_updater, memory_dim, _, _) = _build_tgn_components(config)

        self.fusion_layer = GatedFusion(
            hgt_dim=self.hgt_hidden_dim,
            memory_dim=memory_dim,
            dropout=config.get('fusion_dropout', 0.1),
        )

        self.temporal_attention = _build_temporal_attention(config)
        self.classifier = _build_classifier(self.hgt_hidden_dim, config)

        logger.info("HTGATFraudNoTimeEncoding 初始化完成（消息函数无时间编码）")

    def forward(self, data):
        device = data.x.device
        num_nodes = data.x.size(0)

        snapshots = split_graph_into_snapshots(
            data, num_snapshots=self.num_snapshots,
            strategy=self.config.get('snapshot_strategy', 'edge_based'),
            timestamps=data.timestamps,
        )

        hgt_outputs = _snapshot_loop_with_memory(
            self, data, snapshots,
            use_gated_fusion=True,
            use_time_encoding=False,
        )

        temporal_sequences, temporal_masks = _assemble_temporal(
            hgt_outputs, num_nodes, self.num_snapshots, self.hgt_hidden_dim, device,
        )
        temporal_sequences, temporal_masks = _fill_inactive_with_memory(
            self, temporal_sequences, temporal_masks, num_nodes, self.num_snapshots, device,
        )

        h = self.temporal_attention(temporal_sequences, temporal_masks)
        return self.classifier(h)
