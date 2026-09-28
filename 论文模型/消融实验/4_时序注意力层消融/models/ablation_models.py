# -*- coding: utf-8 -*-
"""
时序注意力层消融实验 - 自定义模型变体

HTGATFraudLastStepOnly: 用"取最后活跃时间步"替代时序注意力聚合
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import logging

from src.models.ht_gat_fraud import HTGATFraud

logger = logging.getLogger(__name__)


class HTGATFraudLastStepOnly(HTGATFraud):
    """
    时序注意力消融变体：只取最后一个活跃时间步的表示

    保留完整的 HGT + TGN 记忆流水线，仅将时序注意力聚合替换为
    取每个节点最后一个活跃时间片的嵌入，验证多时间步注意力聚合的必要性。

    与实验1的 wo_temporal_attention（均值池化）互补：
    - 实验1: 时序注意力 → 均值池化（验证注意力 vs 无差别聚合）
    - 本变体: 时序注意力 → 只取最新状态（验证历史聚合 vs 只看当前）
    """

    def __init__(self, config):
        super().__init__(config)
        logger.info("HTGATFraudLastStepOnly: 时序聚合替换为取最后活跃时间步")

    def forward(self, data, num_snapshots=None, return_attention=False):
        if num_snapshots is None:
            num_snapshots = self.num_snapshots

        num_nodes = data.x.size(0)
        device = data.x.device

        # Step 0~3 与父类完全一致：时间片划分、HGT、TGN记忆、整合时间序列
        # 直接复用父类逻辑到 temporal_sequences 和 temporal_masks 的构建
        if self.use_memory:
            self.memory.reset()

        from src.utils.temporal_split import split_graph_into_snapshots
        snapshots = split_graph_into_snapshots(
            data,
            num_snapshots=num_snapshots,
            strategy=self.config.get('snapshot_strategy', 'edge_based'),
            timestamps=data.timestamps
        )

        hgt_outputs = []
        for i, snapshot in enumerate(snapshots):
            snapshot = snapshot.to(device)

            h = self.hgt(
                node_feature=snapshot.x,
                node_type=snapshot.node_type,
                edge_index=snapshot.edge_index,
                edge_type=snapshot.edge_type
            )

            if self.use_memory:
                active_nodes = snapshot.original_indices
                local_edge_index = snapshot.edge_index
                num_edges = local_edge_index.size(1) if local_edge_index.numel() > 0 else 0
                num_nodes_in_snapshot = h.size(0)

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

                    old_to_new = torch.full((snapshot.original_indices.max().item() + 1,), -1, dtype=torch.long, device=device)
                    old_to_new[torch.where(unique_mask)[0]] = torch.arange(num_nodes_in_snapshot, device=device)
                    valid_edges = (old_to_new[local_edge_index[0]] >= 0) & (old_to_new[local_edge_index[1]] >= 0)
                    local_edge_index = local_edge_index[:, valid_edges]
                    local_edge_index = old_to_new[local_edge_index]
                    num_edges = local_edge_index.size(1)

                current_time = snapshot.timestamps.mean().item()
                node_memory = self.memory.get_memory(active_nodes)

                if num_edges > 0:
                    local_src = local_edge_index[0]
                    local_dst = local_edge_index[1]
                    global_src = active_nodes[local_src]

                    src_embedding = h[local_src]
                    dst_embedding = h[local_dst]

                    src_last_update = self.memory.get_last_update(global_src)
                    time_delta = torch.full_like(src_last_update, current_time) - src_last_update
                    edge_time_encoding = self.time_encoder(time_delta)

                    edge_messages = self.message_function(
                        src_memory=src_embedding,
                        dst_memory=dst_embedding,
                        edge_features=None,
                        time_encoding=edge_time_encoding
                    )

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

                new_memory = self.memory_updater(aggregated_messages, node_memory)
                self.memory.set_memory(active_nodes, new_memory)
                self.memory.update_last_update_time(
                    active_nodes,
                    torch.full((len(active_nodes),), current_time, device=device)
                )

                if new_memory.dtype != h.dtype:
                    new_memory = new_memory.to(h.dtype)
                h = self.fusion_layer(h, new_memory)

            hgt_outputs.append((snapshot.original_indices, h))

        # Step 3: 整合时间序列
        temporal_sequences = torch.zeros(
            num_nodes, num_snapshots, self.hgt_hidden_dim, device=device
        )
        temporal_masks = torch.zeros(num_nodes, num_snapshots, device=device)

        for t, (original_indices, h_t) in enumerate(hgt_outputs):
            if h_t.dtype != temporal_sequences.dtype:
                h_t = h_t.to(temporal_sequences.dtype)
            temporal_sequences[original_indices, t, :] = h_t
            temporal_masks[original_indices, t] = 1.0

        # ============================================================
        # 消融点：用"取最后活跃时间步"替代时序注意力
        # ============================================================
        time_indices = torch.arange(num_snapshots, device=device).float()
        last_active_t = (temporal_masks * time_indices).argmax(dim=1)  # [N]

        # 全不活跃的节点回退到最后一个时间步
        all_inactive = (temporal_masks.sum(dim=1) == 0)
        last_active_t = torch.where(
            all_inactive,
            torch.full_like(last_active_t, num_snapshots - 1),
            last_active_t
        )

        idx = last_active_t.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hgt_hidden_dim)
        h_temporal = temporal_sequences.gather(1, idx).squeeze(1)  # [N, F]

        # Step 5: 分类器
        logits = self.classifier(h_temporal)

        if return_attention:
            return logits, {}
        return logits
