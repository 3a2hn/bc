# -*- coding: utf-8 -*-
"""
TGN记忆模块中间状态捕获

在模型推理过程中，hook住关键节点，捕获：
1. 每个时间片结束后的节点记忆向量 → 记忆演化轨迹
2. GatedFusion的更新门z值 → 门控权重分析
3. 时序注意力的time_agg_weights → 时间片重要性
4. 时序注意力的attn_weights → T×T注意力矩阵

输出为 .npz 文件，供 plot_memory_analysis.py 读取绘图。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import numpy as np
import argparse
import logging
from typing import Dict, List

from config import get_data_dir, get_model_config_for_dataset, SPLIT_CONFIG, MODEL_CONFIG
from src.utils.data_loader import load_fraud_detection_data
from src.utils.node_split import create_node_split
from src.models.ht_gat_fraud import HTGATFraud
from src.utils.temporal_split import split_graph_into_snapshots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TGNStateCapture:
    """
    在模型前向传播中捕获TGN记忆模块的中间状态

    使用方式：
        capture = TGNStateCapture(model, data)
        states = capture.run()
        capture.save('output.npz')
    """

    def __init__(self, model: HTGATFraud, data, device: str = 'cuda:0'):
        self.model = model
        self.data = data
        self.device = device
        self.states = {}

    def run(self) -> Dict[str, np.ndarray]:
        """执行一次前向传播并捕获所有中间状态"""
        self.model.eval()
        device = self.device
        data = self.data
        num_nodes = data.x.size(0)
        num_snapshots = self.model.num_snapshots
        hgt_dim = self.model.hgt_hidden_dim
        working_dim = getattr(self.model, 'working_dim', hgt_dim)

        # ====== 1. 手动执行前向传播，逐步捕获状态 ======
        self.model.memory.reset()

        snapshot_strategy = self.model.config.get('snapshot_strategy', 'edge_based')
        snapshots = split_graph_into_snapshots(
            data, num_snapshots=num_snapshots,
            strategy=snapshot_strategy, timestamps=data.timestamps,
        )

        # 存储容器
        memory_snapshots = []          # [T, N, memory_dim] 每个时间片后的记忆
        gating_z_snapshots = []        # [T] 每个时间片的更新门z均值（按节点）
        gating_z_per_node = []         # [T, N_active] 每个时间片每个活跃节点的z
        gating_z_indices = []          # [T, N_active] 对应的全局节点ID
        hgt_outputs = []

        # hook GatedFusion 捕获更新门z
        z_buffer = {}

        def fusion_hook(module, input, output):
            """捕获GatedFusion的更新门z"""
            h, m = input
            if module.memory_proj is not None:
                m = module.memory_proj(m)
            concat = torch.cat([h, m], dim=-1)
            z = torch.sigmoid(module.W_z(concat))
            z_buffer['z'] = z.detach().cpu()

        hook_handle = None
        if hasattr(self.model, 'fusion_layer'):
            hook_handle = self.model.fusion_layer.register_forward_hook(fusion_hook)

        with torch.no_grad():
            for i, snapshot in enumerate(snapshots):
                snapshot = snapshot.to(device)

                # HGT
                h = self.model.hgt(
                    node_feature=snapshot.x,
                    node_type=snapshot.node_type,
                    edge_index=snapshot.edge_index,
                    edge_type=snapshot.edge_type,
                )

                # TGN记忆处理
                if self.model.use_memory:
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
                    node_memory = self.model.memory.get_memory(active_nodes)

                    # 消息传递
                    if num_edges > 0:
                        local_src = local_edge_index[0]
                        local_dst = local_edge_index[1]
                        global_src = active_nodes[local_src]

                        src_embedding = h[local_src]
                        dst_embedding = h[local_dst]

                        src_last_update = self.model.memory.get_last_update(global_src)
                        time_delta = torch.full_like(src_last_update, current_time) - src_last_update
                        edge_time_encoding = self.model.time_encoder(time_delta)

                        edge_messages = self.model.message_function(
                            src_memory=src_embedding, dst_memory=dst_embedding,
                            edge_features=None, time_encoding=edge_time_encoding,
                        )

                        aggregated_messages = torch.zeros(
                            num_nodes_in_snapshot, edge_messages.size(1),
                            device=device, dtype=edge_messages.dtype,
                        )
                        aggregated_messages.scatter_add_(
                            0, local_dst.unsqueeze(-1).expand_as(edge_messages), edge_messages,
                        )
                        edge_count = torch.zeros(num_nodes_in_snapshot, device=device)
                        edge_count.scatter_add_(0, local_dst, torch.ones(num_edges, device=device))
                        edge_count = edge_count.clamp(min=1).unsqueeze(-1)
                        aggregated_messages = aggregated_messages / edge_count
                    else:
                        msg_dim = self.model.config.get('message_dim', hgt_dim)
                        aggregated_messages = torch.zeros(num_nodes_in_snapshot, msg_dim, device=device)

                    new_memory = self.model.memory_updater(aggregated_messages, node_memory)
                    self.model.memory.set_memory(active_nodes, new_memory)
                    self.model.memory.update_last_update_time(
                        active_nodes, torch.full((len(active_nodes),), current_time, device=device),
                    )

                    # 融合（触发hook捕获z）
                    if new_memory.dtype != h.dtype:
                        new_memory = new_memory.to(h.dtype)
                    h = self.model.fusion_layer(h, new_memory)

                    # 捕获门控z
                    if 'z' in z_buffer:
                        z_mean = z_buffer['z'].mean(dim=-1)  # [N_active] 每个节点的z均值（跨维度）
                        gating_z_per_node.append(z_mean.numpy())
                        gating_z_indices.append(active_nodes.cpu().numpy())
                        gating_z_snapshots.append(z_mean.mean().item())
                        z_buffer.clear()

                    # 捕获当前时间片后的全局记忆状态
                    all_mem = self.model.memory.get_memory(
                        torch.arange(num_nodes, device=device)
                    ).cpu().numpy()
                    memory_snapshots.append(all_mem)

                # HGT后投影（与forward保持一致）
                if getattr(self.model, 'post_hgt_proj', None) is not None:
                    h = self.model.post_hgt_proj(h)

                hgt_outputs.append((snapshot.original_indices, h))

            # ====== 2. 整合时间序列并通过时序注意力 ======
            temporal_sequences = torch.zeros(num_nodes, num_snapshots, working_dim, device=device)
            temporal_masks = torch.zeros(num_nodes, num_snapshots, device=device)

            for t, (indices, h_t) in enumerate(hgt_outputs):
                if h_t.dtype != temporal_sequences.dtype:
                    h_t = h_t.to(temporal_sequences.dtype)
                temporal_sequences[indices, t, :] = h_t
                temporal_masks[indices, t] = 1.0

            # 记忆填充
            if self.model.use_memory:
                inactive_mask = (temporal_masks == 0)
                if inactive_mask.any():
                    all_memory = self.model.memory.get_memory(
                        torch.arange(num_nodes, device=device)
                    )
                    if all_memory.dtype != temporal_sequences.dtype:
                        all_memory = all_memory.to(temporal_sequences.dtype)
                    if all_memory.size(-1) != working_dim:
                        if getattr(self.model, 'post_hgt_proj', None) is not None:
                            all_memory = self.model.post_hgt_proj(all_memory)
                        elif hasattr(self.model.fusion_layer, 'memory_proj') and self.model.fusion_layer.memory_proj is not None:
                            all_memory = self.model.fusion_layer.memory_proj(all_memory)
                    memory_expanded = all_memory.unsqueeze(1).expand(-1, num_snapshots, -1)
                    temporal_sequences = torch.where(
                        inactive_mask.unsqueeze(-1).expand_as(temporal_sequences),
                        memory_expanded, temporal_sequences,
                    )
                    weight = self.model.config.get('fixed_memory_weight', 0.3)
                    temporal_masks = torch.where(
                        inactive_mask,
                        torch.full_like(temporal_masks, weight),
                        temporal_masks,
                    )

            # 时序注意力
            h_temporal = self.model.temporal_attention(temporal_sequences, temporal_masks)

        # 移除hook
        if hook_handle is not None:
            hook_handle.remove()

        # ====== 3. 收集时序注意力权重 ======
        attn_weights = self.model.temporal_attention.get_attention_weights()  # [N, T, T]
        time_agg_weights = self.model.temporal_attention.get_time_aggregation_weights()  # [N, T]

        # ====== 4. 打包结果 ======
        self.states = {
            # 记忆演化: [T, N, memory_dim]
            'memory_snapshots': np.stack(memory_snapshots, axis=0) if memory_snapshots else None,
            # 门控z均值: [T]
            'gating_z_mean': np.array(gating_z_snapshots) if gating_z_snapshots else None,
            # 时序注意力权重: [N, T, T]
            'attn_weights': attn_weights.cpu().numpy() if attn_weights is not None else None,
            # 时间聚合权重: [N, T]
            'time_agg_weights': time_agg_weights.cpu().numpy() if time_agg_weights is not None else None,
            # 时序mask: [N, T]
            'temporal_masks': temporal_masks.cpu().numpy(),
            # 标签
            'labels': data.y.cpu().numpy(),
            # 门控z逐节点（变长，用object数组）
            'gating_z_per_node': gating_z_per_node,
            'gating_z_indices': gating_z_indices,
        }

        logger.info(f"状态捕获完成:")
        if self.states['memory_snapshots'] is not None:
            logger.info(f"  记忆快照: {self.states['memory_snapshots'].shape}")
        logger.info(f"  注意力权重: {self.states['attn_weights'].shape}")
        logger.info(f"  时间聚合权重: {self.states['time_agg_weights'].shape}")

        return self.states

    def save(self, output_path: str):
        """保存捕获的状态到npz文件"""
        save_dict = {}
        for k, v in self.states.items():
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                save_dict[k] = v
            elif isinstance(v, list) and len(v) > 0:
                # 变长数组，逐个保存
                for i, arr in enumerate(v):
                    save_dict[f'{k}_{i}'] = np.array(arr)

        np.savez_compressed(output_path, **save_dict)
        logger.info(f"状态已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='捕获TGN记忆模块中间状态')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径 (.pth)')
    parser.add_argument('--dataset', type=str, default='yelp',
                        choices=['yelp', 'amazon'])
    parser.add_argument('--output', type=str, default=None,
                        help='输出npz文件路径')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    data_dir = get_data_dir(args.dataset)
    data, class_weights = load_fraud_detection_data(data_dir, dataset=args.dataset)
    data = data.to(device)

    # 划分
    splits = create_node_split(
        labels=data.y.cpu(),
        timestamps=data.timestamps.cpu() if hasattr(data, 'timestamps') else None,
        method=SPLIT_CONFIG['split_method'],
        train_ratio=0.5, val_ratio=0.1, test_ratio=0.4,
        seed=42,
    )
    data.train_mask = splits['train_mask'].to(device)
    data.val_mask = splits['val_mask'].to(device)
    data.test_mask = splits['test_mask'].to(device)

    # 创建模型并加载权重（从config.py读取配置）
    config = {**MODEL_CONFIG, 'use_rl_memory_gate': False}
    model_config = get_model_config_for_dataset(args.dataset, config)
    model_config['device'] = device

    model = HTGATFraud(model_config)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    logger.info(f"模型加载完成: {args.checkpoint}")

    # 捕获状态
    capture = TGNStateCapture(model, data, device=device)
    capture.run()

    # 保存
    output_path = args.output or str(
        Path(__file__).parent / f'tgn_states_{args.dataset}.npz'
    )
    capture.save(output_path)


if __name__ == '__main__':
    main()
