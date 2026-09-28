# -*- coding: utf-8 -*-
"""
TGN记忆模块可视化分析

生成4类图表：
  Fig1. 记忆演化轨迹 — 典型节点的记忆向量在时间片上的2D轨迹
  Fig2. 门控融合权重热力图 — 更新门z在[节点×时间片]上的分布
  Fig3. 记忆填充效果对比 — 有/无填充时时间聚合权重的差异
  Fig4. 时序注意力矩阵 — 典型节点的T×T注意力热力图

输入：capture_states.py 生成的 .npz 文件
输出：PNG 图片（300 DPI）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 样式配置
# ============================================================================

# 尝试使用中文字体，失败则回退
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

COLORS = {
    'fraud': '#e74c3c',
    'normal': '#2ecc71',
    'primary': '#3498db',
    'secondary': '#9b59b6',
    'accent': '#f39c12',
    'gray': '#95a5a6',
}


def _select_representative_nodes(labels, temporal_masks, n_per_class=5, seed=42):
    """
    选择代表性节点：高活跃欺诈、低活跃欺诈、高活跃正常、低活跃正常

    Returns
    -------
    dict: {category_name: node_indices}
    """
    rng = np.random.RandomState(seed)
    activity = temporal_masks.sum(axis=1)  # 每个节点的活跃时间片数

    fraud_idx = np.where(labels == 0)[0]
    normal_idx = np.where(labels == 1)[0]

    median_activity = np.median(activity)

    groups = {}
    for name, pool in [('fraud', fraud_idx), ('normal', normal_idx)]:
        high = pool[activity[pool] >= median_activity]
        low = pool[activity[pool] < median_activity]
        if len(high) > 0:
            groups[f'{name}_high_active'] = rng.choice(high, min(n_per_class, len(high)), replace=False)
        if len(low) > 0:
            groups[f'{name}_low_active'] = rng.choice(low, min(n_per_class, len(low)), replace=False)

    return groups


# ============================================================================
# Fig1: 记忆演化轨迹
# ============================================================================

def plot_memory_evolution(states: dict, output_dir: Path, n_nodes=8):
    """
    用PCA将记忆向量降到2D，画出典型节点在时间片上的演化轨迹

    欺诈节点用红色，正常节点用绿色，箭头表示时间方向。
    """
    memory_snapshots = states.get('memory_snapshots')
    if memory_snapshots is None:
        logger.warning("无记忆快照数据，跳过记忆演化图")
        return

    labels = states['labels']
    temporal_masks = states['temporal_masks']
    T, N, D = memory_snapshots.shape

    # 选择代表性节点
    groups = _select_representative_nodes(labels, temporal_masks, n_per_class=n_nodes // 2)
    selected = []
    for v in groups.values():
        selected.extend(v.tolist())
    selected = list(set(selected))[:n_nodes]

    if len(selected) == 0:
        logger.warning("无可选节点，跳过记忆演化图")
        return

    # 提取选中节点在所有时间片的记忆: [T, n_selected, D]
    mem_selected = memory_snapshots[:, selected, :]

    # PCA降维: 将 [T * n_selected, D] → [T * n_selected, 2]
    flat = mem_selected.reshape(-1, D)
    # 过滤全零行（未被更新的记忆）
    nonzero_mask = np.abs(flat).sum(axis=1) > 1e-8
    if nonzero_mask.sum() < 10:
        logger.warning("记忆向量几乎全零，跳过记忆演化图")
        return

    pca = PCA(n_components=2)
    pca.fit(flat[nonzero_mask])
    coords = pca.transform(flat).reshape(T, len(selected), 2)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    for j, node_id in enumerate(selected):
        is_fraud = (labels[node_id] == 0)
        color = COLORS['fraud'] if is_fraud else COLORS['normal']
        label_text = 'Fraud' if is_fraud else 'Normal'
        alpha = 0.8

        traj = coords[:, j, :]  # [T, 2]

        # 找到该节点活跃的时间片
        active_t = np.where(temporal_masks[node_id] > 0.5)[0]
        if len(active_t) == 0:
            continue

        # 画轨迹线
        ax.plot(traj[active_t, 0], traj[active_t, 1],
                '-o', color=color, alpha=alpha, markersize=4, linewidth=1.5,
                label=f'{label_text} #{node_id}' if j < 4 else None)

        # 起点标记
        ax.plot(traj[active_t[0], 0], traj[active_t[0], 1],
                's', color=color, markersize=8, zorder=5)

        # 终点箭头
        if len(active_t) >= 2:
            dx = traj[active_t[-1], 0] - traj[active_t[-2], 0]
            dy = traj[active_t[-1], 1] - traj[active_t[-2], 1]
            ax.annotate('', xy=(traj[active_t[-1], 0], traj[active_t[-1], 1]),
                        xytext=(traj[active_t[-2], 0], traj[active_t[-2], 1]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    ax.set_title('Memory Evolution Trajectories (PCA 2D)', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / 'fig1_memory_evolution.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Fig1 已保存: {out_path}")


# ============================================================================
# Fig2: 门控融合权重热力图
# ============================================================================

def plot_gating_weights(states: dict, output_dir: Path, n_nodes=50):
    """
    绘制更新门z的热力图: [节点 × 时间片]

    z接近1 → 依赖HGT空间信息（蓝色）
    z接近0 → 依赖TGN记忆（红色）

    按欺诈/正常分组，观察门控模式差异。
    """
    gating_z_per_node = states.get('gating_z_per_node')
    gating_z_indices = states.get('gating_z_indices')

    if not gating_z_per_node or len(gating_z_per_node) == 0:
        logger.warning("无门控权重数据，跳过门控热力图")
        return

    labels = states['labels']
    T = len(gating_z_per_node)
    N = len(labels)

    # 构建完整的 [N, T] 门控矩阵（NaN表示不活跃）
    z_matrix = np.full((N, T), np.nan)
    for t in range(T):
        indices = gating_z_indices[t]
        values = gating_z_per_node[t]
        z_matrix[indices, t] = values

    # 选择有足够活跃时间片的节点
    active_count = np.sum(~np.isnan(z_matrix), axis=1)
    valid_nodes = np.where(active_count >= T // 2)[0]

    if len(valid_nodes) == 0:
        logger.warning("无足够活跃节点，跳过门控热力图")
        return

    # 分组选择
    rng = np.random.RandomState(42)
    fraud_valid = valid_nodes[labels[valid_nodes] == 0]
    normal_valid = valid_nodes[labels[valid_nodes] == 1]

    n_half = n_nodes // 2
    fraud_sel = rng.choice(fraud_valid, min(n_half, len(fraud_valid)), replace=False) if len(fraud_valid) > 0 else np.array([], dtype=int)
    normal_sel = rng.choice(normal_valid, min(n_half, len(normal_valid)), replace=False) if len(normal_valid) > 0 else np.array([], dtype=int)

    selected = np.concatenate([fraud_sel, normal_sel])
    if len(selected) == 0:
        return

    z_selected = z_matrix[selected]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 自定义colormap: 红(记忆) → 白 → 蓝(HGT)
    cmap = LinearSegmentedColormap.from_list('memory_hgt',
        [(0, COLORS['fraud']), (0.5, '#ffffff'), (1, COLORS['primary'])])

    im = ax.imshow(z_selected, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   interpolation='nearest')

    # 分隔线
    if len(fraud_sel) > 0 and len(normal_sel) > 0:
        ax.axhline(y=len(fraud_sel) - 0.5, color='black', linewidth=2, linestyle='--')
        ax.text(-0.5, len(fraud_sel) / 2, 'Fraud', ha='right', va='center',
                fontsize=11, fontweight='bold', color=COLORS['fraud'])
        ax.text(-0.5, len(fraud_sel) + len(normal_sel) / 2, 'Normal', ha='right', va='center',
                fontsize=11, fontweight='bold', color=COLORS['normal'])

    ax.set_xlabel('Snapshot (t)', fontsize=12)
    ax.set_ylabel('Node', fontsize=12)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f't{i}' for i in range(T)])
    ax.set_title('Gating Weight z (1=HGT, 0=Memory)', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Update Gate z', fontsize=11)

    plt.tight_layout()
    out_path = output_dir / 'fig2_gating_weights.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Fig2 已保存: {out_path}")


# ============================================================================
# Fig3: 记忆填充效果对比
# ============================================================================

def plot_memory_filling_effect(states: dict, output_dir: Path):
    """
    对比有记忆填充 vs 无填充时，时间聚合权重的分布差异

    左图：有填充时各时间片的平均聚合权重（按欺诈/正常分组）
    右图：活跃 vs 不活跃时间片获得的聚合权重对比
    """
    time_agg = states.get('time_agg_weights')
    temporal_masks = states['temporal_masks']
    labels = states['labels']

    if time_agg is None:
        logger.warning("无时间聚合权重，跳过填充效果图")
        return

    N, T = time_agg.shape
    fraud_mask = (labels == 0)
    normal_mask = (labels == 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图：各时间片的平均聚合权重 ---
    ax = axes[0]
    x = np.arange(T)
    width = 0.35

    fraud_weights = time_agg[fraud_mask].mean(axis=0)
    normal_weights = time_agg[normal_mask].mean(axis=0)

    ax.bar(x - width/2, fraud_weights, width, label='Fraud', color=COLORS['fraud'], alpha=0.8)
    ax.bar(x + width/2, normal_weights, width, label='Normal', color=COLORS['normal'], alpha=0.8)

    ax.set_xlabel('Snapshot (t)', fontsize=12)
    ax.set_ylabel('Mean Aggregation Weight', fontsize=12)
    ax.set_title('Time Aggregation Weights by Class', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f't{i}' for i in range(T)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # --- 右图：活跃 vs 不活跃位置的聚合权重 ---
    ax = axes[1]

    active_positions = (temporal_masks > 0.5)
    inactive_positions = (temporal_masks > 0) & (temporal_masks <= 0.5)  # 被记忆填充的位置
    zero_positions = (temporal_masks == 0)  # 完全不活跃

    categories = []
    values = []

    if active_positions.any():
        w = time_agg[active_positions].mean()
        categories.append('Active\n(mask=1.0)')
        values.append(w)

    if inactive_positions.any():
        w = time_agg[inactive_positions].mean()
        categories.append(f'Memory-filled\n(mask={temporal_masks[inactive_positions].mean():.2f})')
        values.append(w)

    if zero_positions.any():
        w = time_agg[zero_positions].mean()
        categories.append('Inactive\n(mask=0)')
        values.append(w)

    if len(categories) > 0:
        bar_colors = [COLORS['primary'], COLORS['accent'], COLORS['gray']][:len(categories)]
        bars = ax.bar(categories, values, color=bar_colors, alpha=0.8, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Mean Aggregation Weight', fontsize=12)
    ax.set_title('Weight by Position Type', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = output_dir / 'fig3_memory_filling_effect.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Fig3 已保存: {out_path}")


# ============================================================================
# Fig4: 时序注意力矩阵
# ============================================================================

def plot_temporal_attention(states: dict, output_dir: Path, n_examples=4):
    """
    选择典型节点，画T×T的时序注意力热力图

    2×2子图：高活跃欺诈、低活跃欺诈、高活跃正常、低活跃正常
    """
    attn_weights = states.get('attn_weights')
    if attn_weights is None:
        logger.warning("无注意力权重，跳过注意力矩阵图")
        return

    labels = states['labels']
    temporal_masks = states['temporal_masks']
    N, T, _ = attn_weights.shape

    groups = _select_representative_nodes(labels, temporal_masks, n_per_class=3)

    # 选4个代表
    examples = []
    for group_name in ['fraud_high_active', 'fraud_low_active', 'normal_high_active', 'normal_low_active']:
        if group_name in groups and len(groups[group_name]) > 0:
            examples.append((group_name, groups[group_name][0]))
        if len(examples) >= n_examples:
            break

    if len(examples) == 0:
        logger.warning("无可选节点，跳过注意力矩阵图")
        return

    nrows = 2
    ncols = (len(examples) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if len(examples) == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for idx, (group_name, node_id) in enumerate(examples):
        ax = axes[idx]
        attn = attn_weights[node_id]  # [T, T]

        im = ax.imshow(attn, cmap='Blues', vmin=0, interpolation='nearest')

        # 标记活跃时间片
        active_t = np.where(temporal_masks[node_id] > 0.5)[0]
        title_label = group_name.replace('_', ' ').title()
        is_fraud = 'fraud' in group_name
        title_color = COLORS['fraud'] if is_fraud else COLORS['normal']

        ax.set_title(f'{title_label}\nNode #{node_id} (active: {len(active_t)}/{T})',
                     fontsize=11, color=title_color)
        ax.set_xlabel('Key (t)', fontsize=10)
        ax.set_ylabel('Query (t)', fontsize=10)
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels([f't{i}' for i in range(T)], fontsize=8)
        ax.set_yticklabels([f't{i}' for i in range(T)], fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.8)

    # 隐藏多余子图
    for idx in range(len(examples), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Temporal Self-Attention Weights (T×T)', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = output_dir / 'fig4_temporal_attention.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Fig4 已保存: {out_path}")


# ============================================================================
# 主函数
# ============================================================================

def load_states(npz_path: str) -> dict:
    """加载capture_states.py保存的npz文件"""
    data = np.load(npz_path, allow_pickle=True)
    states = {}

    # 标准数组
    for key in ['memory_snapshots', 'gating_z_mean', 'attn_weights',
                'time_agg_weights', 'temporal_masks', 'labels']:
        if key in data:
            states[key] = data[key]
        else:
            states[key] = None

    # 变长数组（gating_z_per_node_0, gating_z_per_node_1, ...）
    z_per_node = []
    z_indices = []
    t = 0
    while f'gating_z_per_node_{t}' in data:
        z_per_node.append(data[f'gating_z_per_node_{t}'])
        t += 1
    t = 0
    while f'gating_z_indices_{t}' in data:
        z_indices.append(data[f'gating_z_indices_{t}'])
        t += 1

    states['gating_z_per_node'] = z_per_node
    states['gating_z_indices'] = z_indices

    return states


def main():
    parser = argparse.ArgumentParser(description='TGN记忆模块可视化分析')
    parser.add_argument('--input', type=str, required=True,
                        help='capture_states.py输出的npz文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='图片输出目录')
    parser.add_argument('--fig', type=str, default='all',
                        choices=['all', '1', '2', '3', '4'],
                        help='生成哪些图（默认全部）')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载状态: {args.input}")
    states = load_states(args.input)

    logger.info(f"输出目录: {output_dir}")

    if args.fig in ('all', '1'):
        plot_memory_evolution(states, output_dir)
    if args.fig in ('all', '2'):
        plot_gating_weights(states, output_dir)
    if args.fig in ('all', '3'):
        plot_memory_filling_effect(states, output_dir)
    if args.fig in ('all', '4'):
        plot_temporal_attention(states, output_dir)

    logger.info("可视化完成")


if __name__ == '__main__':
    main()
