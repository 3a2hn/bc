# -*- coding: utf-8 -*-
"""
图数据分析脚本
解析并展示图的详细统计信息
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import config


def load_graph_data(graph_path):
    """加载图数据"""
    print(f"正在加载图数据: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    print("✓ 图数据加载完成\n")
    return graph_data


def analyze_basic_stats(graph_data):
    """分析基本统计信息"""
    print("=" * 80)
    print("基本统计信息".center(80))
    print("=" * 80)
    
    stats = graph_data['statistics']
    
    print(f"节点数量: {stats['num_nodes']:,}")
    print(f"总边数: {stats['total_edges']:,}")
    print(f"边/节点比: {stats['edge_node_ratio']:.2f}")
    print(f"孤立节点数: {stats['isolated_nodes']:,} ({stats['isolated_nodes']/stats['num_nodes']*100:.2f}%)")
    print()


def analyze_edge_types(graph_data):
    """分析边类型分布"""
    print("=" * 80)
    print("边类型分布".center(80))
    print("=" * 80)
    
    stats = graph_data['statistics']
    
    edge_types = [
        ('R-U-R (用户关联)', stats['rur_edges'], stats['rur_ratio']),
        ('R-T-R (时间关联)', stats['rtr_edges'], stats['rtr_ratio']),
        ('R-S-R (评分关联)', stats['rsr_edges'], stats['rsr_ratio']),
    ]
    
    for name, count, ratio in edge_types:
        bar_length = int(ratio / 100 * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{name:20s}: {count:>10,} 条  ({ratio:>5.2f}%)  {bar}")
    
    print()


def analyze_degree_distribution(graph_data):
    """分析度分布"""
    print("=" * 80)
    print("度分布统计".center(80))
    print("=" * 80)
    
    num_nodes = graph_data['num_nodes']
    edges = graph_data['edges']
    
    # 计算每个节点的度
    degrees = np.zeros(num_nodes, dtype=int)
    for edge in edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    
    stats = graph_data['statistics']
    
    print(f"平均度: {stats['avg_degree']:.2f}")
    print(f"最大度: {stats['max_degree']:,}")
    print(f"最小度: {stats['min_degree']}")
    print(f"度中位数: {np.median(degrees):.2f}")
    print(f"度标准差: {np.std(degrees):.2f}")
    print()
    
    # 度分布直方图
    print("度分布区间:")
    bins = [0, 10, 50, 100, 150, 200, 300, 500, float('inf')]
    labels = ['0-10', '10-50', '50-100', '100-150', '150-200', '200-300', '300-500', '500+']
    
    for i in range(len(bins) - 1):
        count = ((degrees > bins[i]) & (degrees <= bins[i+1])).sum()
        percentage = count / num_nodes * 100
        bar_length = int(percentage / 100 * 40)
        bar = '█' * bar_length
        print(f"  {labels[i]:10s}: {count:>6,} 个节点 ({percentage:>5.2f}%)  {bar}")
    
    print()
    
    # 找出度最高的节点
    top_k = 10
    top_indices = np.argsort(degrees)[-top_k:][::-1]
    print(f"度最高的前{top_k}个节点:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  #{rank:2d}  节点 {idx:6d}: 度 = {degrees[idx]:,}")
    
    print()


def analyze_edge_weights(graph_data):
    """分析边权重分布"""
    print("=" * 80)
    print("边权重统计".center(80))
    print("=" * 80)
    
    # 按边类型分析权重
    for edge_type in ['R-U-R', 'R-T-R', 'R-S-R']:
        attrs = graph_data['attributes_by_type'][edge_type]
        
        if not attrs:
            continue
        
        weights = [attr['weight'] for attr in attrs]
        weights = np.array(weights)
        
        print(f"\n{edge_type} 边权重:")
        print(f"  数量: {len(weights):,}")
        print(f"  均值: {weights.mean():.4f}")
        print(f"  标准差: {weights.std():.4f}")
        print(f"  最小值: {weights.min():.4f}")
        print(f"  最大值: {weights.max():.4f}")
        print(f"  中位数: {np.median(weights):.4f}")
        print(f"  25%分位: {np.percentile(weights, 25):.4f}")
        print(f"  75%分位: {np.percentile(weights, 75):.4f}")
    
    print()


def analyze_time_distribution(graph_data):
    """分析时间分布"""
    print("=" * 80)
    print("时间戳分布".center(80))
    print("=" * 80)
    
    timestamps = graph_data.get('timestamps', None)
    
    if timestamps is not None:
        print(f"节点时间戳统计 (归一化到0-100):")
        print(f"  最小值: {timestamps.min():.2f}")
        print(f"  最大值: {timestamps.max():.2f}")
        print(f"  均值: {timestamps.mean():.2f}")
        print(f"  中位数: {np.median(timestamps):.2f}")
        print(f"  标准差: {np.std(timestamps):.2f}")
        print()
        
        # 时间段分布
        print("时间段分布 (0-100分成10段):")
        bins = np.linspace(0, 100, 11)
        hist, _ = np.histogram(timestamps, bins=bins)
        
        for i in range(len(hist)):
            start = bins[i]
            end = bins[i+1]
            count = hist[i]
            percentage = count / len(timestamps) * 100
            bar_length = int(percentage / 100 * 40)
            bar = '█' * bar_length
            print(f"  [{start:5.1f}, {end:5.1f}): {count:>6,} 个节点 ({percentage:>5.2f}%)  {bar}")
    else:
        print("时间戳数据不可用")
    
    print()


def analyze_multi_edges(graph_data):
    """分析多重边（同一对节点有多种边类型）"""
    print("=" * 80)
    print("多重边分析".center(80))
    print("=" * 80)
    
    # 统计每对节点的边类型
    edge_type_count = defaultdict(set)
    
    for edge_type in ['R-U-R', 'R-T-R', 'R-S-R']:
        edges = graph_data['edges_by_type'][edge_type]
        for edge in edges:
            edge_key = tuple(sorted(edge))
            edge_type_count[edge_key].add(edge_type)
    
    # 统计不同类型组合的数量
    type_combo_count = Counter()
    for edge_key, types in edge_type_count.items():
        combo = tuple(sorted(types))
        type_combo_count[combo] += 1
    
    print(f"总边对数: {len(edge_type_count):,}")
    print()
    
    print("边类型组合分布:")
    for combo, count in sorted(type_combo_count.items(), key=lambda x: -x[1]):
        combo_str = ' + '.join(combo)
        percentage = count / len(edge_type_count) * 100
        print(f"  {combo_str:30s}: {count:>10,} 对 ({percentage:>5.2f}%)")
    
    print()
    
    # 统计有多少条边是多重边
    multi_edge_pairs = sum(1 for types in edge_type_count.values() if len(types) > 1)
    print(f"多重边对数: {multi_edge_pairs:,} ({multi_edge_pairs/len(edge_type_count)*100:.2f}%)")
    print()


def analyze_connectivity(graph_data):
    """分析连通性"""
    print("=" * 80)
    print("连通性分析".center(80))
    print("=" * 80)
    
    num_nodes = graph_data['num_nodes']
    edges = graph_data['edges']
    
    # 构建邻接表
    adj_list = defaultdict(set)
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    # BFS寻找连通分量
    visited = set()
    components = []
    
    def bfs(start):
        queue = [start]
        visited.add(start)
        component = [start]
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        
        return component
    
    for node in range(num_nodes):
        if node not in visited:
            component = bfs(node)
            components.append(component)
    
    print(f"连通分量数: {len(components)}")
    
    if components:
        component_sizes = [len(c) for c in components]
        largest_component = max(component_sizes)
        
        print(f"最大连通分量大小: {largest_component:,} 个节点 ({largest_component/num_nodes*100:.2f}%)")
        print(f"最小连通分量大小: {min(component_sizes):,} 个节点")
        print(f"平均连通分量大小: {np.mean(component_sizes):.2f} 个节点")
        print()
        
        # 连通分量大小分布
        print("连通分量大小分布 (Top 10):")
        sorted_sizes = sorted(component_sizes, reverse=True)[:10]
        for i, size in enumerate(sorted_sizes, 1):
            percentage = size / num_nodes * 100
            print(f"  #{i:2d}: {size:>8,} 个节点 ({percentage:>6.2f}%)")
    
    print()


def analyze_edge_time_diff(graph_data):
    """分析边的时间差分布"""
    print("=" * 80)
    print("边时间差统计".center(80))
    print("=" * 80)
    
    # 只分析R-U-R和R-T-R（它们有time_diff属性）
    for edge_type in ['R-U-R', 'R-T-R']:
        attrs = graph_data['attributes_by_type'][edge_type]
        
        if not attrs:
            continue
        
        time_diffs = [attr.get('time_diff', 0) for attr in attrs]
        time_diffs = np.array(time_diffs)
        
        print(f"\n{edge_type} 边时间差 (天数):")
        print(f"  数量: {len(time_diffs):,}")
        print(f"  均值: {time_diffs.mean():.2f} 天")
        print(f"  标准差: {time_diffs.std():.2f} 天")
        print(f"  最小值: {time_diffs.min():.0f} 天")
        print(f"  最大值: {time_diffs.max():.0f} 天")
        print(f"  中位数: {np.median(time_diffs):.2f} 天")
        
        # 时间差分布区间
        if edge_type == 'R-U-R':
            bins = [0, 1, 7, 30, 90, 180, 365, float('inf')]
            labels = ['<1天', '1-7天', '7-30天', '30-90天', '90-180天', '180-365天', '>365天']
        else:
            bins = [0, 1, 3, 7, 14, 21, 30, float('inf')]
            labels = ['<1天', '1-3天', '3-7天', '7-14天', '14-21天', '21-30天', '>30天']
        
        print(f"\n  时间差区间分布:")
        for i in range(len(bins) - 1):
            count = ((time_diffs >= bins[i]) & (time_diffs < bins[i+1])).sum()
            percentage = count / len(time_diffs) * 100
            bar_length = int(percentage / 100 * 30)
            bar = '█' * bar_length
            print(f"    {labels[i]:12s}: {count:>8,} ({percentage:>5.2f}%)  {bar}")
    
    print()


def main():
    """主函数"""
    graph_path = config.GRAPH_OUTPUT
    
    if not graph_path.exists():
        print(f"错误: 图数据文件不存在: {graph_path}")
        print("请先运行 main.py 生成图数据")
        return
    
    # 加载图数据
    graph_data = load_graph_data(graph_path)
    
    # 执行各项分析
    analyze_basic_stats(graph_data)
    analyze_edge_types(graph_data)
    analyze_degree_distribution(graph_data)
    analyze_edge_weights(graph_data)
    analyze_time_distribution(graph_data)
    analyze_multi_edges(graph_data)
    analyze_connectivity(graph_data)
    analyze_edge_time_diff(graph_data)
    
    print("=" * 80)
    print("分析完成！".center(80))
    print("=" * 80)


if __name__ == '__main__':
    main()
