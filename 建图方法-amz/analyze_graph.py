# -*- coding: utf-8 -*-
"""
图数据分析脚本
分析建图后的数据，统计节点数和各种边的数量
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import Counter
import config_amazon as config


def load_graph_data(graph_path=None):
    """加载图数据"""
    if graph_path is None:
        graph_path = config.GRAPH_OUTPUT
    
    if not Path(graph_path).exists():
        raise FileNotFoundError(f"图数据文件不存在: {graph_path}")
    
    print(f"正在加载图数据: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    return graph_data


def analyze_nodes(graph_data):
    """分析节点信息"""
    print("\n" + "=" * 80)
    print("【节点分析】".center(80))
    print("=" * 80)
    
    num_nodes = graph_data.get('num_nodes', 0)
    print(f"节点总数: {num_nodes:,}")
    
    # 节点类型
    node_type = graph_data.get('node_type', 'unknown')
    print(f"节点类型: {node_type}")
    
    # 节点标签统计（如果有）
    if 'node_labels' in graph_data:
        labels = graph_data['node_labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n节点标签分布:")
        for label, count in zip(unique_labels, counts):
            label_name = "欺诈评论" if label == 0 else "正常评论"
            print(f"  {label_name} (label={label}): {count:,} ({count/len(labels)*100:.2f}%)")
    
    # 节点时间戳信息
    if 'node_timestamps_raw' in graph_data:
        timestamps = graph_data['node_timestamps_raw']
        print(f"\n节点时间戳信息:")
        print(f"  时间戳范围: [{timestamps.min():.0f}, {timestamps.max():.0f}] (毫秒)")
        if 'timestamp_range' in graph_data:
            ts_range = graph_data['timestamp_range']
            print(f"  全局时间范围: ts_min={ts_range['ts_min']:.0f}, ts_max={ts_range['ts_max']:.0f} (毫秒)")
            time_span_days = (ts_range['ts_max'] - ts_range['ts_min']) / (1000 * 86400)
            print(f"  时间跨度: {time_span_days:.2f} 天")


def analyze_edges(graph_data):
    """分析边信息"""
    print("\n" + "=" * 80)
    print("【边分析】".center(80))
    print("=" * 80)
    
    # 总边数
    num_edges = graph_data.get('num_edges', 0)
    num_nodes = graph_data.get('num_nodes', 1)
    print(f"总边数: {num_edges:,}")
    print(f"边/节点比: {num_edges/num_nodes:.2f}")
    
    # 按类型统计边
    if 'edges_by_type' in graph_data:
        edges_by_type = graph_data['edges_by_type']
        print(f"\n各类型边数量:")
        
        edge_type_names = {
            'U-P-U': 'U-P-U (同用户评论)',
            'U-S-U': 'U-S-U (同评分评论)',
            'U-V-U': 'U-V-U (文本相似评论)'
        }
        
        for edge_type, edges in edges_by_type.items():
            edge_count = len(edges)
            edge_name = edge_type_names.get(edge_type, edge_type)
            percentage = (edge_count / num_edges * 100) if num_edges > 0 else 0
            print(f"  {edge_name}: {edge_count:,} ({percentage:.2f}%)")
    
    # 从统计信息中获取边数量
    if 'statistics' in graph_data:
        stats = graph_data['statistics']
        print(f"\n详细边统计（来自statistics）:")
        if 'edge_counts' in stats:
            for edge_type, count in stats['edge_counts'].items():
                edge_name = edge_type_names.get(edge_type, edge_type)
                percentage = (count / num_edges * 100) if num_edges > 0 else 0
                print(f"  {edge_name}: {count:,} ({percentage:.2f}%)")
        
        if 'upu_edges' in stats:
            print(f"\n边类型占比:")
            print(f"  U-P-U: {stats.get('upu_ratio', 0):.2f}%")
            print(f"  U-S-U: {stats.get('usu_ratio', 0):.2f}%")
            print(f"  U-V-U: {stats.get('uvu_ratio', 0):.2f}%")
    
    # 分析边的属性
    if 'edge_attributes' in graph_data:
        attrs = graph_data['edge_attributes']
        print(f"\n边属性分析:")
        
        # 统计边类型（考虑多重边）
        edge_types = []
        for attr in attrs:
            edge_type = attr.get('edge_type', 'unknown')
            # 处理多重边类型（如 "U-P-U+U-S-U"）
            if '+' in edge_type:
                edge_types.extend(edge_type.split('+'))
            else:
                edge_types.append(edge_type)
        
        type_counts = Counter(edge_types)
        print(f"  边类型分布（考虑多重边）:")
        for edge_type, count in type_counts.most_common():
            edge_name = edge_type_names.get(edge_type, edge_type)
            print(f"    {edge_name}: {count:,}")
        
        # 权重统计
        if 'edge_weights' in graph_data:
            weights = graph_data['edge_weights']
            print(f"\n  边权重统计:")
            print(f"    最小值: {weights.min():.4f}")
            print(f"    最大值: {weights.max():.4f}")
            print(f"    平均值: {weights.mean():.4f}")
            print(f"    中位数: {np.median(weights):.4f}")
        
        # 时间差统计
        if 'edge_time_diffs' in graph_data:
            time_diffs = graph_data['edge_time_diffs']
            print(f"\n  边时间差统计（天）:")
            print(f"    最小值: {time_diffs.min():.2f}")
            print(f"    最大值: {time_diffs.max():.2f}")
            print(f"    平均值: {time_diffs.mean():.2f}")
            print(f"    中位数: {np.median(time_diffs):.2f}")


def analyze_graph_structure(graph_data):
    """分析图结构"""
    print("\n" + "=" * 80)
    print("【图结构分析】".center(80))
    print("=" * 80)
    
    num_nodes = graph_data.get('num_nodes', 0)
    num_edges = graph_data.get('num_edges', 0)
    
    if num_nodes == 0:
        print("节点数为0，无法分析图结构")
        return
    
    # 计算度数
    degrees = np.zeros(num_nodes, dtype=int)
    if 'edges' in graph_data:
        edges = graph_data['edges']
        for edge in edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
    
    print(f"度数统计:")
    print(f"  平均度数: {degrees.mean():.2f}")
    print(f"  最大度数: {degrees.max()}")
    print(f"  最小度数: {degrees.min()}")
    print(f"  中位数度数: {np.median(degrees):.2f}")
    
    # 孤立节点
    isolated_nodes = (degrees == 0).sum()
    isolated_ratio = isolated_nodes / num_nodes * 100
    print(f"\n孤立节点:")
    print(f"  数量: {isolated_nodes:,}")
    print(f"  比例: {isolated_ratio:.2f}%")
    
    # 度数分布
    unique_degrees, degree_counts = np.unique(degrees, return_counts=True)
    print(f"\n度数分布（前10个最常见的度数）:")
    degree_dist = list(zip(unique_degrees, degree_counts))
    degree_dist.sort(key=lambda x: x[1], reverse=True)
    for degree, count in degree_dist[:10]:
        print(f"  度数为 {degree}: {count:,} 个节点 ({count/num_nodes*100:.2f}%)")


def load_and_compare_statistics():
    """加载并对比统计信息文件"""
    stats_path = config.STATISTICS_OUTPUT
    if Path(stats_path).exists():
        print("\n" + "=" * 80)
        print("【统计信息文件对比】".center(80))
        print("=" * 80)
        
        with open(stats_path, 'r', encoding='utf-8') as f:
            saved_stats = json.load(f)
        
        print("从 graph_statistics.json 加载的统计信息:")
        print(f"  节点数: {saved_stats.get('num_nodes', 0):,}")
        print(f"  总边数: {saved_stats.get('total_edges', 0):,}")
        if 'edge_counts' in saved_stats:
            print(f"  各类型边数量:")
            for edge_type, count in saved_stats['edge_counts'].items():
                print(f"    {edge_type}: {count:,}")
        print(f"  边/节点比: {saved_stats.get('edge_node_ratio', 0):.2f}")
        print(f"  平均度数: {saved_stats.get('avg_degree', 0):.2f}")
        print(f"  孤立节点: {saved_stats.get('isolated_nodes', 0):,}")


def main():
    """主函数"""
    print("=" * 80)
    print("Amazon评论图数据分析".center(80))
    print("=" * 80)
    
    try:
        # 加载图数据
        graph_data = load_graph_data()
        
        # 分析节点
        analyze_nodes(graph_data)
        
        # 分析边
        analyze_edges(graph_data)
        
        # 分析图结构
        analyze_graph_structure(graph_data)
        
        # 对比统计信息文件
        load_and_compare_statistics()
        
        print("\n" + "=" * 80)
        print("分析完成！".center(80))
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
