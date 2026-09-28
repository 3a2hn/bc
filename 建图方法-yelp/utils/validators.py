# -*- coding: utf-8 -*-
"""
数据验证工具
"""

import numpy as np
from .logger import get_logger


def validate_graph_quality(graph_data):
    """
    验证图数据质量
    
    Parameters
    ----------
    graph_data : dict
        图数据字典，包含 edges, edge_attributes, node_timestamps_norm 等
        
    Returns
    -------
    is_valid : bool
        是否通过验证
    issues : list
        发现的问题列表
    """
    logger = get_logger()
    issues = []
    
    logger.info("开始验证图数据质量...")
    
    # 检查必需字段
    required_fields = ['edges', 'edge_attributes', 'node_timestamps_norm', 'node_labels', 'num_nodes', 'num_edges']
    for field in required_fields:
        if field not in graph_data:
            issues.append(f"缺少必需字段: {field}")
    
    if issues:
        return False, issues
    
    # 检查边列表
    edges = graph_data['edges']
    num_edges = graph_data['num_edges']
    
    if len(edges) != num_edges:
        issues.append(f"edges 长度 ({len(edges)}) 与 num_edges ({num_edges}) 不匹配")
    
    # 检查边属性
    edge_attrs = graph_data['edge_attributes']
    if len(edge_attrs) != num_edges:
        issues.append(f"edge_attributes 长度 ({len(edge_attrs)}) 与 num_edges ({num_edges}) 不匹配")
    
    # 检查边索引范围
    num_nodes = graph_data['num_nodes']
    for i, (u, v) in enumerate(edges):
        if u < 0 or v < 0:
            issues.append(f"边 {i} 包含负索引: ({u}, {v})")
            break
        if u >= num_nodes or v >= num_nodes:
            issues.append(f"边 {i} 索引超出范围: ({u}, {v}), 节点数={num_nodes}")
            break
    
    # 检查节点时间戳
    node_timestamps = graph_data['node_timestamps_norm']
    if len(node_timestamps) != num_nodes:
        issues.append(f"node_timestamps_norm 长度 ({len(node_timestamps)}) 与 num_nodes ({num_nodes}) 不匹配")
    
    if np.isnan(node_timestamps).any():
        issues.append("node_timestamps_norm 包含 NaN 值")
    
    if np.isinf(node_timestamps).any():
        issues.append("node_timestamps_norm 包含 Inf 值")
    
    # 检查边时间戳
    if 'edge_timestamps_norm' in graph_data:
        edge_timestamps = graph_data['edge_timestamps_norm']
        if len(edge_timestamps) != num_edges:
            issues.append(f"edge_timestamps_norm 长度 ({len(edge_timestamps)}) 与 num_edges ({num_edges}) 不匹配")
        
        if np.isnan(edge_timestamps).any():
            issues.append("edge_timestamps_norm 包含 NaN 值")
        
        if np.isinf(edge_timestamps).any():
            issues.append("edge_timestamps_norm 包含 Inf 值")
    
    # 检查标签
    labels = graph_data['node_labels']
    if len(labels) != num_nodes:
        issues.append(f"node_labels 长度 ({len(labels)}) 与 num_nodes ({num_nodes}) 不匹配")
    
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, np.array([0, 1])):
        issues.append(f"node_labels 值异常: {unique_labels}, 应为 [0, 1]")
    
    # 输出结果
    if issues:
        logger.warning(f"图数据验证发现 {len(issues)} 个问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False, issues
    else:
        logger.info("图数据验证通过 ✓")
        return True, []


def validate_features(features):
    """
    验证特征矩阵质量
    
    Parameters
    ----------
    features : numpy.ndarray
        特征矩阵 (num_nodes, num_features)
        
    Returns
    -------
    is_valid : bool
        是否通过验证
    issues : list
        发现的问题列表
    """
    logger = get_logger()
    issues = []
    
    logger.info("开始验证特征矩阵...")
    
    # 检查形状
    if features.ndim != 2:
        issues.append(f"特征矩阵维度错误: {features.ndim}, 应为 2")
        return False, issues
    
    # 检查 NaN
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        issues.append(f"特征矩阵包含 {nan_count} 个 NaN 值")
    
    # 检查 Inf
    inf_count = np.isinf(features).sum()
    if inf_count > 0:
        issues.append(f"特征矩阵包含 {inf_count} 个 Inf 值")
    
    # 检查全零列
    zero_cols = np.where((features == 0).all(axis=0))[0]
    if len(zero_cols) > 0:
        issues.append(f"特征矩阵包含 {len(zero_cols)} 个全零列: {zero_cols[:10]}...")
    
    # 检查常数列
    constant_cols = []
    for i in range(features.shape[1]):
        if len(np.unique(features[:, i])) == 1:
            constant_cols.append(i)
    
    if len(constant_cols) > 0:
        issues.append(f"特征矩阵包含 {len(constant_cols)} 个常数列: {constant_cols[:10]}...")
    
    # 输出结果
    if issues:
        logger.warning(f"特征验证发现 {len(issues)} 个问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False, issues
    else:
        logger.info("特征验证通过 ✓")
        return True, []
