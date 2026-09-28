# -*- coding: utf-8 -*-
"""
图构建器主模块
整合三种边类型，构建完整的时间感知异构图

设计原则（Transductive 设定）：
- 建图阶段只负责构建完整图，不进行 train/val/test 划分
- 划分逻辑完全放到训练阶段统一处理
- 所有时间信息显式保存，供后续模块直接消费

时间戳定义：
- 节点时间戳：评论发布时间
  - node_timestamps_raw: 原始时间戳（秒）
  - node_timestamps_norm: 归一化时间戳 [0, 100]
- 边时间戳：max(t_raw(i), t_raw(j))，即两端节点中较晚的时间戳
  - edge_timestamps_raw: 原始边时间戳（秒）
  - edge_timestamps_norm: 归一化边时间戳 [0, 100]
- 全局时间边界：ts_min, ts_max（用于归一化）

输出数据结构：
- graph.pkl: 完整图数据（不含 train/val/test 划分）
- features.npy: 节点特征矩阵
- labels.npy: 节点标签
"""

import numpy as np
import json
from utils.logger import get_logger
from utils.data_loader import save_pickle
from .rur_builder import build_rur_edges
from .rtr_builder import build_rtr_edges
from .rsr_builder import build_rsr_edges
import config

logger = get_logger()


class GraphBuilder:
    """
    图构建器
    
    职责：构建完整的时间感知异构图，不进行数据划分
    """
    
    def __init__(self, df):
        """
        初始化
        
        Parameters
        ----------
        df : pandas.DataFrame
            预处理后的评论数据
        """
        self.df = df
        self.graph_data = {}
        # 时间戳归一化参数（节点和边共用）
        self.ts_min = None
        self.ts_max = None
        # 节点原始时间戳（秒）
        self.node_timestamps_raw = None
        
    def build_complete_graph(self):
        """
        构建完整图
        
        Returns
        -------
        graph_data : dict
            包含所有边、边属性和统计信息的字典
        """
        logger.info("="*80)
        logger.info("开始构建完整图")
        logger.info("="*80)
        
        # 首先计算全局时间戳范围（用于节点和边的归一化）
        self._compute_timestamp_range()
        
        # 构建三种边
        rur_edges, rur_attrs = build_rur_edges(self.df)
        rtr_edges, rtr_attrs = build_rtr_edges(self.df)
        rsr_edges, rsr_attrs = build_rsr_edges(self.df)
        
        # 合并所有边
        all_edges = rur_edges + rtr_edges + rsr_edges
        all_attrs = rur_attrs + rtr_attrs + rsr_attrs
        
        # 去重（防止多重边，虽然理论上不应该有）
        edge_dict = {}
        for edge, attr in zip(all_edges, all_attrs):
            edge_key = tuple(sorted(edge))
            if edge_key not in edge_dict:
                edge_dict[edge_key] = []
            edge_dict[edge_key].append(attr)
        
        # 合并相同边的属性
        unique_edges = []
        unique_attrs = []
        for edge, attrs_list in edge_dict.items():
            unique_edges.append(edge)
            # 如果同一对节点有多条边，保留所有类型
            if len(attrs_list) == 1:
                unique_attrs.append(attrs_list[0])
            else:
                # 合并多个边属性（取最大时间戳，因为边事件时间应该是最晚的）
                merged_attr = {
                    'edge_types': [a['edge_type'] for a in attrs_list],
                    'weights': [a['weight'] for a in attrs_list],
                    'timestamps': [a['timestamp'] for a in attrs_list],
                    'time_diffs': [a['time_diff'] for a in attrs_list],
                    # 合并边的事件时间取最大值
                    'timestamp': max(a['timestamp'] for a in attrs_list),
                }
                unique_attrs.append(merged_attr)
        
        logger.info(f"\n边去重统计:")
        logger.info(f"  原始边数: {len(all_edges):,}")
        logger.info(f"  去重后边数: {len(unique_edges):,}")
        logger.info(f"  多重边数: {len(all_edges) - len(unique_edges):,}")
        
        # 计算统计信息
        stats = self._calculate_statistics(
            unique_edges, unique_attrs,
            len(rur_edges), len(rtr_edges), len(rsr_edges)
        )
        
        # 提取节点时间戳（归一化到0-100）
        node_timestamps_norm = self._extract_node_timestamps()
        
        # 归一化边时间戳（使用与节点相同的参数）
        self._normalize_edge_timestamps(rur_attrs)
        self._normalize_edge_timestamps(rtr_attrs)
        self._normalize_edge_timestamps(rsr_attrs)
        self._normalize_edge_timestamps(unique_attrs)
        
        # 提取边属性为独立数组（用于快速访问）
        edge_arrays = self._extract_edge_arrays(unique_attrs)
        
        # 提取节点标签
        node_labels = self._extract_node_labels()
        
        # 构建图数据字典（Transductive 设定：不含 train/val/test 划分）
        self.graph_data = {
            # 基本信息
            'num_nodes': len(self.df),
            'num_edges': len(unique_edges),
            
            # 边信息
            'edges': unique_edges,
            'edge_attributes': unique_attrs,
            'edges_by_type': {
                'R-U-R': rur_edges,
                'R-T-R': rtr_edges,
                'R-S-R': rsr_edges,
            },
            'attributes_by_type': {
                'R-U-R': rur_attrs,
                'R-T-R': rtr_attrs,
                'R-S-R': rsr_attrs,
            },
            
            # 节点时间戳
            'node_timestamps_raw': self.node_timestamps_raw,   # [N] 原始时间戳（秒）
            'node_timestamps_norm': node_timestamps_norm,       # [N] 归一化时间戳 [0, 100]
            
            # 边时间戳和属性数组
            'edge_timestamps_raw': edge_arrays['edge_timestamps_raw'],    # [E] 原始边时间戳（秒）
            'edge_timestamps_norm': edge_arrays['edge_timestamps_norm'],  # [E] 归一化边时间戳 [0, 100]
            'edge_time_diffs': edge_arrays['edge_time_diffs'],            # [E] 边时间差（天）
            'edge_weights': edge_arrays['edge_weights'],                  # [E] 边权重
            
            # 节点标签（不含划分）
            'node_labels': node_labels,  # [N] 节点标签 (0=欺诈, 1=正常)
            
            # 全局时间边界（用于归一化）
            'timestamp_range': {
                'ts_min': self.ts_min,
                'ts_max': self.ts_max,
            },
            
            # 统计信息
            'statistics': stats,
            
            # 兼容旧字段（后续可移除）
            'timestamps': node_timestamps_norm,
            'edge_timestamps': edge_arrays['edge_timestamps_norm'],
        }
        
        # 验证边时间戳 >= max(两端节点时间戳)
        self._validate_edge_timestamps(unique_edges, unique_attrs, node_timestamps_norm)
        
        logger.info("="*80)
        logger.info("图构建完成！")
        logger.info("="*80)
        
        return self.graph_data
    
    def _compute_timestamp_range(self):
        """
        计算全局时间戳范围（用于节点和边的归一化）
        同时保存节点的原始时间戳
        """
        timestamps = self.df['timestamp'].values
        
        # 转换为秒数
        if hasattr(timestamps[0], 'timestamp'):
            ts_seconds = np.array([t.timestamp() for t in timestamps], dtype=np.float64)
        else:
            ts_seconds = timestamps.astype(np.float64)
        
        # 保存原始时间戳
        self.node_timestamps_raw = ts_seconds
        
        self.ts_min = float(ts_seconds.min())
        self.ts_max = float(ts_seconds.max())
        
        logger.info(f"\n全局时间戳范围:")
        logger.info(f"  ts_min: {self.ts_min}")
        logger.info(f"  ts_max: {self.ts_max}")
        logger.info(f"  时间跨度: {(self.ts_max - self.ts_min) / 86400:.2f} 天")
    
    def _extract_node_timestamps(self):
        """
        提取节点时间戳（原始和归一化）
        
        Returns
        -------
        timestamps_norm : numpy.ndarray
            归一化的时间戳数组 [N]，范围 [0, 100]
        """
        if self.ts_max == self.ts_min:
            # 防止除零
            timestamps_norm = np.zeros(len(self.node_timestamps_raw), dtype=np.float32)
        else:
            # 使用全局参数归一化
            timestamps_norm = ((self.node_timestamps_raw - self.ts_min) / (self.ts_max - self.ts_min) * 100).astype(np.float32)
        
        logger.info(f"\n节点时间戳提取:")
        logger.info(f"  节点数: {len(timestamps_norm)}")
        logger.info(f"  原始范围: [{self.node_timestamps_raw.min():.2f}, {self.node_timestamps_raw.max():.2f}] (秒)")
        logger.info(f"  归一化范围: [{timestamps_norm.min():.2f}, {timestamps_norm.max():.2f}]")
        logger.info(f"  归一化均值: {timestamps_norm.mean():.2f}")
        
        return timestamps_norm
    
    def _normalize_edge_timestamps(self, edge_attrs):
        """
        归一化边时间戳（使用与节点相同的参数）
        
        Parameters
        ----------
        edge_attrs : list of dict
            边属性列表，会原地修改添加 'timestamp_norm' 字段
        """
        if self.ts_max == self.ts_min:
            for attr in edge_attrs:
                attr['timestamp_norm'] = 0.0
        else:
            for attr in edge_attrs:
                # 边事件时间（秒）
                ts_raw = attr['timestamp']
                # 归一化到 [0, 100]
                ts_norm = (ts_raw - self.ts_min) / (self.ts_max - self.ts_min) * 100
                attr['timestamp_norm'] = ts_norm
    
    def _extract_edge_arrays(self, edge_attrs):
        """
        提取边属性为独立数组（用于快速访问）
        
        Parameters
        ----------
        edge_attrs : list of dict
            边属性列表
            
        Returns
        -------
        edge_arrays : dict
            包含以下数组：
            - edge_timestamps_raw: 原始边时间戳（秒）[E]
            - edge_timestamps_norm: 归一化边时间戳 [0, 100] [E]
            - edge_time_diffs: 边时间差（天）[E]
            - edge_weights: 边权重 [E]
        """
        num_edges = len(edge_attrs)
        
        edge_timestamps_raw = np.zeros(num_edges, dtype=np.float64)
        edge_timestamps_norm = np.zeros(num_edges, dtype=np.float32)
        edge_time_diffs = np.zeros(num_edges, dtype=np.float32)
        edge_weights = np.zeros(num_edges, dtype=np.float32)
        
        for i, attr in enumerate(edge_attrs):
            edge_timestamps_raw[i] = attr['timestamp']
            edge_timestamps_norm[i] = attr['timestamp_norm']
            # 处理合并边的情况
            if 'time_diff' in attr:
                edge_time_diffs[i] = attr['time_diff']
            elif 'time_diffs' in attr:
                edge_time_diffs[i] = np.mean(attr['time_diffs'])
            if 'weight' in attr:
                edge_weights[i] = attr['weight']
            elif 'weights' in attr:
                edge_weights[i] = np.mean(attr['weights'])
        
        logger.info(f"\n边属性数组提取:")
        logger.info(f"  边数: {num_edges}")
        logger.info(f"  时间戳原始范围: [{edge_timestamps_raw.min():.2f}, {edge_timestamps_raw.max():.2f}] (秒)")
        logger.info(f"  时间戳归一化范围: [{edge_timestamps_norm.min():.2f}, {edge_timestamps_norm.max():.2f}]")
        logger.info(f"  时间差范围: [{edge_time_diffs.min():.2f}, {edge_time_diffs.max():.2f}] (天)")
        logger.info(f"  权重范围: [{edge_weights.min():.4f}, {edge_weights.max():.4f}]")
        
        return {
            'edge_timestamps_raw': edge_timestamps_raw,
            'edge_timestamps_norm': edge_timestamps_norm,
            'edge_time_diffs': edge_time_diffs,
            'edge_weights': edge_weights,
        }
    
    def _extract_node_labels(self):
        """
        提取节点标签
        
        Returns
        -------
        labels : numpy.ndarray
            节点标签数组 [N]，0=欺诈，1=正常
        """
        if 'label' in self.df.columns:
            labels = self.df['label'].to_numpy().astype(np.int32)
            
            # 统计标签分布
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"\n节点标签提取:")
            for label, count in zip(unique, counts):
                label_name = "正常" if label == 1 else "欺诈"
                logger.info(f"  {label_name} (label={label}): {count:,} ({count/len(labels)*100:.2f}%)")
            
            return labels
        else:
            logger.warning("数据中没有 'label' 列，返回空标签")
            return np.zeros(len(self.df), dtype=np.int32)
    
    def _validate_edge_timestamps(self, edges, edge_attrs, node_timestamps):
        """
        验证边时间戳 >= max(两端节点时间戳)
        
        Parameters
        ----------
        edges : list of tuple
            边列表
        edge_attrs : list of dict
            边属性列表
        node_timestamps : numpy.ndarray
            节点时间戳数组
        """
        violations = 0
        max_violation = 0.0
        violation_examples = []
        
        for (i, j), attr in zip(edges, edge_attrs):
            edge_ts = attr['timestamp_norm']
            node_max_ts = max(node_timestamps[i], node_timestamps[j])
            
            # 允许小的浮点误差
            diff = node_max_ts - edge_ts
            if diff > 1e-6:
                violations += 1
                max_violation = max(max_violation, diff)
                
                # 记录前几个违规示例
                if len(violation_examples) < 5:
                    violation_examples.append({
                        'edge': (i, j),
                        'edge_ts': edge_ts,
                        'node_i_ts': node_timestamps[i],
                        'node_j_ts': node_timestamps[j],
                        'node_max_ts': node_max_ts,
                        'diff': diff,
                        'edge_type': attr.get('edge_type', 'unknown')
                    })
        
        if violations > 0:
            logger.warning(f"⚠️ 发现 {violations} 条边的时间戳小于两端节点的最大时间戳！")
            logger.warning(f"   最大违规差值: {max_violation:.6f}")
            logger.warning(f"   违规比例: {violations/len(edges)*100:.2f}%")
            
            if violation_examples:
                logger.warning(f"\n   前{len(violation_examples)}个违规示例:")
                for ex in violation_examples:
                    logger.warning(f"     边 {ex['edge']}: edge_ts={ex['edge_ts']:.4f}, "
                                 f"node_max_ts={ex['node_max_ts']:.4f}, "
                                 f"diff={ex['diff']:.6f}, type={ex['edge_type']}")
        else:
            logger.info(f"✓ 边时间戳验证通过: 所有边的 t_norm(e) >= max(t_norm(i), t_norm(j))")
    
    def _calculate_statistics(self, edges, attrs, rur_count, rtr_count, rsr_count):
        """
        计算图统计信息
        
        Parameters
        ----------
        edges : list
            边列表
        attrs : list
            边属性列表
        rur_count, rtr_count, rsr_count : int
            各类型边的数量
            
        Returns
        -------
        stats : dict
            统计信息
        """
        num_nodes = len(self.df)
        num_edges = len(edges)
        
        stats = {
            'num_nodes': num_nodes,
            'total_edges': num_edges,
            'rur_edges': rur_count,
            'rtr_edges': rtr_count,
            'rsr_edges': rsr_count,
        }
        
        # 边/节点比
        edge_node_ratio = num_edges / num_nodes if num_nodes > 0 else 0
        stats['edge_node_ratio'] = edge_node_ratio
        
        # 边类型占比
        if num_edges > 0:
            stats['rur_ratio'] = (rur_count / num_edges) * 100
            stats['rtr_ratio'] = (rtr_count / num_edges) * 100
            stats['rsr_ratio'] = (rsr_count / num_edges) * 100
        else:
            stats['rur_ratio'] = 0
            stats['rtr_ratio'] = 0
            stats['rsr_ratio'] = 0
        
        # 度统计
        degrees = np.zeros(num_nodes, dtype=int)
        for edge in edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        
        stats['avg_degree'] = degrees.mean()
        stats['max_degree'] = degrees.max()
        stats['min_degree'] = degrees.min()
        stats['isolated_nodes'] = (degrees == 0).sum()
        
        # 打印统计信息
        logger.info("\n图统计信息:")
        logger.info(f"  节点数: {num_nodes:,}")
        logger.info(f"  总边数: {num_edges:,}")
        logger.info(f"  R-U-R边: {rur_count:,} ({stats['rur_ratio']:.2f}%)")
        logger.info(f"  R-T-R边: {rtr_count:,} ({stats['rtr_ratio']:.2f}%)")
        logger.info(f"  R-S-R边: {rsr_count:,} ({stats['rsr_ratio']:.2f}%)")
        logger.info(f"  边/节点比: {edge_node_ratio:.2f}")
        logger.info(f"  平均度数: {stats['avg_degree']:.2f}")
        logger.info(f"  最大度数: {stats['max_degree']}")
        logger.info(f"  孤立节点: {stats['isolated_nodes']}")
        
        return stats
    
    def save_graph(self, output_path=None):
        """
        保存图数据
        
        Parameters
        ----------
        output_path : str or Path, optional
            保存路径，默认使用config中的路径
        """
        if output_path is None:
            output_path = config.GRAPH_OUTPUT
        
        save_pickle(self.graph_data, output_path)
        logger.info(f"图数据已保存到: {output_path}")
        
        # 同时保存统计信息为JSON
        stats_path = config.STATISTICS_OUTPUT
        with open(stats_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python类型
            stats_json = {}
            for k, v in self.graph_data['statistics'].items():
                if isinstance(v, (np.integer, np.floating)):
                    stats_json[k] = v.item()
                else:
                    stats_json[k] = v
            json.dump(stats_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"统计信息已保存到: {stats_path}")


def run_graph_building(df):
    """
    运行图构建流程
    
    Parameters
    ----------
    df : pandas.DataFrame
        预处理后的数据
        
    Returns
    -------
    graph_data : dict
        图数据
    """
    builder = GraphBuilder(df)
    graph_data = builder.build_complete_graph()
    builder.save_graph()
    
    return graph_data


if __name__ == '__main__':
    from utils.logger import setup_logger
    from preprocessor import run_preprocessing
    
    setup_logger(log_file=config.LOG_FILE, level='INFO')
    
    # 运行预处理
    df, mappings = run_preprocessing(config.INPUT_CSV)
    
    # 构建图
    graph_data = run_graph_building(df)
    
    print("\n" + "="*80)
    print("图构建完成！")
    print(f"节点数: {graph_data['num_nodes']:,}")
    print(f"边数: {graph_data['statistics']['total_edges']:,}")
    print("="*80)

