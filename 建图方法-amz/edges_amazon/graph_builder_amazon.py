# -*- coding: utf-8 -*-
"""
Amazon图构建器主模块
整合三种边类型（U-P-U, U-S-U, U-V-U），构建以评论为节点的时间感知异构图

边定义：
- U-P-U: 由同一用户发布的评论之间的连接
- U-S-U: 一周内至少共享一次相同星级评分的用户之间的评论连接
- U-V-U: 评论文本相似度(TF-IDF)位于所有用户前5%的用户之间的评论连接
"""

import numpy as np
import json
import pickle
import logging
from pathlib import Path
from .rpr_builder import build_rpr_edges
from .rsr_builder import build_rsr_edges
from .rvr_builder import build_rvr_edges

logger = logging.getLogger(__name__)


class GraphBuilderAmazon:
    """
    Amazon图构建器
    
    构建以评论为节点的时间感知异构图（与Yelp结构相同）
    """
    
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.graph_data = {}
        
        self.ts_min = None
        self.ts_max = None
        self.node_timestamps_raw = None
        
    def build_complete_graph(self):
        """构建完整图"""
        logger.info("=" * 80)
        logger.info("开始构建Amazon评论图（以评论为节点）")
        logger.info("边类型: U-P-U / U-S-U / U-V-U")
        logger.info("=" * 80)
        
        # 首先计算全局时间戳范围
        self._compute_timestamp_range()
        
        # 构建三种边
        logger.info("\n--- 构建 U-P-U 边（同用户评论）---")
        rpr_edges, rpr_attrs = build_rpr_edges(self.df, self.config)

        logger.info("\n--- 构建 U-S-U 边（同评分评论）---")
        rsr_edges, rsr_attrs = build_rsr_edges(self.df, self.config)

        logger.info("\n--- 构建 U-V-U 边（文本相似评论）---")
        rvr_edges, rvr_attrs = build_rvr_edges(self.df, self.config)
        
        # 合并所有边
        all_edges = rpr_edges + rsr_edges + rvr_edges
        all_attrs = rpr_attrs + rsr_attrs + rvr_attrs
        
        logger.info(f"\n合并前边数统计:")
        logger.info(f"  U-P-U: {len(rpr_edges):,}")
        logger.info(f"  U-S-U: {len(rsr_edges):,}")
        logger.info(f"  U-V-U: {len(rvr_edges):,}")
        logger.info(f"  总计: {len(all_edges):,}")
        
        # 去重（保留边类型信息）
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
            if len(attrs_list) == 1:
                unique_attrs.append(attrs_list[0])
            else:
                merged_attr = {
                    'edge_types': [a['edge_type'] for a in attrs_list],
                    'weights': [a['weight'] for a in attrs_list],
                    'timestamps': [a['timestamp'] for a in attrs_list],
                    'time_diffs': [a['time_diff'] for a in attrs_list],
                    'timestamp': max(a['timestamp'] for a in attrs_list),
                    'weight': np.mean([a['weight'] for a in attrs_list]),
                    'time_diff': np.mean([a['time_diff'] for a in attrs_list]),
                    'edge_type': '+'.join(sorted(set(a['edge_type'] for a in attrs_list))),
                }
                unique_attrs.append(merged_attr)
        
        logger.info(f"\n边去重统计:")
        logger.info(f"  原始边数: {len(all_edges):,}")
        logger.info(f"  去重后边数: {len(unique_edges):,}")
        logger.info(f"  多重边数: {len(all_edges) - len(unique_edges):,}")
        
        # 计算统计信息
        stats = self._calculate_statistics(
            unique_edges, unique_attrs,
            len(rpr_edges), len(rsr_edges), len(rvr_edges)
        )
        
        # 提取节点时间戳
        node_timestamps_norm = self._extract_node_timestamps()
        
        # 归一化边时间戳
        self._normalize_edge_timestamps(rpr_attrs)
        self._normalize_edge_timestamps(rsr_attrs)
        self._normalize_edge_timestamps(rvr_attrs)
        self._normalize_edge_timestamps(unique_attrs)
        
        # 提取边属性数组
        edge_arrays = self._extract_edge_arrays(unique_attrs)
        
        # 提取节点标签
        node_labels = self._extract_node_labels()
        
        # 构建图数据字典（与Yelp结构一致）
        self.graph_data = {
            # 基本信息
            'num_nodes': len(self.df),
            'num_edges': len(unique_edges),
            'node_type': 'review',
            
            # 边信息
            'edges': unique_edges,
            'edge_attributes': unique_attrs,
            'edges_by_type': {
                'U-P-U': rpr_edges,
                'U-S-U': rsr_edges,
                'U-V-U': rvr_edges,
            },
            'attributes_by_type': {
                'U-P-U': rpr_attrs,
                'U-S-U': rsr_attrs,
                'U-V-U': rvr_attrs,
            },
            
            # 节点时间戳
            'node_timestamps_raw': self.node_timestamps_raw,
            'node_timestamps_norm': node_timestamps_norm,
            
            # 边时间戳和属性数组
            'edge_timestamps_raw': edge_arrays['edge_timestamps_raw'],
            'edge_timestamps_norm': edge_arrays['edge_timestamps_norm'],
            'edge_time_diffs': edge_arrays['edge_time_diffs'],
            'edge_weights': edge_arrays['edge_weights'],
            
            # 节点标签
            'node_labels': node_labels,
            
            # 全局时间边界
            'timestamp_range': {
                'ts_min': self.ts_min,
                'ts_max': self.ts_max,
            },
            
            # 统计信息
            'statistics': stats,
            
            # 兼容字段
            'timestamps': node_timestamps_norm,
            'edge_timestamps': edge_arrays['edge_timestamps_norm'],
        }
        
        # 验证边时间戳
        self._validate_edge_timestamps(unique_edges, unique_attrs, node_timestamps_norm)
        
        logger.info("=" * 80)
        logger.info("Amazon评论图构建完成！")
        logger.info("=" * 80)
        
        return self.graph_data
    
    def _compute_timestamp_range(self):
        """计算全局时间戳范围"""
        timestamps = self.df['detailed_timestamp'].values.astype(np.float64)
        
        self.node_timestamps_raw = timestamps
        self.ts_min = float(timestamps.min())
        self.ts_max = float(timestamps.max())
        
        logger.info(f"\n全局时间戳范围:")
        logger.info(f"  ts_min: {self.ts_min}")
        logger.info(f"  ts_max: {self.ts_max}")
        logger.info(f"  时间跨度: {(self.ts_max - self.ts_min) / (1000 * 86400):.2f} 天")
    
    def _extract_node_timestamps(self):
        """提取节点时间戳（归一化）"""
        if self.ts_max == self.ts_min:
            timestamps_norm = np.zeros(len(self.node_timestamps_raw), dtype=np.float32)
        else:
            timestamps_norm = ((self.node_timestamps_raw - self.ts_min) / 
                             (self.ts_max - self.ts_min) * 100).astype(np.float32)
        
        logger.info(f"\n节点时间戳提取:")
        logger.info(f"  节点数: {len(timestamps_norm)}")
        logger.info(f"  归一化范围: [{timestamps_norm.min():.2f}, {timestamps_norm.max():.2f}]")
        
        return timestamps_norm
    
    def _normalize_edge_timestamps(self, edge_attrs):
        """归一化边时间戳"""
        if self.ts_max == self.ts_min:
            for attr in edge_attrs:
                attr['timestamp_norm'] = 0.0
        else:
            for attr in edge_attrs:
                ts_raw = attr['timestamp']
                ts_norm = (ts_raw - self.ts_min) / (self.ts_max - self.ts_min) * 100
                attr['timestamp_norm'] = ts_norm
    
    def _extract_edge_arrays(self, edge_attrs):
        """提取边属性为独立数组"""
        num_edges = len(edge_attrs)
        
        edge_timestamps_raw = np.zeros(num_edges, dtype=np.float64)
        edge_timestamps_norm = np.zeros(num_edges, dtype=np.float32)
        edge_time_diffs = np.zeros(num_edges, dtype=np.float32)
        edge_weights = np.zeros(num_edges, dtype=np.float32)
        
        for i, attr in enumerate(edge_attrs):
            edge_timestamps_raw[i] = attr['timestamp']
            edge_timestamps_norm[i] = attr.get('timestamp_norm', 0.0)
            
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
        logger.info(f"  时间戳归一化范围: [{edge_timestamps_norm.min():.2f}, {edge_timestamps_norm.max():.2f}]")
        logger.info(f"  权重范围: [{edge_weights.min():.4f}, {edge_weights.max():.4f}]")
        
        return {
            'edge_timestamps_raw': edge_timestamps_raw,
            'edge_timestamps_norm': edge_timestamps_norm,
            'edge_time_diffs': edge_time_diffs,
            'edge_weights': edge_weights,
        }
    
    def _extract_node_labels(self):
        """提取节点标签（评论级别）"""
        if 'label' in self.df.columns:
            labels = self.df['label'].to_numpy().astype(np.int32)
        elif 'class' in self.df.columns:
            # 反转标签：原始 class=1(欺诈)→label=0, class=0(正常)→label=1，与Yelp一致
            labels = (1 - self.df['class']).to_numpy().astype(np.int32)
        else:
            logger.warning("数据中没有标签列，返回空标签")
            return np.zeros(len(self.df), dtype=np.int32)
        
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"\n节点标签提取:")
        for label, count in zip(unique, counts):
            label_name = "欺诈" if label == 0 else "正常"
            logger.info(f"  {label_name} (label={label}): {count:,} ({count/len(labels)*100:.2f}%)")
        
        return labels
    
    def _validate_edge_timestamps(self, edges, edge_attrs, node_timestamps):
        """验证边时间戳 >= max(两端节点时间戳)
        
        注意：node_timestamps 是 float32，edge_ts 是 float64，
        精度差异约 1e-5，因此容差需设为 0.01（归一化 [0,100] 范围）
        """
        violations = 0
        tolerance = 0.01
        for (i, j), attr in zip(edges, edge_attrs):
            edge_ts = attr.get('timestamp_norm', 0)
            node_max_ts = float(max(node_timestamps[i], node_timestamps[j]))
            if node_max_ts - edge_ts > tolerance:
                violations += 1
        
        if violations > 0:
            logger.warning(f"⚠️ 发现 {violations} 条边的时间戳违规（容差={tolerance}）")
        else:
            logger.info(f"✓ 边时间戳验证通过（容差={tolerance}）")
    
    def _calculate_statistics(self, edges, attrs, rpr_count, rsr_count, rvr_count):
        """计算图统计信息"""
        num_nodes = len(self.df)
        num_edges = len(edges)
        
        stats = {
            'num_nodes': num_nodes,
            'total_edges': num_edges,
            'upu_edges': rpr_count,
            'usu_edges': rsr_count,
            'uvu_edges': rvr_count,
            'edge_counts': {
                'U-P-U': rpr_count,
                'U-S-U': rsr_count,
                'U-V-U': rvr_count,
            },
        }
        
        edge_node_ratio = num_edges / num_nodes if num_nodes > 0 else 0
        stats['edge_node_ratio'] = edge_node_ratio
        
        if num_edges > 0:
            stats['upu_ratio'] = (rpr_count / num_edges) * 100
            stats['usu_ratio'] = (rsr_count / num_edges) * 100
            stats['uvu_ratio'] = (rvr_count / num_edges) * 100
        else:
            stats['upu_ratio'] = 0
            stats['usu_ratio'] = 0
            stats['uvu_ratio'] = 0
        
        # 度统计
        degrees = np.zeros(num_nodes, dtype=int)
        for edge in edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        
        stats['avg_degree'] = float(degrees.mean()) if num_nodes > 0 else 0
        stats['max_degree'] = int(degrees.max()) if num_nodes > 0 else 0
        stats['min_degree'] = int(degrees.min()) if num_nodes > 0 else 0
        stats['isolated_nodes'] = int((degrees == 0).sum())
        
        logger.info("\n图统计信息:")
        logger.info(f"  节点数（评论）: {num_nodes:,}")
        logger.info(f"  总边数: {num_edges:,}")
        logger.info(f"  U-P-U边: {rpr_count:,} ({stats['upu_ratio']:.2f}%)")
        logger.info(f"  U-S-U边: {rsr_count:,} ({stats['usu_ratio']:.2f}%)")
        logger.info(f"  U-V-U边: {rvr_count:,} ({stats['uvu_ratio']:.2f}%)")
        logger.info(f"  边/节点比: {edge_node_ratio:.2f}")
        logger.info(f"  平均度数: {stats['avg_degree']:.2f}")
        logger.info(f"  孤立节点: {stats['isolated_nodes']}")
        
        return stats
    
    def save_graph(self, output_path=None):
        """保存图数据"""
        if output_path is None:
            output_path = self.config.GRAPH_OUTPUT
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph_data, f)
        logger.info(f"图数据已保存到: {output_path}")
        
        # 保存统计信息
        stats_path = self.config.STATISTICS_OUTPUT
        with open(stats_path, 'w', encoding='utf-8') as f:
            stats_json = {}
            for k, v in self.graph_data['statistics'].items():
                if isinstance(v, (np.integer, np.floating)):
                    stats_json[k] = v.item()
                elif isinstance(v, dict):
                    stats_json[k] = {kk: (vv.item() if isinstance(vv, (np.integer, np.floating)) else vv) 
                                    for kk, vv in v.items()}
                else:
                    stats_json[k] = v
            json.dump(stats_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"统计信息已保存到: {stats_path}")


def run_graph_building(df, config):
    """运行图构建流程"""
    builder = GraphBuilderAmazon(df, config)
    graph_data = builder.build_complete_graph()
    builder.save_graph()
    return graph_data
