# -*- coding: utf-8 -*-
"""
Amazon边构建模块
以评论为节点，构建三种类型的边：

Amazon边定义：
- U-P-U: 由同一用户发布的评论之间的连接
- U-S-U: 一周内至少共享一次相同星级评分的用户之间的连接
- U-V-U: 评论文本相似度(TF-IDF)位于所有用户前5%的用户之间的连接
"""

from .rpr_builder import build_rpr_edges
from .rsr_builder import build_rsr_edges
from .rvr_builder import build_rvr_edges
from .graph_builder_amazon import GraphBuilderAmazon, run_graph_building

__all__ = [
    'build_rpr_edges',
    'build_rsr_edges',
    'build_rvr_edges',
    'GraphBuilderAmazon',
    'run_graph_building',
]
