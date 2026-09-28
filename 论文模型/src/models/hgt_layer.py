# -*- coding: utf-8 -*-
"""
HGT层封装
改编自HGT原始实现，添加对我们欺诈检测任务的适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
import math
import logging

logger = logging.getLogger(__name__)


class HGTConv(MessagePassing):
    """
    异构图Transformer卷积层
    
    实现异构图的注意力机制，对不同类型的节点和边使用不同的参数
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_types: int,
        num_relations: int,
        n_heads: int = 8,
        dropout: float = 0.2,
        use_norm: bool = True,
        use_RTE: bool = False,
        **kwargs
    ):
        """
        初始化HGT卷积层
        
        Parameters
        ----------
        in_dim : int
            输入特征维度
        out_dim : int
            输出特征维度
        num_types : int
            节点类型数量
        num_relations : int
            边类型数量
        n_heads : int
            注意力头数
        dropout : float
            Dropout率
        use_norm : bool
            是否使用LayerNorm
        use_RTE : bool
            是否使用相对时间编码
        """
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_RTE = use_RTE
        self.att = None
        
        # 为每种节点类型创建线性变换
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        
        # 关系特定的参数
        self.relation_pri = nn.Parameter(torch.ones(num_relations, n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)
        
        # 相对时间编码（可选）
        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
    
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time=None):
        """
        前向传播
        
        Parameters
        ----------
        node_inp : torch.Tensor
            节点输入特征 [N, F]
        node_type : torch.Tensor
            节点类型 [N]
        edge_index : torch.Tensor
            边索引 [2, E]
        edge_type : torch.Tensor
            边类型 [E]
        edge_time : torch.Tensor
            边时间（可选） [E]
            
        Returns
        -------
        out : torch.Tensor
            输出节点特征 [N, F]
        """
        return self.propagate(
            edge_index,
            node_inp=node_inp,
            node_type=node_type,
            edge_type=edge_type,
            edge_time=edge_time
        )
    
    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        """
        消息函数
        
        j: source, i: target; <j, i>
        """
        data_size = edge_index_i.size(0)
        
        # 创建注意力和消息张量
        res_att = torch.zeros(data_size, self.n_heads, device=node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k, device=node_inp_i.device)
        
        # 遍历所有类型组合
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                
                for relation_type in range(self.num_relations):
                    # 找到符合<source_type, relation_type, target_type>的边
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    
                    # 获取对应的节点表示
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    
                    # 添加时间编码（可选）
                    if self.use_RTE and edge_time is not None:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    
                    # Step 1: 异构互注意力
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
                    att_scores = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    # AMP兼容：确保dtype匹配
                    if res_att.dtype != att_scores.dtype:
                        res_att = res_att.to(att_scores.dtype)
                    res_att[idx] = att_scores
                    
                    # Step 2: 异构消息传递
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    msg = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)
                    # AMP兼容：确保dtype匹配
                    if res_msg.dtype != msg.dtype:
                        res_msg = res_msg.to(msg.dtype)
                    res_msg[idx] = msg
        
        # Softmax注意力权重
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)
    
    def update(self, aggr_out, node_inp, node_type):
        """
        更新函数
        
        Step 3: 目标特定聚合
        x = W[node_type] * gelu(Agg(x)) + x
        """
        aggr_out = functional.gelu(aggr_out)
        # 创建输出张量
        res = torch.zeros(aggr_out.size(0), self.out_dim, device=node_inp.device)
        
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            
            # 带可学习权重的跳跃连接
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                output = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                output = trans_out * alpha + node_inp[idx] * (1 - alpha)
            
            # AMP兼容：确保dtype匹配
            if res.dtype != output.dtype:
                res = res.to(output.dtype)
            res[idx] = output
        
        return res


class RelTemporalEncoding(nn.Module):
    """
    相对时间编码（Sinusoid）
    """
    
    def __init__(self, n_hid, max_len=240, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    
    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class HGTLayer(nn.Module):
    """
    HGT层的高级封装
    
    支持多层HGT的堆叠
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_types: int,
        num_relations: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.2,
        use_norm: bool = True,
        use_RTE: bool = False,
        use_gradient_checkpointing: bool = False
    ):
        """
        初始化HGT层
        
        Parameters
        ----------
        in_dim : int
            输入特征维度
        hidden_dim : int
            隐藏层维度
        num_types : int
            节点类型数量
        num_relations : int
            边类型数量
        n_heads : int
            注意力头数
        n_layers : int
            HGT层数
        dropout : float
            Dropout率
        use_norm : bool
            是否使用LayerNorm
        use_RTE : bool
            是否使用相对时间编码
        """
        super(HGTLayer, self).__init__()

        self.num_types = num_types
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 适配层：将输入特征映射为hidden_dim
        self.adapt_ws = nn.ModuleList()
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))
        
        # HGT卷积层
        self.gcs = nn.ModuleList()
        for l in range(n_layers):
            self.gcs.append(
                HGTConv(
                    hidden_dim,
                    hidden_dim,
                    num_types,
                    num_relations,
                    n_heads,
                    dropout,
                    use_norm,
                    use_RTE
                )
            )
        
        self.drop = nn.Dropout(dropout)
    
    def forward(self, node_feature, node_type, edge_index, edge_type, edge_time=None):
        """
        前向传播
        
        Parameters
        ----------
        node_feature : torch.Tensor
            节点特征 [N, F_in]
        node_type : torch.Tensor
            节点类型 [N]
        edge_index : torch.Tensor
            边索引 [2, E]
        edge_type : torch.Tensor
            边类型 [E]
        edge_time : torch.Tensor
            边时间（可选） [E]
            
        Returns
        -------
        meta_xs : torch.Tensor
            输出节点特征 [N, F_hidden]
        """
        # 适配层：根据节点类型进行不同的变换
        # AMP兼容：动态匹配dtype
        res = torch.zeros(node_feature.size(0), self.hidden_dim, device=node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            transformed = torch.nn.functional.gelu(self.adapt_ws[t_id](node_feature[idx]))
            # 确保res的dtype与变换后的输出匹配（AMP兼容）
            if res.dtype != transformed.dtype:
                res = res.to(transformed.dtype)
            res[idx] = transformed
        
        meta_xs = self.drop(res)
        del res
        
        # 通过HGT层
        for gc in self.gcs:
            if self.use_gradient_checkpointing and self.training:
                # 梯度检查点：不保存中间激活，反向传播时重新计算，节省显存
                meta_xs = grad_checkpoint(
                    gc, meta_xs, node_type, edge_index, edge_type, edge_time,
                    use_reentrant=False
                )
            else:
                meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        
        return meta_xs


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 参数
    num_nodes = 100
    in_dim = 87
    hidden_dim = 128
    num_types = 1  # 只有评论节点
    num_relations = 3  # R-U-R, R-T-R, R-S-R
    n_heads = 8
    n_layers = 2
    
    # 创建模型
    model = HGTLayer(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_types=num_types,
        num_relations=num_relations,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.2,
        use_norm=True,
        use_RTE=False
    )
    
    # 创建测试数据
    node_feature = torch.randn(num_nodes, in_dim)
    node_type = torch.zeros(num_nodes, dtype=torch.long)  # 全部为评论节点
    
    # 创建边（随机）
    num_edges = 500
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    print(f"节点特征形状: {node_feature.shape}")
    print(f"边数量: {num_edges}")
    
    # 前向传播
    output = model(node_feature, node_type, edge_index, edge_type)
    
    print(f"输出形状: {output.shape}")
    print(f"期望形状: ({num_nodes}, {hidden_dim})")
    
    print("\n✓ HGT层测试通过！")
