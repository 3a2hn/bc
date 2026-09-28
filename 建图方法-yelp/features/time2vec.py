# -*- coding: utf-8 -*-
"""
Time2Vec时间编码模块
基于论文: "Time2Vec: Learning a Vector Representation of Time"
用周期性激活函数捕捉时间的周期性特征
"""

import torch
import torch.nn as nn
import numpy as np
from utils.logger import get_logger

logger = get_logger()


class Time2Vec(nn.Module):
    """
    Time2Vec编码器
    
    将标量时间戳转换为向量表示，捕捉周期性模式
    
    公式:
        t2v(τ)[i] = ω₀·τ + φ₀,           if i = 0  (线性项，1维)
                  = sin(ωᵢ·τ + φᵢ),      if i = 1, ..., k-1  (周期项，k-1维)
    
    Parameters
    ----------
    out_features : int
        输出特征维度（默认64，即1个线性项 + 63个周期项）
    use_cos : bool
        是否使用cos代替sin（默认False）
    """
    
    def __init__(self, out_features=64, use_cos=False):
        super(Time2Vec, self).__init__()
        
        self.out_features = out_features
        
        # 线性项的参数 (1维输入 -> 1维输出)
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        
        # 周期项的参数 (1维输入 -> (k-1)维输出)
        self.w = nn.Parameter(torch.randn(1, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))
        
        # 选择周期函数
        self.f = torch.cos if use_cos else torch.sin
        
    def forward(self, tau):
        """
        前向传播
        
        Parameters
        ----------
        tau : torch.Tensor
            输入时间戳，形状 (batch_size, 1)
            
        Returns
        -------
        output : torch.Tensor
            Time2Vec编码，形状 (batch_size, out_features)
        """
        # 线性项: v0 = w0 * tau + b0
        v0 = torch.matmul(tau, self.w0) + self.b0  # (batch_size, 1)
        
        # 周期项: v_i = sin(w_i * tau + b_i) for i=1,...,k-1
        v_periodic = self.f(torch.matmul(tau, self.w) + self.b)  # (batch_size, k-1)
        
        # 拼接线性项和周期项
        return torch.cat([v0, v_periodic], dim=-1)  # (batch_size, k)


class Time2VecEncoder:
    """
    Time2Vec特征编码器（用于特征工程）
    
    将归一化的时间戳编码为Time2Vec向量
    """
    
    def __init__(self, out_features=64, use_cos=False):
        """
        初始化
        
        Parameters
        ----------
        out_features : int
            输出特征维度（默认64：1线性 + 63周期）
        use_cos : bool
            是否使用cos代替sin
        """
        self.out_features = out_features
        self.model = Time2Vec(out_features=out_features, use_cos=use_cos)
        self.model.eval()  # 评估模式
        
        logger.info(f"Time2Vec编码器初始化: 输出{out_features}维 (1线性 + {out_features-1}周期)")
    
    def encode(self, timestamps_norm):
        """
        编码归一化的时间戳
        
        Parameters
        ----------
        timestamps_norm : numpy.ndarray
            归一化的时间戳数组，形状 (n_samples,)，范围[0,1]
            
        Returns
        -------
        time_vectors : numpy.ndarray
            Time2Vec编码向量，形状 (n_samples, out_features)
        """
        # 转换为PyTorch张量
        tau = torch.FloatTensor(timestamps_norm).unsqueeze(1)  # (n, 1)
        
        # 编码
        with torch.no_grad():
            time_vectors = self.model(tau)  # (n, out_features)
        
        # 转换回NumPy
        return time_vectors.numpy()
    
    def save_weights(self, path):
        """保存模型权重"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Time2Vec权重已保存到: {path}")
    
    def load_weights(self, path):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Time2Vec权重已加载: {path}")


def initialize_time2vec_with_frequencies(encoder, frequencies=None):
    """
    使用预定义频率初始化Time2Vec
    
    这样可以让模型捕捉特定的时间周期（如日、周、月）
    
    Parameters
    ----------
    encoder : Time2VecEncoder
        Time2Vec编码器
    frequencies : list of float, optional
        预定义的频率列表（单位：周期/年）
        例如: [365, 52, 12, 4, 1] 表示 日、周、月、季度、年
    """
    if frequencies is None:
        # 默认频率：捕捉多个时间尺度
        frequencies = [
            365.0,  # 日周期
            52.0,   # 周周期
            12.0,   # 月周期
            4.0,    # 季度周期
            1.0,    # 年周期
        ]
    
    # 确保频率数量不超过模型维度-1（因为第一维是线性的）
    n_freq = min(len(frequencies), encoder.out_features - 1)
    
    with torch.no_grad():
        # 设置周期项的权重为预定义频率
        for i in range(n_freq):
            encoder.model.w[0, i] = 2 * np.pi * frequencies[i]
        
        # 初始化偏置为0
        encoder.model.b.zero_()
    
    logger.info(f"Time2Vec已初始化预定义频率: {frequencies[:n_freq]}")


if __name__ == '__main__':
    """测试代码"""
    
    # 创建测试数据
    n_samples = 1000
    timestamps_norm = np.linspace(0, 1, n_samples)  # 归一化时间戳
    
    # 创建编码器（64维：1线性 + 63周期）
    encoder = Time2VecEncoder(out_features=64, use_cos=False)
    
    # 可选：使用预定义频率初始化
    initialize_time2vec_with_frequencies(encoder)
    
    # 编码
    time_vectors = encoder.encode(timestamps_norm)
    
    print(f"\n输入形状: {timestamps_norm.shape}")
    print(f"输出形状: {time_vectors.shape}")
    print(f"输出范围: [{time_vectors.min():.4f}, {time_vectors.max():.4f}]")
    print(f"\n维度分解:")
    print(f"  线性项（第0维）: 1维")
    print(f"  周期项（第1-63维）: 63维")
    print(f"  总计: 64维")
    
    # 可视化（可选）
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 8))
        
        # 绘制前4个维度
        for i in range(4):
            axes[i].plot(timestamps_norm, time_vectors[:, i])
            axes[i].set_ylabel(f'Dim {i}')
            axes[i].grid(True)
        
        axes[-1].set_xlabel('Normalized Time')
        plt.suptitle('Time2Vec Encoding')
        plt.tight_layout()
        plt.savefig('time2vec_visualization.png')
        print("\n可视化已保存到: time2vec_visualization.png")
    except:
        pass
