# -*- coding: utf-8 -*-
"""
节点划分工具（Transductive 设定）

职责：
- 在训练阶段进行 train/val/test 节点划分
- 支持随机分层划分（保持类别比例）
- 支持时序划分（按时间戳排序）

设计原则：
- 划分逻辑与建图阶段解耦
- 使用固定随机种子保证可复现
- 分层划分缓解类别不平衡
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def create_stratified_split(
    labels: torch.Tensor,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    随机分层划分（保持类别比例）
    
    Parameters
    ----------
    labels : torch.Tensor
        节点标签 [N]
    train_ratio : float
        训练集比例（默认0.6）
    val_ratio : float
        验证集比例（默认0.2）
    test_ratio : float
        测试集比例（默认0.2）
    seed : int
        随机种子（默认42）
        
    Returns
    -------
    train_mask : torch.Tensor
        训练集掩码 [N]
    val_mask : torch.Tensor
        验证集掩码 [N]
    test_mask : torch.Tensor
        测试集掩码 [N]
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为1，当前为 {train_ratio + val_ratio + test_ratio}"
    
    n = len(labels)
    indices = np.arange(n)
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # 生成掩码
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    if val_ratio > 0:
        # 有验证集：train vs (val+test)，然后 val vs test
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels_np,
            random_state=seed
        )
        
        # 第二次划分：val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            stratify=labels_np[temp_idx],
            random_state=seed
        )
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    else:
        # 无验证集：只划分 train vs test
        train_idx, test_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels_np,
            random_state=seed
        )
        
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        # val_mask保持全False
    
    # 打印统计信息
    _log_split_stats(labels, train_mask, val_mask, test_mask, "随机分层划分")
    
    return train_mask, val_mask, test_mask


def create_temporal_split(
    timestamps: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    时序划分（按时间戳排序）
    
    Parameters
    ----------
    timestamps : torch.Tensor
        节点时间戳 [N]
    labels : torch.Tensor
        节点标签 [N]
    train_ratio : float
        训练集比例
    val_ratio : float
        验证集比例
    test_ratio : float
        测试集比例
        
    Returns
    -------
    train_mask : torch.Tensor
        训练集掩码 [N]
    val_mask : torch.Tensor
        验证集掩码 [N]
    test_mask : torch.Tensor
        测试集掩码 [N]
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(timestamps)
    
    # 按时间戳排序
    sorted_indices = torch.argsort(timestamps)
    
    # 计算划分点
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # 生成掩码
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[sorted_indices[:train_end]] = True
    val_mask[sorted_indices[train_end:val_end]] = True
    test_mask[sorted_indices[val_end:]] = True
    
    # 打印统计信息
    _log_split_stats(labels, train_mask, val_mask, test_mask, "时序划分")
    
    return train_mask, val_mask, test_mask


def _log_split_stats(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    split_name: str
):
    """打印划分统计信息"""
    n = len(labels)
    
    logger.info(f"\n{split_name}统计:")
    logger.info(f"  总节点数: {n:,}")
    
    for name, mask in [("训练集", train_mask), ("验证集", val_mask), ("测试集", test_mask)]:
        count = mask.sum().item()
        fraud_count = (labels[mask] == 0).sum().item()
        normal_count = (labels[mask] == 1).sum().item()
        fraud_ratio = fraud_count / count * 100 if count > 0 else 0
        
        logger.info(f"  {name}: {count:,} ({count/n*100:.1f}%) | "
                   f"欺诈: {fraud_count:,} ({fraud_ratio:.2f}%) | "
                   f"正常: {normal_count:,}")


def create_node_split(
    labels: torch.Tensor,
    timestamps: Optional[torch.Tensor] = None,
    method: str = 'stratified',
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    便捷函数：创建节点划分
    
    Parameters
    ----------
    labels : torch.Tensor
        节点标签 [N]
    timestamps : torch.Tensor, optional
        节点时间戳 [N]（时序划分时需要）
    method : str
        划分方法：'stratified'（随机分层）或 'temporal'（时序）
    train_ratio : float
        训练集比例
    val_ratio : float
        验证集比例
    test_ratio : float
        测试集比例
    seed : int
        随机种子
        
    Returns
    -------
    splits : Dict[str, torch.Tensor]
        包含 train_mask, val_mask, test_mask 的字典
    """
    if method == 'stratified':
        train_mask, val_mask, test_mask = create_stratified_split(
            labels, train_ratio, val_ratio, test_ratio, seed
        )
    elif method == 'temporal':
        if timestamps is None:
            raise ValueError("时序划分需要提供 timestamps")
        train_mask, val_mask, test_mask = create_temporal_split(
            timestamps, labels, train_ratio, val_ratio, test_ratio
        )
    else:
        raise ValueError(f"未知的划分方法: {method}")
    
    return {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 模拟数据
    n = 1000
    labels = torch.zeros(n, dtype=torch.long)
    labels[:100] = 0  # 10% 欺诈
    labels[100:] = 1  # 90% 正常
    
    timestamps = torch.linspace(0, 100, n)
    
    # 测试随机分层划分
    print("\n" + "="*60)
    print("测试随机分层划分")
    print("="*60)
    splits = create_node_split(labels, method='stratified')
    
    # 测试时序划分
    print("\n" + "="*60)
    print("测试时序划分")
    print("="*60)
    splits = create_node_split(labels, timestamps, method='temporal')
    
    print("\n✓ 节点划分测试通过！")
