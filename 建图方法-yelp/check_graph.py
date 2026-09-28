# -*- coding: utf-8 -*-
"""
检查重建后的graph.pkl是否包含timestamps
"""

import pickle
import numpy as np
from pathlib import Path

graph_file = Path(__file__).parent / 'output' / 'graph.pkl'

print("="*80)
print("检查 graph.pkl 中的 timestamps")
print("="*80)

if not graph_file.exists():
    print(f"\n❌ 文件不存在: {graph_file}")
    exit(1)

print(f"\n文件路径: {graph_file}")
print(f"文件大小: {graph_file.stat().st_size / 1024 / 1024:.2f} MB")

print("\n正在加载...")
with open(graph_file, 'rb') as f:
    graph_data = pickle.load(f)

print("\n包含的键:")
for key in graph_data.keys():
    print(f"  ✓ {key}")

print("\n" + "="*80)
print("时间戳检查结果")
print("="*80)

if 'timestamps' in graph_data:
    timestamps = graph_data['timestamps']
    print(f"\n✅ timestamps 字段存在！")
    print(f"\n基本信息:")
    print(f"  类型: {type(timestamps)}")
    
    if isinstance(timestamps, np.ndarray):
        print(f"  形状: {timestamps.shape}")
        print(f"  数据类型: {timestamps.dtype}")
        print(f"\n统计信息:")
        print(f"  最小值: {timestamps.min():.4f}")
        print(f"  最大值: {timestamps.max():.4f}")
        print(f"  平均值: {timestamps.mean():.4f}")
        print(f"  标准差: {timestamps.std():.4f}")
        print(f"  中位数: {np.median(timestamps):.4f}")
        
        # 检查时间片分布
        print(f"\n时间片分布（10个时间片）:")
        boundaries = np.linspace(timestamps.min(), timestamps.max() + 1e-6, 11)
        
        total_nodes = len(timestamps)
        empty_snapshots = 0
        
        for i in range(10):
            t_start, t_end = boundaries[i], boundaries[i + 1]
            count = ((timestamps >= t_start) & (timestamps < t_end)).sum()
            pct = count / total_nodes * 100
            
            status = "✓" if count > 0 else "⚠️ 空"
            print(f"  {status} 时间片 {i}: [{t_start:6.2f}, {t_end:6.2f}) - {count:6,} 节点 ({pct:5.2f}%)")
            
            if count == 0:
                empty_snapshots += 1
        
        if empty_snapshots > 0:
            print(f"\n⚠️  警告: {empty_snapshots} 个时间片为空！")
        else:
            print(f"\n✅ 所有时间片都包含节点")
        
        print(f"\n节点数匹配检查:")
        print(f"  graph_data['num_nodes']: {graph_data['num_nodes']:,}")
        print(f"  len(timestamps): {len(timestamps):,}")
        
        if len(timestamps) == graph_data['num_nodes']:
            print(f"  ✅ 数量匹配")
        else:
            print(f"  ❌ 数量不匹配！")
else:
    print(f"\n❌ timestamps 字段不存在！")
    print(f"\n可能的原因:")
    print(f"  1. 使用的是旧版本的graph_builder.py")
    print(f"  2. 未重新运行建图流程")
    print(f"\n解决方案:")
    print(f"  运行: python rebuild_graph.py")

print("\n" + "="*80)
print("检查完成")
print("="*80)

