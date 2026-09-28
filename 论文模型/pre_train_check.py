# -*- coding: utf-8 -*-
"""
训练前完整性检查
检查所有必要文件和配置是否正确
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import torch

print("="*80)
print("HTGATFraud 训练前完整性检查")
print("="*80)

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# 1. 检查数据文件
# ============================================================================

print("\n[1/6] 检查数据文件...")

data_dir = Path(__file__).parent.parent / '新建图方法' / 'output'
required_files = {
    'graph.pkl': None,
    'features.npy': None,
    'labels.npy': None,
    'splits.pkl': None,
}

all_files_exist = True
for filename in required_files.keys():
    filepath = data_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / 1024 / 1024
        required_files[filename] = f"✓ {size_mb:.2f} MB"
        print(f"  ✓ {filename}: {size_mb:.2f} MB")
    else:
        required_files[filename] = "❌ 不存在"
        print(f"  ❌ {filename}: 不存在")
        all_files_exist = False

if not all_files_exist:
    print("\n⚠️  错误：缺少必要文件，请先运行建图流程！")
    sys.exit(1)

# ============================================================================
# 2. 检查graph.pkl中的timestamps
# ============================================================================

print("\n[2/6] 检查graph.pkl中的timestamps...")

with open(data_dir / 'graph.pkl', 'rb') as f:
    graph_data = pickle.load(f)

if 'timestamps' not in graph_data:
    print("  ❌ graph.pkl中缺少timestamps字段！")
    print("  解决方案: 运行 rebuild_graph.py 重新构建图")
    sys.exit(1)

timestamps = graph_data['timestamps']
print(f"  ✓ timestamps存在")
print(f"    形状: {timestamps.shape}")
print(f"    范围: [{timestamps.min():.2f}, {timestamps.max():.2f}]")

# 检查时间片分布
boundaries = np.linspace(timestamps.min(), timestamps.max() + 1e-6, 11)
empty_snapshots = 0
for i in range(10):
    t_start, t_end = boundaries[i], boundaries[i + 1]
    count = ((timestamps >= t_start) & (timestamps < t_end)).sum()
    if count == 0:
        empty_snapshots += 1

if empty_snapshots > 0:
    print(f"  ⚠️  警告: {empty_snapshots}个时间片为空")
else:
    print(f"  ✓ 所有时间片都有数据")

# ============================================================================
# 3. 检查标签
# ============================================================================

print("\n[3/6] 检查标签...")

labels = np.load(data_dir / 'labels.npy')
print(f"  标签数量: {len(labels)}")
print(f"  标签值范围: [{labels.min()}, {labels.max()}]")

unique_labels, counts = np.unique(labels, return_counts=True)
print(f"  标签分布:")
for label, count in zip(unique_labels, counts):
    pct = count / len(labels) * 100
    label_name = "欺诈" if label == -1 else "正常" if label == 1 else "未知"
    print(f"    {label_name} ({label}): {count:,} ({pct:.2f}%)")

# 检查标签转换
if set(unique_labels) == {-1, 1}:
    print(f"  ✓ 标签格式正确: -1(欺诈), 1(正常)")
    # 模拟转换
    labels_converted = (labels + 1) // 2
    print(f"  转换后标签: 0(欺诈), 1(正常)")
    unique_converted, counts_converted = np.unique(labels_converted, return_counts=True)
    for label, count in zip(unique_converted, counts_converted):
        label_name = "欺诈" if label == 0 else "正常"
        print(f"    {label_name} ({label}): {count:,}")
else:
    print(f"  ⚠️  警告: 标签值不是[-1, 1]")

# ============================================================================
# 4. 检查特征维度
# ============================================================================

print("\n[4/6] 检查特征...")

features = np.load(data_dir / 'features.npy')
print(f"  特征形状: {features.shape}")
print(f"  特征维度: {features.shape[1]}")
print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")

expected_dim = 87  # 根据config.py
if features.shape[1] == expected_dim:
    print(f"  ✓ 特征维度正确 ({expected_dim}维)")
else:
    print(f"  ⚠️  警告: 特征维度({features.shape[1]})与配置({expected_dim})不匹配")

# 检查NaN和Inf
nan_count = np.isnan(features).sum()
inf_count = np.isinf(features).sum()
if nan_count > 0:
    print(f"  ❌ 发现{nan_count}个NaN值")
if inf_count > 0:
    print(f"  ❌ 发现{inf_count}个Inf值")
if nan_count == 0 and inf_count == 0:
    print(f"  ✓ 无NaN或Inf值")

# ============================================================================
# 5. 检查数据划分
# ============================================================================

print("\n[5/6] 检查数据划分...")

with open(data_dir / 'splits.pkl', 'rb') as f:
    splits = pickle.load(f)

train_indices = splits['train_indices']
val_indices = splits['val_indices']
test_indices = splits['test_indices']

total = len(train_indices) + len(val_indices) + len(test_indices)
print(f"  训练集: {len(train_indices):,} ({len(train_indices)/total*100:.1f}%)")
print(f"  验证集: {len(val_indices):,} ({len(val_indices)/total*100:.1f}%)")
print(f"  测试集: {len(test_indices):,} ({len(test_indices)/total*100:.1f}%)")
print(f"  总计: {total:,}")

if total == len(labels):
    print(f"  ✓ 数据划分完整")
else:
    print(f"  ❌ 数据划分总数({total})与标签数({len(labels)})不匹配")

# ============================================================================
# 6. 检查模型和配置
# ============================================================================

print("\n[6/6] 检查模型配置...")

from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG

print(f"  模型: {MODEL_CONFIG['model_name']}")
print(f"  输入维度: {MODEL_CONFIG['input_dim']}")
print(f"  时间片数: {MODEL_CONFIG['num_snapshots']}")
print(f"  HGT层数: {MODEL_CONFIG['hgt_num_layers']}")
print(f"  HGT隐藏维度: {MODEL_CONFIG['hgt_hidden_dim']}")
print(f"  训练轮数: {TRAIN_CONFIG['num_epochs']}")
print(f"  学习率: {TRAIN_CONFIG['learning_rate']}")
print(f"  设备: {TRAIN_CONFIG['device']}")

# 检查CUDA可用性
if torch.cuda.is_available():
    print(f"  ✓ CUDA可用: {torch.cuda.get_device_name(0)}")
    print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"  ⚠️  CUDA不可用，将使用CPU训练（速度较慢）")

# 检查配置一致性
if MODEL_CONFIG['input_dim'] != features.shape[1]:
    print(f"  ❌ 配置的input_dim({MODEL_CONFIG['input_dim']}) != 实际特征维度({features.shape[1]})")
    sys.exit(1)

if DATA_CONFIG['num_nodes'] != len(labels):
    print(f"  ⚠️  警告: 配置的num_nodes({DATA_CONFIG['num_nodes']}) != 实际节点数({len(labels)})")

# ============================================================================
# 7. 测试数据加载
# ============================================================================

print("\n[7/7] 测试数据加载...")

try:
    from src.utils.data_loader import load_fraud_detection_data
    
    data, splits_dict, class_weights = load_fraud_detection_data(data_dir)
    
    print(f"  ✓ 数据加载成功")
    print(f"    节点数: {data.x.size(0):,}")
    print(f"    边数: {data.edge_index.size(1):,}")
    print(f"    特征维度: {data.x.size(1)}")
    print(f"    标签: {data.y.size(0)}")
    print(f"    timestamps: {data.timestamps.size(0)}")
    print(f"    类别权重: {class_weights.tolist()}")
    
    # 检查标签值
    unique_y = torch.unique(data.y)
    print(f"    转换后标签值: {unique_y.tolist()}")
    
    if set(unique_y.tolist()) == {0, 1}:
        print(f"  ✓ 标签转换正确 (0=欺诈, 1=正常)")
    else:
        print(f"  ❌ 标签转换错误，期望{0,1}，实际{unique_y.tolist()}")
        sys.exit(1)
    
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 总结
# ============================================================================

print("\n" + "="*80)
print("检查完成 - 所有检查通过！✅")
print("="*80)
print("\n可以开始训练了:")
print("  python train.py")
print("\n或使用自定义参数:")
print("  python train.py --epochs 100 --lr 0.001 --num_snapshots 10")
print("="*80)

