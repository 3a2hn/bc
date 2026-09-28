# 实验五：时间片划分策略消融

## 实验目的

分析时间片数量对模型性能的影响，寻找最优的时间片划分粒度。

## 实验设置

| 变体编号 | 变体名称 | 时间片数量 | 说明 |
|----------|----------|------------|------|
| Var-5.1 | Snapshot-4 | 4 | 粗粒度划分 |
| Var-5.2 | Snapshot-8 (Default) | 8 | 默认设置 |
| Var-5.3 | Snapshot-12 | 12 | 中粒度划分 |
| Var-5.4 | Snapshot-16 | 16 | 细粒度划分 |

## 时间片数量的权衡

### 时间片数量增加的优势
- 时间分辨率更高，捕获更细粒度的时序演化
- 每个时间片内节点更少，计算更快

### 时间片数量增加的劣势
- 时序注意力计算复杂度增加（O(T²)）
- 每个时间片内边更稀疏，空间聚合信息量减少
- 被切除的跨时间片边比例增加

## 运行方式

```bash
# 运行所有时间片划分策略消融实验
python run_ablation.py --all

# 运行单个变体
python run_ablation.py --variant snapshot_4
python run_ablation.py --variant snapshot_8
python run_ablation.py --variant snapshot_12
python run_ablation.py --variant snapshot_16
```

## 预期结果

| 变体 | 预期AUC变化 | 分析 |
|------|-------------|------|
| Snapshot-4 | ↓ 1-2% | 时间分辨率过低，演化信息损失 |
| Snapshot-8 | Baseline | 默认设置，平衡点 |
| Snapshot-12 | ↔ 或 ↑ 0-1% | 可能略有提升，但边际收益递减 |
| Snapshot-16 | ↔ 或 ↓ 0-1% | 过于细粒度，稀疏问题突出 |

## 关键观察指标

| 指标 | 说明 |
|------|------|
| AUC | 模型判别能力 |
| 每时间片平均节点数 | 数据稀疏程度 |
| 每时间片平均边数 | 图结构完整性 |
| 训练时间 | 计算效率 |
| 显存占用 | 资源消耗 |

## 文件说明

- `run_ablation.py`: 消融实验运行脚本
- `results/`: 实验结果输出
