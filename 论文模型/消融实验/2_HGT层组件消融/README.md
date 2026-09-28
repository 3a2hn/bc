# 实验二：HGT层组件消融

## 实验目的

深入分析HGT层内部各组件的贡献，包括层数、注意力头数、关系特定参数等。

## 实验设置

| 变体编号 | 变体名称 | 修改内容 | 默认值 → 变体值 |
|----------|----------|----------|-----------------|
| Var-2.1 | HGT-1Layer | 减少HGT层数 | 2层 → 1层 |
| Var-2.2 | HGT-3Layer | 增加HGT层数 | 2层 → 3层 |
| Var-2.3 | w/o Relation-Specific | 移除关系特定参数 | 独立参数 → 共享参数 |
| Var-2.4 | HGT-2Heads | 减少注意力头数 | 4头 → 2头 |
| Var-2.5 | w/o RTE | 移除相对时间编码 | RTE启用 → 禁用 |

## 运行方式

```bash
# 运行所有HGT层消融实验
python run_ablation.py --all

# 运行单个变体
python run_ablation.py --variant hgt_1layer
python run_ablation.py --variant hgt_3layer
python run_ablation.py --variant wo_relation_specific
python run_ablation.py --variant hgt_2heads
python run_ablation.py --variant wo_rte
```

## 预期结果

| 变体 | 预期AUC变化 | 分析 |
|------|-------------|------|
| HGT-1Layer | ↓ 1-2% | 感受野减小，无法聚合远邻居信息 |
| HGT-3Layer | ↔ 或 ↓ 0-1% | 可能出现过平滑，边际收益递减 |
| w/o Relation-Specific | ↓ 2-4% | 异构性建模核心组件，影响显著 |
| HGT-2Heads | ↓ 0-1% | 头数减少可能略微降低性能 |
| w/o RTE | ↓ 0-2% | 边时间信息的补充作用 |

## 文件说明

- `run_ablation.py`: 消融实验运行脚本
- `results/`: 实验结果输出
