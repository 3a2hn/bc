# 实验六：分类器结构消融

## 实验目的

分析分类器结构对最终性能的影响，包括隐藏层维度、BatchNorm等组件。

## 实验设置

| 变体编号 | 变体名称 | 修改内容 | 默认值 → 变体值 |
|----------|----------|----------|-----------------|
| Var-6.1 | Classifier-32 | 分类器隐藏维度 | 48 → 32 |
| Var-6.2 | Classifier-96 | 分类器隐藏维度 | 48 → 96 |
| Var-6.3 | w/o BatchNorm | 移除BatchNorm | 启用 → 禁用 |

## 运行方式

```bash
# 运行所有分类器结构消融实验
python run_ablation.py --all

# 运行单个变体
python run_ablation.py --variant classifier_32
python run_ablation.py --variant classifier_96
python run_ablation.py --variant wo_batchnorm
```

## 预期结果

| 变体 | 预期AUC变化 | 分析 |
|------|-------------|------|
| Classifier-32 | ↓ 0-1% | 容量减小可能略微影响性能 |
| Classifier-96 | ↔ 或 ↓ 0-1% | 更大容量可能导致轻微过拟合 |
| w/o BatchNorm | ↓ 0-2% | 训练稳定性降低，可能影响最终性能 |

## 文件说明

- `run_ablation.py`: 消融实验运行脚本
- `results/`: 实验结果输出
