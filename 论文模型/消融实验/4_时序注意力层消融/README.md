# 实验四：时序注意力层消融

## 实验目的

分析时序注意力层内部各组件的贡献，包括位置编码、因果mask、前馈网络等。

## 实验设置

| 变体编号 | 变体名称 | 修改内容 | 默认值 → 变体值 |
|----------|----------|----------|-----------------|
| Var-4.1 | w/o Position Embedding | 移除位置编码 | 启用 → 禁用 |
| Var-4.2 | w/ Causal Mask | 启用因果mask | 禁用 → 启用 |
| Var-4.3 | Temporal-2Heads | 减少注意力头数 | 4头 → 2头 |
| Var-4.4 | w/o Position FFN | 移除位置前馈网络 | 启用 → 禁用 |

## 运行方式

```bash
# 运行所有时序注意力层消融实验
python run_ablation.py --all

# 运行单个变体
python run_ablation.py --variant wo_position_embedding
python run_ablation.py --variant w_causal_mask
python run_ablation.py --variant temporal_2heads
python run_ablation.py --variant wo_position_ffn
```

## 预期结果

| 变体 | 预期AUC变化 | 分析 |
|------|-------------|------|
| w/o Position Embedding | ↓ 1-3% | 时间顺序信息丢失，注意力可能退化为对称形式 |
| w/ Causal Mask | ↓ 0-2% | 限制信息流动，但更符合实际应用场景 |
| Temporal-2Heads | ↓ 0-1% | 头数减少影响较小 |
| w/o Position FFN | ↓ 0-1% | 表达能力略有下降 |

## 文件说明

- `run_ablation.py`: 消融实验运行脚本
- `results/`: 实验结果输出
