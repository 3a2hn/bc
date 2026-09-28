# 实验一：核心模块消融

## 实验目的

验证HTGATFraud模型中三大核心模块（HGT层、TGN记忆、时序注意力）各自的贡献，以及它们组合使用的必要性。

## 实验设置

| 变体编号 | 变体名称 | HGT | TGN记忆 | 时序注意力 | 说明 |
|----------|----------|-----|---------|------------|------|
| Baseline | **Full Model** | ✅ | ✅ | ✅ | 完整HTGATFraud模型 |
| Var-1.1 | w/o TGN Memory | ✅ | ❌ | ✅ | 移除TGN记忆模块 |
| Var-1.2 | w/o Temporal Attention | ✅ | ✅ | ❌ | 使用简单平均聚合替代时序注意力 |
| Var-1.3 | w/o HGT (Use GCN) | ❌ | ✅ | ✅ | 用标准GCN替代HGT层 |
| Var-1.4 | Static Baseline | ✅ | ❌ | ❌ | 纯静态模型（单时间片） |

## 运行方式

```bash
# 运行所有核心模块消融实验
python run_ablation.py --all

# 运行单个变体
python run_ablation.py --variant wo_tgn_memory
python run_ablation.py --variant wo_temporal_attention
python run_ablation.py --variant wo_hgt_use_gcn
python run_ablation.py --variant static_baseline

# 指定数据集
python run_ablation.py --variant wo_tgn_memory --dataset amazon
```

## 预期结果

| 变体 | 预期AUC变化 | 分析 |
|------|-------------|------|
| w/o TGN Memory | ↓ 2-5% | 跨时间片信息传递受损，尤其影响长期依赖建模 |
| w/o Temporal Attention | ↓ 3-6% | 时序依赖学习能力下降，简单聚合无法捕获复杂模式 |
| w/o HGT (Use GCN) | ↓ 1-3% | 异构性建模缺失，不同关系的区分能力降低 |
| Static Baseline | ↓ 5-10% | 完全忽略时序信息，性能显著下降 |

## 文件说明

- `run_ablation.py`: 消融实验运行脚本
- `models/`: 特定变体的模型实现
- `results/`: 实验结果输出
