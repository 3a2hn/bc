# HTGATFraud 消融实验

## 概述

本目录包含HTGATFraud模型的完整消融实验框架，用于验证模型各组件的有效性和贡献度。

## 目录结构

```
消融实验/
├── README.md                      # 本文件
├── ablation_config.py             # 消融实验配置管理
├── ablation_base.py               # 消融实验基类
├── run_all_ablations.py           # 统一运行器
├── aggregate_results.py           # 结果汇总与可视化
├── results/                       # 总结果目录
│
├── 1_核心模块消融/                # 实验一
│   ├── README.md
│   ├── run_ablation.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── ablation_models.py     # 变体模型实现
│   └── results/
│
├── 2_HGT层组件消融/               # 实验二
│   ├── README.md
│   ├── run_ablation.py
│   └── results/
│
├── 3_TGN记忆模块消融/             # 实验三
│   ├── README.md
│   ├── run_ablation.py
│   └── results/
│
├── 4_时序注意力层消融/            # 实验四
│   ├── README.md
│   ├── run_ablation.py
│   └── results/
│
├── 5_时间片划分策略消融/          # 实验五
│   ├── README.md
│   ├── run_ablation.py
│   └── results/
│
└── 6_分类器结构消融/              # 实验六
    ├── README.md
    ├── run_ablation.py
    └── results/
```

## 实验列表

| 实验编号 | 实验主题 | 消融维度 | 实验数量 | 优先级 |
|----------|----------|----------|----------|--------|
| Exp-1 | 核心模块消融 | 模块级别 | 5 | ⭐⭐⭐ |
| Exp-2 | HGT层组件消融 | 子模块级别 | 5 | ⭐⭐ |
| Exp-3 | TGN记忆模块消融 | 子模块级别 | 4 | ⭐⭐ |
| Exp-4 | 时序注意力层消融 | 组件级别 | 4 | ⭐⭐ |
| Exp-5 | 时间片划分策略消融 | 超参数级别 | 4 | ⭐⭐ |
| Exp-6 | 分类器结构消融 | 结构级别 | 3 | ⭐ |
| **总计** | | | **25** | |

## 快速开始

### 1. 运行所有消融实验

```bash
cd 消融实验
python run_all_ablations.py --all --dataset yelp --epochs 200
```

### 2. 按优先级运行实验

```bash
# 运行高优先级实验（核心验证）
python run_all_ablations.py --priority high --dataset yelp

# 运行中等优先级实验
python run_all_ablations.py --priority medium --dataset yelp
```

### 3. 运行指定类别的实验

```bash
# 只运行核心模块消融
python run_all_ablations.py --category 1_核心模块消融 --dataset yelp

# 运行多个类别
python run_all_ablations.py --category 1_核心模块消融 2_HGT层组件消融 --dataset yelp
```

### 4. 运行单个实验变体

```bash
# 进入对应实验目录
cd 1_核心模块消融

# 运行指定变体
python run_ablation.py --variant wo_tgn_memory --dataset yelp

# 运行该类别所有变体
python run_ablation.py --all --dataset yelp
```

### 5. 汇总实验结果

```bash
cd 消融实验
python aggregate_results.py
```

## 评估指标

| 指标 | 说明 | 重要性 |
|------|------|--------|
| **AUC-ROC** | 主要指标，衡量模型整体判别能力 | ⭐⭐⭐ |
| **F1-Score** | 平衡精确率和召回率 | ⭐⭐⭐ |
| **Recall** | 欺诈检测场景的核心指标 | ⭐⭐ |
| **Precision** | 精确率 | ⭐⭐ |
| **AP** | 精确率-召回率曲线下面积 | ⭐⭐ |

## 实验详情

### 实验一：核心模块消融 (⭐⭐⭐ 高优先级)

验证三大核心模块的贡献：
- **w/o TGN Memory**: 移除TGN记忆模块
- **w/o Temporal Attention**: 用平均聚合替代时序注意力
- **w/o HGT (Use GCN)**: 用标准GCN替代HGT
- **Static Baseline**: 静态单时间片模型

### 实验二：HGT层组件消融

深入分析HGT层：
- **HGT-1Layer / HGT-3Layer**: 层数影响
- **w/o Relation-Specific**: 关系特定参数的重要性
- **HGT-2Heads**: 注意力头数影响
- **w/o RTE**: 相对时间编码的作用

### 实验三：TGN记忆模块消融

分析记忆模块组件：
- **RNN/MLP Updater**: 记忆更新器类型对比
- **Identity Message**: 消息函数的重要性
- **w/o Time Encoding**: 时间编码的作用

### 实验四：时序注意力层消融

分析时序注意力组件：
- **w/o Position Embedding**: 位置编码的重要性
- **w/ Causal Mask**: 因果约束的影响
- **Temporal-2Heads**: 注意力头数
- **w/o Position FFN**: 前馈网络的作用

### 实验五：时间片划分策略消融

寻找最优时间粒度：
- **Snapshot-4/8/12/16**: 不同时间片数量的影响

### 实验六：分类器结构消融

分析分类器设计：
- **Classifier-32/96**: 隐藏维度影响
- **w/o BatchNorm**: 批归一化的作用

## 结果输出

每个实验会生成以下输出：
- `*_results_*.json`: 详细实验结果（含训练历史）
- `*.log`: 训练日志

汇总脚本会生成：
- `ablation_summary.csv`: 结果汇总表
- `ablation_*_auc.png`: AUC对比图
- `component_contribution.png`: 组件贡献度图
- `ablation_results.tex`: LaTeX格式表格

## 注意事项

1. **显存需求**: 确保GPU显存足够（建议8GB+）
2. **运行时间**: 完整实验需要较长时间，建议按优先级分批运行
3. **随机种子**: 默认使用seed=42，保证可复现性
4. **数据集**: 支持yelp和amazon两个数据集
