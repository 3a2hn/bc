# 实验三：TGN记忆模块消融

## 实验目的

分析TGN记忆模块内部各子组件的贡献，包括记忆更新器类型、消息函数类型、时间编码等。

## 实验设置

| 变体编号 | 变体名称 | 修改内容 | 默认值 → 变体值 |
|----------|----------|----------|-----------------|
| Var-3.1 | RNN Updater | 记忆更新器类型 | GRU → RNN |
| Var-3.2 | MLP Updater | 记忆更新器类型 | GRU → MLP |
| Var-3.3 | Identity Message | 消息函数类型 | MLP → Identity |
| Var-3.4 | w/o Time Encoding | 移除时间编码 | 启用 → 禁用 |

## 运行方式

```bash
# 运行所有TGN记忆模块消融实验
python run_ablation.py --all

# 运行单个变体
python run_ablation.py --variant rnn_updater
python run_ablation.py --variant mlp_updater
python run_ablation.py --variant identity_message
python run_ablation.py --variant wo_time_encoding
```

## 预期结果

| 变体 | 预期AUC变化 | 分析 |
|------|-------------|------|
| RNN Updater | ↓ 0-1% | GRU门控机制略优，但差异可能不大 |
| MLP Updater | ↓ 1-2% | 缺乏序列建模能力，长期依赖建模受损 |
| Identity Message | ↓ 1-3% | 消息变换的非线性映射有助于信息提取 |
| w/o Time Encoding | ↓ 1-2% | 时间敏感性降低，忽略了时间间隔信息 |

## 文件说明

- `run_ablation.py`: 消融实验运行脚本
- `results/`: 实验结果输出
