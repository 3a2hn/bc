# -*- coding: utf-8 -*-
"""
训练曲线可视化示例
展示如何使用保存的训练历史绘制自定义图表
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_history(history_file):
    """加载训练历史"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    return history

def plot_metrics_comparison(history, save_dir='./'):
    """绘制所有指标对比图"""
    save_dir = Path(save_dir)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    # 1. Loss vs Accuracy
    ax = axes[0, 0]
    ax2 = ax.twinx()
    line1 = ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    line2 = ax2.plot(epochs, history['val_acc'], 'b-', label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Accuracy', color='b')
    ax.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    ax.set_title('Loss vs Accuracy')
    ax.grid(True, alpha=0.3)
    
    # 2. AUC vs F1-Score
    ax = axes[0, 1]
    ax.plot(epochs, history['val_auc'], 'b-', label='AUC', linewidth=2)
    ax.plot(epochs, history['val_f1'], 'g-', label='F1-Score', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('AUC vs F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Precision vs Recall
    ax = axes[1, 0]
    ax.plot(epochs, history['val_precision'], 'purple', label='Precision', linewidth=2)
    ax.plot(epochs, history['val_recall'], 'orange', label='Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision vs Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 所有指标综合
    ax = axes[1, 1]
    ax.plot(epochs, history['val_auc'], label='AUC', linewidth=2)
    ax.plot(epochs, history['val_f1'], label='F1', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Accuracy', linewidth=2)
    ax.plot(epochs, history['val_precision'], label='Precision', linewidth=2)
    ax.plot(epochs, history['val_recall'], label='Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('All Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 指标对比图已保存: {save_dir / 'metrics_comparison.png'}")

def plot_smooth_curves(history, save_dir='./'):
    """绘制平滑后的曲线（移动平均）"""
    save_dir = Path(save_dir)
    
    def smooth(values, window=5):
        """移动平均平滑"""
        if len(values) < window:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        return smoothed
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 平滑Loss曲线
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', alpha=0.3, label='Train Loss (Raw)')
    ax.plot(epochs, smooth(history['train_loss']), 'b-', linewidth=2, label='Train Loss (Smooth)')
    ax.plot(epochs, history['val_loss'], 'r-', alpha=0.3, label='Val Loss (Raw)')
    ax.plot(epochs, smooth(history['val_loss']), 'r-', linewidth=2, label='Val Loss (Smooth)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Smoothed Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 平滑AUC曲线
    ax = axes[1]
    ax.plot(epochs, history['val_auc'], 'b-', alpha=0.3, label='AUC (Raw)')
    ax.plot(epochs, smooth(history['val_auc']), 'b-', linewidth=2, label='AUC (Smooth)')
    # 标注最高点
    best_idx = np.argmax(smooth(history['val_auc']))
    best_val = smooth(history['val_auc'])[best_idx]
    ax.plot(best_idx + 1, best_val, 'r*', markersize=15)
    ax.annotate(f'Best: {best_val:.4f}\nEpoch: {best_idx+1}',
               xy=(best_idx + 1, best_val),
               xytext=(10, -20),
               textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('Smoothed AUC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'smooth_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 平滑曲线已保存: {save_dir / 'smooth_curves.png'}")

def analyze_training(history):
    """分析训练过程"""
    print("\n" + "="*80)
    print("训练过程分析")
    print("="*80)
    
    # 找到最佳epoch
    best_auc_idx = np.argmax(history['val_auc'])
    best_f1_idx = np.argmax(history['val_f1'])
    
    print(f"\n最佳AUC:")
    print(f"  Epoch: {best_auc_idx + 1}")
    print(f"  AUC: {history['val_auc'][best_auc_idx]:.4f}")
    print(f"  F1: {history['val_f1'][best_auc_idx]:.4f}")
    print(f"  ACC: {history['val_acc'][best_auc_idx]:.4f}")
    print(f"  Recall: {history['val_recall'][best_auc_idx]:.4f}")
    
    print(f"\n最佳F1:")
    print(f"  Epoch: {best_f1_idx + 1}")
    print(f"  AUC: {history['val_auc'][best_f1_idx]:.4f}")
    print(f"  F1: {history['val_f1'][best_f1_idx]:.4f}")
    print(f"  ACC: {history['val_acc'][best_f1_idx]:.4f}")
    print(f"  Recall: {history['val_recall'][best_f1_idx]:.4f}")
    
    # 最终指标
    print(f"\n最终指标 (Epoch {len(history['val_auc'])}):")
    print(f"  AUC: {history['val_auc'][-1]:.4f}")
    print(f"  F1: {history['val_f1'][-1]:.4f}")
    print(f"  ACC: {history['val_acc'][-1]:.4f}")
    print(f"  Recall: {history['val_recall'][-1]:.4f}")
    
    # 过拟合检测
    train_val_gap_auc = history['train_auc'][-1] - history['val_auc'][-1]
    train_val_gap_f1 = history['train_f1'][-1] - history['val_f1'][-1]
    
    print(f"\n过拟合分析:")
    print(f"  训练-验证 AUC差距: {train_val_gap_auc:.4f}")
    print(f"  训练-验证 F1差距: {train_val_gap_f1:.4f}")
    
    if train_val_gap_auc > 0.1 or train_val_gap_f1 > 0.1:
        print(f"  ⚠️ 存在过拟合风险！建议增加正则化或使用早停")
    else:
        print(f"  ✓ 过拟合程度适中")
    
    print("="*80)


if __name__ == '__main__':
    # 使用示例
    print("训练曲线可视化示例")
    print("="*80)
    print("使用方法:")
    print("  1. 从 logs/ 目录找到最新的 history_*.json 文件")
    print("  2. 运行: python plot_training_example.py")
    print("  3. 查看生成的图表")
    print("\n示例代码:")
    print("""
    # 加载训练历史
    history = load_history('logs/history_20251209_220000.json')
    
    # 绘制指标对比图
    plot_metrics_comparison(history, save_dir='./results')
    
    # 绘制平滑曲线
    plot_smooth_curves(history, save_dir='./results')
    
    # 分析训练过程
    analyze_training(history)
    """)
    print("="*80)

