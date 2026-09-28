# -*- coding: utf-8 -*-
"""
HTGATFraud模型评估脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report, roc_curve
)
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import MODEL_CONFIG, EVAL_CONFIG, DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR
from src.models.ht_gat_fraud import HTGATFraud
from src.utils.data_loader import load_fraud_detection_data

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_recall_at_k(y_true, y_prob, k_values=[50, 100, 200, 500]):
    """
    计算Recall@K
    
    Parameters
    ----------
    y_true : np.ndarray
        真实标签
    y_prob : np.ndarray
        预测概率（欺诈概率）
    k_values : list
        K值列表
        
    Returns
    -------
    recall_at_k : dict
        不同K值的Recall
    """
    # 按概率降序排序
    sorted_indices = np.argsort(y_prob)[::-1]
    
    recall_at_k = {}
    num_fraud = (y_true == 0).sum()
    
    for k in k_values:
        if k > len(y_true):
            k = len(y_true)
        
        # 前K个样本
        top_k_indices = sorted_indices[:k]
        top_k_labels = y_true[top_k_indices]
        
        # 计算召回率
        num_fraud_in_top_k = (top_k_labels == 0).sum()
        recall = num_fraud_in_top_k / num_fraud if num_fraud > 0 else 0
        
        recall_at_k[f'Recall@{k}'] = recall
    
    return recall_at_k


def plot_roc_curve(y_true, y_prob, save_path):
    """
    绘制ROC曲线
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=0)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测', linewidth=1)
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC曲线已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['欺诈', '正常'],
        yticklabels=['欺诈', '正常']
    )
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"混淆矩阵已保存: {save_path}")


def plot_attention_heatmap(attention_weights, save_path, num_samples=100):
    """
    绘制时序注意力热图
    """
    # 随机选择一些样本
    if attention_weights.shape[0] > num_samples:
        indices = np.random.choice(attention_weights.shape[0], num_samples, replace=False)
        attention_weights = attention_weights[indices]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_weights, cmap='YlOrRd', cbar_kws={'label': '注意力权重'})
    plt.xlabel('时间片')
    plt.ylabel('节点样本')
    plt.title('时序注意力热图')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"注意力热图已保存: {save_path}")


@torch.no_grad()
def evaluate_model(checkpoint_path=None):
    """
    评估模型
    
    Parameters
    ----------
    checkpoint_path : str, optional
        检查点路径，如果为None则使用最新的
    """
    logger.info("="*80)
    logger.info("开始评估HTGATFraud模型")
    logger.info("="*80)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("\n加载数据...")
    data, splits, class_weights = load_fraud_detection_data(DATA_DIR)
    data = data.to(device)
    
    # 加载模型
    logger.info("\n加载模型...")
    model = HTGATFraud(MODEL_CONFIG)
    
    # 加载检查点
    if checkpoint_path is None:
        # 查找最新的检查点
        checkpoints = list(CHECKPOINTS_DIR.glob('best_model_*.pth'))
        if not checkpoints:
            raise FileNotFoundError("未找到任何检查点文件")
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 前向传播
    logger.info("\n进行预测...")
    logits, attention_dict = model(data, return_attention=True)
    
    # 获取测试集预测
    test_mask = data.test_mask
    test_logits = logits[test_mask]
    test_pred = test_logits.argmax(dim=1).cpu().numpy()
    test_prob = torch.softmax(test_logits, dim=1)[:, 0].cpu().numpy()  # 欺诈概率
    test_true = data.y[test_mask].cpu().numpy()
    
    # 计算基础指标
    logger.info("\n计算评估指标...")
    metrics = {
        'accuracy': accuracy_score(test_true, test_pred),
        'precision': precision_score(test_true, test_pred, pos_label=0),
        'recall': recall_score(test_true, test_pred, pos_label=0),
        'f1': f1_score(test_true, test_pred, pos_label=0),
        'auc': roc_auc_score(test_true, test_prob)
    }
    
    # 计算Recall@K
    recall_at_k = calculate_recall_at_k(
        test_true, test_prob,
        k_values=EVAL_CONFIG['recall_at_k']
    )
    metrics.update(recall_at_k)
    
    # 打印结果
    logger.info("\n" + "="*80)
    logger.info("评估结果")
    logger.info("="*80)
    logger.info(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    logger.info(f"精确率 (Precision): {metrics['precision']:.4f}")
    logger.info(f"召回率 (Recall):    {metrics['recall']:.4f}")
    logger.info(f"F1分数 (F1-Score):  {metrics['f1']:.4f}")
    logger.info(f"AUC:                {metrics['auc']:.4f}")
    logger.info("\nRecall@K:")
    for k in EVAL_CONFIG['recall_at_k']:
        logger.info(f"  Recall@{k:3d}: {metrics[f'Recall@{k}']:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(test_true, test_pred)
    logger.info("\n混淆矩阵:")
    logger.info(f"              预测欺诈  预测正常")
    logger.info(f"真实欺诈:    {cm[0,0]:6d}    {cm[0,1]:6d}")
    logger.info(f"真实正常:    {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f'eval_{timestamp}'
    result_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存指标
    metrics_path = result_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"\n评估指标已保存: {metrics_path}")
    
    # 绘制ROC曲线
    if EVAL_CONFIG.get('visualize_attention', True):
        logger.info("\n生成可视化...")
        plot_roc_curve(test_true, test_prob, result_dir / 'roc_curve.png')
        plot_confusion_matrix(test_true, test_pred, result_dir / 'confusion_matrix.png')
        
        # 绘制注意力热图
        if 'temporal_attention' in attention_dict:
            attn_weights = attention_dict['temporal_attention'][test_mask].cpu().numpy()
            # 只显示第一维的注意力（时间片权重）
            if len(attn_weights.shape) == 3:
                attn_weights = attn_weights.mean(axis=2)  # 平均跨target时间步
            plot_attention_heatmap(attn_weights, result_dir / 'attention_heatmap.png')
    
    # 保存预测结果
    if EVAL_CONFIG.get('save_predictions', True):
        predictions = {
            'y_true': test_true.tolist(),
            'y_pred': test_pred.tolist(),
            'y_prob': test_prob.tolist()
        }
        pred_path = result_dir / 'predictions.json'
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"预测结果已保存: {pred_path}")
    
    # 生成分类报告
    report = classification_report(
        test_true, test_pred,
        target_names=['欺诈', '正常'],
        digits=4
    )
    logger.info("\n分类报告:")
    logger.info("\n" + report)
    
    # 保存报告
    report_path = result_dir / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("HTGATFraud模型评估报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"检查点: {checkpoint_path}\n")
        f.write(f"评估时间: {timestamp}\n\n")
        f.write("评估指标:\n")
        f.write("-"*80 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:20s}: {v:.4f}\n")
        f.write("\n混淆矩阵:\n")
        f.write("-"*80 + "\n")
        f.write(str(cm) + "\n\n")
        f.write("分类报告:\n")
        f.write("-"*80 + "\n")
        f.write(report)
    
    logger.info(f"\n评估报告已保存: {report_path}")
    logger.info(f"所有结果保存在: {result_dir}")
    logger.info("="*80)
    
    return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估HTGATFraud模型')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    args = parser.parse_args()
    
    metrics = evaluate_model(args.checkpoint)
