# -*- coding: utf-8 -*-
"""
HTGATFraud模型训练脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import logging
import json
from datetime import datetime
import numpy as np

from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR
from src.models.ht_gat_fraud import HTGATFraud
from src.utils.data_loader import load_fraud_detection_data

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_epoch(model, data, optimizer, criterion, device):
    """
    训练一个epoch
    
    Parameters
    ----------
    model : HTGATFraud
        模型
    data : Data
        图数据
    optimizer : Optimizer
        优化器
    criterion : Loss
        损失函数
    device : str
        设备
        
    Returns
    -------
    loss : float
        平均损失
    metrics : dict
        训练指标
    """
    model.train()
    data = data.to(device)
    
    # 前向传播
    optimizer.zero_grad()
    logits = model(data)
    
    # 计算损失（只在训练集上）
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪
    if TRAIN_CONFIG.get('max_gradient_norm'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_gradient_norm'])
    
    optimizer.step()
    
    # 计算训练集指标
    with torch.no_grad():
        pred = logits[data.train_mask].argmax(dim=1)
        prob = torch.softmax(logits[data.train_mask], dim=1)[:, 1]
        y_true = data.y[data.train_mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = prob.cpu().numpy()
        
        metrics = {
            'loss': loss.item(),
            'acc': (y_pred == y_true).mean(),
            'auc': roc_auc_score(y_true, y_prob),
            'f1': f1_score(y_true, y_pred),
        }
    
    return loss.item(), metrics


@torch.no_grad()
def evaluate(model, data, criterion, device, mask_name='val_mask'):
    """
    评估模型
    
    Parameters
    ----------
    model : HTGATFraud
        模型
    data : Data
        图数据
    criterion : Loss
        损失函数
    device : str
        设备
    mask_name : str
        使用的mask名称 ('val_mask' 或 'test_mask')
        
    Returns
    -------
    metrics : dict
        评估指标
    """
    model.eval()
    data = data.to(device)
    
    # 前向传播
    logits = model(data)
    mask = getattr(data, mask_name)
    
    # 计算损失
    loss = criterion(logits[mask], data.y[mask])
    
    # 计算指标
    pred = logits[mask].argmax(dim=1)
    prob = torch.softmax(logits[mask], dim=1)[:, 1]
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = prob.cpu().numpy()
    
    metrics = {
        'loss': loss.item(),
        'acc': (y_pred == y_true).mean(),
        'auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    
    return metrics


def train_model():
    """
    主训练函数
    """
    logger.info("="*80)
    logger.info("开始训练HTGATFraud模型")
    logger.info("="*80)
    
    # 设置设备
    device = TRAIN_CONFIG['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，使用CPU")
        device = 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("\n加载数据...")
    data, splits, class_weights = load_fraud_detection_data(DATA_DIR)
    
    # 创建模型
    logger.info("\n创建模型...")
    model = HTGATFraud(MODEL_CONFIG)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数量: {num_params:,}")
    
    # 创建优化器
    if TRAIN_CONFIG['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
    elif TRAIN_CONFIG['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
    
    # 创建损失函数
    if TRAIN_CONFIG['use_class_weights']:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"使用类别权重: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 创建学习率调度器
    scheduler = None
    if TRAIN_CONFIG['use_scheduler']:
        if TRAIN_CONFIG['scheduler_type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  # 监控AUC最大化
                factor=TRAIN_CONFIG['scheduler_factor'],
                patience=TRAIN_CONFIG['scheduler_patience'],
                min_lr=TRAIN_CONFIG['scheduler_min_lr']
            )
    
    # 训练循环
    logger.info("\n开始训练...")
    logger.info("="*80)
    
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss': [],
        'val_auc': [],
        'val_f1': []
    }
    
    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        # 训练
        train_loss, train_metrics = train_epoch(model, data, optimizer, criterion, device)
        
        # 验证
        val_metrics = evaluate(model, data, criterion, device, 'val_mask')
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        
        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{TRAIN_CONFIG['num_epochs']} | "
                f"Train Loss: {train_loss:.4f} AUC: {train_metrics['auc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc']:.4f} "
                f"F1: {val_metrics['f1']:.4f}"
            )
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(val_metrics['auc'])
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0
            
            if TRAIN_CONFIG['save_checkpoints']:
                checkpoint_path = CHECKPOINTS_DIR / f'best_model_auc_{best_val_auc:.4f}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'config': MODEL_CONFIG
                }, checkpoint_path)
                logger.info(f"  ✓ 保存最佳模型: {checkpoint_path.name}")
        else:
            patience_counter += 1
        
        # 早停
        if TRAIN_CONFIG['use_early_stopping'] and patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            logger.info(f"\n早停触发！{TRAIN_CONFIG['early_stopping_patience']} 个epoch无提升")
            break
    
    logger.info("="*80)
    logger.info(f"训练完成！最佳验证AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    
    # 保存训练历史
    history_path = LOGS_DIR / f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(history_path, 'w') as f:
        # 转换numpy类型为Python类型
        history_save = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_save, f, indent=2)
    logger.info(f"训练历史已保存: {history_path}")
    
    # 在测试集上评估
    logger.info("\n在测试集上评估...")
    test_metrics = evaluate(model, data, criterion, device, 'test_mask')
    logger.info(f"测试集结果:")
    logger.info(f"  AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  F1: {test_metrics['f1']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['acc']:.4f}")
    
    return model, history, test_metrics


if __name__ == '__main__':
    model, history, test_metrics = train_model()
