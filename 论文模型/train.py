# -*- coding: utf-8 -*-
"""
HTGATFraud模型训练主脚本
支持命令行参数配置和完整的训练流程
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
import logging
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 不显示图形界面
import matplotlib.pyplot as plt

from config import (
    MODEL_CONFIG, TRAIN_CONFIG, SPLIT_CONFIG, 
    CHECKPOINTS_DIR, LOGS_DIR,
    get_data_dir, get_data_config, get_model_config_for_dataset, 
    DEFAULT_DATASET, DATASET_EDGE_TYPES
)
from src.models.ht_gat_fraud import HTGATFraud
from src.utils.data_loader import load_fraud_detection_data
from src.utils.node_split import create_node_split


class Trainer:
    """HTGATFraud训练器"""
    
    def __init__(self, args):
        """
        初始化训练器
        
        Parameters
        ----------
        args : argparse.Namespace
            命令行参数
        """
        self.args = args
        # 获取数据集类型
        self.dataset = args.dataset if hasattr(args, 'dataset') else DEFAULT_DATASET
        
        # 设置设备
        if args.device == 'cuda' and torch.cuda.is_available():
            gpu_id = args.gpu if hasattr(args, 'gpu') else 0
            self.device = f'cuda:{gpu_id}'
        else:
            self.device = 'cpu'
        
        # 设置日志
        self.setup_logging()
        
        # 获取数据集对应的数据目录
        data_dir = get_data_dir(self.dataset)
        
        # 加载数据
        self.logger.info("="*80)
        self.logger.info(f"数据集: {self.dataset.upper()}")
        self.logger.info(f"数据目录: {data_dir}")
        self.logger.info("加载数据...")
        
        # 获取边类型信息
        edge_config = DATASET_EDGE_TYPES.get(self.dataset, {})
        edge_type_names = edge_config.get('edge_type_names', [])
        self.logger.info(f"边类型: {', '.join(edge_type_names)}")
        
        self.data, self.class_weights = load_fraud_detection_data(data_dir, dataset=self.dataset)
        self.data = self.data.to(self.device)
        self.class_weights = self.class_weights.to(self.device) if args.use_class_weights else None
        
        # 进行节点划分（Transductive 设定）
        self.logger.info("\n进行节点划分（Transductive 设定）...")
        split_method = args.split_method if hasattr(args, 'split_method') else SPLIT_CONFIG['split_method']
        train_ratio = args.train_ratio if hasattr(args, 'train_ratio') else SPLIT_CONFIG['train_ratio']
        val_ratio = args.val_ratio if hasattr(args, 'val_ratio') else SPLIT_CONFIG['val_ratio']
        test_ratio = args.test_ratio if hasattr(args, 'test_ratio') else SPLIT_CONFIG['test_ratio']
        split_seed = args.seed  # 使用训练的随机种子保证一致性
        
        splits = create_node_split(
            labels=self.data.y.cpu(),  # 在CPU上进行划分
            timestamps=self.data.timestamps.cpu() if hasattr(self.data, 'timestamps') else None,
            method=split_method,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed
        )
        
        # 将掩码添加到data对象并移到设备上
        self.data.train_mask = splits['train_mask'].to(self.device)
        self.data.val_mask = splits['val_mask'].to(self.device)
        self.data.test_mask = splits['test_mask'].to(self.device)
        
        # 打印划分统计
        self.logger.info(f"\n划分统计:")
        self.logger.info(f"  方法: {split_method}")
        self.logger.info(f"  训练集: {self.data.train_mask.sum().item():,} ({train_ratio:.0%})")
        if val_ratio > 0:
            self.logger.info(f"  验证集: {self.data.val_mask.sum().item():,} ({val_ratio:.0%})")
        else:
            self.logger.info(f"  验证集: 无（不使用验证集）")
        self.logger.info(f"  测试集: {self.data.test_mask.sum().item():,} ({test_ratio:.0%})")
        self.logger.info(f"  注意: Transductive设定 - 前向传播使用完整图，损失仅在掩码节点上计算")
        
        # 创建模型
        self.logger.info("\n创建模型...")
        self.model = HTGATFraud(self.get_model_config()).to(self.device)
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 创建优化器
        self.optimizer = self.create_optimizer()
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # 创建学习率调度器
        self.scheduler = self.create_scheduler()
        
        # 判断是否使用验证集
        self.use_validation = val_ratio > 0
        
        # 训练历史（记录所有指标）
        self.history = {
            'train_loss': [], 'train_auc': [], 'train_f1': [], 'train_acc': [], 
            'train_precision': [], 'train_recall': [], 'train_ap': [],
            'lr': []
        }
        
        # 如果有验证集，添加验证指标
        if self.use_validation:
            self.history.update({
                'val_loss': [], 'val_auc': [], 'val_f1': [], 'val_acc': [],
                'val_precision': [], 'val_recall': [], 'val_ap': [],
            })
        
        # 最佳模型追踪
        # 无论有无验证集都初始化，避免属性不存在的错误
        self.best_val_metric = 0.0 if args.monitor_metric != 'loss' else float('inf')
        self.best_epoch = 0
        
        # 早停相关变量（仅在有验证集时使用）
        if self.use_validation:
            self.patience_counter = 0
            self.epochs_no_improve = 0  # 无提升的epoch数
        
        # 混合精度训练
        self.use_amp = args.use_amp if hasattr(args, 'use_amp') else TRAIN_CONFIG.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp and self.device == 'cuda' else None
        
        if self.use_amp:
            self.logger.info("✓ 启用混合精度训练（AMP）- 显存降低约40%")
        
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 将数据集名称添加到日志文件名
        log_file = LOGS_DIR / f'train_{self.dataset}_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件: {log_file}")
        
    def get_model_config(self):
        """获取模型配置（合并默认配置和命令行参数）"""
        # 获取数据集特定的模型配置
        config = get_model_config_for_dataset(self.dataset)
        
        # 从命令行参数更新配置
        if self.args.num_snapshots is not None:
            config['num_snapshots'] = self.args.num_snapshots
        if self.args.hgt_hidden_dim is not None:
            config['hgt_hidden_dim'] = self.args.hgt_hidden_dim
        if self.args.hgt_num_layers is not None:
            config['hgt_num_layers'] = self.args.hgt_num_layers
        if self.args.temporal_num_heads is not None:
            config['temporal_num_heads'] = self.args.temporal_num_heads
            
        return config
    
    def create_optimizer(self):
        """创建优化器"""
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"未知的优化器: {self.args.optimizer}")
        
        return optimizer
    
    def create_scheduler(self):
        """创建学习率调度器"""
        if not self.args.use_scheduler:
            return None
            
        if self.args.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.args.scheduler_factor,
                patience=self.args.scheduler_patience,
                min_lr=self.args.scheduler_min_lr
            )
        elif self.args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.scheduler_min_lr
            )
        else:
            raise ValueError(f"未知的调度器: {self.args.scheduler}")
        
        return scheduler
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        self.optimizer.zero_grad()
        
        # 混合精度训练
        if self.use_amp and self.scaler is not None:
            # 使用自动混合精度
            with autocast():
                logits = self.model(self.data)
                train_mask = self.data.train_mask
                loss = self.criterion(logits[train_mask], self.data.y[train_mask])
            
            # 反向传播（AMP版本）
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.args.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准训练
            logits = self.model(self.data)
            train_mask = self.data.train_mask
            loss = self.criterion(logits[train_mask], self.data.y[train_mask])
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
        
        # 计算训练指标
        with torch.no_grad():
            pred = logits[train_mask].argmax(dim=1)
            prob_all = torch.softmax(logits[train_mask], dim=1)
            prob_fraud = prob_all[:, 0]  # 欺诈（类别0）概率
            prob_normal = prob_all[:, 1]  # 正常（类别1）概率
            
            y_true = self.data.y[train_mask].cpu().numpy()
            y_pred = pred.cpu().numpy()
            
            # 对于欺诈检测，我们关注欺诈类（0）的检测性能
            # 但AUC需要正类（1）的概率，所以反转标签或使用1-prob
            y_true_fraud = 1 - y_true  # 转换：0→1(欺诈), 1→0(正常)
            y_prob_fraud = prob_fraud.cpu().numpy()
            
            metrics = {
                'loss': loss.item(),
                'acc': accuracy_score(y_true, y_pred),
                'auc': roc_auc_score(y_true_fraud, y_prob_fraud),  # 欺诈检测AUC
                'ap': average_precision_score(y_true_fraud, y_prob_fraud),  # 平均精度
                'f1': f1_score(y_true, y_pred, pos_label=0),  # 欺诈为正类
                'precision': precision_score(y_true, y_pred, pos_label=0),
                'recall': recall_score(y_true, y_pred, pos_label=0),
            }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, split='val'):
        """评估模型"""
        self.model.eval()
        
        # 前向传播
        logits = self.model(self.data)
        
        # 选择评估集
        if split == 'val':
            mask = self.data.val_mask
        elif split == 'test':
            mask = self.data.test_mask
        else:
            raise ValueError(f"未知的split: {split}")
        
        # 计算损失
        loss = self.criterion(logits[mask], self.data.y[mask])
        
        # 计算指标
        pred = logits[mask].argmax(dim=1)
        prob_all = torch.softmax(logits[mask], dim=1)
        prob_fraud = prob_all[:, 0]  # 欺诈（类别0）概率
        prob_normal = prob_all[:, 1]  # 正常（类别1）概率
        
        y_true = self.data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        # 对于欺诈检测，我们关注欺诈类（0）的检测性能
        # 但AUC需要正类（1）的概率，所以反转标签或使用1-prob
        y_true_fraud = 1 - y_true  # 转换：0→1(欺诈), 1→0(正常)
        y_prob_fraud = prob_fraud.cpu().numpy()
        
        metrics = {
            'loss': loss.item(),
            'acc': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true_fraud, y_prob_fraud),  # 欺诈检测AUC
            'ap': average_precision_score(y_true_fraud, y_prob_fraud),  # 平均精度
            'f1': f1_score(y_true, y_pred, pos_label=0),
            'precision': precision_score(y_true, y_pred, pos_label=0),
            'recall': recall_score(y_true, y_pred, pos_label=0),
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'best_val_metric': self.best_val_metric,
            'model_config': self.get_model_config(),
            'args': vars(self.args)
        }
        
        if is_best:
            # 保存最佳模型
            filename = f'best_model_epoch{epoch}_auc{val_metrics["auc"]:.4f}.pth'
            filepath = CHECKPOINTS_DIR / filename
            
            # 删除之前的最佳模型
            for f in CHECKPOINTS_DIR.glob('best_model_*.pth'):
                f.unlink()
            
            torch.save(checkpoint, filepath)
            self.logger.info(f"  ✓ 保存最佳模型: {filename}")
        
        elif epoch % self.args.save_frequency == 0:
            # 定期保存
            filename = f'checkpoint_epoch{epoch}.pth'
            filepath = CHECKPOINTS_DIR / filename
            torch.save(checkpoint, filepath)
            self.logger.info(f"  保存检查点: {filename}")
    
    def train(self):
        """完整训练流程"""
        self.logger.info("\n" + "="*80)
        self.logger.info("开始训练")
        self.logger.info("="*80)
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"训练模式: Transductive（前向传播在完整图上，损失仅在train_mask上）")
        self.logger.info(f"训练轮数: {self.args.epochs}")
        self.logger.info(f"学习率: {self.args.lr}")
        
        if self.use_validation:
            if self.args.early_stopping:
                self.logger.info(f"早停机制: ✓ 启用")
                self.logger.info(f"  - 监控指标: {self.args.monitor_metric.upper()}")
                self.logger.info(f"  - 耐心值: {self.args.early_stopping_patience} epochs")
                if hasattr(self.args, 'min_delta'):
                    self.logger.info(f"  - 最小变化阈值: {self.args.min_delta}")
            else:
                self.logger.info(f"早停机制: ✗ 未启用")
        else:
            self.logger.info(f"验证集: ✗ 无验证集，固定训练{self.args.epochs}个epoch")
            self.logger.info(f"早停机制: ✗ 未启用（无验证集）")
        
        self.logger.info("="*80 + "\n")
        
        # 训练循环
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            train_metrics = self.train_epoch()
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            # 记录训练指标到历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['train_ap'].append(train_metrics['ap'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            
            if self.use_validation:
                # 有验证集：执行验证和早停逻辑
                val_metrics = self.evaluate('val')
                
                # 记录验证指标
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_auc'].append(val_metrics['auc'])
                self.history['val_ap'].append(val_metrics['ap'])
                self.history['val_f1'].append(val_metrics['f1'])
                self.history['val_acc'].append(val_metrics['acc'])
                self.history['val_precision'].append(val_metrics['precision'])
                self.history['val_recall'].append(val_metrics['recall'])
                
                # 打印进度
                if epoch % self.args.print_freq == 0 or epoch == 1:
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.args.epochs} | LR: {current_lr:.6f}"
                    )
                    self.logger.info(
                        f"  Train: Loss={train_metrics['loss']:.4f} | "
                        f"AUC={train_metrics['auc']:.4f} AP={train_metrics['ap']:.4f} "
                        f"F1={train_metrics['f1']:.4f} ACC={train_metrics['acc']:.4f} Recall={train_metrics['recall']:.4f}"
                    )
                    self.logger.info(
                        f"  Val:   Loss={val_metrics['loss']:.4f} | "
                        f"AUC={val_metrics['auc']:.4f} AP={val_metrics['ap']:.4f} "
                        f"F1={val_metrics['f1']:.4f} ACC={val_metrics['acc']:.4f} Recall={val_metrics['recall']:.4f}"
                    )
                
                # 更新学习率
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['auc'])
                    else:
                        self.scheduler.step()
                
                # 检查是否是最佳模型
                current_metric = val_metrics[self.args.monitor_metric]
                
                # 根据监控指标判断是否提升（loss越小越好，其他指标越大越好）
                if self.args.monitor_metric == 'loss':
                    is_best = current_metric < self.best_val_metric
                    improved = self.best_val_metric - current_metric
                else:
                    is_best = current_metric > self.best_val_metric
                    improved = current_metric - self.best_val_metric
                
                if is_best:
                    self.logger.info(f"  🎯 验证{self.args.monitor_metric.upper()}提升: "
                                   f"{self.best_val_metric:.4f} → {current_metric:.4f} "
                                   f"(+{abs(improved):.4f})")
                    self.best_val_metric = current_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.epochs_no_improve = 0
                    
                    # 保存最佳模型
                    if self.args.save_checkpoints:
                        self.save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1
                    self.epochs_no_improve += 1
                    
                    if self.args.early_stopping and epoch % 10 == 0:
                        self.logger.info(f"  ⏳ 已 {self.epochs_no_improve} 个epoch无提升 "
                                       f"(耐心值: {self.args.early_stopping_patience})")
                
                # 定期保存
                if self.args.save_checkpoints and epoch % self.args.save_frequency == 0:
                    self.save_checkpoint(epoch, val_metrics, is_best=False)
                
                # 早停检查
                if self.args.early_stopping and self.patience_counter >= self.args.early_stopping_patience:
                    self.logger.info(f"\n" + "="*80)
                    self.logger.info(f"⛔ 早停触发！")
                    self.logger.info(f"   - 监控指标: {self.args.monitor_metric.upper()}")
                    self.logger.info(f"   - 最佳值: {self.best_val_metric:.4f} (Epoch {self.best_epoch})")
                    self.logger.info(f"   - 当前值: {current_metric:.4f} (Epoch {epoch})")
                    self.logger.info(f"   - 已 {self.args.early_stopping_patience} 个epoch无提升")
                    self.logger.info("="*80)
                    break
            else:
                # 无验证集：简单训练模式
                # 打印进度
                if epoch % self.args.print_freq == 0 or epoch == 1:
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.args.epochs} | LR: {current_lr:.6f} | "
                        f"Loss={train_metrics['loss']:.4f} AUC={train_metrics['auc']:.4f} "
                        f"AP={train_metrics['ap']:.4f} F1={train_metrics['f1']:.4f} "
                        f"ACC={train_metrics['acc']:.4f} Recall={train_metrics['recall']:.4f}"
                    )
                
                # 更新学习率（cosine等不需要验证指标的调度器）
                if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
                
                # 定期保存（使用训练指标）
                if self.args.save_checkpoints and epoch % self.args.save_frequency == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_metrics': train_metrics,
                        'model_config': self.get_model_config(),
                        'args': vars(self.args)
                    }
                    filename = f'checkpoint_epoch{epoch}.pth'
                    filepath = CHECKPOINTS_DIR / filename
                    torch.save(checkpoint, filepath)
                
                # 保存最后一个epoch的模型作为最终模型
                if epoch == self.args.epochs:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_metrics': train_metrics,
                        'model_config': self.get_model_config(),
                        'args': vars(self.args)
                    }
                    filename = f'final_model_epoch{epoch}.pth'
                    filepath = CHECKPOINTS_DIR / filename
                    
                    # 删除之前的最终模型
                    for f in CHECKPOINTS_DIR.glob('final_model_*.pth'):
                        f.unlink()
                    
                    torch.save(checkpoint, filepath)
                    self.logger.info(f"  ✓ 保存最终模型: {filename}")
        
        # 训练完成
        self.logger.info("\n" + "="*80)
        self.logger.info(f"训练完成！")
        if self.use_validation:
            self.logger.info(f"最佳{self.args.monitor_metric.upper()}: {self.best_val_metric:.4f} (Epoch {self.best_epoch})")
        else:
            self.logger.info(f"完成 {self.args.epochs} 个epoch的训练（无验证集）")
        self.logger.info("="*80)
        
        # 保存训练历史
        self.save_history()
        
        # 绘制并保存训练曲线
        if self.args.save_plots:
            self.plot_training_curves()
        
        # 在测试集上评估
        if self.args.eval_test:
            self.logger.info("\n在测试集上评估最佳模型...")
            test_metrics = self.evaluate('test')
            
            self.logger.info("\n" + "="*80)
            self.logger.info("测试集最终评估结果")
            self.logger.info("="*80)
            self.logger.info(f"  AUC:       {test_metrics['auc']:.4f}  (ROC曲线下面积)")
            self.logger.info(f"  AP:        {test_metrics['ap']:.4f}  (平均精度)")
            self.logger.info(f"  F1-Score:  {test_metrics['f1']:.4f}  (精确率和召回率的调和平均)")
            self.logger.info(f"  Accuracy:  {test_metrics['acc']:.4f}  (准确率)")
            self.logger.info(f"  Precision: {test_metrics['precision']:.4f}  (精确率/查准率)")
            self.logger.info(f"  Recall:    {test_metrics['recall']:.4f}  (召回率/查全率)")
            self.logger.info("="*80)
            
            # 保存测试结果
            test_results = {
                'total_epochs': self.args.epochs,
                'best_epoch': self.best_epoch if self.use_validation else self.args.epochs,
                'best_val_metric': self.best_val_metric if self.use_validation else None,
                'use_validation': self.use_validation,
                'test_metrics': test_metrics,
                'args': vars(self.args)
            }
            
            results_file = LOGS_DIR / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"\n测试结果已保存: {results_file}")
    
    def save_history(self):
        """保存训练历史"""
        history_file = LOGS_DIR / f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # 转换为可序列化格式
        history_save = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_save, f, indent=2)
        
        self.logger.info(f"训练历史已保存: {history_file}")
    
    def plot_training_curves(self):
        """绘制并保存训练曲线"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 设置中文字体（如果可用）
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass
            
            # 创建大图：3x3布局
            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
            fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
            
            epochs = range(1, len(self.history['train_loss']) + 1)
            
            # 1. Loss曲线
            ax = axes[0, 0]
            ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if self.use_validation and 'val_loss' in self.history:
                ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. AUC曲线
            ax = axes[0, 1]
            ax.plot(epochs, self.history['train_auc'], 'b-', label='Train AUC', linewidth=2)
            if self.use_validation and 'val_auc' in self.history:
                ax.plot(epochs, self.history['val_auc'], 'r-', label='Val AUC', linewidth=2)
                # 标注最佳点
                best_idx = np.argmax(self.history['val_auc'])
                ax.plot(best_idx + 1, self.history['val_auc'][best_idx], 'r*', markersize=15, 
                       label=f'Best: {self.history["val_auc"][best_idx]:.4f}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('AUC')
            ax.set_title('AUC Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. AP曲线（新增）
            ax = axes[0, 2]
            ax.plot(epochs, self.history['train_ap'], 'b-', label='Train AP', linewidth=2)
            if self.use_validation and 'val_ap' in self.history:
                ax.plot(epochs, self.history['val_ap'], 'r-', label='Val AP', linewidth=2)
                # 标注最佳点
                best_idx = np.argmax(self.history['val_ap'])
                ax.plot(best_idx + 1, self.history['val_ap'][best_idx], 'r*', markersize=15, 
                       label=f'Best: {self.history["val_ap"][best_idx]:.4f}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Precision')
            ax.set_title('AP Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. F1-Score曲线
            ax = axes[1, 0]
            ax.plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
            if self.use_validation and 'val_f1' in self.history:
                ax.plot(epochs, self.history['val_f1'], 'r-', label='Val F1', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1-Score')
            ax.set_title('F1-Score Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 5. Accuracy曲线
            ax = axes[1, 1]
            ax.plot(epochs, self.history['train_acc'], 'b-', label='Train ACC', linewidth=2)
            if self.use_validation and 'val_acc' in self.history:
                ax.plot(epochs, self.history['val_acc'], 'r-', label='Val ACC', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 6. Recall曲线
            ax = axes[1, 2]
            ax.plot(epochs, self.history['train_recall'], 'b-', label='Train Recall', linewidth=2)
            if self.use_validation and 'val_recall' in self.history:
                ax.plot(epochs, self.history['val_recall'], 'r-', label='Val Recall', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Recall')
            ax.set_title('Recall Curve (Fraud Detection)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 7. Precision曲线
            ax = axes[2, 0]
            ax.plot(epochs, self.history['train_precision'], 'b-', label='Train Precision', linewidth=2)
            if self.use_validation and 'val_precision' in self.history:
                ax.plot(epochs, self.history['val_precision'], 'r-', label='Val Precision', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Precision')
            ax.set_title('Precision Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 8. Learning Rate曲线
            ax = axes[2, 1]
            ax.plot(epochs, self.history['lr'], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # 9. 隐藏最后一个子图（预留）
            axes[2, 2].axis('off')
            
            plt.tight_layout()
            
            # 保存图片
            plot_file = LOGS_DIR / f'training_curves_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"训练曲线已保存: {plot_file}")
            
            # 额外保存单独的Loss曲线（高分辨率）
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if self.use_validation and 'val_loss' in self.history:
                ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Loss Convergence Curve', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            loss_file = LOGS_DIR / f'loss_curve_{timestamp}.png'
            plt.savefig(loss_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Loss收敛曲线已保存: {loss_file}")
            
        except Exception as e:
            self.logger.warning(f"绘制训练曲线失败: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HTGATFraud模型训练')
    
    # 数据集选择
    parser.add_argument('--dataset', type=str, default='yelp',
                       choices=['yelp', 'amazon'],
                       help='选择数据集: yelp（Yelp欺诈检测）或 amazon（Amazon欺诈检测）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数（默认200，无验证集）')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'adamw', 'sgd'], help='优化器')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 模型参数（与 config.py 中 MODEL_CONFIG 的默认值保持一致，避免显存占用过大）
    parser.add_argument('--num_snapshots', type=int, default=8, help='时间片数量')
    parser.add_argument('--hgt_hidden_dim', type=int, default=96, help='HGT隐藏维度')
    parser.add_argument('--hgt_num_layers', type=int, default=2, help='HGT层数')
    parser.add_argument('--temporal_num_heads', type=int, default=None, help='时序注意力头数')
    
    # 学习率调度
    parser.add_argument('--use_scheduler', action='store_true', default=True, help='使用学习率调度')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['plateau', 'cosine'], help='调度器类型（无验证集推荐cosine）')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Plateau调度器耐心值')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='学习率衰减因子')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6, help='最小学习率')
    
    # 早停配置（仅在有验证集时生效）
    parser.add_argument('--early_stopping', action='store_true', default=False, 
                       help='是否启用早停机制（仅在有验证集时生效）')
    parser.add_argument('--no_early_stopping', dest='early_stopping', action='store_false',
                       help='禁用早停机制')
    parser.add_argument('--early_stopping_patience', type=int, default=50, 
                       help='早停耐心值：验证指标连续多少个epoch无提升后停止训练')
    parser.add_argument('--monitor_metric', type=str, default='auc', 
                       choices=['auc', 'ap', 'f1', 'loss', 'precision', 'recall'], 
                       help='早停监控的指标 (auc/ap/f1/loss/precision/recall)')
    parser.add_argument('--min_delta', type=float, default=0.0001,
                       help='判断指标提升的最小变化阈值')
    
    # 类别权重
    parser.add_argument('--use_class_weights', action='store_true', default=True, 
                       help='使用类别权重处理不平衡')
    
    # 保存和日志
    parser.add_argument('--save_checkpoints', action='store_true', default=True, help='保存检查点')
    parser.add_argument('--save_frequency', type=int, default=10, help='保存频率（epoch）')
    parser.add_argument('--print_freq', type=int, default=5, help='打印频率（epoch）')
    
    # 评估
    parser.add_argument('--eval_test', action='store_true', default=True, 
                       help='训练结束后在测试集上评估')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='训练设备')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID（当device=cuda时生效）')
    
    # 内存优化
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='使用自动混合精度训练（降低显存）')
    
    # 可视化
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='保存训练曲线图')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 节点划分参数（Transductive 设定）
    parser.add_argument('--split_method', type=str, default='stratified',
                       choices=['stratified', 'temporal'],
                       help='节点划分方法：stratified（随机分层）或 temporal（时序）')
    parser.add_argument('--train_ratio', type=float, default=0.4, help='训练集比例（默认40%）')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='验证集比例（默认0，不使用验证集）')
    parser.add_argument('--test_ratio', type=float, default=0.6, help='测试集比例（默认60%）')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建训练器并训练
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
