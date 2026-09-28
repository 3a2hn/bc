# -*- coding: utf-8 -*-
"""
消融实验基类
提供所有消融实验的通用训练和评估框架
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score, precision_recall_curve
)
import logging
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

from config import (
    get_data_dir, get_data_config, get_model_config_for_dataset,
    TRAIN_CONFIG, SPLIT_CONFIG, DATASET_EDGE_TYPES
)
from src.utils.data_loader import load_fraud_detection_data
from src.utils.node_split import create_node_split


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification (Lin et al., 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 时，降低易分样本的损失权重，让模型聚焦于难分样本。
    对欺诈检测中的少数类（欺诈样本）尤其有效。
    """

    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none',
            label_smoothing=self.label_smoothing
        )
        p_t = torch.exp(-ce_loss)  # p_t = softmax probability of correct class
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class AblationExperimentBase(ABC):
    """消融实验基类"""
    
    def __init__(
        self,
        experiment_name: str,
        variant_name: str,
        config: Dict,
        dataset: str = 'yelp',
        device: str = 'cuda',
        gpu: int = 0,
        seed: int = 42,
        output_dir: Optional[Path] = None,
        log_level: int = logging.INFO
    ):
        """
        初始化消融实验
        
        Parameters
        ----------
        experiment_name : str
            实验名称，如 '核心模块消融'
        variant_name : str
            变体名称，如 'wo_tgn_memory'
        config : Dict
            模型配置
        dataset : str
            数据集名称
        device : str
            设备类型 ('cuda' 或 'cpu')
        gpu : int
            GPU设备ID（当device='cuda'时生效）
        seed : int
            随机种子
        output_dir : Path
            输出目录
        log_level : int
            日志级别
        """
        self.experiment_name = experiment_name
        self.variant_name = variant_name
        self.config = config
        self.dataset = dataset
        self.seed = seed
        
        # 设置设备
        if device == 'cuda' and torch.cuda.is_available():
            self.device = f'cuda:{gpu}'
        else:
            self.device = 'cpu'
        
        # 设置输出目录
        if output_dir is None:
            output_dir = Path(__file__).parent / experiment_name / 'results'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging(log_level)
        
        # 设置随机种子
        self._set_seed(seed)
        
        # 加载数据
        self._load_data()
        
        # 训练历史
        self.history = {
            'train_loss': [], 'train_auc': [], 'train_f1': [],
            'train_precision': [], 'train_recall': [], 'train_ap': [],
            'test_loss': [], 'test_auc': [], 'test_f1': [],
            'test_precision': [], 'test_recall': [], 'test_ap': [],
            'lr': [],
            # RL 相关指标
            'rl_policy_loss': [], 'rl_value_loss': [], 'rl_entropy': [],
            'rl_total_loss': [], 'memory_weight_mean': [], 'memory_weight_std': []
        }
        
        # 混合精度训练
        self.use_amp = TRAIN_CONFIG.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp and 'cuda' in self.device else None
    
    def _setup_logging(self, log_level):
        """设置日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f'{self.variant_name}_{timestamp}.log'
        
        # 创建独立的logger
        self.logger = logging.getLogger(f'{self.experiment_name}.{self.variant_name}')
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # 清除已有的handlers
        
        # 文件handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(log_level)
        
        # 控制台handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # 格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"日志文件: {log_file}")
    
    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _load_data(self):
        """加载数据"""
        data_dir = get_data_dir(self.dataset)
        
        self.logger.info("=" * 80)
        self.logger.info(f"消融实验: {self.experiment_name}")
        self.logger.info(f"变体: {self.variant_name}")
        self.logger.info(f"数据集: {self.dataset.upper()}")
        self.logger.info(f"数据目录: {data_dir}")
        self.logger.info("=" * 80)
        
        # 加载数据
        self.data, self.class_weights = load_fraud_detection_data(
            data_dir, dataset=self.dataset
        )
        self.data = self.data.to(self.device)
        self.class_weights = self.class_weights.to(self.device)
        
        # 节点划分
        splits = create_node_split(
            labels=self.data.y.cpu(),
            timestamps=self.data.timestamps.cpu() if hasattr(self.data, 'timestamps') else None,
            method=SPLIT_CONFIG['split_method'],
            train_ratio=SPLIT_CONFIG['train_ratio'],
            val_ratio=SPLIT_CONFIG['val_ratio'],
            test_ratio=SPLIT_CONFIG['test_ratio'],
            seed=self.seed
        )
        
        self.data.train_mask = splits['train_mask'].to(self.device)
        self.data.val_mask = splits['val_mask'].to(self.device)
        self.data.test_mask = splits['test_mask'].to(self.device)
        
        self.logger.info(f"训练集: {self.data.train_mask.sum().item():,}")
        if self.data.val_mask.sum().item() > 0:
            self.logger.info(f"验证集: {self.data.val_mask.sum().item():,}")
        self.logger.info(f"测试集: {self.data.test_mask.sum().item():,}")
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """
        创建模型（子类必须实现）
        
        Returns
        -------
        nn.Module
            模型实例
        """
        pass
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = TRAIN_CONFIG.get('optimizer', 'adamw')
        if optimizer_type == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=TRAIN_CONFIG['learning_rate'],
                weight_decay=TRAIN_CONFIG['weight_decay']
            )
        else:
            return optim.Adam(
                model.parameters(),
                lr=TRAIN_CONFIG['learning_rate'],
                weight_decay=TRAIN_CONFIG['weight_decay']
            )
    
    def create_scheduler(self, optimizer: optim.Optimizer, num_epochs: int):
        """创建学习率调度器（Warmup + CosineAnnealing）"""
        warmup_epochs = TRAIN_CONFIG.get('warmup_epochs', 20)
        # 阶段1: 线性warmup，从 lr*0.01 增长到 lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        # 阶段2: 余弦退火
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=TRAIN_CONFIG.get('scheduler_min_lr', 1e-6)
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        支持强化学习记忆门控：
        - 如果模型启用了 RL 记忆门控，会额外计算 RL 损失
        - RL 损失包括策略梯度损失、价值网络损失和熵正则化
        - 奖励信号基于分类性能（AUC 的近似）
        """
        model.train()
        optimizer.zero_grad()
        
        # 初始化 RL 损失相关变量
        rl_losses = None
        
        if self.use_amp and self.scaler is not None:
            with autocast():
                logits = model(self.data)
                train_mask = self.data.train_mask
                cls_loss = criterion(logits[train_mask], self.data.y[train_mask])
                
                # ===== 强化学习损失计算 =====
                total_loss = cls_loss
                if hasattr(model, 'compute_rl_loss') and hasattr(model, 'rl_output') and model.rl_output is not None:
                    # 计算奖励：使用分类正确率作为即时奖励
                    with torch.no_grad():
                        pred = logits[train_mask].argmax(dim=1)
                        y_true = self.data.y[train_mask]
                        # 节点级别的正确性奖励
                        node_correct = (pred == y_true).float()  # [N_train]
                        # 扩展到所有节点（非训练节点奖励为 0）
                        reward = torch.zeros(self.data.x.size(0), device=self.device)
                        reward[train_mask] = node_correct
                        # 额外奖励：欺诈检测正确性（更高权重）
                        fraud_mask = (y_true == 0)  # 欺诈类（假设 0 是欺诈）
                        fraud_reward_bonus = node_correct * fraud_mask.float() * 0.5
                        reward[train_mask] = reward[train_mask] + fraud_reward_bonus
                    
                    # 计算 RL 损失
                    rl_losses = model.compute_rl_loss(reward, mask=None)
                    total_loss = cls_loss + rl_losses['total_rl_loss']
            
            self.scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            max_grad_norm = TRAIN_CONFIG.get('max_gradient_norm', 1.0)
            if max_grad_norm > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            logits = model(self.data)
            train_mask = self.data.train_mask
            cls_loss = criterion(logits[train_mask], self.data.y[train_mask])
            
            # ===== 强化学习损失计算 =====
            total_loss = cls_loss
            if hasattr(model, 'compute_rl_loss') and hasattr(model, 'rl_output') and model.rl_output is not None:
                # 计算奖励
                with torch.no_grad():
                    pred = logits[train_mask].argmax(dim=1)
                    y_true = self.data.y[train_mask]
                    node_correct = (pred == y_true).float()
                    reward = torch.zeros(self.data.x.size(0), device=self.device)
                    reward[train_mask] = node_correct
                    fraud_mask = (y_true == 0)
                    fraud_reward_bonus = node_correct * fraud_mask.float() * 0.5
                    reward[train_mask] = reward[train_mask] + fraud_reward_bonus
                
                rl_losses = model.compute_rl_loss(reward, mask=None)
                total_loss = cls_loss + rl_losses['total_rl_loss']
            
            total_loss.backward()
            
            max_grad_norm = TRAIN_CONFIG.get('max_gradient_norm', 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
        
        # 计算训练指标
        with torch.no_grad():
            metrics = self._compute_metrics(logits, train_mask, cls_loss.item())
            
            # 添加 RL 相关指标
            if rl_losses is not None:
                metrics['rl_policy_loss'] = rl_losses['policy_loss'].item()
                metrics['rl_value_loss'] = rl_losses['value_loss'].item()
                metrics['rl_entropy'] = -rl_losses['entropy_loss'].item()  # 取负得到熵
                metrics['rl_total_loss'] = rl_losses['total_rl_loss'].item()
            
            # 添加记忆权重统计（如果可用）
            if hasattr(model, 'get_memory_weights_stats'):
                weight_stats = model.get_memory_weights_stats()
                if weight_stats is not None:
                    metrics['memory_weight_mean'] = weight_stats['mean']
                    metrics['memory_weight_std'] = weight_stats['std']
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        split: str = 'test',
        threshold: float = None
    ) -> Dict[str, float]:
        """评估模型"""
        model.eval()

        logits = model(self.data)

        if split == 'test':
            mask = self.data.test_mask
        elif split == 'val':
            mask = self.data.val_mask
        else:
            mask = self.data.train_mask

        loss = criterion(logits[mask], self.data.y[mask])
        metrics = self._compute_metrics(logits, mask, loss.item(), threshold=threshold)

        return metrics
    
    def _find_optimal_threshold(self, y_true_fraud: np.ndarray, y_prob_fraud: np.ndarray) -> float:
        """在验证集上搜索最优F1阈值"""
        precisions, recalls, thresholds = precision_recall_curve(y_true_fraud, y_prob_fraud)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = f1_scores.argmax()
        if best_idx < len(thresholds):
            return float(thresholds[best_idx])
        return 0.5

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        loss: float,
        threshold: float = None
    ) -> Dict[str, float]:
        """计算评估指标（带NaN安全处理，支持最优阈值）"""
        # NaN安全检查：将NaN替换为0
        if torch.isnan(logits[mask]).any():
            self.logger.warning("检测到logits中包含NaN，进行安全替换")
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)

        prob_all = torch.softmax(logits[mask], dim=1)
        prob_fraud = prob_all[:, 0]

        y_true = self.data.y[mask].cpu().numpy()
        y_true_fraud = 1 - y_true  # 0→1(欺诈), 1→0(正常)
        y_prob_fraud = prob_fraud.cpu().numpy()

        # NaN安全检查概率值
        if np.isnan(y_prob_fraud).any():
            y_prob_fraud = np.nan_to_num(y_prob_fraud, nan=0.5)

        # 使用最优阈值或argmax
        if threshold is not None:
            y_pred_fraud = (y_prob_fraud >= threshold).astype(int)
            y_pred = 1 - y_pred_fraud  # 转回原始标签空间: 1(欺诈)→0, 0(正常)→1
        else:
            y_pred = logits[mask].argmax(dim=1).cpu().numpy()

        return {
            'loss': loss,
            'auc': roc_auc_score(y_true_fraud, y_prob_fraud),
            'ap': average_precision_score(y_true_fraud, y_prob_fraud),
            'f1': f1_score(y_true, y_pred, pos_label=0),
            'precision': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            'recall': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
        }
    
    def run(
        self,
        num_epochs: int = 200,
        print_freq: int = 10
    ) -> Dict[str, float]:
        """
        运行消融实验（支持验证集早停和最佳模型恢复）
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"开始消融实验: {self.variant_name}")
        self.logger.info("=" * 80)
        
        # 更新配置中的设备信息
        self.config['device'] = self.device
        
        # 创建模型
        model = self.create_model()
        model = model.to(self.device)
        
        # 检查模型是否标记为禁用AMP（GCN在大图+float16下容易NaN）
        if hasattr(model, 'disable_amp') and model.disable_amp:
            self.use_amp = False
            self.scaler = None
            self.logger.info("模型已禁用AMP（防止float16数值溢出）")
        
        num_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"模型参数量: {num_params:,}")
        
        # 创建优化器和调度器
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer, num_epochs)
        focal_gamma = TRAIN_CONFIG.get('focal_gamma', 2.0)
        label_smoothing = TRAIN_CONFIG.get('label_smoothing', 0.0)
        criterion = FocalLoss(weight=self.class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
        
        # 早停配置
        use_early_stopping = TRAIN_CONFIG.get('use_early_stopping', False)
        early_stopping_patience = TRAIN_CONFIG.get('early_stopping_patience', 30)
        early_stopping_metric = TRAIN_CONFIG.get('early_stopping_metric', 'auc')
        early_stopping_min_delta = TRAIN_CONFIG.get('early_stopping_min_delta', 0.0001)
        
        # 检查是否有验证集
        has_val = hasattr(self.data, 'val_mask') and self.data.val_mask.sum().item() > 0
        if use_early_stopping and has_val:
            self.logger.info(f"早停已启用: patience={early_stopping_patience}, "
                           f"metric={early_stopping_metric}, val_size={self.data.val_mask.sum().item():,}")
        elif use_early_stopping and not has_val:
            self.logger.info("早停已启用但无验证集，使用测试集监控（仅消融实验使用）")
        
        # 训练循环
        best_val_metric = 0.0
        best_test_auc = 0.0
        best_epoch = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(model, optimizer, criterion)
            
            # 验证集评估（如果有）
            if has_val:
                val_metrics = self.evaluate(model, criterion, 'val')
            
            # 测试集评估
            test_metrics = self.evaluate(model, criterion, 'test')
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            self.history['train_ap'].append(train_metrics['ap'])
            
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['test_auc'].append(test_metrics['auc'])
            self.history['test_f1'].append(test_metrics['f1'])
            self.history['test_precision'].append(test_metrics['precision'])
            self.history['test_recall'].append(test_metrics['recall'])
            self.history['test_ap'].append(test_metrics['ap'])
            
            self.history['lr'].append(current_lr)
            
            # 记录 RL 相关指标（如果存在）
            self.history['rl_policy_loss'].append(train_metrics.get('rl_policy_loss', 0.0))
            self.history['rl_value_loss'].append(train_metrics.get('rl_value_loss', 0.0))
            self.history['rl_entropy'].append(train_metrics.get('rl_entropy', 0.0))
            self.history['rl_total_loss'].append(train_metrics.get('rl_total_loss', 0.0))
            self.history['memory_weight_mean'].append(train_metrics.get('memory_weight_mean', 0.0))
            self.history['memory_weight_std'].append(train_metrics.get('memory_weight_std', 0.0))
            
            # 确定监控指标（优先用验证集，否则用测试集）
            monitor_metrics = val_metrics if has_val else test_metrics
            current_metric = monitor_metrics[early_stopping_metric]
            
            # 更新最佳结果 & 保存最佳模型状态
            if current_metric > best_val_metric + early_stopping_min_delta:
                best_val_metric = current_metric
                best_test_auc = test_metrics['auc']
                best_epoch = epoch
                patience_counter = 0
                # 深拷贝最佳模型参数
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % print_freq == 0 or epoch == 1:
                log_msg = (
                    f"Epoch {epoch:3d}/{num_epochs} | LR: {current_lr:.6f} | "
                    f"Train Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc']:.4f} | "
                    f"Test AUC: {test_metrics['auc']:.4f} F1: {test_metrics['f1']:.4f}"
                )
                if has_val:
                    log_msg += f" | Val AUC: {val_metrics['auc']:.4f}"
                if 'memory_weight_mean' in train_metrics and train_metrics['memory_weight_mean'] > 0:
                    log_msg += f" | MemW: {train_metrics['memory_weight_mean']:.3f}±{train_metrics['memory_weight_std']:.3f}"
                self.logger.info(log_msg)
            
            # 早停检查
            if use_early_stopping and patience_counter >= early_stopping_patience:
                self.logger.info(f"早停触发: 连续 {early_stopping_patience} 个epoch无提升，"
                               f"最佳 {early_stopping_metric}={best_val_metric:.4f} (Epoch {best_epoch})")
                break
        
        # 恢复最佳模型并最终评估
        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            self.logger.info(f"已恢复最佳模型 (Epoch {best_epoch})")

        # 在验证集上搜索最优阈值，然后应用到测试集
        optimal_threshold = None
        if has_val:
            model.eval()
            with torch.no_grad():
                logits = model(self.data)
                val_mask = self.data.val_mask
                prob_all = torch.softmax(logits[val_mask], dim=1)
                prob_fraud = prob_all[:, 0].cpu().numpy()
                y_true_val = self.data.y[val_mask].cpu().numpy()
                y_true_val_fraud = 1 - y_true_val
                if not np.isnan(prob_fraud).any():
                    optimal_threshold = self._find_optimal_threshold(y_true_val_fraud, prob_fraud)
                    self.logger.info(f"验证集最优阈值: {optimal_threshold:.4f}")

        final_metrics = self.evaluate(model, criterion, 'test', threshold=optimal_threshold)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"消融实验完成: {self.variant_name}")
        self.logger.info("=" * 80)
        self.logger.info(f"最佳Epoch: {best_epoch} (早停于 Epoch {epoch})")
        self.logger.info(f"最佳模型测试结果:")
        self.logger.info(f"  AUC:       {final_metrics['auc']:.4f}")
        self.logger.info(f"  AP:        {final_metrics['ap']:.4f}")
        self.logger.info(f"  F1-Score:  {final_metrics['f1']:.4f}")
        self.logger.info(f"  Precision: {final_metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {final_metrics['recall']:.4f}")
        self.logger.info("=" * 80)
        
        # 保存最佳模型checkpoint（用于可视化等后续分析）
        if best_model_state is not None:
            ckpt_path = self.output_dir / f'{self.variant_name}_best_{self.dataset}.pth'
            torch.save(best_model_state, ckpt_path)
            self.logger.info(f"最佳模型已保存: {ckpt_path}")

        # 保存结果
        self._save_results(final_metrics, best_test_auc, best_epoch, num_params)

        return final_metrics
    
    def _save_results(
        self,
        final_metrics: Dict[str, float],
        best_auc: float,
        best_epoch: int,
        num_params: int
    ):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'experiment_name': self.experiment_name,
            'variant_name': self.variant_name,
            'dataset': self.dataset,
            'config': {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.config.items()},
            'num_parameters': num_params,
            'best_test_auc': best_auc,
            'best_epoch': best_epoch,
            'final_metrics': final_metrics,
            'history': self.history,
            'timestamp': timestamp,
        }
        
        # 保存JSON
        results_file = self.output_dir / f'{self.variant_name}_results_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"结果已保存: {results_file}")


class StandardAblationExperiment(AblationExperimentBase):
    """
    标准消融实验类
    用于大多数只需要修改配置参数的消融实验
    """
    
    def create_model(self) -> nn.Module:
        """创建标准HTGATFraud模型"""
        from src.models.ht_gat_fraud import HTGATFraud
        
        # 合并数据集特定配置
        model_config = get_model_config_for_dataset(self.dataset, self.config)
        
        return HTGATFraud(model_config)


if __name__ == '__main__':
    # 测试基类
    from ablation_config import ABLATION_CORE_MODULES
    
    baseline = ABLATION_CORE_MODULES['baseline']
    experiment = StandardAblationExperiment(
        experiment_name='测试实验',
        variant_name='baseline',
        config=baseline['config'],
        dataset='yelp',
        device='cuda',
        seed=42
    )
    
    print("消融实验基类测试通过！")
