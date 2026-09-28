# -*- coding: utf-8 -*-
"""
HTGATFraud实验运行脚本 - 不同训练集比例
====================================

运行 HTGATFraud 模型在不同训练集比例(20%, 40%, 60%)下的实验
"""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime
import logging


# 配置
TRAIN_RATIOS = [0.5, 0.6, 0.7]  # 改为与消融实验类似的比例
VAL_RATIO = 0.1  # 使用10%验证集
EPOCHS = 200
SEED = 42
GPU_ID = 0  # GPU设备ID


def setup_logging():
    """设置日志"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'experiments_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def run_experiment(train_ratio, val_ratio, epochs=200, seed=42, logger=None):
    """
    运行单个实验

    Parameters
    ----------
    train_ratio : float
        训练集比例
    val_ratio : float
        验证集比例
    epochs : int
        训练轮数
    seed : int
        随机种子
    logger : logging.Logger
        日志记录器

    Returns
    -------
    dict or None
        训练结果
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    
    logger.info(f"\n{'='*80}")
    logger.info(f"运行实验: HTGATFraud | Train={train_ratio*100:.0f}% Val={val_ratio*100:.0f}% Test={test_ratio*100:.0f}%")
    logger.info(f"{'='*80}")

    train_script = Path(__file__).parent / 'train.py'

    # 构建命令（使用验证集和早停，与消融实验一致）
    cmd = [
        sys.executable,
        str(train_script),
        '--train_ratio', str(train_ratio),
        '--val_ratio', str(val_ratio),
        '--test_ratio', str(test_ratio),
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--device', 'cuda',
        '--gpu', str(GPU_ID),
        '--eval_test',
        '--save_plots',
        '--use_amp'  # 启用混合精度训练
    ]
    
    try:
        logger.info(f"命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=72000  # 20小时超时
        )
        
        if result.returncode != 0:
            logger.error(f"✗ 训练失败")
            logger.error(f"stdout: {result.stdout[-2000:] if result.stdout else 'None'}")
            logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'None'}")
            return None
        
        logger.info(f"✓ 训练完成")
        
        # 查找最新的测试结果文件
        logs_dir = Path(__file__).parent / 'logs'
        result_files = sorted(logs_dir.glob('test_results_*.json'), 
                             key=lambda p: p.stat().st_mtime, reverse=True)
        
        if result_files:
            with open(result_files[0], 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 添加训练比例信息
            result_data['train_ratio'] = train_ratio
            result_data['test_ratio'] = test_ratio
            
            return result_data
        else:
            logger.warning(f"未找到测试结果文件")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ 训练超时（>2小时）")
        return None
    except Exception as e:
        logger.error(f"✗ 运行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def print_results_table(all_results, logger):
    """打印结果表格"""
    logger.info("\n" + "="*90)
    logger.info("实验结果汇总")
    logger.info("="*90)
    
    # 表头
    logger.info(f"\n{'Model':<15} {'Train%':<8} {'AUC':<8} {'AP':<8} {'F1':<8} {'ACC':<8} "
               f"{'Precision':<11} {'Recall':<8}")
    logger.info("-"*90)
    
    # 数据行
    for ratio_key, result in all_results['results'].items():
        if result and 'test_metrics' in result:
            m = result['test_metrics']
            ap_val = m.get('ap', 0.0)
            logger.info(f"{'HTGATFraud':<15} {ratio_key:<8} {m['auc']:<8.4f} {ap_val:<8.4f} "
                      f"{m['f1']:<8.4f} {m['acc']:<8.4f} {m['precision']:<11.4f} {m['recall']:<8.4f}")
        else:
            logger.info(f"{'HTGATFraud':<15} {ratio_key:<8} {'FAILED':<8}")
    
    logger.info("="*90)
    
    # 趋势分析
    logger.info("\n按训练比例分析（随训练集比例的变化）")
    logger.info("-"*80)
    
    ratios = []
    aucs = []
    aps = []
    f1s = []
    
    for ratio_key in ['20%', '40%', '60%']:
        if ratio_key in all_results['results'] and all_results['results'][ratio_key]:
            m = all_results['results'][ratio_key]['test_metrics']
            ratios.append(ratio_key)
            aucs.append(m['auc'])
            aps.append(m.get('ap', 0.0))
            f1s.append(m['f1'])
    
    if ratios:
        logger.info(f"\nHTGATFraud:")
        logger.info(f"  Train Ratio:  {' -> '.join(ratios)}")
        logger.info(f"  AUC:          {' -> '.join([f'{a:.4f}' for a in aucs])}")
        logger.info(f"  AP:           {' -> '.join([f'{a:.4f}' for a in aps])}")
        logger.info(f"  F1:           {' -> '.join([f'{f:.4f}' for f in f1s])}")
        
        if len(aucs) >= 2:
            auc_gain = aucs[-1] - aucs[0]
            ap_gain = aps[-1] - aps[0] if aps[0] > 0 else 0.0
            f1_gain = f1s[-1] - f1s[0]
            logger.info(f"  增益(20%->60%): AUC={auc_gain:+.4f}, AP={ap_gain:+.4f}, F1={f1_gain:+.4f}")


def main():
    """主函数"""
    logger = setup_logging()
    base_dir = Path(__file__).parent
    
    logger.info("="*80)
    logger.info("HTGATFraud模型 - 多比例训练集实验")
    logger.info("="*80)
    logger.info(f"模型: HTGATFraud")
    logger.info(f"训练集比例: {', '.join([f'{r*100:.0f}%' for r in TRAIN_RATIOS])}")
    logger.info(f"验证集: {VAL_RATIO*100:.0f}% (使用早停)")
    logger.info(f"训练轮数: {EPOCHS}")
    logger.info(f"随机种子: {SEED}")
    logger.info(f"设备: cuda:{GPU_ID}")
    logger.info("="*80)
    
    # 存储所有结果
    all_results = {
        'experiment': 'htgatfraud_train_ratio_comparison',
        'model': 'HTGATFraud',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'train_ratios': TRAIN_RATIOS,
            'val_ratio': VAL_RATIO,
            'epochs': EPOCHS,
            'seed': SEED,
            'device': f'cuda:{GPU_ID}',
        },
        'results': {}
    }
    
    # 对每个比例进行实验
    for train_ratio in TRAIN_RATIOS:
        ratio_key = f'{train_ratio*100:.0f}%'
        
        result = run_experiment(
            train_ratio=train_ratio,
            val_ratio=VAL_RATIO,
            epochs=EPOCHS,
            seed=SEED,
            logger=logger
        )
        
        if result:
            all_results['results'][ratio_key] = result
            
            # 打印简要结果
            metrics = result['test_metrics']
            ap_str = f"AP={metrics.get('ap', 0.0):.4f} | " if 'ap' in metrics else ""
            logger.info(f"  结果: AUC={metrics['auc']:.4f} | {ap_str}F1={metrics['f1']:.4f} | "
                      f"ACC={metrics['acc']:.4f} | Recall={metrics['recall']:.4f}")
        else:
            all_results['results'][ratio_key] = None
            logger.warning(f"  结果: 失败")
    
    # 保存汇总结果
    timestamp = all_results['timestamp']
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    summary_file = results_dir / f'htgatfraud_summary_{timestamp}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n汇总结果已保存: {summary_file}")
    
    # 打印结果表格
    print_results_table(all_results, logger)
    
    logger.info("\n" + "="*80)
    logger.info("所有实验完成！")
    logger.info("="*80)


if __name__ == '__main__':
    main()
