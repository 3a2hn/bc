# -*- coding: utf-8 -*-
"""
日志工具
"""

import logging
import sys
from pathlib import Path

# 全局 logger 实例
_logger = None


def setup_logger(log_file=None, level='INFO'):
    """
    设置日志系统
    
    Parameters
    ----------
    log_file : str, optional
        日志文件路径
    level : str, default='INFO'
        日志级别
    """
    global _logger
    
    # 创建 logger
    _logger = logging.getLogger('graph_builder')
    _logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的 handlers
    _logger.handlers.clear()
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    
    # 文件 handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
    
    return _logger


def get_logger():
    """
    获取全局 logger 实例
    
    Returns
    -------
    logger : logging.Logger
        日志对象
    """
    global _logger
    
    if _logger is None:
        _logger = setup_logger()
    
    return _logger
