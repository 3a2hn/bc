# -*- coding: utf-8 -*-
"""
数据加载和保存工具
"""

import pickle
import numpy as np
from pathlib import Path
from .logger import get_logger


def save_pickle(data, filepath):
    """
    保存数据为 pickle 文件
    
    Parameters
    ----------
    data : any
        要保存的数据
    filepath : str or Path
        保存路径
    """
    logger = get_logger()
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"数据已保存到: {filepath}")


def load_pickle(filepath):
    """
    从 pickle 文件加载数据
    
    Parameters
    ----------
    filepath : str or Path
        文件路径
        
    Returns
    -------
    data : any
        加载的数据
    """
    logger = get_logger()
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"数据已从 {filepath} 加载")
    
    return data


def save_numpy(array, filepath):
    """
    保存 numpy 数组
    
    Parameters
    ----------
    array : numpy.ndarray
        要保存的数组
    filepath : str or Path
        保存路径
    """
    logger = get_logger()
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(filepath, array)
    
    logger.info(f"数组已保存到: {filepath} (shape={array.shape}, dtype={array.dtype})")


def load_numpy(filepath):
    """
    加载 numpy 数组
    
    Parameters
    ----------
    filepath : str or Path
        文件路径
        
    Returns
    -------
    array : numpy.ndarray
        加载的数组
    """
    logger = get_logger()
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    array = np.load(filepath)
    
    logger.info(f"数组已从 {filepath} 加载 (shape={array.shape}, dtype={array.dtype})")
    
    return array
