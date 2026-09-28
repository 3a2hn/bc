# -*- coding: utf-8 -*-
"""
Amazon数据预处理器
以评论为节点的图构建预处理（与Yelp结构一致）

核心功能：
1. 加载和清洗评论数据
2. 解析时间戳
3. 创建ID映射
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class AmazonPreprocessor:
    """Amazon数据预处理器（评论节点图，与Yelp一致）"""
    
    def __init__(self, config):
        """
        初始化预处理器
        
        Args:
            config: 配置模块
        """
        self.config = config
        self.input_path = config.INPUT_CSV
        self.df = None
        self.user_to_idx = {}
        self.product_to_idx = {}
        
    def load_data(self):
        """加载CSV数据"""
        logger.info(f"正在加载数据: {self.input_path}")

        try:
            self.df = pd.read_csv(self.input_path, encoding='utf-8-sig')
        except:
            self.df = pd.read_csv(self.input_path, encoding='utf-8')

        logger.info(f"成功加载 {len(self.df)} 条评论")
        logger.info(f"数据列: {list(self.df.columns)}")

        # 适配新数据集列名：user_id -> reviewerID, text -> reviewText, rating -> overall
        col_mapping = {
            'user_id': 'reviewerID',
            'text': 'reviewText',
            'rating': 'overall',
        }
        for old_col, new_col in col_mapping.items():
            if old_col in self.df.columns and new_col not in self.df.columns:
                self.df.rename(columns={old_col: new_col}, inplace=True)
                logger.info(f"列名映射: {old_col} -> {new_col}")

        # 适配时间戳：unix_timestamp（秒级）-> detailed_timestamp（毫秒级）
        if 'unix_timestamp' in self.df.columns and 'detailed_timestamp' not in self.df.columns:
            self.df['detailed_timestamp'] = self.df['unix_timestamp'] * 1000
            logger.info("已将 unix_timestamp（秒）转换为 detailed_timestamp（毫秒）")

        if 'reviewerID' in self.df.columns:
            logger.info(f"用户数: {self.df['reviewerID'].nunique()}")
        if 'asin' in self.df.columns:
            logger.info(f"产品数: {self.df['asin'].nunique()}")

        if 'label' in self.df.columns:
            logger.info(f"标签分布:\n{self.df['label'].value_counts()}")
        elif 'class' in self.df.columns:
            logger.info(f"标签分布:\n{self.df['class'].value_counts()}")

        return self
    
    def clean_data(self):
        """清洗数据"""
        logger.info("开始数据清洗...")
        
        initial_count = len(self.df)
        
        # 标准化列名
        self._standardize_columns()

        # 删除缺失值
        required_columns = ['reviewerID', 'asin', 'reviewText', 'overall',
                          'detailed_timestamp', 'label']

        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            logger.warning(f"缺少列: {missing_cols}")
            if 'unixReviewTime' in self.df.columns and 'detailed_timestamp' in missing_cols:
                self.df['detailed_timestamp'] = self.df['unixReviewTime'] * 1000
                missing_cols.remove('detailed_timestamp')

        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

        self.df = self.df.dropna(subset=[col for col in required_columns if col in self.df.columns])
        logger.info(f"删除缺失值后: {len(self.df)} 条")
        
        # 删除重复评论
        before_dedup = len(self.df)
        self.df = self.df.drop_duplicates(
            subset=['reviewerID', 'asin', 'reviewText'],
            keep='first'
        )
        logger.info(f"删除重复评论后: {len(self.df)} 条")
        
        # 验证评分范围
        self.df = self.df[self.df['overall'].isin([1.0, 2.0, 3.0, 4.0, 5.0])]
        
        # 验证时间戳
        self.df = self.df[self.df['detailed_timestamp'] > 0]
        
        # 反转标签：原始 0=正常,1=欺诈 → 统一 0=欺诈,1=正常
        self.df['label'] = 1 - self.df['label'].astype(int)
        self.df = self.df[self.df['label'].isin([0, 1])]
        
        logger.info(f"数据清洗完成，保留 {len(self.df)} 条评论")
        
        return self
    
    def _standardize_columns(self):
        """标准化列名，兼容新旧数据集格式"""
        # --- 时间戳字段适配 ---
        # 新数据集使用 timestamp_aligned（秒级），需转换为 detailed_timestamp（毫秒级）
        if 'timestamp_aligned' in self.df.columns and 'detailed_timestamp' not in self.df.columns:
            self.df['detailed_timestamp'] = self.df['timestamp_aligned'] * 1000
            logger.info("已将 timestamp_aligned（秒）转换为 detailed_timestamp（毫秒）")
        
        # --- helpful 字段适配 ---
        # 新数据集 helpful 为字符串 "[helpful_votes, total_votes]"，需解析
        if 'helpful' in self.df.columns and 'helpful_votes' not in self.df.columns:
            import ast
            def parse_helpful(val):
                try:
                    parsed = ast.literal_eval(str(val))
                    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                        return int(parsed[0]), int(parsed[1])
                except:
                    pass
                return 0, 0
            
            parsed = self.df['helpful'].apply(parse_helpful)
            self.df['helpful_votes'] = parsed.apply(lambda x: x[0])
            self.df['total_votes'] = parsed.apply(lambda x: x[1])
            logger.info("已从 helpful 字段解析出 helpful_votes 和 total_votes")
        
        # --- 兜底：确保 helpful_votes / total_votes 存在 ---
        if 'helpful_votes' not in self.df.columns:
            self.df['helpful_votes'] = 0
        if 'total_votes' not in self.df.columns:
            self.df['total_votes'] = 0
        
        self.df['helpful_votes'] = self.df['helpful_votes'].fillna(0).astype(int)
        self.df['total_votes'] = self.df['total_votes'].fillna(0).astype(int)
    
    def parse_timestamps(self):
        """解析和标准化时间戳"""
        logger.info("解析时间戳...")
        
        # detailed_timestamp是毫秒级时间戳
        self.df['timestamp'] = pd.to_datetime(
            self.df['detailed_timestamp'] / 1000,
            unit='s'
        )
        
        # 提取时间特征
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day'] = self.df['timestamp'].dt.day
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['dayofweek'] = self.df['timestamp'].dt.dayofweek
        
        # 创建日期列
        self.df['date'] = self.df['timestamp'].dt.date
        
        # 创建年月列
        self.df['year_month'] = self.df['timestamp'].dt.to_period('M')
        
        # 使用detailed_timestamp作为date_unix
        self.df['date_unix'] = self.df['detailed_timestamp']
        
        # 计算归一化时间戳
        min_ts = self.df['date_unix'].min()
        max_ts = self.df['date_unix'].max()
        self.df['timestamp_norm'] = (self.df['date_unix'] - min_ts) / (max_ts - min_ts)
        
        logger.info(f"时间范围: {self.df['timestamp'].min()} 到 {self.df['timestamp'].max()}")
        logger.info(f"时间跨度: {(self.df['timestamp'].max() - self.df['timestamp'].min()).days} 天")
        
        return self
    
    def create_mappings(self):
        """创建ID映射"""
        logger.info("创建ID映射...")
        
        # 用户ID映射
        unique_users = sorted(self.df['reviewerID'].unique())
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.df['user_idx'] = self.df['reviewerID'].map(self.user_to_idx)
        
        # 产品ID映射
        unique_products = sorted(self.df['asin'].unique())
        self.product_to_idx = {prod: idx for idx, prod in enumerate(unique_products)}
        self.df['product_idx'] = self.df['asin'].map(self.product_to_idx)
        
        # 按时间排序，重建评论索引
        self.df = self.df.sort_values('date_unix').reset_index(drop=True)
        self.df['review_idx'] = self.df.index
        
        logger.info(f"映射创建完成:")
        logger.info(f"  - {len(self.user_to_idx)} 个用户")
        logger.info(f"  - {len(self.product_to_idx)} 个产品")
        logger.info(f"  - {len(self.df)} 条评论（节点）")
        
        return self
    
    def process_all(self):
        """执行完整的预处理流程"""
        self.load_data()
        self.clean_data()
        self.parse_timestamps()
        self.create_mappings()
        
        # 打印最终统计
        logger.info("="*60)
        logger.info("预处理完成!")
        logger.info(f"评论数（节点数）: {len(self.df)}")
        logger.info(f"用户数: {len(self.user_to_idx)}")
        logger.info(f"产品数: {len(self.product_to_idx)}")
        logger.info(f"欺诈评论: {(self.df['label']==0).sum()} ({(self.df['label']==0).mean()*100:.2f}%)")
        logger.info(f"正常评论: {(self.df['label']==1).sum()} ({(self.df['label']==1).mean()*100:.2f}%)")
        logger.info("="*60)
        
        return self.df
    
    def get_mappings(self):
        """获取ID映射字典"""
        return {
            'user_to_idx': self.user_to_idx,
            'product_to_idx': self.product_to_idx,
            'idx_to_user': {idx: user for user, idx in self.user_to_idx.items()},
            'idx_to_product': {idx: prod for prod, idx in self.product_to_idx.items()},
        }
    
    def save_mappings(self):
        """保存映射"""
        with open(self.config.MAPPINGS_OUTPUT, 'wb') as f:
            pickle.dump(self.get_mappings(), f)
        logger.info(f"ID映射已保存: {self.config.MAPPINGS_OUTPUT}")


def preprocess_amazon_data(config):
    """
    预处理Amazon数据的便捷函数
    
    Args:
        config: 配置模块
        
    Returns:
        df: 评论级别DataFrame
        mappings: ID映射字典
    """
    preprocessor = AmazonPreprocessor(config)
    df = preprocessor.process_all()
    mappings = preprocessor.get_mappings()
    preprocessor.save_mappings()
    
    return df, mappings


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import config_amazon as config
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df, mappings = preprocess_amazon_data(config)
    
    print("\n数据预览:")
    print(df.head())
    print("\n数据列:")
    print(df.columns.tolist())
