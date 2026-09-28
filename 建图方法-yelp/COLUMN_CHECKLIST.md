# 列名检查清单

## Preprocessor 创建的列

### 原始列（保留）
- `json_user_id` - 用户ID
- `json_business_id` - 商家ID  
- `json_stars` - 评分 (1-5)
- `timestamp` - 时间戳 (datetime)
- `label` - 标签 (0=欺诈, 1=正常)

### 重命名列
- `text` - 评论文本 (从 json_text 重命名)

### 时间特征列
- `year` - 年份
- `month` - 月份
- `day` - 日期
- `hour` - 小时
- `dayofweek` - 星期几
- `date` - 日期（不含时间）
- `year_month` - 年月 (Period类型)
- `date_unix` - Unix时间戳（秒）
- `timestamp_norm` - 归一化时间戳 [0, 1]

### 索引列
- `user_idx` - 用户索引 (0-based)
- `business_idx` - 商家索引 (0-based)
- `review_idx` - 评论索引 (0-based)

## Edges 模块需要的列

### RUR Builder
- `user_idx` ✓
- `review_idx` ✓
- `timestamp` ✓

### RTR Builder
- `business_idx` ✓
- `review_idx` ✓
- `timestamp` ✓
- `year_month` ✓

### RSR Builder
- `business_idx` ✓
- `json_stars` ✓
- `review_idx` ✓
- `timestamp` ✓

## Features 模块需要的列

### User Features
- `json_user_id` ✓
- `json_business_id` ✓
- `json_stars` ✓
- `timestamp` ✓
- `date` ✓
- `review_idx` ✓

### Product Features
- `json_business_id` ✓
- `json_stars` ✓
- `timestamp` ✓
- `date` ✓

### Review Features
- `json_business_id` ✓
- `json_user_id` ✓
- `json_stars` ✓
- `text` ✓
- `timestamp` ✓
- `hour` ✓
- `dayofweek` ✓
- `review_idx` ✓

### Temporal Features
- `json_user_id` ✓
- `timestamp` ✓

## 状态

✅ 所有必需的列都已创建
✅ 列名已统一（除了 text 从 json_text 重命名）
✅ 数据类型正确
