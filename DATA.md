# 数据文件获取说明

本仓库不会提交大文件。队友 clone 后先做两件事：

1. 安装前端依赖：

   ```
   cd 系统
   npm install
   ```

2. 按下面的清单补齐数据文件。

## 需要补齐的文件

把 `获取方式` 换成真实网盘链接、服务器路径或生成命令，然后提交这个文件。

| 文件或目录 | 说明 | 获取方式 |
| --- | --- | --- |
| `系统/node_modules/` | 前端依赖，不用手动下载 | `cd 系统; npm install` |
| `建图方法-amz/output/` | Amazon 建图输出 | TODO：填网盘链接或生成命令 |
| `建图方法-yelp/output/` | Yelp 建图输出 | TODO：填网盘链接或生成命令 |
| `建图方法-amz/*.csv` | 原始或中间 CSV | TODO：填网盘链接或数据来源 |
| `建图方法-yelp/*.csv` | 原始或中间 CSV | TODO：填网盘链接或数据来源 |
| `系统/storage/datasets/` | 系统数据集 | TODO：填网盘链接或服务器路径 |
| `系统/storage/models/` | 模型检查点 | TODO：填网盘链接或生成命令 |
| `论文模型/**/*.npz` | 消融实验可视化数据 | TODO：填网盘链接或生成命令 |
| `论文模型/**/*.pkl`、`*.pth` | 模型和中间文件 | TODO：填网盘链接或生成命令 |

## 队友的操作

```
git clone https://github.com/3a2hn/bc.git
cd bc
cd 系统
npm install
cd ..
```

然后打开 `DATA.md`，按清单把数据文件放到对应目录。
