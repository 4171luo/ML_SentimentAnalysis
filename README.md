# 基于机器学习的新闻评论情感分析

本项目是毕业设计实现代码。任务目标是对中文新闻评论进行情感二分类，并提供可演示的桌面端界面。项目包含数据预处理、模型训练、结果对比、最终模型保存、评论抓取、前端预测与可视化展示几个部分。

## 1. 项目概览

项目当前包含三类模型：

- 朴素贝叶斯
- 随机森林
- MLP 两层神经网络

项目当前使用两类文本特征：

- TF-IDF
- Word2Vec 平均词向量

其中：

- 朴素贝叶斯支持 `TF-IDF` 和 `Word2Vec`
- 随机森林支持 `TF-IDF` 和 `Word2Vec`
- MLP 使用 `TF-IDF`

## 2. 目录结构

```text
ML_SentimentAnalysis/
├─ data/
│  ├─ train/                 # 训练数据
│  └─ test/                  # 测试评论样本与爬取结果
├─ docs/
│  ├─ 第四章图表/            # 论文第4章图表输出
│  └─ 特征示例表/            # TF-IDF示例表与高权重词示例表
├─ models/                   # 训练完成后的模型文件
├─ resources/                # 停用词表、词表等资源文件
├─ results/                  # 实验结果CSV
├─ src/                      # 源代码
├─ README.md
└─ requirements.txt
```

## 3. 数据文件说明

### 3.1 训练数据

- `data/train/weibo_senti_100k.csv`
- 来源：`weibo_senti_100k`
- 字段：
  - `label`：情感标签
  - `review`：评论文本

### 3.2 测试数据

- `data/test/test*.csv`
- 字段：
  - `uname`
  - `create_time`
  - `comment_contents`

这些文件可以由 `src/app/spider.py` 抓取新闻评论后生成，也可以直接作为前端测试输入。

## 4. 主要代码文件说明

### 4.1 训练脚本

- `src/training/train_nb.py`
  - 朴素贝叶斯训练脚本
  - 对比 `TF-IDF + MultinomialNB` 与 `Word2Vec + GaussianNB`

- `src/training/train_rf.py`
  - 随机森林训练脚本
  - 对比 `TF-IDF` 与 `Word2Vec`

- `src/training/train_nn.py`
  - MLP 训练脚本
  - 使用 `TF-IDF` 作为输入特征

### 4.2 模型保存脚本

- `src/models/save_final_models.py`
  - 根据 `results/` 中的最优结果重新训练全量模型
  - 将最终可部署模型保存到 `models/`

### 4.3 前端与爬虫

- `src/app/demo.py`
  - Tkinter 图形界面
  - 支持输入新闻 URL 或本地 CSV
  - 支持选择模型并展示预测结果、折线图、词云图

- `src/app/spider.py`
  - 凤凰新闻评论抓取脚本
  - 将评论保存到 `data/test/`

### 4.4 论文图表脚本

- `src/visualization/plot_dataset_distribution.py`
  - 生成图4-1 数据集正负样本分布图
  - 输出到 `docs/第四章图表/`

- `src/visualization/plot_tfidf_top_terms.py`
  - 生成图4-2 TF-IDF Top-N词项柱状图
  - 输出到 `docs/第四章图表/`

## 5. 资源与结果文件说明

### 5.1 资源文件

- `resources/stoplist.txt`
  - 中文停用词表

- `resources/vocab.txt`
  - 词表资源

### 5.2 实验结果

- `results/nb_metrics.csv`
  - 朴素贝叶斯实验结果

- `results/rf_grid.csv`
  - 随机森林网格搜索结果

- `results/nn_metrics.csv`
  - MLP 实验结果

### 5.3 论文素材

- `docs/第四章图表/图4-1_数据集正负样本分布图.png`
- `docs/第四章图表/图4-2_TF-IDF Top-N词项柱状图.png`
- `docs/特征示例表/TF-IDF示例表.csv`
- `docs/特征示例表/TF-IDF示例表.xlsx`
- `docs/特征示例表/TF-IDF高权重词示例.csv`
- `docs/特征示例表/TF-IDF高权重词示例.xlsx`

## 6. 环境配置

建议使用 Python 3.12。

### 6.1 创建虚拟环境

```bash
conda create -n ML python=3.12
conda activate ML
```

### 6.2 安装依赖

```bash
pip install -r requirements.txt
```

如果下载较慢，可以使用镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6.3 补充说明

`requirements.txt` 中当前未包含 `torch`。如果需要训练或加载 MLP 模型，请额外安装 PyTorch。

## 7. 使用方法

### 7.1 训练模型

```bash
python src/training/train_nb.py
python src/training/train_rf.py
python src/training/train_nn.py
```

### 7.2 保存最终部署模型

```bash
python src/models/save_final_models.py
```

运行完成后，最终模型会保存到 `models/`。

### 7.3 启动前端演示

```bash
python src/app/demo.py
```

前端支持两种输入方式：

- 输入本地 CSV 文件路径
- 输入凤凰新闻 URL，程序自动抓取评论后分析

### 7.4 生成论文图表

生成图4-1：

```bash
python src/visualization/plot_dataset_distribution.py
```

生成图4-2：

```bash
python src/visualization/plot_tfidf_top_terms.py
```

## 8. 当前默认模型文件

`models/` 目录中当前保存了以下主要文件：

- 朴素贝叶斯模型及向量器
- 随机森林模型及向量器
- Word2Vec 模型
- MLP 权重文件
- MLP 配置文件

这些文件可直接供 `src/app/demo.py` 调用。

## 9. 说明

本项目当前的训练集来自微博情感数据，前端演示测试数据主要来自新闻评论。两者场景不同，后续撰写论文时应如实说明数据来源与任务关系。
