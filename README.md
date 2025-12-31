# 此为本人本科毕业设计，仅供大家参考
### 使用SKLearn，WordCloud等库完成
### 具体演示视频可以看我发在bilibili上的视频 P1 大概五分半开始的演示
### [【[毕设] 基于机器学习的新闻评论情感分析方法研究】](https://www.bilibili.com/video/BV1j142197rt/?share_source=copy_web&vd_source=35c49b46a86899e58b3f6414dd834db3)
## ⚠⚠⚠ NEWS!
### 新增论文： [基于机器学习的新闻评论情感分析方法研究-论文定稿.pdf](基于机器学习的新闻评论情感分析方法研究-论文定稿.pdf)
### 新增PPT：  [基于机器学习的新闻评论情感分析方法研究-答辩PPT.pptx](基于机器学习的新闻评论情感分析方法研究-答辩PPT.pptx)
### 推荐将论文和PPT下载后再打开或编辑，尽管PPT嵌入了字体但是还是会出现字体显示错误的问题，可以自行更换显示错误的字体。
# 环境配置
### 这里以conda创建虚拟环境为例，如果你是使用conda创建虚拟环境，就从这里开始
```bash
conda create -n ML python=3.12 #这里的ML可以是自定义的环境名
conda activate ML
```
### 如果你使用的不是conda，就从这里开始，如果你使用的是conda，那就继续
### 确保你现在在项目根目录
```bash
pip install -r requirements.txt
```
### 如果出现网络问题，可以尝试使用镜像
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
# 文件说明
### 训练数据集：`data/train/weibo_senti_100k.csv`，来自 https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/weibo_senti_100k

### 测试数据集：`data/test/test*.csv`，由 `src/spider.py` 爬取生成

### 停用词：`resources/stoplist.txt`

### 训练脚本：`src/train_nb.py`（朴素贝叶斯）、`src/train_rf.py`（随机森林）、`src/train_nn.py`（两层 MLP）
### 指标输出：`results/nb_metrics.csv`、`results/rf_grid.csv`、`results/nn_metrics.csv`

### 前端入口：`src/demo.py`

### 模型输出：`models/`（如 `tfidf_vectorizer.pkl`、`naive_bayes.pkl`、`random_forest.pkl`）

# 使用方法

### 训练模型（8:2，TF-IDF 与 Word2vec 对比；MLP 使用 TF-IDF）
```bash
python src/train_nb.py
python src/train_rf.py
python src/train_nn.py
```

### 使用TKInter完成前端编写，运行 `src/demo.py`，即可使用。页面包括链接输入框、模型选择下拉栏、预测输出框，以及折线图和词云图。

### 选择训练好的模型，再输入 `data/test/test.csv` 或在线凤凰新闻网址，例如：https://sports.ifeng.com/c/8YRoRaeBxRf ，即可输出预测结果。


# 学习路径
参考B站 [黑马程序员Python教程，4天快速入门Python数据挖掘，系统精讲+实战案例](https://www.bilibili.com/video/BV1xt411v7z9/?share_source=copy_web&vd_source=35c49b46a86899e58b3f6414dd834db3)

参考B站 [黑马程序员3天快速入门python机器学习](https://www.bilibili.com/video/BV1nt411r7tj/?share_source=copy_web)

参考B站 [尚硅谷Python爬虫教程小白零基础速通（含python基础+爬虫案例）](https://www.bilibili.com/video/BV1Db4y1m7Ho/?share_source=copy_web&vd_source=35c49b46a86899e58b3f6414dd834db3)

和ChatGPT的辅助完成，仅供大家参考
