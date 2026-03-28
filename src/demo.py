import re
import os
import glob
import json
import jieba
import joblib
import numpy as np
import urllib.parse
import torch
from PIL import ImageTk
from wordcloud import WordCloud
import pandas as pd
import tkinter as tk
import urllib.request
from tkinter import ttk
from tkinter import font
from datetime import datetime
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torch import nn

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_TEST_DIR = os.path.join(ROOT_DIR, "data", "test")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")

old_chart = None
# 在程序开始时加载停用词列表
with open(STOPLIST_PATH, 'r', encoding='UTF-8') as stop_path:
    stop_words = set(stop_path.read().split())

# 获取URL的后十一位字符的函数
def get_lasteleven(n_url):
    # 返回URL的最后11个字符（凤凰新闻评论接口需要）
    return n_url[-11:]

# 调用函数使得每一页都有自己的请求方法
def create_page(page, url):
    # 构造分页请求参数
    data = {
        'p': page,
        'pageSize': 20,
    }
    # 拼接评论接口地址
    base_url = 'https://comment.ifeng.com/get.php?orderby=create_time&docUrl=ucms_' + url + '&format=js&job=1&'
    data_url = base_url + urllib.parse.urlencode(data)
    return data_url

# 抓取评论信息的函数
def fetch_comments(url, headers):
    # 构造请求并发送
    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request)
    html = response.read().decode('utf-8')
    # 从响应中提取JSON字符串（接口返回 JS 赋值形式）
    json_str = html.split('=', 1)[1].strip(';')
    # 解析JSON数据并返回评论列表
    data = json.loads(json_str)
    return data['comments']

# 将Unix时间戳转换为可读日期和时间的函数
def convert_timestamp(timestamp):
    # 将Unix时间戳转换为UTC时间
    utc_time = datetime.utcfromtimestamp(int(timestamp))
    # 格式化时间字符串（按需调整时区）
    local_time = utc_time.strftime('%Y-%m-%d %H:%M:%S')
    return local_time

# 检查文件是否存在并返回一个不重名的文件名的函数
def get_unique_filename(filename):
    # 如果文件已存在，则在文件名中添加当前时间戳
    if os.path.exists(filename):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename_without_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        new_filename = f"{filename_without_ext}_{timestamp}{extension}"
        return new_filename
    return filename

# 获取指定目录下最新的CSV文件
def get_latest_csv(directory_path):
    # 获取目录下所有CSV文件
    list_of_files = glob.glob(os.path.join(directory_path, '*.csv'))
    # 按创建时间排序，获取最新的文件
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# 加载数据
def load_data(file_path):
    # 读取 CSV，自动兼容不同编码
    try:
        dataset = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            dataset = pd.read_csv(file_path, encoding='gb18030')
        except UnicodeDecodeError:
            return None, "无法读取数据集，请检查文件编码。"
    return dataset, None


# 加载模型并预测
def load_model_and_predict(model_path, X_test_vect):
    # 加载模型并返回预测
    model = joblib.load(model_path)
    return model.predict(X_test_vect)


def build_mlp(input_dim, hidden1, hidden2, dropout):
    # 与训练脚本保持一致的两层 MLP 结构。
    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden2, 1),
    )


def load_mlp_and_predict(config_path, model_path, X_test_vect):
    # 读取网络结构配置，重建模型并加载训练好的权重。
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlp(
        input_dim=config["input_dim"],
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout=config["dropout"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if hasattr(X_test_vect, "toarray"):
        X_test_array = X_test_vect.toarray().astype(np.float32)
    else:
        X_test_array = np.asarray(X_test_vect, dtype=np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_array).to(device))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return (probs >= 0.5).astype(int)

def spider_main(n_url):
    # 爬取指定新闻 URL 的评论并保存到 data/test
    print("正在获取评论数据，请稍后。。。\n")
    url = get_lasteleven(n_url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0',
        'Cookie': 'userid=1710776144523_vy8wjt1321; prov=cn0791; city=0791; weather_city=jx_nc; sid=A8CC8F3965C4183C0100AAED43B901EE; IF_TIME=1710776362857397; IF_USER=%E5%87%A4%E5%87%B0%E7%BD%91%E5%8F%8BEB78bAP; IF_REAL=1; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2218e524678e79e-0a7efd23795df7-4c657b58-2073600-18e524678e8189d%22%2C%22first_id%22%3A%22%22%2C%22props%22%3A%7B%7D%2C%22%24device_id%22%3A%2218e524678e79e-0a7efd23795df7-4c657b58-2073600-18e524678e8189d%22%7D; region_ip=183.217.29.x; region_ver=1.2'
    }
    # 在循环外部创建一个空的 DataFrame，用于累积多页评论
    all_comments = pd.DataFrame()
    page = 1
    while True:
        # 生成分页链接并抓取当前页
        page_url = create_page(page, url)
        comments = fetch_comments(page_url, headers)
        # 如果返回为空，表示到达最后一页
        if not comments:
            break
        comments_data = [{
            'uname': comment['uname'],
            'create_time': convert_timestamp(comment['create_time']),
            'comment_contents': comment['comment_contents']
        } for comment in comments]
        # 转换为 DataFrame 后追加到总表
        df = pd.DataFrame(comments_data)
        all_comments = pd.concat([all_comments, df], ignore_index=True)
        page += 1  # 增加页码
    # 循环结束后，将累积的数据保存到 CSV 文件
    os.makedirs(DATA_TEST_DIR, exist_ok=True)
    output_path = os.path.join(DATA_TEST_DIR, 'test.csv')
    unique_filename = get_unique_filename(output_path)
    all_comments.to_csv(unique_filename, index=False, encoding='utf_8_sig')
    print(f'所有评论信息已保存到csv文件：{unique_filename}')

def clean_and_segment_text(input_text):
    # 文本清洗 + 分词 + 去停用词
    print("正在清洗和分词文本。。。")
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', input_text)  # 只保留中文字符
    words = jieba.cut(cleaned_text)
    global stop_words
    words = filter(lambda w: w not in stop_words and w.strip(), words)
    return ' '.join(words)

# 主函数：分析情感
def analyze_sentiment():
    # 内部函数：获取URL并分析情感
    def fetch_and_analyze():
        # 读取输入内容：URL 或 CSV 路径
        n_url = url_entry.get()
        if os.path.isfile(n_url):
            try:
                # 加载文件并获取'comment_contents'列
                test_data = pd.read_csv(n_url)
                X_new_test = test_data['comment_contents']
                X_new_test_tokenized = X_new_test.apply(clean_and_segment_text)
            except Exception as e:
                messagebox.showerror("错误", f"加载文件时出错: {e}")
        else:
            if not n_url:
                messagebox.showerror("错误", "请输入有效的 URL 或文件名。")
                return
            # 输入的是 URL，先爬取评论生成 CSV
            spider_main(n_url)
            latest_csv = get_latest_csv(DATA_TEST_DIR)
            test_data, error_msg = load_data(latest_csv)
            if error_msg:
                messagebox.showerror("错误", error_msg)
                return
            X_new_test = test_data['comment_contents']
            X_new_test_tokenized = X_new_test.apply(clean_and_segment_text)
        selected_algorithm = algorithm_combobox.get()
        if selected_algorithm == "朴素贝叶斯":
            vect_path = os.path.join(MODELS_DIR, "nb_tfidf_vectorizer.pkl")
            model_path = os.path.join(MODELS_DIR, "naive_bayes.pkl")
            required_paths = [vect_path, model_path]
        elif selected_algorithm == "随机森林":
            vect_path = os.path.join(MODELS_DIR, "rf_tfidf_vectorizer.pkl")
            model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
            required_paths = [vect_path, model_path]
        elif selected_algorithm == "MLP":
            vect_path = os.path.join(MODELS_DIR, "nn_tfidf_vectorizer.pkl")
            config_path = os.path.join(MODELS_DIR, "nn_config.json")
            model_path = os.path.join(MODELS_DIR, "nn_model.pt")
            required_paths = [vect_path, config_path, model_path]
        else:
            messagebox.showerror("错误", "请选择有效的模型。")
            return

        # 检查当前模型文件是否已训练并保存。
        missing_files = [path for path in required_paths if not os.path.exists(path)]
        if missing_files:
            missing_text = "\n".join(missing_files)
            messagebox.showerror("错误", f"缺少模型文件，请先完成训练并保存：\n{missing_text}")
            return

        vect = joblib.load(vect_path)
        X_new_test_vect = vect.transform(X_new_test_tokenized)
        if selected_algorithm == "MLP":
            new_predictions = load_mlp_and_predict(config_path, model_path, X_new_test_vect)
        else:
            new_predictions = load_model_and_predict(model_path, X_new_test_vect)

        # 预测情感并将结果映射为中文标签
        test_data['predicted_sentiment'] = new_predictions
        test_data['predicted_sentiment'] = (test_data['predicted_sentiment'].map({1: '积极', 0: '消极'}))
        result_text.delete(1.0, tk.END)

        # 输出每条评论的预测结果
        for index, row in test_data.iterrows():
            result_text.insert(tk.END, f"评论: {row['comment_contents']} - 情感: {row['predicted_sentiment']}\n")
        # 绘制情感分析结果
        plot_sentiment_analysis(test_data)
        # 初始化词云图像
        generate_wordcloud(test_data)

    # 函数：绘制情感分析结果
    def plot_sentiment_analysis(data):
        # 将评论时间转换为日期，并按天聚合数量
        data['create_time'] = pd.to_datetime(data['create_time'])
        daily_comment_count = data.groupby(data['create_time'].dt.date).size()
        # 检查是否有旧的图表，如果有，则销毁
        global old_chart
        if old_chart is not None:
            old_chart.get_tk_widget().destroy()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        fig, ax = plt.subplots(figsize=(8, 4))
        # 绘制折线图
        ax.plot(daily_comment_count.index, daily_comment_count.values, marker='o', linestyle='-', color='b')
        ax.set_xlabel('日期')
        ax.set_ylabel('评论数量')
        ax.set_title('每日评论数量变化')
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # 嵌入到 Tkinter 窗口
        chart = FigureCanvasTkAgg(fig, master=root)
        chart.draw()
        chart.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        # 更新全局变量 old_chart 为当前图表
        old_chart = chart

    # 函数：生成词云
    def generate_wordcloud(test_data):
        # 基于评论文本生成词云
        X_new_test = test_data['comment_contents']
        X_new_test_tokenized = X_new_test.apply(clean_and_segment_text)
        # 将分词结果拼接为大字符串
        text = ' '.join(X_new_test_tokenized)
        wordcloud = WordCloud(font_path='simhei.ttf', width=800, height=400, background_color='white').generate(
            text)
        wordcloud_image = wordcloud.to_image()
        wordcloud_image_tk = ImageTk.PhotoImage(image=wordcloud_image)
        label.config(image=wordcloud_image_tk)
        label.image = wordcloud_image_tk

    # Tkinter GUI设置
    root = tk.Tk()
    # 定义字体和大小
    custom_font = font.Font(family='宋体', size=14)  # 您可以根据需要更改字体大小
    root.title("情感分析")
    url_label = tk.Label(root, text="请输入 URL 或 文件名: ")
    url_entry = tk.Entry(root, width=160, font=custom_font)
    algorithm_combobox = ttk.Combobox(root, values=["朴素贝叶斯", "随机森林", "MLP"], font=custom_font)
    result_text = tk.Text(root, height=20, width=160, font=custom_font)
    analyze_button = tk.Button(root, text="分析情感", command=fetch_and_analyze)
    label = tk.Label(root)  # 用于显示词云图像的标签

    # 打包GUI元素
    url_label.pack()
    url_entry.pack()
    algorithm_combobox.set("朴素贝叶斯")
    algorithm_combobox.pack()
    result_text.pack()
    analyze_button.pack()
    # 将词云图像标签放置在右侧
    label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    root.mainloop()

# 调用主函数
if __name__ == "__main__":
    analyze_sentiment()
