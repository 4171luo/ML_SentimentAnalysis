import os
import json
import pandas as pd
import urllib.request
import urllib.parse
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_TEST_DIR = os.path.join(ROOT_DIR, "data", "test")

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
    else:
        return filename

if __name__ == '__main__':
    # 读取用户输入的新闻链接
    or_url = input('输入URL: ')
    url = get_lasteleven(or_url)
    print("正在获取评论数据，请稍后。。。\n")
    # 伪装浏览器请求，避免被简单反爬
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
        # 整理为结构化字段，便于后续分析
        comments_data = [
            {
                'uname': comment['uname'],
                'create_time': convert_timestamp(comment['create_time']),
                'comment_contents': comment['comment_contents']
            }
            for comment in comments
        ]
        # 转换为 DataFrame 后追加到总表
        df = pd.DataFrame(comments_data)
        all_comments = pd.concat([all_comments, df], ignore_index=True)
        page += 1  # 增加页码
    # 循环结束后，将累积的数据保存到 CSV 文件
    os.makedirs(DATA_TEST_DIR, exist_ok=True)
    output_path = os.path.join(DATA_TEST_DIR, "test.csv")
    unique_filename = get_unique_filename(output_path)
    all_comments.to_csv(unique_filename, index=False, encoding='utf_8_sig')
    print(f'所有评论信息已保存到csv文件：{unique_filename}')
