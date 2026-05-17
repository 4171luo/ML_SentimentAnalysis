import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import jieba
except ModuleNotFoundError:
    jieba = None

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")
INPUT_PATH = os.path.join(ROOT_DIR, "data", "test", "test.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "docs", "第四章图表", "图4-2_TF-IDF Top-N词项柱状图.png")
SAMPLE_TABLE_PATH = os.path.join(ROOT_DIR, "docs", "特征示例表", "TF-IDF高权重词示例.csv")


def load_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(f.read().split())


def resolve_chinese_font():
    candidate_paths = [
        os.path.join(ROOT_DIR, "simhei.ttf"),
        os.path.join(ROOT_DIR, "resources", "simhei.ttf"),
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def clean_and_segment_text(text, stop_words):
    if jieba is None:
        raise ModuleNotFoundError("jieba is not installed.")
    cleaned_text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    words = jieba.cut(cleaned_text)
    filtered_words = [word for word in words if word and word not in stop_words]
    return " ".join(filtered_words)


def pick_typical_comment(df):
    if "comment_contents" not in df.columns:
        raise ValueError("Dataset must contain a 'comment_contents' column.")

    comments = df["comment_contents"].dropna().astype(str)
    for comment in comments:
        chinese_chars = re.findall(r"[\u4e00-\u9fa5]", comment)
        if len(chinese_chars) >= 8:
            return comment
    raise ValueError("No suitable typical comment found.")


def load_sample_top_terms(path):
    df = load_dataset(path)
    valid_rows = df[df["预处理后文本"].notna()].copy()
    if valid_rows.empty:
        raise ValueError("No valid rows found in the TF-IDF sample table.")

    row = valid_rows.iloc[0]
    terms = []
    values = []
    for idx in range(1, 16):
        term_col = f"高权重词{idx}"
        value_col = f"TF-IDF值{idx}"
        if term_col in row and value_col in row and pd.notna(row[term_col]) and pd.notna(row[value_col]):
            terms.append(str(row[term_col]))
            values.append(float(row[value_col]))

    if not terms:
        raise ValueError("No TF-IDF top terms found in the sample table.")

    return str(row["原始评论"]), str(row["预处理后文本"]), terms[:8], values[:8]


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if jieba is None:
        original_comment, processed_comment, sample_terms, sample_values = load_sample_top_terms(SAMPLE_TABLE_PATH)
        terms = sample_terms[::-1]
        values = sample_values[::-1]
    else:
        stop_words = load_stopwords(STOPLIST_PATH)
        df = load_dataset(INPUT_PATH)

        original_comment = pick_typical_comment(df)
        processed_comment = clean_and_segment_text(original_comment, stop_words)
        if not processed_comment.strip():
            raise ValueError("The selected comment becomes empty after preprocessing.")

        vectorizer = TfidfVectorizer()
        tfidf_vector = vectorizer.fit_transform([processed_comment])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_vector.toarray()[0]

        term_scores = sorted(
            zip(feature_names, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        top_terms = term_scores[:8]
        terms = [item[0] for item in top_terms][::-1]
        values = [item[1] for item in top_terms][::-1]

    font_path = resolve_chinese_font()
    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars = ax.barh(terms, values, color="#5b8ff9")

    ax.set_title("TF-IDF Top-N词项柱状图", fontsize=14, pad=12)
    ax.set_xlabel("TF-IDF值", fontsize=11)
    ax.set_ylabel("特征词", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.01,
        f"典型评论：{original_comment}",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Selected comment: {original_comment}")
    print(f"Processed comment: {processed_comment}")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
