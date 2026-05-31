import os
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd

try:
    import jieba
except ModuleNotFoundError:
    jieba = None


BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")
DATA_PATH = os.path.join(ROOT_DIR, "data", "train", "weibo_senti_100k.csv")
VECTORIZER_PATH = os.path.join(ROOT_DIR, "models", "nb_tfidf_vectorizer.pkl")
OUTPUT_DIR = os.path.join(ROOT_DIR, "docs", "第四章图表")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "图4-2_TF-IDF Top-N词项柱状图.png")
TABLE_PATH = os.path.join(ROOT_DIR, "docs", "特征示例表", "TF-IDF_TopN词项图数据.csv")
TOP_N = 10
MAX_SCAN_ROWS = 1200


def load_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(f.read().split())


def clean_and_segment_text(text, stop_words):
    if jieba is None:
        raise ModuleNotFoundError("jieba is not installed.")
    cleaned_text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    words = jieba.cut(cleaned_text)
    return [word for word in words if word and word not in stop_words]


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


def setup_font():
    font_path = resolve_chinese_font()
    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


def get_top_terms_from_row(vectorizer, tfidf_row):
    feature_names = vectorizer.get_feature_names_out()
    pairs = sorted(
        zip(tfidf_row.indices, tfidf_row.data),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(feature_names[index], float(value)) for index, value in pairs[:TOP_N]]


def select_display_sample(df, stop_words, vectorizer):
    candidates = []
    scan_df = df.head(MAX_SCAN_ROWS)

    prepared_rows = []
    for row_idx, row in scan_df.iterrows():
        review = str(row["review"])
        chinese_len = len(re.findall(r"[\u4e00-\u9fa5]", review))
        if chinese_len < 35 or chinese_len > 100:
            continue

        tokens = clean_and_segment_text(review, stop_words)
        if len(tokens) < TOP_N:
            continue

        processed_text = " ".join(tokens)
        prepared_rows.append(
            {
                "row_idx": row_idx,
                "label": int(row["label"]),
                "review": review,
                "processed_text": processed_text,
                "tokens": tokens,
            }
        )

    if not prepared_rows:
        raise ValueError("No suitable text sample found before TF-IDF transform.")

    tfidf_matrix = vectorizer.transform([item["processed_text"] for item in prepared_rows])
    for item_idx, item in enumerate(prepared_rows):
        top_terms = get_top_terms_from_row(vectorizer, tfidf_matrix[item_idx])
        if len(top_terms) < TOP_N:
            continue

        top_values = [value for _, value in top_terms]
        max_value = max(top_values)
        min_value = min(top_values)
        if max_value >= 0.95 or min_value <= 0:
            continue

        candidates.append(
            {
                "row_idx": item["row_idx"],
                "label": item["label"],
                "review": item["review"],
                "processed_text": item["processed_text"],
                "tokens": item["tokens"],
                "top_terms": top_terms,
                "score": len(top_terms) * 10 + len(item["tokens"]) + (max_value - min_value),
            }
        )

    if not candidates:
        raise ValueError("No suitable sample found for TF-IDF Top-N plotting.")

    return sorted(candidates, key=lambda item: item["score"], reverse=True)[0]


def save_table(sample):
    data = [
        {
            "样本序号": sample["row_idx"],
            "情感标签": sample["label"],
            "情感含义": "积极" if sample["label"] == 1 else "消极",
            "原始评论": sample["review"],
            "预处理后文本": sample["processed_text"],
            "词项": term,
            "TF-IDF值": round(value, 6),
        }
        for term, value in sample["top_terms"]
    ]
    pd.DataFrame(data).to_csv(TABLE_PATH, index=False, encoding="utf_8_sig")


def plot_top_terms(sample):
    setup_font()
    terms = [term for term, _ in sample["top_terms"]][::-1]
    values = [value for _, value in sample["top_terms"]][::-1]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    bars = ax.barh(terms, values, color="#4e79a7")

    ax.set_title("TF-IDF Top-N词项柱状图", fontsize=14, pad=12)
    ax.set_xlabel("TF-IDF值", fontsize=11)
    ax.set_ylabel("特征词", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.18)

    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.012,
        f"样本{sample['row_idx']}：{sample['review']}",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)

    stop_words = load_stopwords(STOPLIST_PATH)
    df = load_dataset(DATA_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    sample = select_display_sample(df, stop_words, vectorizer)
    save_table(sample)
    plot_top_terms(sample)

    print(f"Selected sample index: {sample['row_idx']}")
    print(f"Label: {sample['label']}")
    print(f"Original comment: {sample['review']}")
    print(f"Processed text: {sample['processed_text']}")
    print(f"Saved table to: {TABLE_PATH}")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
