import os
import re

import jieba
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "train", "weibo_senti_100k.csv")
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")
VECTORIZER_PATH = os.path.join(ROOT_DIR, "models", "nb_tfidf_vectorizer.pkl")
OUT_DIR = os.path.join(ROOT_DIR, "docs", "特征示例表")
COORD_CSV_PATH = os.path.join(OUT_DIR, "TF-IDF稀疏矩阵坐标表示.csv")
ANNOTATION_CSV_PATH = os.path.join(OUT_DIR, "TF-IDF稀疏矩阵标注说明.csv")
PPT_TXT_PATH = os.path.join(OUT_DIR, "TF-IDF稀疏矩阵PPT文本.txt")


def load_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(f.read().split())


def clean_and_segment(text, stop_words):
    cleaned = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    words = jieba.cut(cleaned)
    return [word for word in words if word and word not in stop_words]


def select_samples(df, stop_words, vectorizer, sample_count=3):
    selected = []
    feature_names = vectorizer.get_feature_names_out()

    for original_idx, row in df.iterrows():
        review = str(row["review"])
        chinese_len = len(re.findall(r"[\u4e00-\u9fa5]", review))
        if chinese_len < 25 or chinese_len > 90:
            continue

        tokens = clean_and_segment(review, stop_words)
        if len(tokens) < 8:
            continue

        processed_text = " ".join(tokens)
        tfidf_row = vectorizer.transform([processed_text])
        if tfidf_row.nnz < 8:
            continue

        top_pairs = sorted(
            zip(tfidf_row.indices, tfidf_row.data),
            key=lambda item: item[1],
            reverse=True,
        )
        top_words = [feature_names[index] for index, _ in top_pairs[:5]]
        if len(set(top_words)) < 5:
            continue

        selected.append(
            {
                "original_idx": int(original_idx),
                "label": int(row["label"]),
                "review": review,
                "processed_text": processed_text,
            }
        )
        if len(selected) == sample_count:
            break

    if len(selected) < sample_count:
        raise ValueError("Not enough suitable samples found for sparse matrix display.")
    return selected


def build_outputs(samples, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    texts = [sample["processed_text"] for sample in samples]
    matrix = vectorizer.transform(texts)

    coord_rows = []
    annotation_rows = []
    ppt_lines = [
        f"TF-IDF稀疏矩阵维度：{matrix.shape[0]} × {matrix.shape[1]}",
        f"非零元素数量：{matrix.nnz}",
        "",
        "坐标格式片段：",
    ]

    for display_row_idx, sample in enumerate(samples):
        row = matrix[display_row_idx]
        sorted_by_weight = sorted(
            zip(row.indices, row.data),
            key=lambda item: item[1],
            reverse=True,
        )

        for col_idx, value in sorted_by_weight[:8]:
            word = feature_names[col_idx]
            coord_rows.append(
                {
                    "矩阵行号": display_row_idx,
                    "词项列号": int(col_idx),
                    "TF-IDF值": round(float(value), 6),
                }
            )
            annotation_rows.append(
                {
                    "矩阵行号": display_row_idx,
                    "原始样本序号": sample["original_idx"],
                    "情感标签": sample["label"],
                    "情感含义": "积极" if sample["label"] == 1 else "消极",
                    "原始评论": sample["review"],
                    "预处理后文本": sample["processed_text"],
                    "词项列号": int(col_idx),
                    "对应词项": word,
                    "TF-IDF值": round(float(value), 6),
                    "含义说明": f"第{display_row_idx}行样本中，词汇表第{int(col_idx)}列词项“{word}”的TF-IDF值为{float(value):.4f}",
                }
            )
            ppt_lines.append(f"({display_row_idx}, {int(col_idx)})    {float(value):.6f}    # {word}")

    return pd.DataFrame(coord_rows), pd.DataFrame(annotation_rows), "\n".join(ppt_lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stop_words = load_stopwords(STOPLIST_PATH)
    df = load_dataset(DATA_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    samples = select_samples(df, stop_words, vectorizer)
    coord_df, annotation_df, ppt_text = build_outputs(samples, vectorizer)

    coord_df.to_csv(COORD_CSV_PATH, index=False, encoding="utf_8_sig")
    annotation_df.to_csv(ANNOTATION_CSV_PATH, index=False, encoding="utf_8_sig")
    with open(PPT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(ppt_text)

    print(f"Saved coordinate CSV: {COORD_CSV_PATH}")
    print(f"Saved annotation CSV: {ANNOTATION_CSV_PATH}")
    print(f"Saved PPT text: {PPT_TXT_PATH}")
    print(ppt_text)


if __name__ == "__main__":
    main()
