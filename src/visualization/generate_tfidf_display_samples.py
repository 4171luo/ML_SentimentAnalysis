import os
import re

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "train", "weibo_senti_100k.csv")
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")
OUTPUT_PATH = os.path.join(ROOT_DIR, "docs", "特征示例表", "TF-IDF展示样本.csv")

MAX_FEATURES = 50000
SAMPLES_PER_LABEL = 5
TOP_TERMS = 5
MIN_RAW_LEN = 20
MAX_RAW_LEN = 90
MIN_UNIQUE_TOKENS = 6
MIN_NONZERO_TERMS = 5


def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(f.read().split())


def load_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def clean_raw_text(text):
    text = str(text).replace("\ufeff", "").strip()
    return re.sub(r"\s+", " ", text)


def tokenize_text(text, stop_words):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    words = jieba.cut(text)
    return [word for word in words if word and word not in stop_words]


def is_readable_sample(raw_text):
    if not (MIN_RAW_LEN <= len(raw_text) <= MAX_RAW_LEN):
        return False
    noisy_patterns = ["http", "回复@", "转发微博", "//@", "【", "】"]
    return not any(pattern in raw_text for pattern in noisy_patterns)


def build_candidate_rows(df, tokens_list, X, feature_names):
    rows = []
    for sample_idx, row in df.iterrows():
        raw_text = clean_raw_text(row["review"])
        tokens = tokens_list[sample_idx]
        tfidf_row = X[sample_idx]

        if not is_readable_sample(raw_text):
            continue
        if len(set(tokens)) < MIN_UNIQUE_TOKENS:
            continue
        if tfidf_row.nnz < MIN_NONZERO_TERMS:
            continue

        weighted_terms = sorted(
            zip(tfidf_row.indices, tfidf_row.data),
            key=lambda item: item[1],
            reverse=True,
        )
        top_terms = [
            (feature_names[term_idx], round(float(value), 6))
            for term_idx, value in weighted_terms[:TOP_TERMS]
        ]
        if len(top_terms) < TOP_TERMS:
            continue

        rows.append(
            {
                "样本序号": int(sample_idx),
                "情感标签": int(row["label"]),
                "情感含义": "积极" if int(row["label"]) == 1 else "消极",
                "原始评论": raw_text,
                "预处理后文本": " ".join(tokens),
                "非零TF-IDF词项数": int(tfidf_row.nnz),
                "_unique_tokens": len(set(tokens)),
                "_raw_len": len(raw_text),
                "_top_terms": top_terms,
            }
        )
    return rows


def select_display_samples(candidate_rows):
    selected = []
    for label in [0, 1]:
        label_rows = [row for row in candidate_rows if row["情感标签"] == label]
        label_rows = sorted(
            label_rows,
            key=lambda row: (
                abs(row["_raw_len"] - 55),
                -row["_unique_tokens"],
                row["样本序号"],
            ),
        )
        selected.extend(label_rows[:SAMPLES_PER_LABEL])
    return sorted(selected, key=lambda row: (row["情感标签"], row["样本序号"]))


def flatten_rows(rows):
    flattened = []
    for row in rows:
        output_row = {
            "样本序号": row["样本序号"],
            "情感标签": row["情感标签"],
            "情感含义": row["情感含义"],
            "原始评论": row["原始评论"],
            "预处理后文本": row["预处理后文本"],
            "非零TF-IDF词项数": row["非零TF-IDF词项数"],
        }
        for idx, (term, value) in enumerate(row["_top_terms"], start=1):
            output_row[f"高权重词{idx}"] = term
            output_row[f"TF-IDF值{idx}"] = value
        flattened.append(output_row)
    return flattened


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    stop_words = load_stopwords(STOPLIST_PATH)
    df = load_dataset(DATA_PATH)
    if "review" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'label' columns.")

    df = df[["label", "review"]].copy()
    df["review"] = df["review"].apply(clean_raw_text)
    tokens_list = df["review"].apply(lambda text: tokenize_text(text, stop_words)).tolist()
    processed_texts = [" ".join(tokens) for tokens in tokens_list]

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()

    candidate_rows = build_candidate_rows(df, tokens_list, X, feature_names)
    selected_rows = select_display_samples(candidate_rows)

    label_counts = pd.Series([row["情感标签"] for row in selected_rows]).value_counts()
    if len(selected_rows) != SAMPLES_PER_LABEL * 2:
        raise ValueError(f"Expected 10 samples, got {len(selected_rows)}.")
    if label_counts.get(0, 0) != SAMPLES_PER_LABEL or label_counts.get(1, 0) != SAMPLES_PER_LABEL:
        raise ValueError("Expected 5 negative and 5 positive samples.")

    output_df = pd.DataFrame(flatten_rows(selected_rows))
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf_8_sig")
    print(f"Saved TF-IDF display samples to: {OUTPUT_PATH}")
    print(f"TF-IDF matrix shape: {X.shape}")


if __name__ == "__main__":
    main()
