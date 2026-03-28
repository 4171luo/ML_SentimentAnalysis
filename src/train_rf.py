import os
import re
import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "train", "weibo_senti_100k.csv")
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "rf_grid.csv")


def load_stopwords(path):
    # 加载停用词为集合，便于快速查找。
    with open(path, "r", encoding="utf-8") as f:
        return set(f.read().split())


def tokenize_text(text, stop_words):
    # 仅保留中文字符，分词并去除停用词。
    text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    words = jieba.cut(text)
    return [w for w in words if w and w not in stop_words]


def load_dataset(path):
    # 优先使用 UTF-8，失败后回退到 GB 编码。
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def pick_columns(df):
    # 支持常见的文本与标签列名。
    if "review" in df.columns:
        text_col = "review"
    elif "comment_contents" in df.columns:
        text_col = "comment_contents"
    else:
        raise ValueError("Text column not found: expected 'review' or 'comment_contents'")

    if "label" in df.columns:
        label_col = "label"
    elif "sentiment" in df.columns:
        label_col = "sentiment"
    else:
        raise ValueError("Label column not found: expected 'label' or 'sentiment'")

    return text_col, label_col


def average_word_vectors(tokens, model, vector_size):
    # 将词向量平均为固定长度句向量。
    vec = np.zeros(vector_size, dtype="float32")
    count = 0
    for token in tokens:
        # 只累加词表内的向量，避免 OOV 产生错误。
        if token in model.wv:
            vec += model.wv[token]
            count += 1
    if count:
        vec /= count
    return vec


def print_report(y_true, y_pred, label_map):
    # 自定义清晰版报告：逐类 + 总体，附简短注释。
    labels = list(label_map.keys())
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    macro = (
        prec.mean(),
        rec.mean(),
        f1.mean(),
        support.sum(),
    )
    weights = support / support.sum()
    weighted = (
        (prec * weights).sum(),
        (rec * weights).sum(),
        (f1 * weights).sum(),
        support.sum(),
    )

    print("\n" + "=" * 28)
    print("评估报告")
    print("=" * 28)
    for i, label in enumerate(labels):
        name = label_map[label]
        print(
            f"类别 {label}（{name}）: "
            f"precision={prec[i]:.2f}（精准率） "
            f"recall={rec[i]:.2f}（召回率） "
            f"f1={f1[i]:.2f}（综合指标） "
            f"support={support[i]}（该类样本数）"
        )
    print("-" * 28)
    print(
        f"accuracy={acc:.2f}（准确率） support={support.sum()}（总样本数）"
    )
    print(
        f"macro avg: precision={macro[0]:.2f} recall={macro[1]:.2f} f1={macro[2]:.2f} "
        f"support={macro[3]}（对每类等权平均）"
    )
    print(
        f"weighted avg: precision={weighted[0]:.2f} recall={weighted[1]:.2f} "
        f"f1={weighted[2]:.2f} support={weighted[3]}（按样本数加权）"
    )
    print("=" * 28 + "\n")


def main():
    # 写入指标前确保目录存在。
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    stop_words = load_stopwords(STOPLIST_PATH)
    df = load_dataset(DATA_PATH)
    text_col, label_col = pick_columns(df)

    # 先做分词，便于复用在不同特征方案中。
    tokens_list = df[text_col].apply(lambda x: tokenize_text(x, stop_words)).tolist()

    y = df[label_col]

    # 按索引拆分训练/测试，保持分词结果与标签对齐。
    indices = list(range(len(tokens_list)))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )

    # 根据划分索引还原训练/测试的分词列表。
    tokens_train = [tokens_list[i] for i in train_idx]
    tokens_test = [tokens_list[i] for i in test_idx]

    # 先构建随机森林与参数网格，供 TF-IDF 与 Word2vec 复用。
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 20, 40],
        "max_features": ["sqrt", "log2"],
    }
    # k折交叉验证
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results_frames = []

    # TF-IDF 特征
    # 将分词结果拼接为空格分隔的文本供 TF-IDF 使用。
    texts_train = [" ".join(tokens) for tokens in tokens_train]
    texts_test = [" ".join(tokens) for tokens in tokens_test]
    vectorizer = TfidfVectorizer(max_features=50000)
    X_train_tfidf = vectorizer.fit_transform(texts_train)
    X_test_tfidf = vectorizer.transform(texts_test)

    # 使用交叉验证搜索 TF-IDF 下的最优参数。
    grid_tfidf = GridSearchCV(
        rf_model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_tfidf.fit(X_train_tfidf, y_train)
    best_tfidf = grid_tfidf.best_estimator_
    print("RF TF-IDF Best Params:", grid_tfidf.best_params_)
    pred_tfidf = best_tfidf.predict(X_test_tfidf)
    acc_tfidf = accuracy_score(y_test, pred_tfidf)
    f1_tfidf = f1_score(y_test, pred_tfidf, average="macro")
    print(f"RF TF-IDF test acc={acc_tfidf:.4f} f1_macro={f1_tfidf:.4f}")
    print_report(y_test, pred_tfidf, {0: "消极", 1: "积极"})

    results_tfidf = pd.DataFrame(grid_tfidf.cv_results_)
    for _, row in results_tfidf.iterrows():
        is_best = int(row["rank_test_score"] == 1)
        results_frames.append({
            "model_name": "random_forest",
            "feature_type": "tfidf",
            "tuning_method": "grid_search_cv",
            "selection_score_name": "cv_accuracy",
            "selection_score": row["mean_test_score"],
            "val_accuracy": np.nan,
            "val_f1_macro": np.nan,
            "test_accuracy": acc_tfidf if is_best else np.nan,
            "test_f1_macro": f1_tfidf if is_best else np.nan,
            "alpha": np.nan,
            "var_smoothing": np.nan,
            "n_estimators": row["param_n_estimators"],
            "max_depth": row["param_max_depth"],
            "max_features": row["param_max_features"],
            "hidden1": np.nan,
            "hidden2": np.nan,
            "dropout": np.nan,
            "lr": np.nan,
            "epochs": np.nan,
            "is_best": is_best,
        })

    # Word2vec 平均向量特征（稠密向量）
    # 仅使用训练集训练 Word2vec，避免验证集信息泄露。
    w2v_model = Word2Vec(
        sentences=tokens_train,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        seed=42,
    )
    # 将每条评论转换为平均词向量。
    X_train_w2v = np.vstack([average_word_vectors(t, w2v_model, 100) for t in tokens_train])
    X_test_w2v = np.vstack([average_word_vectors(t, w2v_model, 100) for t in tokens_test])

    # 使用交叉验证搜索 Word2vec 下的最优参数。
    grid_w2v = GridSearchCV(
        rf_model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_w2v.fit(X_train_w2v, y_train)
    best_w2v = grid_w2v.best_estimator_
    print("RF W2V Best Params:", grid_w2v.best_params_)
    pred_w2v = best_w2v.predict(X_test_w2v)
    acc_w2v = accuracy_score(y_test, pred_w2v)
    f1_w2v = f1_score(y_test, pred_w2v, average="macro")
    print(f"RF W2V test acc={acc_w2v:.4f} f1_macro={f1_w2v:.4f}")
    print_report(y_test, pred_w2v, {0: "消极", 1: "积极"})

    results_w2v = pd.DataFrame(grid_w2v.cv_results_)
    for _, row in results_w2v.iterrows():
        is_best = int(row["rank_test_score"] == 1)
        results_frames.append({
            "model_name": "random_forest",
            "feature_type": "word2vec",
            "tuning_method": "grid_search_cv",
            "selection_score_name": "cv_accuracy",
            "selection_score": row["mean_test_score"],
            "val_accuracy": np.nan,
            "val_f1_macro": np.nan,
            "test_accuracy": acc_w2v if is_best else np.nan,
            "test_f1_macro": f1_w2v if is_best else np.nan,
            "alpha": np.nan,
            "var_smoothing": np.nan,
            "n_estimators": row["param_n_estimators"],
            "max_depth": row["param_max_depth"],
            "max_features": row["param_max_features"],
            "hidden1": np.nan,
            "hidden2": np.nan,
            "dropout": np.nan,
            "lr": np.nan,
            "epochs": np.nan,
            "is_best": is_best,
        })

    # 合并不同特征的网格结果，统一输出便于后续统计。
    pd.DataFrame(results_frames).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"Saved grid results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
