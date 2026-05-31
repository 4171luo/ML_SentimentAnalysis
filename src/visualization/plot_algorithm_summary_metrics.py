import os
import re

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from torch import nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "train", "weibo_senti_100k.csv")
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")
NB_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "nb_metrics.csv")
RF_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "rf_grid.csv")
NN_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "nn_metrics.csv")
OUT_DIR = os.path.join(ROOT_DIR, "docs", "第四章图表")
CSV_PATH = os.path.join(ROOT_DIR, "results", "algorithm_summary_metrics.csv")
PRECISION_FIG = os.path.join(OUT_DIR, "图4-9_不同算法测试集Precision对比图.png")
METRICS_FIG = os.path.join(OUT_DIR, "图4-10_不同算法Accuracy_Recall_F1对比图.png")


def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(f.read().split())


def load_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def tokenize_text(text, stop_words):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    words = jieba.cut(text)
    return [word for word in words if word and word not in stop_words]


def average_word_vectors(tokens, model, vector_size):
    vec = np.zeros(vector_size, dtype="float32")
    count = 0
    for token in tokens:
        if token in model.wv:
            vec += model.wv[token]
            count += 1
    if count:
        vec /= count
    return vec


def build_mlp(input_dim, hidden1, hidden2, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden2, 1),
    )


class SparseTfidfDataset(torch.utils.data.Dataset):
    def __init__(self, X_sparse, y_array):
        self.X = X_sparse
        self.y = y_array

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].toarray().astype("float32").squeeze(0)
        y = np.array(self.y[idx], dtype="float32")
        return torch.from_numpy(x), torch.from_numpy(y)


def train_mlp_and_predict(X_train, y_train, X_test, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlp(
        input_dim=X_train.shape[1],
        hidden1=int(config["hidden1"]),
        hidden2=int(config["hidden2"]),
        dropout=float(config["dropout"]),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))

    train_dataset = SparseTfidfDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(int(config["epochs"])):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X_test.shape[0], 256):
            end = min(start + 256, X_test.shape[0])
            batch_x = X_test[start:end].toarray().astype("float32")
            logits = model(torch.from_numpy(batch_x).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            preds.append((probs >= 0.5).astype(int))
    return np.concatenate(preds)


def select_best_row(results_path, model_name):
    df = pd.read_csv(results_path)
    subset = df[(df["model_name"] == model_name) & (df["is_best"] == 1)].copy()
    subset = subset.dropna(subset=["test_accuracy", "test_f1_macro"])
    if subset.empty:
        raise ValueError(f"No best row found for {model_name}.")
    return subset.sort_values(
        by=["test_f1_macro", "test_accuracy", "selection_score"],
        ascending=[False, False, False],
    ).iloc[0]


def calc_metrics(model_label, feature_type, y_true, y_pred):
    return {
        "算法": model_label,
        "特征": feature_type,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


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


def plot_precision(df):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.bar(df["算法"], df["Precision"], color=["#5b8ff9", "#5ad8a6", "#f6bd16"])
    ax.set_title("不同算法测试集Precision对比图", fontsize=14, pad=12)
    ax.set_xlabel("算法", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_ylim(0.78, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PRECISION_FIG, bbox_inches="tight")


def plot_metrics(df):
    metrics = ["Accuracy", "Recall", "F1"]
    x = np.arange(len(df))
    width = 0.24
    colors = ["#5b8ff9", "#5ad8a6", "#f6bd16"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    for idx, metric in enumerate(metrics):
        values = df[metric].astype(float).values
        bars = ax.bar(x + (idx - 1) * width, values, width=width, label=metric, color=colors[idx])
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("不同算法Accuracy、Recall和F1分数对比图", fontsize=14, pad=12)
    ax.set_xlabel("算法", fontsize=11)
    ax.set_ylabel("指标值", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df["算法"])
    ax.set_ylim(0.78, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(METRICS_FIG, bbox_inches="tight")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    setup_font()

    stop_words = load_stopwords(STOPLIST_PATH)
    df = load_dataset(DATA_PATH)
    tokens_list = df["review"].apply(lambda text: tokenize_text(text, stop_words)).tolist()
    labels = df["label"].values

    indices = list(range(len(tokens_list)))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    tokens_train = [tokens_list[i] for i in train_idx]
    tokens_test = [tokens_list[i] for i in test_idx]
    texts_train = [" ".join(tokens) for tokens in tokens_train]
    texts_test = [" ".join(tokens) for tokens in tokens_test]

    rows = []

    nb_best = select_best_row(NB_RESULTS_PATH, "naive_bayes")
    if nb_best["feature_type"] == "tfidf":
        vectorizer = TfidfVectorizer(max_features=50000)
        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)
        model = MultinomialNB(alpha=float(nb_best["alpha"]))
        feature_name = "TF-IDF"
    else:
        w2v_model = Word2Vec(sentences=tokens_train, vector_size=100, window=5, min_count=2, workers=4, seed=42)
        X_train = np.vstack([average_word_vectors(tokens, w2v_model, 100) for tokens in tokens_train])
        X_test = np.vstack([average_word_vectors(tokens, w2v_model, 100) for tokens in tokens_test])
        model = GaussianNB(var_smoothing=float(nb_best["var_smoothing"]))
        feature_name = "Word2Vec"
    model.fit(X_train, y_train)
    rows.append(calc_metrics("朴素贝叶斯", feature_name, y_test, model.predict(X_test)))

    rf_best = select_best_row(RF_RESULTS_PATH, "random_forest")
    if rf_best["feature_type"] == "tfidf":
        vectorizer = TfidfVectorizer(max_features=50000)
        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)
        feature_name = "TF-IDF"
    else:
        w2v_model = Word2Vec(sentences=tokens_train, vector_size=100, window=5, min_count=2, workers=4, seed=42)
        X_train = np.vstack([average_word_vectors(tokens, w2v_model, 100) for tokens in tokens_train])
        X_test = np.vstack([average_word_vectors(tokens, w2v_model, 100) for tokens in tokens_test])
        feature_name = "Word2Vec"
    rf_model = RandomForestClassifier(
        n_estimators=int(rf_best["n_estimators"]),
        max_depth=None if pd.isna(rf_best["max_depth"]) else int(float(rf_best["max_depth"])),
        max_features=rf_best["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rows.append(calc_metrics("随机森林", feature_name, y_test, rf_model.predict(X_test)))

    nn_best = select_best_row(NN_RESULTS_PATH, "mlp")
    vectorizer = TfidfVectorizer(max_features=20000, dtype=np.float32)
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    nn_preds = train_mlp_and_predict(X_train, y_train, X_test, nn_best)
    rows.append(calc_metrics("MLP神经网络", "TF-IDF", y_test, nn_preds))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(CSV_PATH, index=False, encoding="utf_8_sig")
    plot_precision(summary_df)
    plot_metrics(summary_df)

    print(f"Saved summary metrics to: {CSV_PATH}")
    print(f"Saved precision figure to: {PRECISION_FIG}")
    print(f"Saved metrics figure to: {METRICS_FIG}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
