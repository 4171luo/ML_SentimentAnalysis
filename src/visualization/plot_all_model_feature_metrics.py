import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from plot_algorithm_summary_metrics import (
    DATA_PATH,
    NB_RESULTS_PATH,
    NN_RESULTS_PATH,
    RF_RESULTS_PATH,
    ROOT_DIR,
    STOPLIST_PATH,
    average_word_vectors,
    calc_metrics,
    load_dataset,
    load_stopwords,
    setup_font,
    tokenize_text,
    train_mlp_and_predict,
)


OUT_DIR = os.path.join(ROOT_DIR, "docs", "第四章图表")
CSV_PATH = os.path.join(ROOT_DIR, "results", "all_model_feature_metrics.csv")
METRICS_FIG = os.path.join(OUT_DIR, "图4-10_五种模型特征组合Accuracy_Recall_F1对比图.png")
PRECISION_FIG = os.path.join(OUT_DIR, "图4-9_五种模型特征组合Precision对比图.png")


def select_best_row(results_path, model_name, feature_type):
    df = pd.read_csv(results_path)
    subset = df[
        (df["model_name"] == model_name)
        & (df["feature_type"] == feature_type)
        & (df["is_best"] == 1)
    ].copy()
    subset = subset.dropna(subset=["test_accuracy", "test_f1_macro"])
    if subset.empty:
        raise ValueError(f"No best row found for {model_name}/{feature_type}.")
    return subset.sort_values(
        by=["test_f1_macro", "test_accuracy", "selection_score"],
        ascending=[False, False, False],
    ).iloc[0]


def build_tfidf_features(texts_train, texts_test, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, dtype=np.float32)
    return vectorizer.fit_transform(texts_train), vectorizer.transform(texts_test)


def build_word2vec_features(tokens_train, tokens_test):
    w2v_model = Word2Vec(
        sentences=tokens_train,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        seed=42,
    )
    X_train = np.vstack([average_word_vectors(tokens, w2v_model, 100) for tokens in tokens_train])
    X_test = np.vstack([average_word_vectors(tokens, w2v_model, 100) for tokens in tokens_test])
    return X_train, X_test


def add_display_labels(df):
    df = df.copy()
    df["模型特征组合"] = df["算法"] + "\n(" + df["特征"] + ")"
    return df


def plot_precision(df):
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    colors = ["#5b8ff9", "#5ad8a6", "#f6bd16", "#e8684a", "#6dc8ec"]
    bars = ax.bar(df["模型特征组合"], df["Precision"], color=colors)
    ax.set_title("五种模型特征组合Precision对比图", fontsize=14, pad=12)
    ax.set_xlabel("模型-特征组合", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_ylim(0.78, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(PRECISION_FIG, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(df):
    metrics = ["Accuracy", "Recall", "F1"]
    x = np.arange(len(df))
    width = 0.24
    colors = ["#5b8ff9", "#5ad8a6", "#f6bd16"]

    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
    for idx, metric in enumerate(metrics):
        values = df[metric].astype(float).values
        bars = ax.bar(x + (idx - 1) * width, values, width=width, label=metric, color=colors[idx])
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("五种模型特征组合Accuracy、Recall和F1分数对比图", fontsize=14, pad=12)
    ax.set_xlabel("模型-特征组合", fontsize=11)
    ax.set_ylabel("指标值", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df["模型特征组合"])
    ax.set_ylim(0.78, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(METRICS_FIG, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
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

    nb_tfidf_best = select_best_row(NB_RESULTS_PATH, "naive_bayes", "tfidf")
    X_train, X_test = build_tfidf_features(texts_train, texts_test, max_features=50000)
    nb_tfidf = MultinomialNB(alpha=float(nb_tfidf_best["alpha"]))
    nb_tfidf.fit(X_train, y_train)
    rows.append(calc_metrics("朴素贝叶斯", "TF-IDF", y_test, nb_tfidf.predict(X_test)))

    nb_w2v_best = select_best_row(NB_RESULTS_PATH, "naive_bayes", "word2vec")
    X_train, X_test = build_word2vec_features(tokens_train, tokens_test)
    nb_w2v = GaussianNB(var_smoothing=float(nb_w2v_best["var_smoothing"]))
    nb_w2v.fit(X_train, y_train)
    rows.append(calc_metrics("朴素贝叶斯", "Word2Vec", y_test, nb_w2v.predict(X_test)))

    rf_tfidf_best = select_best_row(RF_RESULTS_PATH, "random_forest", "tfidf")
    X_train, X_test = build_tfidf_features(texts_train, texts_test, max_features=50000)
    rf_tfidf = RandomForestClassifier(
        n_estimators=int(rf_tfidf_best["n_estimators"]),
        max_depth=None if pd.isna(rf_tfidf_best["max_depth"]) else int(float(rf_tfidf_best["max_depth"])),
        max_features=rf_tfidf_best["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    rf_tfidf.fit(X_train, y_train)
    rows.append(calc_metrics("随机森林", "TF-IDF", y_test, rf_tfidf.predict(X_test)))

    rf_w2v_best = select_best_row(RF_RESULTS_PATH, "random_forest", "word2vec")
    X_train, X_test = build_word2vec_features(tokens_train, tokens_test)
    rf_w2v = RandomForestClassifier(
        n_estimators=int(rf_w2v_best["n_estimators"]),
        max_depth=None if pd.isna(rf_w2v_best["max_depth"]) else int(float(rf_w2v_best["max_depth"])),
        max_features=rf_w2v_best["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    rf_w2v.fit(X_train, y_train)
    rows.append(calc_metrics("随机森林", "Word2Vec", y_test, rf_w2v.predict(X_test)))

    nn_best = select_best_row(NN_RESULTS_PATH, "mlp", "tfidf")
    X_train, X_test = build_tfidf_features(texts_train, texts_test, max_features=20000)
    nn_preds = train_mlp_and_predict(X_train, y_train, X_test, nn_best)
    rows.append(calc_metrics("MLP神经网络", "TF-IDF", y_test, nn_preds))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(CSV_PATH, index=False, encoding="utf_8_sig")
    display_df = add_display_labels(summary_df)
    plot_precision(display_df)
    plot_metrics(display_df)

    print(f"Saved summary metrics to: {CSV_PATH}")
    print(f"Saved precision figure to: {PRECISION_FIG}")
    print(f"Saved metrics figure to: {METRICS_FIG}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
