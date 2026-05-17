import json
import os
import re

import jieba
import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from torch import nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "train", "weibo_senti_100k.csv")
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
NB_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "nb_metrics.csv")
RF_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "rf_grid.csv")
NN_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "nn_metrics.csv")
TFIDF_MAX_FEATURES = 20000


def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(f.read().split())


def tokenize_text(text, stop_words):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    words = jieba.cut(text)
    return [w for w in words if w and w not in stop_words]


def load_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def pick_columns(df):
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


def select_best_row(results_path, model_name, feature_type):
    df = pd.read_csv(results_path)
    subset = df[(df["model_name"] == model_name) & (df["feature_type"] == feature_type)].copy()
    if subset.empty:
        raise ValueError(f"No result rows found for {model_name}/{feature_type}")

    best_subset = subset[subset["is_best"] == 1].copy()
    if best_subset.empty:
        raise ValueError(f"No best row found for {model_name}/{feature_type}")

    return best_subset.sort_values(
        by=["selection_score", "test_f1_macro", "test_accuracy"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]


def save_nb_models(tokens_list, labels):
    tfidf_best = select_best_row(NB_RESULTS_PATH, "naive_bayes", "tfidf")
    w2v_best = select_best_row(NB_RESULTS_PATH, "naive_bayes", "word2vec")

    texts = [" ".join(tokens) for tokens in tokens_list]

    tfidf_vectorizer = TfidfVectorizer(max_features=50000)
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    nb_tfidf = MultinomialNB(alpha=float(tfidf_best["alpha"]))
    nb_tfidf.fit(X_tfidf, labels)

    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "nb_tfidf_vectorizer.pkl"))
    joblib.dump(nb_tfidf, os.path.join(MODELS_DIR, "nb_tfidf_model.pkl"))
    # 兼容当前 demo.py 的旧命名
    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "naive_bayes_tfidf_vectorizer.pkl"))
    joblib.dump(nb_tfidf, os.path.join(MODELS_DIR, "naive_bayes.pkl"))

    w2v_model = Word2Vec(
        sentences=tokens_list,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        seed=42,
    )
    X_w2v = np.vstack([average_word_vectors(tokens, w2v_model, 100) for tokens in tokens_list])
    nb_w2v = GaussianNB(var_smoothing=float(w2v_best["var_smoothing"]))
    nb_w2v.fit(X_w2v, labels)

    w2v_model.save(os.path.join(MODELS_DIR, "nb_word2vec.model"))
    joblib.dump(nb_w2v, os.path.join(MODELS_DIR, "nb_word2vec_model.pkl"))


def save_rf_models(tokens_list, labels):
    tfidf_best = select_best_row(RF_RESULTS_PATH, "random_forest", "tfidf")
    w2v_best = select_best_row(RF_RESULTS_PATH, "random_forest", "word2vec")

    texts = [" ".join(tokens) for tokens in tokens_list]

    tfidf_vectorizer = TfidfVectorizer(max_features=50000)
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    rf_tfidf = RandomForestClassifier(
        n_estimators=int(tfidf_best["n_estimators"]),
        max_depth=None if pd.isna(tfidf_best["max_depth"]) else int(float(tfidf_best["max_depth"])),
        max_features=tfidf_best["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    rf_tfidf.fit(X_tfidf, labels)

    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "rf_tfidf_vectorizer.pkl"))
    joblib.dump(rf_tfidf, os.path.join(MODELS_DIR, "rf_tfidf_model.pkl"))
    # 兼容当前 demo.py 的旧命名
    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "random_forest_tfidf_vectorizer.pkl"))
    joblib.dump(rf_tfidf, os.path.join(MODELS_DIR, "random_forest.pkl"))

    w2v_model = Word2Vec(
        sentences=tokens_list,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        seed=42,
    )
    X_w2v = np.vstack([average_word_vectors(tokens, w2v_model, 100) for tokens in tokens_list])
    rf_w2v = RandomForestClassifier(
        n_estimators=int(w2v_best["n_estimators"]),
        max_depth=None if pd.isna(w2v_best["max_depth"]) else int(float(w2v_best["max_depth"])),
        max_features=w2v_best["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    rf_w2v.fit(X_w2v, labels)

    w2v_model.save(os.path.join(MODELS_DIR, "rf_word2vec.model"))
    joblib.dump(rf_w2v, os.path.join(MODELS_DIR, "rf_word2vec_model.pkl"))


def save_nn_model(tokens_list, labels):
    best = select_best_row(NN_RESULTS_PATH, "mlp", "tfidf")
    texts = [" ".join(tokens) for tokens in tokens_list]

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, dtype=np.float32)
    X_tfidf = vectorizer.fit_transform(texts)

    config = {
        "input_dim": int(X_tfidf.shape[1]),
        "hidden1": int(best["hidden1"]),
        "hidden2": int(best["hidden2"]),
        "dropout": float(best["dropout"]),
        "lr": float(best["lr"]),
        "epochs": int(best["epochs"]),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlp(
        input_dim=config["input_dim"],
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCEWithLogitsLoss()

    dataset = SparseTfidfDataset(X_tfidf, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(config["epochs"]):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "nn_tfidf_vectorizer.pkl"))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "nn_model.pt"))
    with open(os.path.join(MODELS_DIR, "nn_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    stop_words = load_stopwords(STOPLIST_PATH)
    df = load_dataset(DATA_PATH)
    text_col, label_col = pick_columns(df)

    tokens_list = df[text_col].apply(lambda x: tokenize_text(x, stop_words)).tolist()
    labels = df[label_col].values

    save_nb_models(tokens_list, labels)
    print("Saved Naive Bayes models.")
    save_rf_models(tokens_list, labels)
    print("Saved Random Forest models.")
    save_nn_model(tokens_list, labels)
    print("Saved MLP model.")
    print(f"All final models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
