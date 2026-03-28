import os
import re
import jieba
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "train", "weibo_senti_100k.csv")
STOPLIST_PATH = os.path.join(ROOT_DIR, "resources", "stoplist.txt")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "nn_metrics.csv")
TFIDF_MAX_FEATURES = 20000


def load_stopwords(path):
    # 加载停用词为集合，便于快速查找。
    with open(path, "r", encoding="utf-8") as f:
        return set(f.read().split())


def tokenize_text(text, stop_words):
    # 仅保留中文字符，分词并去除停用词。
    text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    words = jieba.cut(text)
    return [w for w in words if w and w not in stop_words]

# 加载数据集
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


def build_mlp(input_dim, hidden1, hidden2, dropout):
    # 两层 MLP：输入 -> 隐层1 -> 隐层2 -> 输出
    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden2, 1),
    )


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


class SparseTfidfDataset(torch.utils.data.Dataset):
    # 按需把稀疏矩阵行转换为 dense，避免整体内存爆炸。
    def __init__(self, X_sparse, y_array):
        self.X = X_sparse
        self.y = y_array

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].toarray().astype("float32").squeeze(0)
        y = np.array(self.y[idx], dtype="float32")
        return torch.from_numpy(x), torch.from_numpy(y)


def eval_model(model, X_sparse, batch_size, device):
    # 使用批量预测，避免一次性转为 dense。
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, X_sparse.shape[0], batch_size):
            end = min(start + batch_size, X_sparse.shape[0])
            batch_x = X_sparse[start:end].toarray().astype("float32")
            logits = model(torch.from_numpy(batch_x).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_preds.append((probs >= 0.5).astype(int))
    return np.concatenate(all_preds)


def train_one_config(X_train, y_train, X_test, y_test, config):
    # 单次超参数配置训练 + 评估。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型参数移动到制定设备（device）
    model = build_mlp(
        input_dim=X_train.shape[1],
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout=config["dropout"],
    ).to(device)
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    # 优化器Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_dataset = SparseTfidfDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(config["epochs"]):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            logits = model(batch_x)
            # 计算损失值（预测值/真实值）
            loss = criterion(logits, batch_y.unsqueeze(1))
            # 反向传播
            loss.backward()
            #更新参数
            optimizer.step()

    preds = eval_model(model, X_test, batch_size=256, device=device)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    return preds, acc, f1


def main():
    # 写入指标前确保目录存在。
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    stop_words = load_stopwords(STOPLIST_PATH)
    df = load_dataset(DATA_PATH)
    text_col, label_col = pick_columns(df)

    # 先做分词，便于复用在不同特征方案中。
    tokens_list = df[text_col].apply(lambda x: tokenize_text(x, stop_words)).tolist()
    texts = [" ".join(tokens) for tokens in tokens_list]
    y = df[label_col].values

    # 先按 8:2 拆分训练验证集与测试集，测试集仅用于最终评估。
    X_train_val_texts, X_test_texts, y_train_val, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    # 再从训练验证集中划分验证集，用于超参数选择。
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        X_train_val_texts,
        y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=y_train_val,
    )

    # TF-IDF 特征化（仅在训练集拟合）
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, dtype=np.float32)
    X_train = vectorizer.fit_transform(X_train_texts)
    X_val = vectorizer.transform(X_val_texts)
    X_test = vectorizer.transform(X_test_texts)

    # 小范围网格搜索
    grid = [
        {"hidden1": 256, "hidden2": 128, "dropout": 0.2, "lr": 1e-3, "epochs": 5},
        {"hidden1": 256, "hidden2": 128, "dropout": 0.5, "lr": 1e-3, "epochs": 5},
        {"hidden1": 128, "hidden2": 64, "dropout": 0.2, "lr": 5e-4, "epochs": 10},
        {"hidden1": 128, "hidden2": 64, "dropout": 0.5, "lr": 5e-4, "epochs": 10},
    ]

    rows = []
    best = None
    for config in grid:
        preds, acc, f1 = train_one_config(X_train, y_train, X_val, y_val, config)
        rows.append({
            "model_name": "mlp",
            "feature_type": "tfidf",
            "tuning_method": "holdout_validation",
            "selection_score_name": "val_f1_macro",
            "selection_score": f1,
            "val_accuracy": acc,
            "val_f1_macro": f1,
            "test_accuracy": np.nan,
            "test_f1_macro": np.nan,
            "alpha": np.nan,
            "var_smoothing": np.nan,
            "n_estimators": np.nan,
            "max_depth": np.nan,
            "max_features": np.nan,
            "hidden1": config["hidden1"],
            "hidden2": config["hidden2"],
            "dropout": config["dropout"],
            "lr": config["lr"],
            "epochs": config["epochs"],
            "is_best": 0,
        })
        if best is None or f1 > best["val_f1_macro"]:
            best = rows[-1]

        print(
            f"MLP config={config} val_acc={acc:.4f} val_f1_macro={f1:.4f}"
        )
        print_report(y_val, preds, {0: "消极", 1: "积极"})

    # 使用最佳参数在训练集+验证集上重新训练，并在测试集做最终评估。
    best_config = {
        "hidden1": best["hidden1"],
        "hidden2": best["hidden2"],
        "dropout": best["dropout"],
        "lr": best["lr"],
        "epochs": best["epochs"],
    }
    X_train_val = vectorizer.fit_transform(X_train_val_texts)
    X_test = vectorizer.transform(X_test_texts)
    test_preds, test_acc, test_f1 = train_one_config(
        X_train_val, y_train_val, X_test, y_test, best_config
    )

    print(f"Best config on val set: {best_config}")
    print(f"Final test acc={test_acc:.4f} test_f1_macro={test_f1:.4f}")
    print_report(y_test, test_preds, {0: "消极", 1: "积极"})

    for row in rows:
        if (
            row["hidden1"] == best_config["hidden1"]
            and row["hidden2"] == best_config["hidden2"]
            and row["dropout"] == best_config["dropout"]
            and row["lr"] == best_config["lr"]
            and row["epochs"] == best_config["epochs"]
        ):
            row["test_accuracy"] = test_acc
            row["test_f1_macro"] = test_f1
            row["is_best"] = 1
        else:
            row["test_accuracy"] = None
            row["test_f1_macro"] = None

    # 保存指标表
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"Saved metrics to: {RESULTS_PATH}")
    print(f"Best config (by val_f1_macro): {best_config}")


if __name__ == "__main__":
    main()
