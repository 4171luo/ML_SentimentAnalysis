import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULT_PATH = os.path.join(ROOT_DIR, "results", "rf_grid.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "docs", "第四章图表", "图4-4_随机森林TF-IDF参数搜索结果图.png")


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


def load_tfidf_grid(path):
    df = pd.read_csv(path)
    df = df[df["feature_type"] == "tfidf"].copy()
    if df.empty:
        raise ValueError("No TF-IDF grid search results found in rf_grid.csv.")
    return df


def build_label(row):
    depth = "None" if pd.isna(row["max_depth"]) else int(float(row["max_depth"]))
    return f"n={int(row['n_estimators'])},d={depth},f={row['max_features']}"


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = load_tfidf_grid(RESULT_PATH)

    df = df.sort_values(by="selection_score", ascending=True)
    labels = [build_label(row) for _, row in df.iterrows()]
    scores = df["selection_score"].astype(float).tolist()

    font_path = resolve_chinese_font()
    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.bar(range(len(scores)), scores, color="#5b8ff9")

    ax.set_title("随机森林在TF-IDF特征下的参数搜索结果图", fontsize=14, pad=12)
    ax.set_xlabel("参数组合", fontsize=11)
    ax.set_ylabel("交叉验证准确率", fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(min(scores) - 0.02, max(scores) + 0.02)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    best_idx = int(df["selection_score"].idxmax())
    best_rank = list(df.index).index(best_idx)
    ax.bar(best_rank, scores[best_rank], color="#5ad8a6")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
