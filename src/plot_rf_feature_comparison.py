import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULT_PATH = os.path.join(ROOT_DIR, "results", "rf_grid.csv")
OUTPUT_PATH = os.path.join(
    ROOT_DIR, "docs", "第四章图表", "图4-5_随机森林模型两种特征下的性能对比图.png"
)


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


def load_best_results(path):
    df = pd.read_csv(path)
    best_rows = df[df["is_best"] == 1].copy()
    best_rows = best_rows.dropna(subset=["test_accuracy", "test_f1_macro"])

    best_rows = (
        best_rows.sort_values(
            by=["feature_type", "selection_score", "test_f1_macro", "test_accuracy"],
            ascending=[True, False, False, False],
        )
        .groupby("feature_type", as_index=False)
        .first()
    )

    expected_features = ["tfidf", "word2vec"]
    if sorted(best_rows["feature_type"].tolist()) != sorted(expected_features):
        raise ValueError("Could not find complete best-result rows for TF-IDF and Word2Vec.")

    return {
        "TF-IDF": (
            float(best_rows.loc[best_rows["feature_type"] == "tfidf", "test_accuracy"].iloc[0]),
            float(best_rows.loc[best_rows["feature_type"] == "tfidf", "test_f1_macro"].iloc[0]),
        ),
        "Word2Vec": (
            float(best_rows.loc[best_rows["feature_type"] == "word2vec", "test_accuracy"].iloc[0]),
            float(best_rows.loc[best_rows["feature_type"] == "word2vec", "test_f1_macro"].iloc[0]),
        ),
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    metrics = load_best_results(RESULT_PATH)

    font_path = resolve_chinese_font()
    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    feature_names = list(metrics.keys())
    accuracy_values = [metrics[name][0] for name in feature_names]
    f1_values = [metrics[name][1] for name in feature_names]

    x = range(len(feature_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars1 = ax.bar([i - width / 2 for i in x], accuracy_values, width=width, label="Accuracy", color="#5b8ff9")
    bars2 = ax.bar([i + width / 2 for i in x], f1_values, width=width, label="Macro-F1", color="#5ad8a6")

    ax.set_title("随机森林模型两种特征下的性能对比图", fontsize=14, pad=12)
    ax.set_xlabel("特征方式", fontsize=11)
    ax.set_ylabel("指标值", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(feature_names)
    ax.set_ylim(0.80, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.001,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
