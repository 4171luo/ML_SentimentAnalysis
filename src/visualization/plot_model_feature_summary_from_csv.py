import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CSV_PATH = os.path.join(ROOT_DIR, "results", "algorithm_summary_metrics.csv")
OUT_DIR = os.path.join(ROOT_DIR, "docs", "第四章图表")
PRECISION_FIG = os.path.join(OUT_DIR, "图4-9_不同模型特征组合Precision对比图.png")
METRICS_FIG = os.path.join(OUT_DIR, "图4-10_不同模型特征组合Accuracy_Recall_F1对比图.png")


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


def add_display_labels(df):
    df = df.copy()
    df["模型特征组合"] = df["算法"] + "\n(" + df["特征"] + ")"
    return df


def plot_precision(df):
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    colors = ["#5b8ff9", "#5ad8a6", "#f6bd16"]
    bars = ax.bar(df["模型特征组合"], df["Precision"], color=colors)
    ax.set_title("不同模型特征组合Precision对比图", fontsize=14, pad=12)
    ax.set_xlabel("模型-特征组合", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_ylim(0.78, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PRECISION_FIG, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(df):
    metrics = ["Accuracy", "Recall", "F1"]
    x = np.arange(len(df))
    width = 0.24
    colors = ["#5b8ff9", "#5ad8a6", "#f6bd16"]

    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=200)
    for idx, metric in enumerate(metrics):
        values = df[metric].astype(float).values
        bars = ax.bar(x + (idx - 1) * width, values, width=width, label=metric, color=colors[idx])
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("不同模型特征组合Accuracy、Recall和F1分数对比图", fontsize=14, pad=12)
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
    df = pd.read_csv(CSV_PATH)
    df = add_display_labels(df)
    plot_precision(df)
    plot_metrics(df)
    print(f"Saved precision figure to: {PRECISION_FIG}")
    print(f"Saved metrics figure to: {METRICS_FIG}")
    print(df[["算法", "特征", "Accuracy", "Precision", "Recall", "F1"]].to_string(index=False))


if __name__ == "__main__":
    main()
