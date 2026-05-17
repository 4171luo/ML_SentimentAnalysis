import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "train", "weibo_senti_100k.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "docs", "第四章图表", "图4-1_数据集正负样本分布图.png")


def load_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


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


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = load_dataset(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    counts = df["label"].value_counts().sort_index()
    label_names = ["负向(0)", "正向(1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    colors = ["#d95f5f", "#4c9f70"]

    font_path = resolve_chinese_font()
    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.bar(label_names, values, color=colors, width=0.55)

    ax.set_title("数据集正负样本分布图", fontsize=14, pad=12)
    ax.set_xlabel("情感类别", fontsize=11)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.005,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
