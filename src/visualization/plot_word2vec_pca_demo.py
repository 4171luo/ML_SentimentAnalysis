import os

import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "rf_word2vec.model")
OUT_DIR = os.path.join(ROOT_DIR, "docs", "第四章图表")
OUTPUT_PATH = os.path.join(OUT_DIR, "图4-6_Word2Vec词向量PCA降维示例图.png")
CSV_PATH = os.path.join(ROOT_DIR, "docs", "特征示例表", "Word2Vec_PCA展示词.csv")

WORD_GROUPS = {
    "积极倾向词": [
        "开心", "快乐", "幸福", "喜欢", "支持", "感谢",
        "可爱", "鼓掌", "加油", "美好", "温柔", "不错",
    ],
    "消极倾向词": [
        "难过", "失望", "伤心", "讨厌", "恶心", "生气",
        "无语", "糟糕", "骗子", "汉奸", "肮脏", "坏",
    ],
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


def collect_existing_words(model):
    rows = []
    for group_name, words in WORD_GROUPS.items():
        for word in words:
            if word in model.wv:
                rows.append(
                    {
                        "词项": word,
                        "类别": group_name,
                        "词频": int(model.wv.get_vecattr(word, "count")),
                    }
                )
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    model = Word2Vec.load(MODEL_PATH)
    rows = collect_existing_words(model)
    if len(rows) < 4:
        raise ValueError("Not enough selected words exist in the Word2Vec vocabulary.")

    words = [row["词项"] for row in rows]
    vectors = [model.wv[word] for word in words]
    coords = PCA(n_components=2, random_state=42).fit_transform(vectors)

    for row, coord in zip(rows, coords):
        row["PCA维度1"] = round(float(coord[0]), 6)
        row["PCA维度2"] = round(float(coord[1]), 6)
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False, encoding="utf_8_sig")

    font_path = resolve_chinese_font()
    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    color_map = {
        "积极倾向词": "#4c9f70",
        "消极倾向词": "#d95f5f",
    }

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    for group_name in WORD_GROUPS:
        group_rows = [row for row in rows if row["类别"] == group_name]
        xs = [row["PCA维度1"] for row in group_rows]
        ys = [row["PCA维度2"] for row in group_rows]
        ax.scatter(
            xs,
            ys,
            s=80,
            color=color_map[group_name],
            label=group_name,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
        )
        for row in group_rows:
            ax.text(row["PCA维度1"] + 0.015, row["PCA维度2"] + 0.015, row["词项"], fontsize=10)

    ax.set_title("Word2Vec词向量PCA降维示例图", fontsize=14, pad=12)
    ax.set_xlabel("PCA维度1", fontsize=11)
    ax.set_ylabel("PCA维度2", fontsize=11)
    ax.grid(linestyle="--", alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")

    print(f"Saved figure to: {OUTPUT_PATH}")
    print(f"Saved PCA word table to: {CSV_PATH}")


if __name__ == "__main__":
    main()
