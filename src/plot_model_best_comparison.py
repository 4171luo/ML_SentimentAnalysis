import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
NB_PATH = os.path.join(ROOT_DIR, "results", "nb_metrics.csv")
RF_PATH = os.path.join(ROOT_DIR, "results", "rf_grid.csv")
NN_PATH = os.path.join(ROOT_DIR, "results", "nn_metrics.csv")
OUT_ACC = os.path.join(ROOT_DIR, "docs", "第四章图表", "图4-7_各模型最优版本Accuracy对比图.png")
OUT_F1 = os.path.join(ROOT_DIR, "docs", "第四章图表", "图4-8_各模型最优版本Macro-F1对比图.png")


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


def pick_best_row(df, model_name):
    subset = df[(df["model_name"] == model_name) & (df["is_best"] == 1)].copy()
    subset = subset.dropna(subset=["test_accuracy", "test_f1_macro"])
    if subset.empty:
        raise ValueError(f"No best rows found for {model_name}.")
    subset = subset.sort_values(
        by=["selection_score", "test_f1_macro", "test_accuracy"],
        ascending=[False, False, False],
    )
    return subset.iloc[0]


def load_best_metrics():
    nb = pd.read_csv(NB_PATH)
    rf = pd.read_csv(RF_PATH)
    nn = pd.read_csv(NN_PATH)

    nb_best = pick_best_row(nb, "naive_bayes")
    rf_best = pick_best_row(rf, "random_forest")
    nn_best = pick_best_row(nn, "mlp")

    models = ["朴素贝叶斯", "随机森林", "MLP"]
    accuracy = [
        float(nb_best["test_accuracy"]),
        float(rf_best["test_accuracy"]),
        float(nn_best["test_accuracy"]),
    ]
    macro_f1 = [
        float(nb_best["test_f1_macro"]),
        float(rf_best["test_f1_macro"]),
        float(nn_best["test_f1_macro"]),
    ]
    return models, accuracy, macro_f1


def plot_metric(models, values, title, ylabel, output_path):
    font_path = resolve_chinese_font()
    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.bar(models, values, color=["#5b8ff9", "#5ad8a6", "#f6bd16"])

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("模型", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(min(values) - 0.03, max(values) + 0.03)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")


def main():
    os.makedirs(os.path.dirname(OUT_ACC), exist_ok=True)
    models, accuracy, macro_f1 = load_best_metrics()

    plot_metric(
        models,
        accuracy,
        "各模型最优版本Accuracy对比图",
        "Accuracy",
        OUT_ACC,
    )
    plot_metric(
        models,
        macro_f1,
        "各模型最优版本Macro-F1对比图",
        "Macro-F1",
        OUT_F1,
    )


if __name__ == "__main__":
    main()
