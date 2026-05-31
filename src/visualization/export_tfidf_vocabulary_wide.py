import os

import joblib
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
VECTORIZER_PATH = os.path.join(ROOT_DIR, "models", "nb_tfidf_vectorizer.pkl")
OUT_DIR = os.path.join(ROOT_DIR, "docs", "特征示例表")
CSV_PATH = os.path.join(OUT_DIR, "TF-IDF词汇表宽表.csv")
XLSX_PATH = os.path.join(OUT_DIR, "TF-IDF词汇表宽表.xlsx")


def build_wide_vocabulary_table(feature_names, groups=10):
    rows_per_group = (len(feature_names) + groups - 1) // groups
    data = {}

    for group_idx in range(groups):
        start = group_idx * rows_per_group
        end = min(start + rows_per_group, len(feature_names))
        indices = list(range(start, end))
        terms = feature_names[start:end].tolist()

        padding = rows_per_group - len(indices)
        if padding:
            indices.extend([""] * padding)
            terms.extend([""] * padding)

        data[f"索引{group_idx + 1}"] = indices
        data[f"词项{group_idx + 1}"] = terms

    return pd.DataFrame(data)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    vectorizer = joblib.load(VECTORIZER_PATH)
    feature_names = vectorizer.get_feature_names_out()
    wide_df = build_wide_vocabulary_table(feature_names, groups=10)

    wide_df.to_csv(CSV_PATH, index=False, encoding="utf_8_sig")
    try:
        with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
            wide_df.to_excel(writer, index=False, sheet_name="TF-IDF词汇表")
            worksheet = writer.sheets["TF-IDF词汇表"]
            worksheet.freeze_panes = "A2"
            for col in worksheet.columns:
                worksheet.column_dimensions[col[0].column_letter].width = 14
    except ImportError:
        XLSX_PATH_MESSAGE = "openpyxl is not installed, skipped xlsx output."
    else:
        XLSX_PATH_MESSAGE = XLSX_PATH

    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved XLSX: {XLSX_PATH_MESSAGE}")


if __name__ == "__main__":
    main()
