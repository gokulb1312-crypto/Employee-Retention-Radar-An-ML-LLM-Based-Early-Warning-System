"""Part 1: clean the employee data and save reproducible EDA outputs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # generate files without requiring a desktop/Tk installation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "employee_retention_raw.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = BASE_DIR / "plots"


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply documented, reproducible cleaning decisions to the raw dataset."""
    data = frame.copy()
    data.columns = data.columns.str.strip()
    data = data.drop_duplicates().dropna(how="all")

    # EmployeeID is an identifier; keep it as an integer and do not use it for imputation.
    data["EmployeeID"] = pd.to_numeric(data["EmployeeID"], errors="raise").astype(int)
    for column in data.select_dtypes(include="number").columns:
        if column != "EmployeeID":
            data[column] = data[column].fillna(data[column].median())
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].fillna(data[column].mode().iat[0]).str.strip()
    return data


def save_plots(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x="Attrition", hue="Attrition", legend=False, ax=ax)
    ax.set(title="Employee attrition distribution", xlabel="Attrition (1 = left)")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "attrition_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=data, x="Age", hue="Attrition", bins=20, kde=True, ax=ax)
    ax.set(title="Age distribution by attrition")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "age_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    corr = data.select_dtypes(include="number").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Numeric feature correlation matrix")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=data, x="Attrition", y="MonthlyIncome", ax=ax)
    ax.set(title="Monthly income by attrition")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "income_by_attrition.png", dpi=160); plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    cleaned = clean_data(raw)
    cleaned.to_csv(OUTPUT_DIR / "employee_retention_cleaned.csv", index=False)
    cleaned.describe(include="all").transpose().to_csv(OUTPUT_DIR / "summary_statistics.csv")
    save_plots(cleaned)
    print(f"Raw rows: {len(raw)} | Clean rows: {len(cleaned)}")
    print("Attrition counts:\n", cleaned["Attrition"].value_counts().sort_index())
    print(f"Saved cleaned data, summary statistics, and plots under {BASE_DIR}")


if __name__ == "__main__":
    main()
