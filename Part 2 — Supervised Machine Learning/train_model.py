"""Part 2: train, tune, evaluate, and save an attrition-risk classifier."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # generate files without requiring a desktop/Tk installation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Part 1 — Data Preparation & Exploration" / "outputs" / "employee_retention_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def evaluate(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predicted = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    metrics = {"model": name, "f1": round(float(f1_score(y_test, predicted)), 4), "roc_auc": round(float(roc_auc_score(y_test, probability)), 4)}
    print(name, metrics)
    print(classification_report(y_test, predicted, digits=3))
    return metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run Part 1 first: cleaned data is missing.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    y = data["Attrition"].astype(int)
    x = data.drop(columns=["Attrition", "EmployeeID"])  # avoid identifier leakage
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = x.select_dtypes(exclude="number").columns.tolist()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    baseline = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
    baseline.fit(x_train, y_train)
    baseline_metrics = evaluate("logistic_regression_baseline", baseline, x_test, y_test)

    tuned = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))])
    search = GridSearchCV(tuned, {"classifier__n_estimators": [200, 400], "classifier__max_depth": [None, 8, 14], "classifier__min_samples_leaf": [1, 3]}, scoring="f1", cv=5, n_jobs=-1)
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    tuned_metrics = evaluate("tuned_random_forest", best_model, x_test, y_test)
    deployment_model = baseline if baseline_metrics["f1"] >= tuned_metrics["f1"] else best_model
    deployment_name = baseline_metrics["model"] if deployment_model is baseline else tuned_metrics["model"]
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps({"baseline": baseline_metrics, "tuned": tuned_metrics, "best_parameters": search.best_params_, "deployment_model": deployment_name}, indent=2), encoding="utf-8")
    joblib.dump(deployment_model, OUTPUT_DIR / "attrition_model.joblib")
    print(f"Deployment model selected by held-out F1: {deployment_name}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_estimator(best_model, x_test, y_test, ax=axes[0], cmap="Blues")
    axes[0].set_title("Tuned model confusion matrix")
    RocCurveDisplay.from_estimator(best_model, x_test, y_test, ax=axes[1])
    axes[1].set_title("Tuned model ROC curve")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "model_evaluation.png", dpi=160); plt.close(fig)

    feature_names = best_model.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(best_model.named_steps["classifier"].feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=importances.values, y=importances.index, ax=ax)
    ax.set(title="Top random-forest feature importances", xlabel="Importance", ylabel="Feature")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160); plt.close(fig)
    print(f"Saved model and evaluation outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
