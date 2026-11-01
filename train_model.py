from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def get_dataset():
    if DATA_PATH.exists():
        print(f"Loading dataset: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("CSV not found. Attempting to fetch the Pima dataset using ucimlrepo...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=228)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        # Normalize common UCI/Kaggle feature-name variants.
        rename_map = {
            "preg": "Pregnancies", "npreg": "Pregnancies", "pregnancies": "Pregnancies",
            "plas": "Glucose", "glucose": "Glucose",
            "pres": "BloodPressure", "bloodpressure": "BloodPressure",
            "skin": "SkinThickness", "skinthickness": "SkinThickness",
            "insu": "Insulin", "serum": "Insulin", "insulin": "Insulin",
            "mass": "BMI", "bmi": "BMI",
            "pedi": "DiabetesPedigreeFunction", "diabetespedigreefunction": "DiabetesPedigreeFunction",
            "age": "Age"
        }
        X.columns = [rename_map.get(str(c).strip().lower(), str(c).strip()) for c in X.columns]
        if len(y.columns) != 1:
            raise ValueError("Expected exactly one target column from the dataset.")
        frame = X.copy()
        target = y.iloc[:, 0]
        if not pd.api.types.is_numeric_dtype(target):
            target_text = target.astype(str).str.strip().str.lower()
            target = target_text.map({
                "tested_positive": 1, "positive": 1, "yes": 1, "diabetic": 1, "1": 1,
                "tested_negative": 0, "negative": 0, "no": 0, "non-diabetic": 0, "0": 0
            })
        frame["Outcome"] = pd.to_numeric(target, errors="coerce").values
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to {DATA_PATH}")
        return frame
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset automatically. Download the Pima Indians Diabetes CSV "
            "and save it as data/diabetes.csv. See README.md. Original error: " + str(exc)
        )

def main():
    df = get_dataset()
    # Standardize expected target spelling.
    if "Outcome" not in df.columns:
        candidates = [c for c in df.columns if str(c).strip().lower() in ("outcome", "class", "diabetes", "target")]
        if candidates:
            df = df.rename(columns={candidates[0]: "Outcome"})
    missing = [c for c in FEATURES + ["Outcome"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found: {df.columns.tolist()}")

    df = df[FEATURES + ["Outcome"]].copy()
    for col in FEATURES + ["Outcome"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Outcome"])
    if not pd.api.types.is_numeric_dtype(df["Outcome"]):
        outcome_text = df["Outcome"].astype(str).str.strip().str.lower()
        df["Outcome"] = outcome_text.map({
            "tested_positive": 1, "positive": 1, "yes": 1, "diabetic": 1, "1": 1,
            "tested_negative": 0, "negative": 0, "no": 0, "non-diabetic": 0, "0": 0
        })
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")
    df = df.dropna(subset=["Outcome"])
    df["Outcome"] = df["Outcome"].astype(int)
    df = df[df["Outcome"].isin([0, 1])]
    if df.empty:
        raise ValueError("No valid rows found after checking the target column.")

    X = df[FEATURES].copy()
    y = df["Outcome"].copy()
    # In the Pima dataset, zero is used for some unrecorded clinical measurements.
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_split=5, class_weight="balanced", random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    }

    comparison = {}
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        scores = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob)
        }
        comparison[name] = scores
        fitted[name] = pipe
        print(name, {k: round(v, 3) for k, v in scores.items()})

    # Select by ROC-AUC; this is a project comparison criterion, not a clinical threshold.
    best_name = max(comparison, key=lambda name: comparison[name]["roc_auc"])
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, best_pred)),
        "precision": float(precision_score(y_test, best_pred, zero_division=0)),
        "recall": float(recall_score(y_test, best_pred, zero_division=0)),
        "f1": float(f1_score(y_test, best_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, best_prob))
    }

    joblib.dump(best_model, MODEL_DIR / "diabetes_pipeline.joblib")
    report = {
        "best_model": best_name,
        "selection_metric": "ROC-AUC",
        "test_metrics": test_metrics,
        "model_comparison": comparison,
        "features": FEATURES,
        "test_size": 0.20,
        "random_state": 42
    }
    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSelected model: {best_name}")
    print(f"Saved pipeline: {MODEL_DIR / 'diabetes_pipeline.joblib'}")
    print(f"Saved metrics: {REPORT_DIR / 'model_metrics.json'}")

if __name__ == "__main__":
    main()
