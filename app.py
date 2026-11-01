from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "models" / "diabetes_pipeline.joblib"
METRICS_PATH = BASE_DIR / "reports" / "model_metrics.json"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
DISPLAY_NAMES = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Glucose concentration (mg/dL)",
    "BloodPressure": "Diastolic blood pressure (mmHg)",
    "SkinThickness": "Skin thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/mL)",
    "BMI": "Body Mass Index (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age (years)"
}

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def show_setup_message():
    st.warning("The trained model or dataset is not available yet.")
    st.markdown("""
    **First-time setup**
    1. Place the Pima Indians Diabetes CSV at `data/diabetes.csv`.
    2. Run `python train_model.py` in the project terminal.
    3. Refresh this dashboard.
    See `README.md` for dataset and setup instructions.
    """)

st.title("🩺 Diabetes Risk Prediction Dashboard")
st.caption("A supervised machine-learning learning project with an interactive Streamlit interface.")
st.info("Educational demonstration only. This model is not a medical diagnostic device and must not be used to make treatment decisions.")

df = load_dataset()
model = load_model()

if df is None:
    show_setup_message()
    st.stop()

missing_features = [c for c in FEATURES if c not in df.columns]
target_col = "Outcome" if "Outcome" in df.columns else (" outcome" if " outcome" in df.columns else None)
if missing_features or target_col is None:
    st.error("Dataset columns do not match the expected Pima diabetes schema. Check data/README.md.")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

overview_tab, prediction_tab, model_tab, about_tab = st.tabs(
    ["📊 Dataset Overview", "🔍 Predict", "📈 Model Evaluation", "ℹ️ About"]
)

with overview_tab:
    st.subheader("Dataset at a glance")
    total = len(df)
    positive = int((df[target_col] == 1).sum())
    negative = int((df[target_col] == 0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{total:,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Positive labels", f"{positive:,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("#### Dataset preview")
    st.dataframe(df.head(15), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Outcome distribution")
        counts = df[target_col].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots()
        ax.bar(["No diabetes label (0)", "Diabetes label (1)"], counts.values)
        ax.set_ylabel("Number of records")
        ax.tick_params(axis="x", rotation=12)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Glucose distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Glucose"].dropna(), bins=20)
        ax.set_xlabel("Glucose (mg/dL)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Descriptive statistics")
    st.dataframe(df[FEATURES].describe().T, use_container_width=True)
    st.markdown("#### Missing / zero-value review")
    st.write("In this dataset, zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI are treated as invalid measurements during training and imputed using training-set medians.")
    st.dataframe(df[FEATURES].isna().sum().rename("Missing values").to_frame(), use_container_width=True)

with prediction_tab:
    st.subheader("Enter the measurements")
    st.write("Enter values from a valid clinical record. The default values are examples, not recommendations.")
    if model is None:
        show_setup_message()
    else:
        with st.form("prediction_form"):
            a, b = st.columns(2)
            with a:
                pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=1)
                glucose = st.number_input("Glucose concentration (mg/dL)", min_value=0.0, max_value=300.0, value=120.0)
                bp = st.number_input("Diastolic blood pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0)
                skin = st.number_input("Skin thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
            with b:
                insulin = st.number_input("2-hour serum insulin (mu U/mL)", min_value=0.0, max_value=900.0, value=80.0)
                bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=80.0, value=28.0)
                dpf = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
                age = st.number_input("Age (years)", min_value=21, max_value=100, value=30)
            submitted = st.form_submit_button("Predict (educational estimate)", type="primary", use_container_width=True)

        if submitted:
            row = pd.DataFrame([{
                "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
                "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age
            }], columns=FEATURES)
            pred = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
            if pred == 1:
                st.warning("The model classified this input in the positive class.")
            else:
                st.success("The model classified this input in the negative class.")
            if probability is not None:
                st.metric("Model-estimated positive-class probability", f"{probability * 100:.1f}%")
                st.progress(min(max(probability, 0.0), 1.0))
            st.caption("This is a model output based on a limited historical dataset, not an individual's medical risk or a diagnosis. Consult a qualified healthcare professional for interpretation.")

with model_tab:
    st.subheader("Training and evaluation results")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"Selected model: **{metrics.get('best_model', 'Not specified')}**")
        score_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cols = st.columns(len(score_keys))
        for col, key in zip(cols, score_keys):
            value = metrics.get("test_metrics", {}).get(key)
            col.metric(key.replace("_", " ").title(), f"{value:.3f}" if isinstance(value, (int, float)) else "N/A")
        comparison = metrics.get("model_comparison", {})
        if comparison:
            comp_df = pd.DataFrame(comparison).T
            st.markdown("#### Model comparison on the held-out test set")
            st.dataframe(comp_df.round(3), use_container_width=True)
        st.caption("Metrics describe this dataset split only. They do not establish clinical validity or performance on other populations.")
    else:
        st.info("Run `python train_model.py` to generate the evaluation report.")

with about_tab:
    st.subheader("Project information")
    st.markdown("""
    **Objective:** demonstrate a supervised binary-classification workflow for the Pima Indians Diabetes dataset.

    **Algorithms compared:** Logistic Regression, Decision Tree, Random Forest and Support Vector Machine.

    **Workflow:** data loading → validation → preprocessing → train/test split → model comparison → saved pipeline → Streamlit prediction dashboard.

    **Important limitations**
    - The dataset is historical and represents a limited population; it may not generalize to other people.
    - The model can be wrong and may reflect biases or limitations in the source data.
    - A predicted class or probability is not a diagnosis, screening recommendation, or treatment plan.
    """)
    st.markdown("**Dataset source:** Pima Indians Diabetes Database. See the README for attribution and access details.")
