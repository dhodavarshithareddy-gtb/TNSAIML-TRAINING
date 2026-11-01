# Diabetes Risk Prediction Using Supervised Learning

A beginner-friendly machine-learning project with a Streamlit dashboard. It compares Logistic Regression, Decision Tree, Random Forest and Support Vector Machine classifiers using the Pima Indians Diabetes dataset.

> **Educational use only:** this project is not a medical device. Predictions are not diagnoses and must not be used for treatment or healthcare decisions.

## 1. Features

- Dataset preview, row/column metrics and descriptive statistics
- Outcome distribution and glucose histogram
- Data quality / missing-value review
- Supervised model training and comparison
- Saved preprocessing + model pipeline
- Interactive prediction form
- Positive-class probability display when supported
- Model evaluation metrics: accuracy, precision, recall, F1-score and ROC-AUC

## 2. Project structure

```text
diabetes_risk_prediction_streamlit/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── README.md
├── models/
│   └── (generated after training)
└── reports/
    └── (generated after training)
```

## 3. Dataset

This project uses the Pima Indians Diabetes dataset, a binary-classification dataset with eight input features and an `Outcome` target (0 or 1). The dataset concerns a specific historical study population and should not be assumed representative of all people.

The training script first attempts to fetch the dataset through the `ucimlrepo` Python package (dataset ID 228). If that fails, download the CSV described in `data/README.md` and save it as `data/diabetes.csv`.

Expected columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

## 4. Installation (Windows)

Open Command Prompt or VS Code terminal in this project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Train and evaluate models

```bash
python train_model.py
```

The script:
1. Loads the CSV (or attempts to fetch it).
2. Checks the expected feature and target columns.
3. Treats zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI as missing measurements.
4. Splits data into training and test sets using a stratified 80:20 split.
5. Fits preprocessing and classifier pipelines.
6. Compares four supervised classifiers.
7. Selects the model with the highest test ROC-AUC for this run.
8. Saves `models/diabetes_pipeline.joblib` and `reports/model_metrics.json`.

Do not compare results across different datasets or splits as if they were directly equivalent. For a serious study, use cross-validation, calibration, subgroup evaluation, external validation and expert review.

## 6. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Streamlit opens the dashboard in your browser. Stop the local server with `Ctrl+C`.

## 7. Dashboard tabs

- **Dataset Overview:** sample rows, summary metrics, outcome distribution and feature statistics.
- **Predict:** enter the eight features and submit the form.
- **Model Evaluation:** view the selected model and held-out test metrics.
- **About:** project workflow, algorithms and limitations.

## 8. Important implementation notes

- The same preprocessing steps are stored with the estimator in a scikit-learn `Pipeline`.
- The Streamlit app loads the model with `st.cache_resource`.
- The dataset is loaded with `st.cache_data`.
- Input column names and order must match the training features.
- Do not publish private patient information or expose a deployed model without appropriate security controls.
- Model probability is not the same as an individual's clinically validated risk.

## 9. Suggested future enhancements

- Add cross-validation and a confusion matrix.
- Add ROC and precision-recall curves.
- Add explainability (for example, permutation importance or SHAP) with careful interpretation.
- Add authentication and role-based access before handling any sensitive information.
- Deploy only for a clearly labelled educational demonstration.

## 10. Dataset attribution

Pima Indians Diabetes dataset, originally associated with the National Institute of Diabetes and Digestive and Kidney Diseases. Dataset access: https://archive.ics.uci.edu/dataset/228/pima+indians+diabetes

Please retain the dataset's attribution and license information when redistributing the data.
