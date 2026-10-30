from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

MODEL_DIR = Path(__file__).parent / "model"
TARGET = "Target"
# must match the notebook so app scores equal the README table
AVERAGE = "macro"
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


st.set_page_config(page_title="ML Classifier Sandbox", layout="wide")
st.title("ML Classifier Sandbox")
st.caption(
    "Predicting student outcomes (Dropout / Enrolled / Graduate) with five pre-trained classifiers. "
    "Data: UCI *Predict Students' Dropout and Academic Success*."
)

with st.sidebar:
    st.header("1. Upload test data")
    uploaded = st.file_uploader("CSV with all feature columns and the `Target` column", type="csv")
    st.header("2. Choose a model")
    model_name = st.selectbox("Classifier", list(MODELS))

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin. The repo's `test_data.csv` works out of the box.")
    st.stop()

try:
    # sep=None sniffs ',' or ';'; utf-8-sig strips the BOM the raw UCI file starts with
    data = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()
data.columns = data.columns.str.strip()

model = load_model(MODELS[model_name])
missing = [col for col in [*model.feature_names_in_, TARGET] if col not in data.columns]
if missing:
    st.error(f"The CSV is missing {len(missing)} required column(s): " + ", ".join(f"`{c}`" for c in missing))
    st.stop()

st.subheader("Uploaded data")
rows, cols, classes = st.columns(3)
rows.metric("Rows", f"{len(data):,}")
cols.metric("Columns", data.shape[1])
classes.metric("Classes in Target", data[TARGET].nunique())
st.dataframe(data.head(20))

st.subheader(f"Model: {model_name}")
st.code(repr(model.named_steps["classifier"]), language="python")

X, y_true = data[list(model.feature_names_in_)], data[TARGET]
unknown = sorted(map(str, set(y_true) - set(model.classes_)))
if unknown:
    st.error(
        f"`{TARGET}` contains labels the model was never trained on: {', '.join(unknown)}. "
        f"Expected: {', '.join(model.classes_)}."
    )
    st.stop()

try:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
except ValueError as exc:
    st.error(f"Prediction failed - check for missing or non-numeric values. Details: {exc}")
    st.stop()

auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=AVERAGE, labels=model.classes_)
if pd.isna(auc):
    auc = None  # sklearn returns NaN when a class is absent from the uploaded rows

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": auc,
    "Precision": precision_score(y_true, y_pred, average=AVERAGE, zero_division=0),
    "Recall": recall_score(y_true, y_pred, average=AVERAGE, zero_division=0),
    "F1": f1_score(y_true, y_pred, average=AVERAGE, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

st.subheader("Evaluation metrics")
st.caption(f"Precision, Recall and F1 are {AVERAGE}-averaged over the classes; AUC is one-vs-rest.")
for col, (name, value) in zip(st.columns(len(metrics)), metrics.items()):
    col.metric(name, "n/a" if value is None else f"{value:.4f}")
if auc is None:
    st.warning("AUC needs every class present in the uploaded rows, so it is not shown.")
