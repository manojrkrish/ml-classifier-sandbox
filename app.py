from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "model"
TARGET = "Target"
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
