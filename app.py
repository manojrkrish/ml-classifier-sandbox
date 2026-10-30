from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"
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


def evaluate(model, X, y_true):
    y_pred = model.predict(X)
    auc = roc_auc_score(
        y_true, model.predict_proba(X), multi_class="ovr", average=AVERAGE, labels=model.classes_
    )
    return y_pred, {
        "Accuracy": accuracy_score(y_true, y_pred),
        # sklearn returns NaN when a class is absent from the uploaded rows
        "AUC": None if pd.isna(auc) else auc,
        "Precision": precision_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=AVERAGE, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.set_page_config(page_title="ML Classifier Sandbox", layout="wide")
st.title("ML Classifier Sandbox")
st.caption(
    "Predicting student outcomes (Dropout / Enrolled / Graduate) with five pre-trained classifiers. "
    "Data: UCI *Predict Students' Dropout and Academic Success*."
)

with st.sidebar:
    st.header("1. Upload test data")
    uploaded = st.file_uploader("CSV with all feature columns and the `Target` column", type="csv")
    st.download_button(
        "No file? Download sample test_data.csv",
        SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
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
with st.expander("Preview first 20 rows"):
    st.dataframe(data.head(20))

X, y_true = data[list(model.feature_names_in_)], data[TARGET]
unknown = sorted(map(str, set(y_true) - set(model.classes_)))
if unknown:
    st.error(
        f"`{TARGET}` contains labels the model was never trained on: {', '.join(unknown)}. "
        f"Expected: {', '.join(model.classes_)}."
    )
    st.stop()

# Decision Tree tolerates NaN but the other models don't; reject up front so all five behave the same
nan_cols = X.columns[X.isna().any()].tolist()
if nan_cols:
    st.error("Missing values in: " + ", ".join(f"`{c}`" for c in nan_cols))
    st.stop()

try:
    y_pred, metrics = evaluate(model, X, y_true)
except ValueError as exc:
    st.error(f"Prediction failed - check for non-numeric values. Details: {exc}")
    st.stop()

selected_tab, compare_tab = st.tabs([f"Selected model: {model_name}", "Compare all models"])

with selected_tab:
    st.code(repr(model.named_steps["classifier"]), language="python")

    st.subheader("Evaluation metrics")
    st.caption(f"Precision, Recall and F1 are {AVERAGE}-averaged over the classes; AUC is one-vs-rest.")
    for col, (name, value) in zip(st.columns(len(metrics)), metrics.items()):
        col.metric(name, "n/a" if value is None else f"{value:.4f}")
    if metrics["AUC"] is None:
        st.warning("AUC needs every class present in the uploaded rows, so it is not shown.")

    cm_col, report_col = st.columns(2)
    with cm_col:
        st.subheader("Confusion matrix")
        as_share = st.toggle("Show as share of each actual class")
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_, normalize="true" if as_share else None)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt=".0%" if as_share else "d", cmap="Blues", cbar=False,
            xticklabels=model.classes_, yticklabels=model.classes_, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="y", rotation=0)
        st.pyplot(fig)
        plt.close(fig)
    with report_col:
        st.subheader("Classification report")
        report = classification_report(y_true, y_pred, labels=model.classes_, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.drop(index="accuracy", errors="ignore")
        st.dataframe(report_df.style.format("{:.4f}").format("{:.0f}", subset=["support"]))

with compare_tab:
    st.caption("Every saved model scored on the same uploaded rows (best per metric highlighted).")
    comparison = pd.DataFrame(
        {name: evaluate(load_model(file), X, y_true)[1] for name, file in MODELS.items()}
    ).T.astype(float)
    best = comparison["F1"].idxmax()
    st.success(f"Best macro-F1 on this data: **{best}** ({comparison.loc[best, 'F1']:.4f})")
    st.dataframe(comparison.style.highlight_max(axis=0, color="#c8e6c9").format("{:.4f}", na_rep="n/a"))
    st.bar_chart(comparison.T, stack=False, x_label="Metric", y_label="Score")
