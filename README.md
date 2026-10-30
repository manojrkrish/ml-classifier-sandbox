# ml-classifier-sandbox

Interactive benchmarking of ML classifiers, deployed with Streamlit.

- **Live app:** https://ml-classifier-sandbox.streamlit.app/
- **GitHub repository:** https://github.com/manojrkrish/ml-classifier-sandbox

---

## a. Problem statement

Higher-education institutions lose students every year, and by the time a dropout shows up in the records it is usually too late to help. This project predicts, from information available at enrollment and from first-year academic performance, whether a student will end up as a **Dropout**, stay **Enrolled** beyond the normal course duration, or **Graduate**.

It is a **3-class classification** problem. Five classifiers are trained on the same data, evaluated on the same held-out test set with six metrics, and served through a Streamlit app that loads the saved models and scores an uploaded test CSV without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | [UCI Machine Learning Repository, dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| Instances | 4,424 students |
| Features | 36 (all numeric; 9 are nominal integer codes, 8 are 0/1 flags) |
| Target | `Target`: Dropout / Enrolled / Graduate |
| Missing values / duplicates | 0 / 0 |

**Class distribution** (imbalanced; `Enrolled` is the minority class):

| Class | Count | Share |
|---|---|---|
| Graduate | 2,209 | 49.9% |
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |

**Feature groups:**

| Group | # | Examples |
|---|---|---|
| Demographic | 6 | Marital status, Nacionality, Gender, Age at enrollment, Displaced, International |
| Socio-economic | 8 | Mother's/Father's qualification and occupation, Scholarship holder, Debtor, Tuition fees up to date, Educational special needs |
| Academic path at enrollment | 7 | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification (+ grade), Admission grade |
| Academic performance | 12 | Curricular units 1st/2nd sem: credited, enrolled, evaluations, approved, grade, without evaluations |
| Macro-economic | 3 | Unemployment rate, Inflation rate, GDP |

**Preprocessing** (fit on the training split only, saved inside each model pipeline):

- Column names stripped (the raw header has a trailing tab in `Daytime/evening attendance`).
- 9 nominal integer-coded columns (course, nationality, parents' qualification/occupation, ...) one-hot encoded; codes seen fewer than 10 times are grouped as "infrequent", which also absorbs unseen codes at prediction time.
- The other 27 columns standardized. 36 raw features become 131 model inputs.
- Stratified 80/20 split (`random_state=5326`): 3,539 training rows, 885 test rows saved as [`test_data.csv`](test_data.csv) (raw values + `Target`).

## c. GitHub repository link

https://github.com/manojrkrish/ml-classifier-sandbox

```
ml-classifier-sandbox/
├── app.py                  # Streamlit app
├── train.ipynb             # EDA, preprocessing, training, evaluation
├── requirements.txt        # pinned dependencies
├── README.md               # this file
├── test_data.csv           # held-out test set (885 rows) for the app
├── data/
│   └── students_dropout.csv    # raw UCI dataset
├── model/
│   ├── logistic_regression.pkl # each .pkl = preprocessing + classifier pipeline
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── metrics_summary.csv     # test-set metrics for all models
└── .streamlit/config.toml  # app theme
```

## d. Models used

All five models are scikit-learn `Pipeline(preprocessor → classifier)` objects, trained on the same 3,539 rows and scored on the same 885 held-out rows. Hyperparameters were chosen by 5-fold stratified cross-validation **on the training set only** (scoring: macro F1), so the test set never influenced model selection.

| Model | Final hyperparameters | Tuning note |
|---|---|---|
| Logistic Regression | `max_iter=1000` | Default regularization |
| Decision Tree | `max_depth=10, min_samples_leaf=10` | Unlimited depth overfit (CV F1 0.631 vs 0.675) |
| KNN | `n_neighbors=7` (uniform weights) | Tried k = 3–45; CV F1 fell as k grew |
| Naive Bayes (Gaussian) | `var_smoothing=0.1` | Default `1e-9` collapsed (CV F1 0.27) |
| SVM | `kernel="linear", C=0.3, probability=True` | Linear beat RBF (CV F1 0.707 vs 0.697) |

### Comparison table (test set, 885 rows)

Precision, Recall and F1 are **macro-averaged** over the three classes so the minority `Enrolled` class counts equally; AUC is one-vs-rest macro. Best value per metric in bold.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.7785** | **0.8979** | 0.7266 | **0.6975** | 0.7067 | 0.6330 |
| Decision Tree | 0.7492 | 0.8497 | 0.6819 | 0.6564 | 0.6622 | 0.5825 |
| KNN | 0.7356 | 0.8405 | 0.6719 | 0.6397 | 0.6473 | 0.5590 |
| Naive Bayes | 0.7141 | 0.8232 | 0.6648 | 0.6516 | 0.6558 | 0.5334 |
| Support Vector Machine | **0.7785** | 0.8973 | **0.7289** | 0.6967 | **0.7072** | **0.6331** |

Per-class recall on the test set (159 actual `Enrolled` students):

| Model | Dropout | Enrolled | Graduate |
|---|---|---|---|
| Logistic Regression | 0.761 | 0.409 | 0.923 |
| Decision Tree | 0.761 | 0.308 | 0.900 |
| KNN | 0.708 | 0.302 | 0.910 |
| Naive Bayes | 0.680 | **0.440** | 0.835 |
| Support Vector Machine | 0.743 | 0.415 | **0.932** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong and stable: ties SVM for best accuracy (0.7785) and has the best AUC (0.8979) and macro recall. Once categories are one-hot encoded and grades scaled, the classes are close to linearly separable, so a simple linear model does very well. Fast to train and easy to interpret. |
| Decision Tree | Mid-table (accuracy 0.7492). An unconstrained tree overfit badly, so depth and leaf size had to be limited. Its AUC (0.8497) is noticeably lower than the linear models because each leaf outputs one coarse probability, so it ranks students less smoothly. Weak on `Enrolled` (recall 0.308). |
| KNN | Weakest F1 (0.6473) and lowest `Enrolled` recall (0.302). With 131 inputs, mostly 0/1 one-hot columns, distances between students carry little signal. Macro F1 fell steadily as k grew (0.613 at k = 7, 0.547 at k = 45) because minority-class neighbours get outvoted: at k = 45 it predicted `Graduate` for 74.8% of students (actual 49.9%) and `Enrolled` for only 3.8% (actual 17.9%), so a small k worked best. The saved model is also the largest (3.7 MB) because KNN stores the whole training set. |
| Naive Bayes | Lowest accuracy (0.7141), AUC and MCC. Its independence assumption is clearly violated (1st- and 2nd-semester results correlate at 0.78–0.94), and the default smoothing failed completely on the one-hot columns. It does, however, have the **highest `Enrolled` recall (0.440)**, because it predicts `Enrolled` more often (170 times vs 159 actual), trading accuracy on the other classes for minority-class coverage. |
| Support Vector Machine | Best precision (0.7289), F1 (0.7072) and MCC (0.6331), tied best accuracy, and the best `Graduate` recall (0.932). A linear kernel beat RBF in cross-validation, confirming the near-linear structure. Practically tied with Logistic Regression: no metric differs by more than 0.0023. |
| **Overall winner** | **Support Vector Machine**, narrowly, on F1 and MCC, the metrics that best reflect performance on an imbalanced 3-class problem. Logistic Regression is effectively tied and is the practical alternative (simpler, faster, best AUC). All models struggle with `Enrolled` (recall 0.30–0.44): these students look like Graduates on paper but haven't finished on time, which is the main limitation and the obvious target for future work (e.g. class weighting or resampling). |

## Streamlit app

- **Upload test data (CSV):** comma or semicolon separated; the repo's `test_data.csv` can also be downloaded from the sidebar.
- **Model selection dropdown:** any of the five saved models (loaded with `joblib`, cached, never retrained).
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC for the selected model.
- **Confusion matrix** (counts or share of each actual class) and **classification report** (per-class precision/recall/F1/support).
- **Compare all models:** every model scored on the uploaded rows, best per metric highlighted, grouped bar chart.
- **Input validation:** clear messages for missing columns, unknown target labels, missing or non-numeric values.

## Run locally

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv` in the sidebar.

To retrain the models, open `train.ipynb` (needs `ipykernel` in the venv) and run all cells. This rewrites `test_data.csv`, the five `.pkl` files and `metrics_summary.csv`. The `.pkl` files require **scikit-learn 1.9.1**, as pinned in `requirements.txt`.
