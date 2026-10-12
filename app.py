"""Interactive demo: heart-disease risk prediction with a Random Forest.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "rf_heart.joblib"
FIG = ROOT / "figures"

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="wide")


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/prepare_data.py` then `python src/train.py` first.")
    st.stop()

bundle = load()
model = bundle["model"]
rf = model.named_steps["clf"]
tm = bundle["test_metrics"]

st.title("Heart Disease Risk Prediction")
st.caption("Random Forest classifier trained on the UCI Heart Disease dataset (920 patients). "
           "For educational purposes only; not intended for clinical use.")

tab_pred, tab_model, tab_figs = st.tabs(["Prediction", "Model Details", "Evaluation"])

# ------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Patient Information")
    age = st.slider("Age", 20, 80, 54)
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    cp = st.selectbox("Chest pain type", ["ASY", "NAP", "ATA", "TA"],
                      format_func=lambda v: {"ASY": "ASY: Asymptomatic", "NAP": "NAP: Non-anginal pain",
                                             "ATA": "ATA: Atypical angina", "TA": "TA: Typical angina"}[v])
    bp = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
    chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl", [0, 1], horizontal=True,
                   format_func=lambda v: "Yes" if v else "No")
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxhr = st.slider("Maximum heart rate achieved (bpm)", 60, 210, 140)
    exang = st.radio("Exercise-induced angina", ["N", "Y"], horizontal=True,
                     format_func=lambda v: "Yes" if v == "Y" else "No")
    oldpeak = st.slider("ST depression (Oldpeak)", -2.5, 6.5, 1.0, 0.1)
    slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])
    threshold = st.slider("Decision threshold", 0.2, 0.8, 0.5, 0.05,
                          help="Lower thresholds increase sensitivity (fewer missed cases) at the cost of more false positives.")

patient = pd.DataFrame([{
    "Age": age, "RestingBP": bp, "Cholesterol": chol, "FastingBS": fbs, "MaxHR": maxhr,
    "Oldpeak": oldpeak, "Sex": sex, "ChestPainType": cp, "RestingECG": ecg,
    "ExerciseAngina": exang, "ST_Slope": slope,
}])[bundle["features"]]

prob = model.predict_proba(patient)[0, 1]
Xt = model.named_steps["prep"].transform(patient)
votes = np.array([t.predict(Xt)[0] for t in rf.estimators_])

# ------------------------------------------------------------------ predict tab
with tab_pred:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted probability of heart disease", f"{prob:.1%}")
    c2.metric("Trees predicting heart disease", f"{int(votes.sum())} of {len(votes)}")
    c3.metric(f"Classification (threshold {threshold:.2f})", "Positive" if prob >= threshold else "Negative")
    st.progress(float(prob))
    if prob >= threshold:
        st.warning("The model classifies this patient as likely to have heart disease. "
                   "In a clinical setting, this would indicate the need for further diagnostic testing.")
    else:
        st.info("The model classifies this patient as unlikely to have heart disease.")
    st.caption("The predicted probability is the average of the class probabilities across all trees in the forest.")

# ------------------------------------------------------------------ model tab
with tab_model:
    st.subheader("Ensemble Composition")
    st.write(f"The forest consists of {len(rf.estimators_)} decision trees. Each tree is trained on a "
             f"bootstrap sample of the training data and considers a random subset of features at each split "
             f"(max_features = {bundle['best_params'].get('max_features')}). The chart below shows the "
             "probability of heart disease estimated by each individual tree for the current patient, sorted "
             "in ascending order.")
    tree_probs = np.array([t.predict_proba(Xt)[0, 1] for t in rf.estimators_])
    st.bar_chart(pd.DataFrame({"Probability of heart disease": np.sort(tree_probs)}), height=250)
    st.subheader("Hyperparameters")
    st.caption("Selected by grid search with 5-fold stratified cross-validation (ROC-AUC).")
    st.dataframe(pd.DataFrame({"Parameter": list(bundle["best_params"]),
                               "Value": [str(v) for v in bundle["best_params"].values()]}),
                 hide_index=True)
    st.subheader("Test-Set Performance")
    st.caption("Evaluated on a held-out test set of 184 patients (threshold 0.50).")
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
              "specificity": "Specificity", "f1": "F1 score", "roc_auc": "ROC-AUC"}
    st.dataframe(pd.DataFrame({v: [round(tm[k], 3)] for k, v in labels.items()}), hide_index=True)

# ------------------------------------------------------------------ figures tab
with tab_figs:
    figs = sorted(FIG.glob("*.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(f), caption=f.stem.split("_", 1)[1].replace("_", " ").title(),
                      width="stretch")
