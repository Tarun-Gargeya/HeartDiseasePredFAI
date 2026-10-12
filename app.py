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

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")


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

st.title("❤️ Heart Disease Prediction — Random Forest")
st.caption("FAI Case Study · UCI Heart Disease data (920 patients) · "
           "Educational demo, **not** a medical device.")

tab_pred, tab_model, tab_figs = st.tabs(["🩺 Predict", "🌲 How the forest decides", "📊 Results"])

# ------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Patient details")
    age = st.slider("Age", 20, 80, 54)
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    cp = st.selectbox("Chest pain type", ["ASY", "NAP", "ATA", "TA"],
                      format_func=lambda v: {"ASY": "ASY – Asymptomatic", "NAP": "NAP – Non-anginal pain",
                                             "ATA": "ATA – Atypical angina", "TA": "TA – Typical angina"}[v])
    bp = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
    chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl?", [0, 1], horizontal=True,
                   format_func=lambda v: "Yes" if v else "No")
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxhr = st.slider("Max heart rate achieved", 60, 210, 140)
    exang = st.radio("Exercise-induced angina?", ["N", "Y"], horizontal=True,
                     format_func=lambda v: "Yes" if v == "Y" else "No")
    oldpeak = st.slider("Oldpeak (ST depression)", -2.5, 6.5, 1.0, 0.1)
    slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])
    threshold = st.slider("Decision threshold", 0.2, 0.8, 0.5, 0.05,
                          help="Lower = catch more patients (higher recall) at the cost of more false alarms.")

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
    c1.metric("Predicted risk of heart disease", f"{prob:.0%}")
    c2.metric("Trees voting 'disease'", f"{int(votes.sum())} / {len(votes)}")
    c3.metric("Decision (threshold " + f"{threshold:.2f})", "⚠️ At risk" if prob >= threshold else "✅ Low risk")
    st.progress(float(prob))
    if prob >= threshold:
        st.error("The model flags this patient as **likely to have heart disease**. "
                 "In a real setting this would prompt further clinical tests.")
    else:
        st.success("The model predicts **no heart disease** for this patient.")
    st.caption("Risk = average of the class probabilities from all trees in the forest.")

# ------------------------------------------------------------------ model tab
with tab_model:
    st.subheader("Votes from individual trees")
    st.write(f"The forest has **{len(rf.estimators_)} trees**, each trained on a different bootstrap sample "
             f"and considering only a random subset of features at every split "
             f"(`max_features={bundle['best_params'].get('max_features')}`). "
             "The final prediction combines all of them.")
    tree_probs = np.array([t.predict_proba(Xt)[0, 1] for t in rf.estimators_])
    st.bar_chart(pd.DataFrame({"P(disease) per tree": np.sort(tree_probs)}), height=250)
    st.subheader("Best hyperparameters (from 5-fold grid search)")
    st.json(bundle["best_params"])
    st.subheader("Held-out test-set performance")
    st.dataframe(pd.DataFrame({k: [tm[k]] for k in
                               ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")}).round(3),
                 hide_index=True)

# ------------------------------------------------------------------ figures tab
with tab_figs:
    figs = sorted(FIG.glob("*.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(f), caption=f.stem.split("_", 1)[1].replace("_", " ").title(),
                      width="stretch")
