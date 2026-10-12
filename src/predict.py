"""Command-line demo: predict heart-disease risk for one patient.

Example:
  python src/predict.py --age 61 --sex M --cp ASY --bp 140 --chol 260 --fbs 0 \
      --ecg Normal --maxhr 120 --exang Y --oldpeak 2.0 --slope Flat
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL = Path(__file__).resolve().parent.parent / "models" / "rf_heart.joblib"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--sex", choices=["M", "F"], required=True)
    p.add_argument("--cp", choices=["TA", "ATA", "NAP", "ASY"], required=True, help="chest pain type")
    p.add_argument("--bp", type=float, required=True, help="resting blood pressure (mm Hg)")
    p.add_argument("--chol", type=float, required=True, help="serum cholesterol (mg/dl)")
    p.add_argument("--fbs", type=int, choices=[0, 1], required=True, help="fasting blood sugar > 120")
    p.add_argument("--ecg", choices=["Normal", "ST", "LVH"], required=True)
    p.add_argument("--maxhr", type=float, required=True)
    p.add_argument("--exang", choices=["Y", "N"], required=True, help="exercise-induced angina")
    p.add_argument("--oldpeak", type=float, required=True)
    p.add_argument("--slope", choices=["Up", "Flat", "Down"], required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()

    bundle = joblib.load(MODEL)
    model = bundle["model"]
    x = pd.DataFrame([{
        "Age": a.age, "RestingBP": a.bp, "Cholesterol": a.chol, "FastingBS": a.fbs, "MaxHR": a.maxhr,
        "Oldpeak": a.oldpeak, "Sex": a.sex, "ChestPainType": a.cp, "RestingECG": a.ecg,
        "ExerciseAngina": a.exang, "ST_Slope": a.slope,
    }])[bundle["features"]]

    prob = model.predict_proba(x)[0, 1]
    rf = model.named_steps["clf"]
    xt = model.named_steps["prep"].transform(x)
    votes = int(np.sum([t.predict(xt)[0] for t in rf.estimators_]))

    print(f"Risk of heart disease : {prob:.1%}")
    print(f"Trees voting disease  : {votes} / {len(rf.estimators_)}")
    print(f"Prediction (t={a.threshold}) : {'HEART DISEASE' if prob >= a.threshold else 'NO HEART DISEASE'}")


if __name__ == "__main__":
    main()
