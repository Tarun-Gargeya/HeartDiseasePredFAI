"""Combine the four UCI Heart Disease files into one clean CSV (data/heart.csv).

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
Files: processed.{cleveland,hungarian,switzerland,va}.data (920 patients total)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.{}.data"
SOURCES = ["cleveland", "hungarian", "switzerland", "va"]
RAW_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load_raw(source: str) -> pd.DataFrame:
    path = DATA / f"processed.{source}.data"
    src = path if path.exists() else URL.format(source)
    df = pd.read_csv(src, header=None, names=RAW_COLS, na_values="?")
    df["Source"] = source.capitalize()
    return df


def main() -> None:
    df = pd.concat([load_raw(s) for s in SOURCES], ignore_index=True)

    out = pd.DataFrame({
        "Age": df["age"].astype(int),
        "Sex": df["sex"].map({1: "M", 0: "F"}),
        "ChestPainType": df["cp"].map({1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}),
        "RestingBP": df["trestbps"],
        "Cholesterol": df["chol"],
        "FastingBS": df["fbs"],
        "RestingECG": df["restecg"].map({0: "Normal", 1: "ST", 2: "LVH"}),
        "MaxHR": df["thalach"],
        "ExerciseAngina": df["exang"].map({1: "Y", 0: "N"}),
        "Oldpeak": df["oldpeak"],
        "ST_Slope": df["slope"].map({1: "Up", 2: "Flat", 3: "Down"}),
        "Source": df["Source"],
        # num: 0 = no disease, 1-4 = increasing severity -> binary target
        "HeartDisease": (df["num"] > 0).astype(int),
    })

    # Physiologically impossible zeros are really missing values
    out.loc[out["Cholesterol"] == 0, "Cholesterol"] = np.nan
    out.loc[out["RestingBP"] == 0, "RestingBP"] = np.nan

    out.to_csv(DATA / "heart.csv", index=False)
    print(f"Saved {len(out)} rows -> {DATA / 'heart.csv'}")
    print("Missing values per column:\n", out.isna().sum()[out.isna().sum() > 0])
    print("Target balance:\n", out["HeartDisease"].value_counts())


if __name__ == "__main__":
    main()
