"""Full experiment: EDA -> preprocessing -> baselines -> Random Forest -> tuning
-> evaluation -> interpretation. Saves figures/, results/ and models/.

Run:  python src/train.py
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
FIG, RES, MOD = ROOT / "figures", ROOT / "results", ROOT / "models"
for d in (FIG, RES, MOD):
    d.mkdir(exist_ok=True)

NUM = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CAT = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET = "HeartDisease"

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4C72B0", 1: "#C44E52"}


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=300)
    plt.close()


# --------------------------------------------------------------------------- EDA
def eda(df: pd.DataFrame) -> dict:
    info = {
        "n_rows": len(df),
        "n_features": len(NUM) + len(CAT),
        "class_counts": df[TARGET].value_counts().sort_index().to_dict(),
        "missing": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "rows_per_source": df["Source"].value_counts().to_dict(),
        "disease_rate_per_source": df.groupby("Source")[TARGET].mean().round(3).to_dict(),
        "sex_counts": df["Sex"].value_counts().to_dict(),
        "disease_rate_by_sex": df.groupby("Sex")[TARGET].mean().round(3).to_dict(),
        "numeric_summary": df[NUM].describe().round(2).to_dict(),
    }

    # Class balance
    ax = sns.countplot(x=TARGET, data=df, hue=TARGET, palette=PALETTE, legend=False)
    ax.set_xticks([0, 1], ["No disease (0)", "Heart disease (1)"])
    ax.set(xlabel="", ylabel="Patients", title="Class distribution")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    savefig("01_class_distribution.png")

    # Numeric distributions by class
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, NUM):
        sns.histplot(df, x=col, hue=TARGET, kde=col != "FastingBS", palette=PALETTE,
                     element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(col)
    fig.suptitle("Numeric feature distributions by class")
    savefig("02_numeric_distributions.png")

    # Disease rate per categorical value
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for ax, col in zip(axes, CAT):
        rate = df.groupby(col)[TARGET].mean().sort_values()
        ax.bar(rate.index.astype(str), rate.values, color="#C44E52")
        ax.set(title=col, ylim=(0, 1))
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    axes[0].set_ylabel("Heart-disease rate")
    fig.suptitle("Heart-disease rate per category")
    savefig("03_categorical_disease_rate.png")

    # Correlation heatmap (one-hot encoded)
    enc = pd.get_dummies(df[NUM + CAT + [TARGET]], columns=CAT, dtype=float)
    plt.figure(figsize=(13, 11))
    sns.heatmap(enc.corr(), cmap="RdBu_r", center=0, annot=True, fmt=".1f",
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    plt.title("Correlation matrix (one-hot encoded features)")
    savefig("04_correlation_heatmap.png")
    info["top_target_correlations"] = (enc.corr()[TARGET].drop(TARGET)
                                       .sort_values(key=abs, ascending=False).head(10)
                                       .round(3).to_dict())
    return info


# ----------------------------------------------------------------- preprocessing
def make_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num_steps), NUM),
                              ("cat", Pipeline(cat_steps), CAT)],
                             verbose_feature_names_out=False)


def metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "heart.csv")
    results = {"eda": eda(df)}

    X, y = df[NUM + CAT], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    results["split"] = {"train": len(X_tr), "test": len(X_te),
                        "train_pos_rate": round(y_tr.mean(), 3), "test_pos_rate": round(y_te.mean(), 3)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ------------------------------------------------------- baselines + default RF
    models = {
        "Logistic Regression": Pipeline([("prep", make_preprocessor(True)),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "KNN (k=7)": Pipeline([("prep", make_preprocessor(True)),
                               ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "Decision Tree": Pipeline([("prep", make_preprocessor(False)),
                                   ("clf", DecisionTreeClassifier(random_state=SEED))]),
        "Random Forest (default)": Pipeline([("prep", make_preprocessor(False)),
                                             ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
    }

    # --------------------------------------------------------------- tuning
    rf_pipe = Pipeline([("prep", make_preprocessor(False)),
                        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1))])
    grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 4, 6, 8, 10],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", 0.5],
        "clf__class_weight": [None, "balanced"],
    }
    t0 = time.time()
    search = GridSearchCV(rf_pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    results["tuning"] = {
        "search_space": {k.replace("clf__", ""): [str(x) for x in v] for k, v in grid.items()},
        "n_combinations": n_combos, "cv_folds": 5, "scoring": "roc_auc",
        "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        "best_cv_roc_auc": search.best_score_,
        "search_seconds": round(time.time() - t0, 1),
    }
    print("Best params:", results["tuning"]["best_params"], f"CV AUC={search.best_score_:.4f}")

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    tuned = Pipeline([("prep", make_preprocessor(False)),
                      ("clf", RandomForestClassifier(**best_params, oob_score=True,
                                                     random_state=SEED, n_jobs=-1))])
    models["Random Forest (tuned)"] = tuned

    # ------------------------------------------------------------ evaluation
    table, probs = {}, {}
    for name, model in models.items():
        cv_acc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = metrics(y_te, pred, prob)
        m["train_accuracy"] = accuracy_score(y_tr, model.predict(X_tr))
        m["cv_accuracy_mean"], m["cv_accuracy_std"] = cv_acc.mean(), cv_acc.std()
        m["cv_roc_auc_mean"], m["cv_roc_auc_std"] = cv_auc.mean(), cv_auc.std()
        table[name], probs[name] = m, prob
        print(f"{name:26s} acc={m['accuracy']:.3f} rec={m['recall']:.3f} "
              f"f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")
    results["models"] = table
    results["oob_score"] = tuned.named_steps["clf"].oob_score_

    pd.DataFrame(table).T.round(4).to_csv(RES / "model_comparison.csv")

    # Confusion matrix (tuned RF)
    pred_rf = (probs["Random Forest (tuned)"] >= 0.5).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred_rf),
                           display_labels=["No disease", "Disease"]).plot(cmap="Blues", colorbar=False)
    plt.title("Confusion matrix — tuned Random Forest (test set)")
    plt.grid(False)
    savefig("05_confusion_matrix.png")

    # ROC curves
    plt.figure(figsize=(7, 6))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, lw=2.5 if "tuned" in name else 1.5,
                 label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate (1 − specificity)")
    plt.ylabel("True positive rate (recall)")
    plt.title("ROC curves — test set")
    plt.legend(loc="lower right", fontsize=9)
    savefig("06_roc_curves.png")

    # Precision-recall curve (tuned RF)
    prec, rec, _ = precision_recall_curve(y_te, probs["Random Forest (tuned)"])
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–recall curve — tuned RF (AP={table['Random Forest (tuned)']['avg_precision']:.3f})")
    savefig("07_precision_recall.png")

    # Model comparison bar chart
    comp = pd.DataFrame(table).T[["accuracy", "recall", "f1", "roc_auc"]]
    ax = comp.plot(kind="bar", figsize=(10, 5.5), ylim=(0.6, 1.0), rot=15, width=0.8)
    ax.grid(axis="x", visible=False)
    plt.title("Model comparison — test set", pad=32); plt.ylabel("Score")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    savefig("08_model_comparison.png")

    # ---------------------------------------------------------- interpretation
    rf = tuned.named_steps["clf"]
    feat_names = tuned.named_steps["prep"].get_feature_names_out()
    mdi = pd.Series(rf.feature_importances_, index=feat_names).sort_values()
    perm = permutation_importance(tuned, X_te, y_te, n_repeats=30, random_state=SEED,
                                  scoring="roc_auc", n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=X_te.columns).sort_values()
    results["feature_importance_mdi"] = mdi.sort_values(ascending=False).round(4).to_dict()
    results["feature_importance_permutation"] = perm_s.sort_values(ascending=False).round(4).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    mdi.tail(15).plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Gini importance (MDI) — top 15 encoded features")
    perm_s.plot(kind="barh", ax=axes[1], color="#C44E52",
                xerr=pd.Series(perm.importances_std, index=X_te.columns)[perm_s.index])
    axes[1].set_title("Permutation importance (drop in test ROC-AUC)")
    savefig("09_feature_importance.png")

    # One shallow tree from the forest
    plt.figure(figsize=(22, 9))
    small = RandomForestClassifier(n_estimators=1, max_depth=3, random_state=SEED)
    small.fit(tuned.named_steps["prep"].transform(X_tr), y_tr)
    plot_tree(small.estimators_[0], feature_names=feat_names, class_names=["No", "Yes"],
              filled=True, rounded=True, fontsize=10, impurity=True)
    plt.title("Example decision tree from a Random Forest (depth limited to 3 for display)")
    savefig("10_example_tree.png")

    # ---------------------------------------------------------- analysis
    # OOB / test error vs number of trees
    Xtr_t = tuned.named_steps["prep"].transform(X_tr)
    Xte_t = tuned.named_steps["prep"].transform(X_te)
    n_list = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
    oob_err, test_err = [], []
    base = {k: v for k, v in best_params.items() if k != "n_estimators"}
    for n in n_list:
        m = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=SEED,
                                   n_jobs=-1, **base).fit(Xtr_t, y_tr)
        oob_err.append(1 - m.oob_score_)
        test_err.append(1 - m.score(Xte_t, y_te))
    tree_err = 1 - table["Decision Tree"]["accuracy"]
    plt.figure(figsize=(8, 5))
    plt.plot(n_list, oob_err, "o-", label="OOB error")
    plt.plot(n_list, test_err, "s-", label="Test error")
    plt.axhline(tree_err, color="grey", ls="--", label=f"Single decision tree test error ({tree_err:.3f})")
    plt.xscale("log"); plt.xlabel("Number of trees (log scale)"); plt.ylabel("Error rate")
    plt.title("Error vs number of trees")
    plt.legend()
    savefig("11_error_vs_trees.png")
    results["error_vs_trees"] = {"n_trees": n_list, "oob_error": oob_err, "test_error": test_err}

    # Threshold analysis (recall vs specificity)
    p = probs["Random Forest (tuned)"]
    thr_rows = []
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        r = metrics(y_te, (p >= t).astype(int), p)
        thr_rows.append({"threshold": t, **{k: r[k] for k in
                                             ("accuracy", "precision", "recall", "specificity", "f1", "fn", "fp")}})
    thr = pd.DataFrame(thr_rows)
    thr.round(4).to_csv(RES / "threshold_analysis.csv", index=False)
    results["threshold_analysis"] = thr_rows
    plt.figure(figsize=(8, 5))
    for col in ("accuracy", "precision", "recall", "specificity"):
        plt.plot(thr["threshold"], thr[col], "o-", label=col)
    plt.axvline(0.5, color="grey", ls=":")
    plt.xlabel("Decision threshold"); plt.ylabel("Score")
    plt.title("Effect of decision threshold — tuned RF")
    plt.legend()
    savefig("12_threshold_analysis.png")

    # ---------------------------------------------------------- save
    # Final demo model: retrain tuned pipeline on ALL data
    final = Pipeline([("prep", make_preprocessor(False)),
                      ("clf", RandomForestClassifier(**best_params, random_state=SEED, n_jobs=-1))])
    final.fit(X, y)
    joblib.dump({"model": final, "features": NUM + CAT, "best_params": best_params,
                 "test_metrics": table["Random Forest (tuned)"]}, MOD / "rf_heart.joblib")

    with open(RES / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"OOB score: {results['oob_score']:.4f}")
    print("Saved figures/, results/, models/rf_heart.joblib")


if __name__ == "__main__":
    main()
