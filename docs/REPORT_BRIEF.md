# Heart Disease Prediction using Random Forest — FAI Case Study

> **What this file is:** the complete brief and source material for the FAI case-study report (~10 pages). The code in this repo has been **run**; every number below is a real result from `src/train.py` (seed 42) and is also stored in [`results/metrics.json`](../results/metrics.json) and [`results/model_comparison.csv`](../results/model_comparison.csv). All figures referenced are in [`figures/`](../figures/).
>
> **To produce the report:** follow the structure in **§7**, use the section content in **§8**, insert the figures listed in **§9**, cite the references in **§10**. Do not invent or round numbers differently from those given here.

---

## 1. Assignment

- **Course:** Foundations of Artificial Intelligence (FAI)
- **Task:** Pick one ML algorithm, apply it to one case study (an application), and write a ~10-page report.
- **Algorithm:** **Random Forest** (supervised, ensemble of decision trees)
- **Application:** **Predicting the presence of heart disease** from 11 routine clinical attributes (binary classification)

---

## 2. Repository & How to Run the Demo

```
FAI/
├── README.md               ← this brief
├── requirements.txt
├── app.py                  ← interactive Streamlit demo (web UI)
├── src/
│   ├── prepare_data.py     ← downloads/combines 4 UCI files → data/heart.csv
│   ├── train.py            ← full experiment: EDA, baselines, RF, tuning, evaluation, plots
│   └── predict.py          ← command-line demo for one patient
├── data/                   ← raw UCI files + heart.csv (920 rows)
├── figures/                ← 12 PNG figures (300 dpi) for the report
├── results/                ← metrics.json, model_comparison.csv, threshold_analysis.csv
└── models/rf_heart.joblib  ← trained pipeline (preprocessing + tuned RF, fit on all data)
```

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python src/prepare_data.py         # build data/heart.csv
python src/train.py                # ~3 min: trains everything, writes figures/ results/ models/
streamlit run app.py               # opens the web demo at http://localhost:8501

# CLI demo
python src/predict.py --age 61 --sex M --cp ASY --bp 140 --chol 260 --fbs 0 --ecg Normal --maxhr 120 --exang Y --oldpeak 2.0 --slope Flat
```

**Demo app features** (useful for a screenshot in the report/appendix): sidebar sliders for all 11 patient attributes → predicted risk %, number of trees voting "disease" out of 300, adjustable decision threshold, a chart of each tree's individual probability (shows the ensemble voting), best hyperparameters, test metrics, and a gallery of all result figures.

Environment used: Python 3.14, scikit-learn 1.9.1, pandas 3.0.6, Streamlit 1.65.

---

## 3. Dataset

**Source:** UCI Machine Learning Repository — *Heart Disease* dataset (Janosi, Steinbrunn, Pfisterer, Detrano, 1988). The four processed sub-datasets were combined:

| Source | Patients | Heart-disease rate |
|---|---|---|
| Cleveland Clinic Foundation (USA) | 303 | 45.9% |
| Hungarian Institute of Cardiology, Budapest | 294 | 36.1% |
| V.A. Medical Center, Long Beach (USA) | 200 | 74.5% |
| University Hospitals Zurich & Basel (Switzerland) | 123 | 93.5% |
| **Total** | **920** | **55.3%** |

(This is the same underlying data as the popular Kaggle "Heart Failure Prediction" dataset, 918 rows, which is the same 4 sources with missing values filled in.)

**Target:** original attribute `num` (0 = < 50% vessel narrowing, 1–4 = > 50% narrowing in at least one major vessel, i.e. angiographic disease) → binarised: `HeartDisease = 1 if num > 0`.
**Class balance:** 509 disease (55.3%) vs 411 no disease (44.7%), mildly imbalanced.

| Feature | Type | Description | Missing |
|---|---|---|---|
| Age | numeric | years (28–77, mean 53.5) | 0 |
| Sex | categorical | M (726) / F (194) | 0 |
| ChestPainType | categorical | TA typical angina, ATA atypical angina, NAP non-anginal pain, ASY asymptomatic | 0 |
| RestingBP | numeric | resting blood pressure, mm Hg (mean 132.3) | 60 |
| Cholesterol | numeric | serum cholesterol, mg/dl (mean 246.8) | 202 (22%) |
| FastingBS | binary | fasting blood sugar > 120 mg/dl | 90 |
| RestingECG | categorical | Normal / ST (ST-T abnormality) / LVH (left-ventricular hypertrophy) | 2 |
| MaxHR | numeric | maximum heart rate achieved during exercise test (mean 137.6) | 55 |
| ExerciseAngina | categorical | exercise-induced angina Y/N | 55 |
| Oldpeak | numeric | ST depression induced by exercise relative to rest (mean 0.9) | 62 |
| ST_Slope | categorical | slope of peak-exercise ST segment: Up / Flat / Down | 309 (34%) |

**Excluded:** `ca` (number of major vessels) and `thal` (thalassemia). They are missing for most non-Cleveland patients, and `ca` is measured by fluoroscopy (the same invasive procedure that defines the label), so including it would leak the answer.
**Data-quality note:** Cholesterol = 0 (172 rows, almost all from Switzerland) and RestingBP = 0 (1 row) are physiologically impossible. They were recoded as missing.

**Key EDA findings** (from `results/metrics.json`):
- Strongest correlations with the target: ChestPainType_ASY **+0.52**, ExerciseAngina_Y **+0.43**, ChestPainType_ATA −0.40, MaxHR **−0.40**, Oldpeak **+0.39**, ST_Slope_Flat +0.34, Sex_M +0.31, Age +0.28.
- Disease rate: men **63.2%** vs women **25.8%**. The dataset is 79% male.
- Patients with *asymptomatic* chest pain have the highest disease rate. This is counter-intuitive, but it's a known clinical pattern ("silent ischaemia").
- Disease rate varies strongly by hospital (36% to 94%), which is a source of dataset shift.

---

## 4. Methodology (what `src/train.py` does)

1. **Split:** stratified 80/20 → **736 train / 184 test**. Disease rate is 55.3% in train and 55.4% in test. The test set is never used for training or tuning.
2. **Preprocessing (inside a scikit-learn `Pipeline`, so it's fitted on training folds only → no leakage):**
   - Numeric: median imputation. Standardisation is applied only for Logistic Regression and KNN; trees don't need it.
   - Categorical: missing values become their own category `"Unknown"` (missingness is informative), then one-hot encoding → 22 encoded features.
3. **Models compared:** Logistic Regression, KNN (k = 7), single Decision Tree, Random Forest (default: 100 trees), Random Forest (tuned).
4. **Hyperparameter tuning:** `GridSearchCV`, **540 combinations × 5-fold stratified CV = 2,700 fits**, scoring = ROC-AUC (97 s):

   | Hyperparameter | Values searched | **Best** |
   |---|---|---|
   | n_estimators | 100, 300 | **300** |
   | max_depth | None, 4, 6, 8, 10 | **6** |
   | min_samples_split | 2, 5, 10 | **10** |
   | min_samples_leaf | 1, 2, 4 | **1** |
   | max_features | sqrt, log2, 0.5 | **sqrt** (≈ 4 of 22 features per split) |
   | class_weight | None, balanced | **balanced** |

   Best cross-validated ROC-AUC = **0.874**.
5. **Evaluation:** accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC and average precision on the held-out test set; 5-fold CV mean ± std on the training set; out-of-bag (OOB) score.
6. **Interpretation:** Gini importance (MDI), permutation importance (30 repeats, drop in test ROC-AUC), and a plot of one example tree.
7. **Extra experiments:** error vs number of trees (5 to 500, OOB and test), and decision-threshold analysis (0.30 to 0.60).
8. **Demo model:** the tuned pipeline is refit on all 920 rows and saved to `models/rf_heart.joblib`.

---

## 5. Results (real numbers)

### 5.1 Test-set comparison (184 patients)

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Avg. Precision | Train Acc. |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | **0.821** | 0.822 | 0.863 | 0.768 | 0.842 | 0.900 | 0.918 | 0.810 |
| KNN (k=7) | 0.815 | 0.798 | **0.892** | 0.720 | **0.843** | 0.879 | 0.861 | 0.830 |
| Decision Tree | 0.647 | 0.687 | 0.667 | 0.622 | 0.677 | 0.644 | 0.643 | **1.000** |
| Random Forest (default) | 0.804 | 0.800 | 0.863 | 0.732 | 0.830 | 0.901 | 0.912 | 1.000 |
| **Random Forest (tuned)** | 0.799 | **0.828** | 0.804 | **0.793** | 0.816 | **0.908** | **0.927** | 0.878 |

### 5.2 5-fold cross-validation on the training set (mean ± std)

| Model | CV Accuracy | CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.788 ± 0.036 | 0.873 ± 0.023 |
| KNN (k=7) | 0.792 ± 0.031 | 0.849 ± 0.022 |
| Decision Tree | 0.732 ± 0.041 | 0.727 ± 0.046 |
| Random Forest (default) | 0.798 ± 0.027 | 0.860 ± 0.025 |
| **Random Forest (tuned)** | 0.794 ± 0.034 | **0.874 ± 0.024** |

**OOB accuracy (tuned RF, 300 trees): 0.785.** This is close to the CV accuracy (0.794) and the test accuracy (0.799), which confirms that OOB is a reliable built-in estimate of generalisation.

### 5.3 Confusion matrix — tuned RF, threshold 0.5

|  | Predicted: No disease | Predicted: Disease |
|---|---|---|
| **Actual: No disease (82)** | TN = 65 | FP = 17 |
| **Actual: Disease (102)** | FN = 20 | TP = 82 |

### 5.4 Feature importance (tuned RF)

| Rank | Gini / MDI (encoded features) | | Permutation (drop in test AUC, original features) | |
|---|---|---|---|---|
| 1 | ChestPainType_ASY | 0.164 | ChestPainType | 0.089 |
| 2 | ExerciseAngina_N | 0.105 | ExerciseAngina | 0.027 |
| 3 | MaxHR | 0.101 | Sex | 0.020 |
| 4 | Oldpeak | 0.099 | Oldpeak | 0.011 |
| 5 | ExerciseAngina_Y | 0.091 | MaxHR | 0.009 |
| 6 | Age | 0.086 | ST_Slope | 0.009 |
| 7 | ChestPainType_ATA | 0.059 | Cholesterol | 0.002 |
| 8 | Cholesterol | 0.055 | Age | 0.002 |
| 9 | ST_Slope_Flat | 0.041 | RestingBP | 0.001 |
| 10 | RestingBP | 0.038 | FastingBS | 0.001 |
| — | | | RestingECG | −0.001 (no value) |

### 5.5 Error vs number of trees (tuned settings)

| Trees | 5 | 10 | 20 | 30 | 50 | 100 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|---|---|---|
| OOB error | 0.247 | 0.220 | 0.226 | 0.209 | 0.204 | 0.209 | 0.205 | 0.215 | 0.205 |
| Test error | 0.217 | 0.201 | 0.207 | 0.174 | 0.179 | 0.201 | 0.190 | 0.201 | 0.207 |

For comparison, the single decision tree has a test error of **0.353**. Error drops quickly up to about 30–50 trees and then plateaus around 0.20; adding more trees does not cause overfitting.

### 5.6 Decision-threshold analysis (tuned RF, test set)

| Threshold | Accuracy | Precision | Recall | Specificity | F1 | Missed patients (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|
| 0.30 | 0.810 | 0.756 | **0.971** | 0.610 | **0.850** | **3** | 32 |
| 0.35 | 0.804 | 0.762 | 0.941 | 0.634 | 0.842 | 6 | 30 |
| 0.40 | 0.804 | 0.775 | 0.912 | 0.671 | 0.838 | 9 | 27 |
| 0.45 | 0.804 | 0.795 | 0.873 | 0.720 | 0.832 | 13 | 23 |
| 0.50 | 0.799 | 0.828 | 0.804 | 0.793 | 0.816 | 20 | 17 |
| 0.55 | 0.804 | 0.859 | 0.775 | 0.842 | 0.814 | 23 | 13 |
| 0.60 | 0.794 | 0.864 | 0.745 | 0.854 | 0.800 | 26 | 12 |

### 5.7 Demo predictions (CLI)
- 61-year-old man, asymptomatic chest pain, BP 140, cholesterol 260, exercise angina, MaxHR 120, Oldpeak 2.0, flat ST slope → **risk 93.0%, 299 / 300 trees vote "disease"** → HEART DISEASE.
- 40-year-old woman, atypical angina, BP 120, cholesterol 200, no exercise angina, MaxHR 170, Oldpeak 0, upsloping ST → **risk 3.1%, 0 / 300 trees** → NO HEART DISEASE.

---

## 6. Interpretation — the key messages for the report

1. **Random Forest fixes the single tree's overfitting.** The unpruned decision tree reaches 100% training accuracy but only 64.7% test accuracy (AUC 0.644). Averaging 100+ decorrelated trees raises test accuracy to about 80% (+15–16 points) and AUC to about 0.90. This is the bagging/variance-reduction theory working as predicted.
2. **Tuning acts as regularisation.** Limiting depth to 6 and requiring at least 10 samples per split lowers training accuracy from 100% to 87.8%. It also gives the best ranking quality of all models (**ROC-AUC 0.908, AP 0.927**) and the best precision (0.828) and specificity (0.793).
3. **Honest comparison:** at the default 0.5 threshold, Logistic Regression has slightly higher accuracy (0.821 vs 0.799). However, the gaps between LR, KNN and RF are **smaller than one CV standard deviation (about ±0.03)**, and their CV ROC-AUCs are practically tied (0.873 vs 0.874). So on this small, mostly linearly separable dataset the models perform statistically equivalently. RF's advantages are its handling of mixed and missing data without assumptions, its built-in OOB estimate and its feature-importance insight, not a large accuracy gain.
4. **The threshold matters more than the algorithm.** In screening, a missed patient (FN) is far worse than a false alarm. Lowering the threshold from 0.5 to 0.35 cuts missed patients from **20 to 6** (recall 0.80 → 0.94) while accuracy barely changes (0.799 → 0.804). The recommended operating point for a screening tool is **0.35–0.40**.
5. **Clinically sensible predictors.** Chest-pain type (especially *asymptomatic*), exercise-induced angina, low MaxHR and high Oldpeak (ST depression) dominate. These are all exercise-stress-test findings that cardiologists already use. Cholesterol, resting BP, fasting sugar and resting ECG add little once these are known.
6. **ST_Slope ranks lower than expected** (it's often the top feature in the Kaggle version) because 34% of its values are missing in our raw data. Our "Unknown" category retains some of the signal (ST_Slope_Unknown MDI 0.017).
7. **Sex is the 3rd-most important feature by permutation.** This reflects the 79%-male sample and the higher disease rate in men. It has fairness implications (see Ethics).
8. **MDI vs permutation disagree on Age** (6th vs 8th; permutation ≈ 0). MDI is known to inflate continuous, high-cardinality features, which is why permutation importance is reported alongside it.

---

## 7. Report Structure (target ≈ 10 pages, A4, 11–12 pt, 1.15–1.5 spacing, figures included)

| # | Section | ≈ Length | Figures / tables |
|---|---|---|---|
| — | Title page | 1 page | Title: *"Heart Disease Prediction using Random Forest: An FAI Case Study"*, student name, roll no., course, instructor, date |
| — | Abstract | 150–200 words | — |
| 1 | Introduction | 0.75 page | — |
| 2 | Problem Statement & Objectives | 0.33 page | — |
| 3 | Literature Review | 0.75 page | — |
| 4 | The Random Forest Algorithm | 2 pages | Fig. A (RF architecture diagram, draw it), pseudocode box, Fig. 10 (example tree) |
| 5 | Dataset & Exploratory Analysis | 1 page | Table: features; Fig. 1, Fig. 3 (+ Fig. 4 optional) |
| 6 | Methodology | 1 page | Fig. B (pipeline flowchart, draw it); table of hyperparameter grid; metric formulas |
| 7 | Results | 1.5 pages | Tables 5.1, 5.3; Figs. 6, 8 (or 5), 9, 11 |
| 8 | Discussion | 0.75 page | Table 5.6 / Fig. 12 |
| 9 | Ethics, Limitations | 0.5 page | — |
| 10 | Conclusion & Future Work | 0.4 page | — |
| — | References | 0.5 page | IEEE style |
| — | Appendix (optional) | — | Demo app screenshot, code snippets, repo link |

---

## 8. Section-by-Section Content

### Abstract (draft, adjust wording)
Cardiovascular disease is the leading cause of death worldwide, and early identification of at-risk patients can save lives. This case study applies the **Random Forest** algorithm, an ensemble of decorrelated decision trees trained on bootstrap samples, to predict the presence of heart disease from 11 routine clinical attributes. We combined the four UCI Heart Disease datasets (Cleveland, Hungary, Switzerland, Long Beach VA; 920 patients). Missing values were handled inside a leakage-free preprocessing pipeline, and the model was tuned with a 2,700-fit grid search under 5-fold cross-validation. On a held-out test set of 184 patients, the tuned Random Forest achieved a **ROC-AUC of 0.908**, accuracy of 79.9%, precision of 82.8% and specificity of 79.3%. This was the best ranking performance of five models, and it far outperformed a single decision tree (64.7% accuracy). Lowering the decision threshold to 0.35 raised recall to 94.1%, cutting missed diagnoses from 20 to 6. Feature-importance analysis identified asymptomatic chest pain, exercise-induced angina, maximum heart rate and ST depression as the key predictors, which agrees with clinical knowledge. We discuss the bias–variance behaviour of the ensemble, fairness concerns arising from a 79%-male dataset, and the limitations of the study.

### 1. Introduction
- CVD burden: an estimated **17.9 million deaths in 2019, about 32% of all global deaths** (WHO). Coronary artery disease is the main form. Definitive diagnosis (coronary angiography) is invasive and costly.
- Motivation for ML: non-invasive clinical measurements (age, chest-pain type, exercise-test results) carry predictive signal. ML can learn decision rules from historical patients to support triage, deciding who needs angiography first.
- Place within AI: supervised learning from examples (Russell & Norvig, Ch. 19). Decision trees are a classic symbolic learner, and ensembles are one of the most successful practical ML ideas.
- Why Random Forest: strong on small tabular data, handles mixed types, needs little preprocessing, resists overfitting, and gives interpretable feature importance.
- Objectives (bullet list): (1) explain the RF algorithm; (2) build an end-to-end pipeline on real clinical data; (3) compare RF with baseline classifiers; (4) interpret which clinical factors drive predictions; (5) discuss the clinical trade-off between missed cases and false alarms, and the ethical implications.
- One-sentence outline of the remaining sections.

### 2. Problem Statement
Given a patient described by a feature vector **x** = (age, sex, chest-pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise angina, oldpeak, ST slope), learn a function *f*: **x** → {0, 1}. The output is 1 if the patient has angiographic heart disease (> 50% narrowing of a major vessel). We also want a calibrated-ish probability P(y = 1 | **x**) so the decision threshold can be set according to clinical cost. A false negative (sending a sick patient home) is more costly than a false positive (an extra test), so **recall/sensitivity** is weighted heavily alongside ROC-AUC.

### 3. Literature Review (write ~1 paragraph each, cite §10)
- **Detrano et al. (1989):** origin of the Cleveland data; a logistic-regression-based probability model reached about 77% accuracy, and they validated it across the Hungarian, Swiss and VA hospitals, the same multi-centre setting we use.
- **Breiman (2001):** introduced Random Forests and showed error comparable to AdaBoost with more robustness to noise. **Breiman (1996)** introduced bagging.
- **Mohan, Thirumalai & Srivastava (2019, IEEE Access):** hybrid Random Forest + linear model (HRFLM) on the Cleveland data, reporting about 88.7% accuracy. This shows RF is a common and strong choice for this problem.
- **General finding in the literature:** on the Cleveland subset (303 clean patients), many papers report 83–90% accuracy for RF, SVM and LR. Our ~80% is lower because we use the harder **multi-centre 920-patient data with heavy missingness** and exclude the leaky `ca`/`thal` attributes. This is an honest, more realistic evaluation.
- **Lundberg & Lee (2017), Louppe et al. (2013):** model interpretation (SHAP; theory and biases of MDI importance), relevant to explainability in healthcare.
- **Gap/justification:** many student studies report only accuracy on a single split. This study adds CV variance, OOB estimation, permutation importance, threshold analysis and an ethics discussion.

### 4. The Random Forest Algorithm (core theory section, about 2 pages)

**4.1 Decision trees.** A tree recursively partitions feature space with axis-aligned tests (e.g. "Oldpeak ≤ 1.2?"). At each node it chooses the split that maximally reduces impurity:
- Gini impurity: **G(t) = 1 − Σₖ pₖ²**
- Entropy: **H(t) = −Σₖ pₖ log₂ pₖ**; information gain = H(parent) − Σ (n_child / n_parent) · H(child)

A fully grown tree has **low bias but high variance**. Our single tree scored 100% train vs 64.7% test, a textbook example. Include **Fig. 10** (example tree).

**4.2 Bagging (Breiman 1996).** Create *B* bootstrap samples (draw *n* rows with replacement), train one tree on each, and aggregate by majority vote (or averaged probabilities). Each bootstrap contains about 63.2% of the unique rows (since 1 − (1 − 1/n)ⁿ → 1 − 1/e). The other ~36.8% are **out-of-bag**.

**4.3 Random feature subspace → Random Forest (Breiman 2001).** At every split, only a random subset of *m* features is considered (classification default *m* = √d; here √22 ≈ 4). This **decorrelates** the trees. For *B* identically distributed trees with variance σ² and pairwise correlation ρ:

**Var(average) = ρσ² + ((1 − ρ) / B)·σ²**

Increasing *B* removes the second term, and feature randomness lowers ρ, shrinking the first. Bias stays roughly that of a single tree, so total error falls. **Fig. 11** shows this empirically: error falls from 0.25 (5 trees) and plateaus around 0.20 after about 50 trees, versus 0.353 for a single tree.

**4.4 Pseudocode** (put in a box/figure):
```
Input: training set D = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ, number of trees B, features per split m
for b = 1 … B:
    D_b ← bootstrap sample of n rows drawn from D with replacement
    grow tree T_b on D_b:
        at each node:
            select m of the d features at random
            pick the (feature, threshold) among them with largest Gini decrease
            split; recurse on children
        stop at max_depth, or if node is pure, or < min_samples_split samples
return forest {T₁ … T_B}

predict(x):  P(y=1|x) = (1/B) Σ_b P_b(y=1|x);   ŷ = 1 if P ≥ threshold else 0
```

**4.5 Out-of-bag error.** Each training row is predicted only by the trees that did *not* see it, which gives a free validation estimate. Ours: OOB accuracy 0.785 vs 5-fold CV 0.794 vs test 0.799.

**4.6 Feature importance.**
- *Mean Decrease in Impurity (MDI):* total Gini decrease from splits on a feature, averaged over trees. Fast, but biased towards continuous / high-cardinality features.
- *Permutation importance:* shuffle one feature in the test set and measure the drop in score. Model-agnostic and unbiased, but slower.

**4.7 Hyperparameters.** n_estimators, max_depth, max_features (*m*), min_samples_split, min_samples_leaf, bootstrap, class_weight, criterion. Explain how depth and leaf-size limits trade a little bias for lower variance.

**4.8 Complexity.** Training ≈ O(B · m · n log n); prediction ≈ O(B · depth). Trees are independent, so training is embarrassingly parallel.

**4.9 Advantages / disadvantages** (table):

| Advantages | Disadvantages |
|---|---|
| High accuracy on tabular data with little tuning | Less interpretable than a single tree ("black box" of 300 trees) |
| Robust to overfitting and noise; variance reduction | Larger memory footprint and slower prediction than linear models |
| Handles mixed numeric/categorical data; no scaling needed | Cannot extrapolate beyond the training range |
| Built-in OOB error and feature importance | MDI importance is biased; probabilities are not perfectly calibrated |
| Parallelisable | Often edged out by gradient boosting (XGBoost/LightGBM) |

### 5. Dataset & Exploratory Analysis
Use §3 of this file: source table, feature table, target definition, missing-value table, the reason for excluding `ca`/`thal`, and the impossible-zero cholesterol issue. Insert **Fig. 1** (class distribution) and **Fig. 3** (disease rate per category). Comment on ASY chest pain, exercise angina and flat ST slope having high disease rates, and on the male/female difference. Optionally add **Fig. 4** (correlation heatmap) or **Fig. 2** (numeric distributions: patients with disease have lower MaxHR and higher Oldpeak).

### 6. Methodology
Use §4 of this file. Draw a **pipeline flowchart (Fig. B)**: Raw UCI files → merge & clean → stratified 80/20 split → [Pipeline: impute → one-hot → classifier] → 5-fold Grid Search (train only) → evaluate on test → interpret → save model → demo app. Include the hyperparameter table and the metric formulas:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall (sensitivity) = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F1 = 2PR / (P + R)
- ROC-AUC = area under the TPR-vs-FPR curve = probability that a random diseased patient is ranked above a random healthy one

Tools: Python, scikit-learn, pandas, matplotlib/seaborn, Streamlit.

### 7. Results
Use §5 of this file: Table 5.1 (main comparison), CV table 5.2 (shows variance), best hyperparameters, OOB score, confusion matrix (Table 5.3 or **Fig. 5**), **Fig. 6** (ROC curves), **Fig. 8** (comparison bar chart), **Fig. 9** (feature importance), **Fig. 11** (error vs trees). State the numbers plainly and keep the interpretation for the Discussion.

### 8. Discussion
Use §6 of this file (points 1–8): variance reduction vs a single tree; tuning as regularisation; honest comparison with LR (statistically tied); threshold trade-off with **Table 5.6 / Fig. 12** and a recommended screening threshold of 0.35–0.40; clinical plausibility of the top features (stress-test findings); ST_Slope missingness; MDI vs permutation; and comparison with literature (lower than Cleveland-only papers because of multi-centre, missing-heavy data and no leaky features).

### 9. Ethical Considerations & Limitations
- **Bias/fairness:** 79% of patients are male, and Sex is the 3rd-most important feature. The model may be less reliable for women, whose heart disease often presents atypically. Future work should report metrics per subgroup.
- **Dataset shift:** disease prevalence ranges from 36% to 94% across hospitals. The data is from 1981–1988, so performance on modern or different populations is unknown. External validation is required.
- **Missing data:** 22% missing cholesterol and 34% missing ST slope. Imputation choices affect the results.
- **Small sample:** 920 patients and a 184-patient test set. A ±3% CV standard deviation means small differences between models are not significant.
- **Not a diagnostic device:** the model is a decision-support aid. It must not replace clinicians, and regulatory approval would be needed before clinical use.
- **Explainability & accountability:** clinicians need to understand why a patient was flagged. Feature importance helps globally; per-patient explanations (SHAP) would be needed in practice.
- **Privacy:** real patient data must be handled under HIPAA/GDPR. The UCI data is anonymised.
- **Threshold choice is an ethical decision:** it trades missed patients against unnecessary invasive tests.

### 10. Conclusion & Future Work
- Summary: RF achieved ROC-AUC 0.908 and about 80% accuracy on a realistic multi-centre dataset. It dramatically outperformed a single tree (+15 points accuracy), matched the strongest baseline, and identified clinically meaningful risk factors. Threshold tuning to 0.35 gives 94% sensitivity for screening.
- Future work: gradient boosting (XGBoost/LightGBM) comparison; SHAP per-patient explanations; probability calibration (Platt/isotonic); subgroup fairness evaluation; better imputation (MICE/KNN); larger modern EHR datasets and external validation; deployment of the Streamlit prototype as a clinical decision-support tool.

---

## 9. Figures (all in `figures/`, 300 dpi)

| File | Content | Use in section |
|---|---|---|
| `01_class_distribution.png` | Count of disease vs no disease (509 / 411) | 5 |
| `02_numeric_distributions.png` | Histograms/KDE of 6 numeric features by class | 5 (optional) |
| `03_categorical_disease_rate.png` | Disease rate per category for 5 categorical features | 5 |
| `04_correlation_heatmap.png` | Correlation matrix of one-hot encoded features | 5 (optional) |
| `05_confusion_matrix.png` | Confusion matrix, tuned RF, test set | 7 |
| `06_roc_curves.png` | ROC curves of all 5 models with AUCs | 7 |
| `07_precision_recall.png` | Precision–recall curve, tuned RF (AP 0.927) | 7 (optional) |
| `08_model_comparison.png` | Bar chart: accuracy/recall/F1/AUC per model | 7 |
| `09_feature_importance.png` | MDI (left) and permutation importance (right) | 7 |
| `10_example_tree.png` | One depth-3 tree from a forest | 4 |
| `11_error_vs_trees.png` | OOB and test error vs number of trees, with single-tree line | 4 or 7 |
| `12_threshold_analysis.png` | Accuracy/precision/recall/specificity vs threshold | 8 |
| *(to draw)* Fig. A | RF architecture: Training data → B bootstrap samples → B trees (random feature subsets) → majority vote → prediction | 4 |
| *(to draw)* Fig. B | Pipeline flowchart (see §8, section 6) | 6 |
| *(screenshot)* | Streamlit demo (`streamlit run app.py`) | Appendix |

Number the figures sequentially in the report, caption every figure, and refer to each one in the text.

---

## 10. References (IEEE style)

[1] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
[2] L. Breiman, "Bagging predictors," *Machine Learning*, vol. 24, no. 2, pp. 123–140, 1996.
[3] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York: Springer, 2009.
[4] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ: Pearson, 2020.
[5] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
[6] R. Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease," *American Journal of Cardiology*, vol. 64, no. 5, pp. 304–310, 1989.
[7] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1988. doi: 10.24432/C52P4X.
[8] World Health Organization, "Cardiovascular diseases (CVDs)," Fact sheet, 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
[9] S. Mohan, C. Thirumalai, and G. Srivastava, "Effective heart disease prediction using hybrid machine learning techniques," *IEEE Access*, vol. 7, pp. 81542–81554, 2019.
[10] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
[11] G. Louppe, L. Wehenkel, A. Sutera, and P. Geurts, "Understanding variable importances in forests of randomized trees," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2013.
[12] fedesoriano, "Heart Failure Prediction Dataset," Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

*(Check volume/page details and the accuracy figures attributed to [6] and [9] against the original papers before final submission.)*

---

## 11. Final Checklist

- [x] Working code: data prep, training/evaluation, CLI demo, Streamlit web demo
- [x] Real results in `results/` and 12 figures in `figures/`
- [ ] Report written following §7–§8 (~10 pages), with every number taken from §5
- [ ] Draw Fig. A (RF architecture) and Fig. B (pipeline flowchart)
- [ ] Screenshot of the Streamlit demo for the appendix
- [ ] All figures captioned, numbered and referenced in the text; tables numbered
- [ ] In-text citations [1]–[12]; references verified
- [ ] Title page, page numbers, spell-check; export to PDF
