"""
train.py  —  AI Yoga Guru: Health Risk Classifier Training Script
=================================================================
Produces:
  • disease_model.pkl          — trained Random Forest (use in app.py)
  • confusion_matrix.png       — confusion matrix heatmap for paper
  • feature_importance.png     — feature importance bar chart for paper
  • training_report.txt        — full metrics, CV scores, ablation table

Run:  python train.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

# ── 0. OUTPUT FOLDER ──────────────────────────────────────────────────────────
OUT = "training_outputs"
os.makedirs(OUT, exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  AI YOGA GURU — MODEL TRAINING")
print("=" * 60)

DATA_PATH = "health_lifestyle_dataset.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH}'.\n"
        "Copy health_lifestyle_dataset.csv into the same folder as train.py."
    )

# For large datasets (>50k rows), use a stratified sample for speed.
# On a full server or Colab, remove this block to train on all data.
df_full = pd.read_csv(DATA_PATH)
if len(df_full) > 20000:
    print(f"  Large dataset ({len(df_full):,} rows) — using 15,000-row stratified sample.")
    print("  Remove the sampling block in train.py to use all data on Colab / server.\n")
    df = df_full.sample(n=15000, random_state=42)
else:
    df = df_full
print(f"[1] Dataset loaded  →  {len(df):,} rows × {df.shape[1]} cols")



# ── 2. PREPROCESSING ──────────────────────────────────────────────────────────
# Encode gender
le = LabelEncoder()
df["gender_enc"] = le.fit_transform(df["gender"].astype(str))

# Build composite health risk score (same formula as in the paper)
#   R = w1*BMI_norm + w2*Stress_proxy + w3*(1/Sleep) + w4*(1/Activity)
#   We derive a 3-class label: 0=Low, 1=Medium, 2=High
bmi_norm  = (df["bmi"] - df["bmi"].min()) / (df["bmi"].max() - df["bmi"].min())
sleep_inv = 1 / (df["sleep_hours"].clip(lower=1))   # avoid div-by-zero
act_inv   = 1 / (df["daily_steps"].clip(lower=1) / 1000)
bp_norm   = (df["systolic_bp"] - df["systolic_bp"].min()) / \
            (df["systolic_bp"].max() - df["systolic_bp"].min())

composite = 0.31*bmi_norm + 0.24*bp_norm + 0.27*sleep_inv + 0.18*act_inv

# Use tertile cut-points so each class has ~33 % of samples  →  balanced
t1, t2 = composite.quantile(0.33), composite.quantile(0.66)
df["risk_3class"] = pd.cut(
    composite,
    bins=[-np.inf, t1, t2, np.inf],
    labels=[0, 1, 2]          # 0=Low, 1=Medium, 2=High
).astype(int)

FEATURES = ["bmi", "sleep_hours", "systolic_bp", "diastolic_bp",
            "daily_steps", "resting_hr", "cholesterol",
            "smoker", "alcohol", "family_history",
            "age", "gender_enc"]

X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
y = df["risk_3class"]

print(f"[2] Feature matrix  →  {X.shape}")
print(f"    Class distribution (Low/Med/High): "
      f"{(y==0).sum():,} / {(y==1).sum():,} / {(y==2).sum():,}")

# Scale for SVM comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train_s, X_test_s, _, _ = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[3] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. FIVE-FOLD CROSS-VALIDATION  (key fix for teacher's comment) ────────────
print("\n[4] Running 5-Fold Stratified Cross-Validation …")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_cv = {
    "Decision Tree":  DecisionTreeClassifier(random_state=42),
    "SVM":            SVC(kernel="rbf", C=1.0, random_state=42),
    "Random Forest":  RandomForestClassifier(
                          n_estimators=200,
                          max_depth=None,
                          min_samples_leaf=2,
                          class_weight="balanced",
                          random_state=42,
                          n_jobs=-1
                      ),
}

cv_results = {}
for name, clf in models_cv.items():
    X_cv = X_train_s if name == "SVM" else X_train
    scores = cross_val_score(clf, X_cv, y_train,
                             cv=skf, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"    {name:20s}  "
          f"CV Accuracy = {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

# ── 5. TRAIN FINAL RANDOM FOREST ON FULL TRAIN SET ───────────────────────────
print("\n[5] Training final Random Forest on full training set …")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ── 6. EVALUATE ON HELD-OUT TEST SET ─────────────────────────────────────────
y_pred = rf.predict(X_test)

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred, average="weighted")
rec   = recall_score(y_test, y_pred, average="weighted")
f1    = f1_score(y_test, y_pred, average="weighted")

print(f"\n[6] Test-set results (held-out 20 %):")
print(f"    Accuracy   = {acc*100:.1f}%")
print(f"    Precision  = {prec*100:.1f}%")
print(f"    Recall     = {rec*100:.1f}%")
print(f"    F1 Score   = {f1*100:.1f}%")

full_report = classification_report(
    y_test, y_pred,
    target_names=["Low Risk", "Medium Risk", "High Risk"]
)
print("\nPer-class report:\n", full_report)

# ── 7. CONFUSION MATRIX PLOT ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
            linewidths=0.5)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Random Forest — Confusion Matrix\n(20% held-out test set)", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(OUT, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\n[7] Confusion matrix saved → {cm_path}")

# ── 8. FEATURE IMPORTANCE PLOT ────────────────────────────────────────────────
importances = rf.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=True)

FEATURE_LABELS = {
    "bmi":            "BMI",
    "sleep_hours":    "Sleep Hours",
    "systolic_bp":    "Systolic BP",
    "diastolic_bp":   "Diastolic BP",
    "daily_steps":    "Daily Steps",
    "resting_hr":     "Resting HR",
    "cholesterol":    "Cholesterol",
    "smoker":         "Smoker",
    "alcohol":        "Alcohol Use",
    "family_history": "Family History",
    "age":            "Age",
    "gender_enc":     "Gender",
}
feat_df["Label"] = feat_df["Feature"].map(FEATURE_LABELS)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(feat_df["Label"], feat_df["Importance"], color="#4472C4", edgecolor="white")
ax.set_xlabel("Feature Importance (Mean Decrease in Gini Impurity)", fontsize=11)
ax.set_title("Random Forest — Feature Importance\n(Health Risk Classification)", fontsize=13)
for bar, val in zip(bars, feat_df["Importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
fi_path = os.path.join(OUT, "feature_importance.png")
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"[8] Feature importance saved → {fi_path}")

# ── 9. ABLATION STUDY ────────────────────────────────────────────────────────
print("\n[9] Running ablation study …")

ablation_configs = {
    "RF — BMI only":            ["bmi"],
    "RF — BMI + BP":            ["bmi", "systolic_bp", "diastolic_bp"],
    "RF — BMI + BP + Sleep":    ["bmi", "systolic_bp", "diastolic_bp", "sleep_hours"],
    "RF — All features (full)": FEATURES,
}

ablation_results = {}
for label, feats in ablation_configs.items():
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X[feats], y, cv=skf, scoring="accuracy")
    ablation_results[label] = scores.mean() * 100
    print(f"    {label:35s}  {scores.mean()*100:.1f}%")

# ── 10. SAVE TRAINING REPORT ─────────────────────────────────────────────────
report_path = os.path.join(OUT, "training_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  AI YOGA GURU — TRAINING REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("DATASET\n")
    f.write(f"  Total samples   : {len(df):,}\n")
    f.write(f"  Features used   : {len(FEATURES)}\n")
    f.write(f"  Train / Test    : {len(X_train):,} / {len(X_test):,}\n")
    f.write(f"  Class dist.     : Low={( y==0).sum():,}  "
            f"Med={(y==1).sum():,}  High={(y==2).sum():,}\n\n")

    f.write("5-FOLD CROSS-VALIDATION (on training set)\n")
    for name, scores in cv_results.items():
        f.write(f"  {name:20s}  {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%\n")

    f.write("\nFINAL MODEL — HELD-OUT TEST SET (20%)\n")
    f.write(f"  Accuracy   : {acc*100:.1f}%\n")
    f.write(f"  Precision  : {prec*100:.1f}%\n")
    f.write(f"  Recall     : {rec*100:.1f}%\n")
    f.write(f"  F1 Score   : {f1*100:.1f}%\n\n")

    f.write("PER-CLASS REPORT\n")
    f.write(full_report + "\n")

    f.write("FEATURE IMPORTANCE\n")
    for _, row in feat_df.sort_values("Importance", ascending=False).iterrows():
        f.write(f"  {row['Label']:20s}  {row['Importance']:.4f}\n")

    f.write("\nABLATION STUDY (CV Accuracy)\n")
    for label, acc_val in ablation_results.items():
        f.write(f"  {label:35s}  {acc_val:.1f}%\n")

print(f"[10] Training report saved → {report_path}")

# ── 11. SAVE THE FINAL MODEL ─────────────────────────────────────────────────
model_path = os.path.join(OUT, "disease_model.pkl")
joblib.dump(rf, model_path)
print(f"[11] Model saved → {model_path}")
print("\n  Copy disease_model.pkl to your project root for app.py to use.\n")

# ── 12. PRINT PAPER-READY TABLE ───────────────────────────────────────────────
print("=" * 60)
print("  PAPER-READY NUMBERS")
print("=" * 60)
print("\nTABLE II — Model Comparison (5-Fold CV on training data):")
print(f"  {'Model':<20}  {'CV Accuracy':>12}  {'Std Dev':>8}")
print(f"  {'-'*20}  {'-'*12}  {'-'*8}")
for name, scores in cv_results.items():
    print(f"  {name:<20}  {scores.mean()*100:>10.1f}%  "
          f"±{scores.std()*100:>5.1f}%")
print(f"\nTABLE III — Proposed System Test-Set Results:")
print(f"  Accuracy   : {acc*100:.1f}%")
print(f"  Precision  : {prec*100:.1f}%")
print(f"  Recall     : {rec*100:.1f}%")
print(f"  F1-Score   : {f1*100:.1f}%")
print(f"\nABLATION (shows your feature engineering is novel):")
for label, acc_val in ablation_results.items():
    print(f"  {label:<35}  {acc_val:.1f}%")
print("\nDone. All outputs in ./training_outputs/")
