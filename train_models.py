"""
train_models.py
===============
Student Performance Prediction System
Data Analytics & Visualization (DAV) Mini Project

Workflow:
  Dataset -> Preprocessing -> EDA -> Feature Selection ->
  Train/Test Split -> Decision Tree -> Random Forest ->
  Evaluation -> Save Models

Dataset columns (all numeric / binary encoded):
  StudentID, Age, Gender, Ethnicity, ParentalEducation,
  StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
  Extracurricular, Sports, Music, Volunteering, GPA, GradeClass

Target: Performance_Level derived from GPA
  Low    -> GPA < 1.5
  Medium -> 1.5 <= GPA < 3.0
  High   -> GPA >= 3.0

GPA and GradeClass are excluded from features to prevent data leakage.
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
DATA_PATH = "data/student_performance.csv"
print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n[2] Data Cleaning")

before = len(df)
df.drop_duplicates(inplace=True)
print(f"    Duplicates removed: {before - len(df)}")

missing = df.isnull().sum().sum()
print(f"    Missing values: {missing}")
if missing > 0:
    df.dropna(inplace=True)
    print(f"    Rows after dropping NaNs: {len(df)}")

# ─────────────────────────────────────────────
# 3. CREATE TARGET VARIABLE - Performance_Level
# ─────────────────────────────────────────────
# Threshold explanation:
#   GPA scale is 0.0 to 4.0
#   Low    : GPA < 1.5  (~D/F range)
#   Medium : 1.5 <= GPA < 3.0  (~C/B range)
#   High   : GPA >= 3.0  (~A/B+ range)
def gpa_to_performance(gpa):
    if gpa < 1.5:
        return "Low"
    elif gpa < 3.0:
        return "Medium"
    else:
        return "High"

df["Performance_Level"] = df["GPA"].apply(gpa_to_performance)
print("\n[3] Performance_Level distribution:")
dist = df["Performance_Level"].value_counts()
for label, count in dist.items():
    pct = count / len(df) * 100
    print(f"    {label:8s}: {count:4d} ({pct:.1f}%)")

# ─────────────────────────────────────────────
# 4. FEATURE SELECTION
# ─────────────────────────────────────────────
# Exclude: StudentID (identifier), GPA (used for target creation),
#          GradeClass (derived from GPA -> leakage), Performance_Level (target)
FEATURE_COLS = [
    "Age", "Gender", "Ethnicity", "ParentalEducation",
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering"
]
TARGET_COL = "Performance_Level"

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

print(f"\n[4] Features selected: {len(FEATURE_COLS)}")
print(f"    {FEATURE_COLS}")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[5] Train/Test split (80/20)")
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# 6. DECISION TREE CLASSIFIER
# ─────────────────────────────────────────────
print("\n[6] Training Decision Tree Classifier ...")

dt_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    ))
])
dt_pipeline.fit(X_train, y_train)
dt_preds = dt_pipeline.predict(X_test)

dt_acc       = accuracy_score(y_test, dt_preds)
dt_precision = precision_score(y_test, dt_preds, average="weighted", zero_division=0)
dt_recall    = recall_score(y_test, dt_preds, average="weighted", zero_division=0)
dt_f1        = f1_score(y_test, dt_preds, average="weighted", zero_division=0)
dt_cm        = confusion_matrix(y_test, dt_preds, labels=["Low", "Medium", "High"])

print(f"    Accuracy  : {dt_acc:.4f}")
print(f"    Precision : {dt_precision:.4f}")
print(f"    Recall    : {dt_recall:.4f}")
print(f"    F1 Score  : {dt_f1:.4f}")

# ─────────────────────────────────────────────
# 7. RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────
print("\n[7] Training Random Forest Classifier ...")

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

rf_acc       = accuracy_score(y_test, rf_preds)
rf_precision = precision_score(y_test, rf_preds, average="weighted", zero_division=0)
rf_recall    = recall_score(y_test, rf_preds, average="weighted", zero_division=0)
rf_f1        = f1_score(y_test, rf_preds, average="weighted", zero_division=0)
rf_cm        = confusion_matrix(y_test, rf_preds, labels=["Low", "Medium", "High"])

print(f"    Accuracy  : {rf_acc:.4f}")
print(f"    Precision : {rf_precision:.4f}")
print(f"    Recall    : {rf_recall:.4f}")
print(f"    F1 Score  : {rf_f1:.4f}")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
rf_clf = rf_pipeline.named_steps["classifier"]
feature_importances = dict(zip(FEATURE_COLS, rf_clf.feature_importances_.tolist()))
sorted_fi = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
print("\n[8] Feature Importances (Random Forest):")
for feat, imp in sorted_fi.items():
    print(f"    {feat:22s}: {imp:.4f}")

# ─────────────────────────────────────────────
# 9. SAVE MODELS
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/decision_tree.pkl", "wb") as f:
    pickle.dump(dt_pipeline, f)
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)
with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print("\n[9] Models saved to models/")

# ─────────────────────────────────────────────
# 10. SAVE METRICS AS JSON (used by Flask API)
# ─────────────────────────────────────────────
perf_dist = df["Performance_Level"].value_counts().to_dict()
study_by_perf_full = {}
absences_by_perf = {}
gpa_by_perf = {}

for level in ["Low", "Medium", "High"]:
    subset = df[df["Performance_Level"] == level]
    vals_study = subset["StudyTimeWeekly"].tolist()
    vals_abs   = subset["Absences"].tolist()
    vals_gpa   = subset["GPA"].tolist()

    study_by_perf_full[level] = {
        "min": round(float(np.min(vals_study)), 2),
        "q1":  round(float(np.percentile(vals_study, 25)), 2),
        "median": round(float(np.median(vals_study)), 2),
        "q3":  round(float(np.percentile(vals_study, 75)), 2),
        "max": round(float(np.max(vals_study)), 2),
        "mean": round(float(np.mean(vals_study)), 2)
    }
    absences_by_perf[level] = round(float(np.mean(vals_abs)), 2)
    gpa_by_perf[level] = {
        "min": round(float(np.min(vals_gpa)), 2),
        "q1":  round(float(np.percentile(vals_gpa, 25)), 2),
        "median": round(float(np.median(vals_gpa)), 2),
        "q3":  round(float(np.percentile(vals_gpa, 75)), 2),
        "max": round(float(np.max(vals_gpa)), 2),
        "mean": round(float(np.mean(vals_gpa)), 2)
    }

# Scatter sample for study hours vs GPA (max 300 points)
sample_df = df.sample(min(300, len(df)), random_state=42)
scatter_data = [
    {"x": round(float(row["StudyTimeWeekly"]), 2),
     "y": round(float(row["GPA"]), 2),
     "level": row["Performance_Level"]}
    for _, row in sample_df.iterrows()
]

# Correlation (numeric features + GPA)
corr_cols = FEATURE_COLS + ["GPA"]
corr_df = df[corr_cols].corr().round(3)

metrics = {
    "dataset": {
        "total_students": int(len(df)),
        "total_features": len(FEATURE_COLS),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_columns": FEATURE_COLS,
        "performance_distribution": {k: int(v) for k, v in perf_dist.items()}
    },
    "decision_tree": {
        "accuracy":  round(float(dt_acc), 4),
        "precision": round(float(dt_precision), 4),
        "recall":    round(float(dt_recall), 4),
        "f1_score":  round(float(dt_f1), 4),
        "confusion_matrix": dt_cm.tolist()
    },
    "random_forest": {
        "accuracy":  round(float(rf_acc), 4),
        "precision": round(float(rf_precision), 4),
        "recall":    round(float(rf_recall), 4),
        "f1_score":  round(float(rf_f1), 4),
        "confusion_matrix": rf_cm.tolist(),
        "feature_importances": sorted_fi
    },
    "visualizations": {
        "study_hours_by_performance": study_by_perf_full,
        "absences_by_performance": absences_by_perf,
        "gpa_by_performance": gpa_by_perf,
        "scatter_study_gpa": scatter_data,
        "correlation_labels": corr_cols,
        "correlation_matrix": corr_df.values.tolist()
    }
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n[10] Metrics saved to models/metrics.json")
print("\n" + "=" * 60)
print("TRAINING COMPLETE - Application is ready to launch!")
print("    Run:  python app.py")
print("=" * 60)
