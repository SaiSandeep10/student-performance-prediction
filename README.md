# 🎓 Student Performance Prediction System

> **Data Analytics & Visualization (DAV) Mini Project**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-prediction-5f385yovaugwbnyqtqp7f3.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Live Demo

**👉 [https://student-performance-prediction-5f385yovaugwbnyqtqp7f3.streamlit.app](https://student-performance-prediction-5f385yovaugwbnyqtqp7f3.streamlit.app)**

---

## 📌 Problem Statement

Student academic performance depends on many interrelated factors including study habits, attendance, parental support, and extracurricular involvement. Traditional analysis methods are insufficient to model these complex relationships. This project builds a machine learning system that:

- Analyzes a dataset of **2392 student records**
- Identifies patterns that correlate with academic outcomes
- Predicts whether a student is likely to achieve **Low**, **Medium**, or **High** performance
- Provides an **interactive Streamlit web app** for live prediction and visualization

---

## 🎯 Objective

Develop a complete, working web-based Data Analytics and Visualization mini project that demonstrates:

| Component | Implementation |
|---|---|
| Data Preprocessing | Cleaning, feature selection, StandardScaler |
| EDA | Distribution analysis, correlation, visual statistics |
| Machine Learning | Decision Tree + Random Forest |
| Model Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Visualization | Plotly interactive charts |
| Web Application | Streamlit (deployed on Streamlit Cloud) |

---

## 📊 Dataset

**File:** `data/student_performance.csv`  
**Source:** Student Performance Factors Dataset  
**Records:** 2392 students | **Features:** 15 columns

| Column | Description | Values |
|---|---|---|
| StudentID | Unique identifier | 1001–3392 |
| Age | Student age | 15–18 |
| Gender | Gender (encoded) | 0=Male, 1=Female |
| Ethnicity | Ethnicity group | 0–3 |
| ParentalEducation | Parents' highest education | 0=None … 4=Higher |
| StudyTimeWeekly | Hours studied per week | 0–40 (float) |
| Absences | Number of school absences | 0–30 |
| Tutoring | Receives tutoring | 0=No, 1=Yes |
| ParentalSupport | Level of parental support | 0–4 |
| Extracurricular | Participates in activities | 0=No, 1=Yes |
| Sports | Participates in sports | 0=No, 1=Yes |
| Music | Participates in music | 0=No, 1=Yes |
| Volunteering | Does volunteering | 0=No, 1=Yes |
| GPA | Grade Point Average | 0.0–4.0 |
| GradeClass | Letter grade class | 0=A … 4=F |

---

## 🎯 Target Variable: Performance_Level

Created from **GPA** using threshold-based classification:

| Performance Level | GPA Range | Interpretation |
|---|---|---|
| 🔴 **Low** | GPA < 1.5 | D/F range — struggling student |
| 🟡 **Medium** | 1.5 ≤ GPA < 3.0 | C/B range — average student |
| 🟢 **High** | GPA ≥ 3.0 | A/B+ range — high achiever |

> **Note:** GPA and GradeClass are **excluded** from features to prevent data leakage.

---

## 🔬 Features Used for Prediction

```
Age, Gender, Ethnicity, ParentalEducation,
StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
Extracurricular, Sports, Music, Volunteering
```
Total: **12 features**

---

## 🤖 Machine Learning Algorithms

### 1. Decision Tree Classifier
- **Why?** Interpretable, visualizable, no data distribution assumptions
- **Parameters:** `max_depth=10`, `min_samples_split=10`, `class_weight='balanced'`
- **Pipeline:** StandardScaler → DecisionTreeClassifier

### 2. Random Forest Classifier
- **Why?** Ensemble reduces overfitting, provides feature importance, robust to noise
- **Parameters:** `n_estimators=200`, `max_depth=12`, `class_weight='balanced'`
- **Pipeline:** StandardScaler → RandomForestClassifier

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Web App | Python 3.x, Streamlit |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Visualizations | Plotly |
| Fonts | Google Fonts (Inter) |
| Deployment | Streamlit Cloud |
| Model Storage | Pickle (.pkl files) |

---

## 📐 Data Preprocessing

1. **Load** CSV from `data/student_performance.csv`
2. **Remove duplicates** — checked and cleaned
3. **Handle missing values** — drop rows with NaN
4. **Create target variable** — GPA → Performance_Level
5. **Feature selection** — 12 input features (exclude StudentID, GPA, GradeClass)
6. **Train/test split** — 80% train / 20% test, `random_state=42`, stratified
7. **StandardScaler** — fitted on training set, same scaler applied at prediction time via Pipeline

---

## 📂 Project Structure

```
student-performance-prediction/
│
├── streamlit_app.py           # ⭐ Main Streamlit app (3 pages)
├── train_models.py            # ML training script
├── requirements.txt           # Python dependencies
├── README.md
│
├── data/
│   └── student_performance.csv
│
├── models/
│   ├── decision_tree.pkl      # Trained DT pipeline
│   ├── random_forest.pkl      # Trained RF pipeline
│   ├── feature_columns.pkl    # Feature list
│   └── metrics.json           # Evaluation results + chart data
│
└── .streamlit/
    └── config.toml            # Dark purple theme config
```

---

## ⚙️ Installation & Running Locally

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/SaiSandeep10/student-performance-prediction.git
cd student-performance-prediction
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Retrain the models
```bash
python train_models.py
```

### Step 4 — Launch the Streamlit app
```bash
streamlit run streamlit_app.py
```

### Step 5 — Open in browser
```
http://localhost:8501
```

---

## 📱 App Pages

| Page | Description |
|---|---|
| 🏠 **Dashboard** | Hero section, 4 metric cards, model comparison table, radar chart, performance pie chart |
| 🔮 **Prediction** | Algorithm selector, 12-field student form, probability bar chart, smart insights |
| 📊 **Analytics** | Feature importance, confusion matrices, study time box plots, scatter plot, correlation heatmap, GPA distribution |

---

## 📈 Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP+TN)/total | Overall correct predictions |
| Precision | TP/(TP+FP) | Of predicted positives, how many are correct |
| Recall | TP/(TP+FN) | Of actual positives, how many were found |
| F1 Score | 2·(P·R)/(P+R) | Harmonic mean of Precision and Recall |
| Confusion Matrix | Grid of actual vs predicted | Per-class breakdown |

---

## 🔮 Future Enhancements

1. Add more ML algorithms (SVM, KNN, Gradient Boosting)
2. Hyperparameter tuning with GridSearchCV
3. Cross-validation (k-fold) for more robust evaluation
4. SHAP values for explainable AI
5. Student cohort comparison
6. Export prediction report as PDF
7. Database integration for storing predictions
8. Authentication for teacher/admin portal

---

## 📝 Academic Note

This project was developed as a **Data Analytics & Visualization (DAV) laboratory mini project** demonstrating the complete ML workflow from raw data to a deployed web application.
