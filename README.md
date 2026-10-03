# Student Performance Prediction System

**Data Analytics & Visualization (DAV) Mini Project**

A web-based machine learning application that predicts student academic performance (Low / Medium / High) using Decision Tree and Random Forest classifiers, served through a Flask web interface with interactive visualizations.

---

## Problem Statement

Student academic performance depends on many interrelated factors including study habits, attendance, parental support, and extracurricular involvement. Traditional analysis methods are insufficient to model these complex relationships. This project builds a machine learning system that:

- Analyzes a dataset of 2392 student records
- Identifies patterns that correlate with academic outcomes
- Predicts whether a student is likely to achieve Low, Medium, or High performance
- Provides an interactive web interface for live prediction and visualization

---

## Objective

Develop a complete, working web-based Data Analytics and Visualization mini project that demonstrates:

| Component | Implementation |
|---|---|
| Data Preprocessing | Cleaning, feature selection, StandardScaler |
| EDA | Distribution analysis, correlation, visual statistics |
| Machine Learning | Decision Tree + Random Forest |
| Model Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Visualization | Chart.js interactive charts |
| Web Application | Flask + HTML/CSS/JS |

---

## Dataset

**File:** `data/student_performance.csv`  
**Source:** Student Performance Factors Dataset  
**Records:** 2392 students  
**Features:** 15 columns

| Column | Description | Values |
|---|---|---|
| StudentID | Unique identifier | 1001–3392 |
| Age | Student age | 15–18 |
| Gender | Gender (encoded) | 0=Male, 1=Female |
| Ethnicity | Ethnicity group | 0–3 |
| ParentalEducation | Parents' highest education | 0=None, 1=HS, 2=College, 3=Bachelor's, 4=Higher |
| StudyTimeWeekly | Hours studied per week | 0–40 (float) |
| Absences | Number of school absences | 0–30 |
| Tutoring | Receives tutoring | 0=No, 1=Yes |
| ParentalSupport | Level of parental support | 0–4 |
| Extracurricular | Participates in activities | 0=No, 1=Yes |
| Sports | Participates in sports | 0=No, 1=Yes |
| Music | Participates in music | 0=No, 1=Yes |
| Volunteering | Does volunteering | 0=No, 1=Yes |
| GPA | Grade Point Average | 0.0–4.0 |
| GradeClass | Letter grade class | 0=A, 1=B, 2=C, 3=D, 4=F |

---

## Target Variable: Performance_Level

Created from **GPA** using threshold-based classification:

| Performance Level | GPA Range | Interpretation |
|---|---|---|
| **Low** | GPA < 1.5 | D/F range — struggling student |
| **Medium** | 1.5 ≤ GPA < 3.0 | C/B range — average student |
| **High** | GPA ≥ 3.0 | A/B+ range — high achiever |

> **Note:** GPA and GradeClass are **excluded** from features to prevent data leakage.

---

## Features Used for Prediction

```
Age, Gender, Ethnicity, ParentalEducation,
StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
Extracurricular, Sports, Music, Volunteering
```

Total: **12 features**

---

## Machine Learning Algorithms

### 1. Decision Tree Classifier
- **Why?** Interpretable, visualizable, no data distribution assumptions
- **Parameters:** max_depth=10, min_samples_split=10, class_weight='balanced'
- **Pipeline:** StandardScaler → DecisionTreeClassifier

### 2. Random Forest Classifier
- **Why?** Ensemble reduces overfitting, provides feature importance, robust to noise
- **Parameters:** n_estimators=200, max_depth=12, class_weight='balanced'
- **Pipeline:** StandardScaler → RandomForestClassifier

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.x, Flask |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Frontend | HTML5, CSS3, JavaScript |
| Visualizations | Chart.js 4.x |
| Fonts | Google Fonts (Inter) |
| Model Storage | Pickle (.pkl files) |

---

## Data Preprocessing

1. **Load** CSV from `data/student_performance.csv`
2. **Remove duplicates** — checked and cleaned
3. **Handle missing values** — drop rows with NaN
4. **Create target variable** — GPA → Performance_Level
5. **Feature selection** — 12 input features (exclude StudentID, GPA, GradeClass)
6. **Train/test split** — 80% train / 20% test, `random_state=42`, stratified
7. **StandardScaler** — fitted on training set, same scaler applied at prediction time via Pipeline

---

## Model Training

Run once:
```bash
python train_models.py
```

This generates:
- `models/decision_tree.pkl` — Trained DT pipeline
- `models/random_forest.pkl` — Trained RF pipeline
- `models/feature_columns.pkl` — Ordered feature list
- `models/metrics.json` — All evaluation metrics + visualization data

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP+TN)/(total) | Overall correct predictions |
| Precision | TP/(TP+FP) | Of predicted positives, how many are correct |
| Recall | TP/(TP+FN) | Of actual positives, how many were found |
| F1 Score | 2*(P*R)/(P+R) | Harmonic mean of Precision and Recall |
| Confusion Matrix | Grid of actual vs predicted | Per-class breakdown |

---

## Web Application Features

| Feature | Description |
|---|---|
| Dashboard | Project stats, model accuracy, performance distribution |
| Prediction Form | 12-field input form for student details |
| Decision Tree Prediction | Live prediction via `/predict/decision-tree` |
| Random Forest Prediction | Live prediction via `/predict/random-forest` |
| Probability Display | Confidence % and per-class probabilities |
| Analytics Dashboard | 8+ interactive Chart.js visualizations |
| Model Comparison | Table + Radar chart comparing both algorithms |
| Responsive Design | Works on desktop, tablet, and mobile |

---

## Project Structure

```
student-performance-prediction/
│
├── app.py                     # Flask backend (routes + prediction)
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
├── templates/
│   ├── index.html             # Dashboard
│   ├── prediction.html        # Prediction form
│   └── analytics.html        # Visualization dashboard
│
├── static/
│   ├── css/style.css
│   └── js/
│       ├── script.js          # Shared utilities
│       └── analytics.js       # Chart.js visualizations
│
└── screenshots/
```

---

## Installation & Running

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the models (do this once)
```bash
python train_models.py
```

### Step 3 — Launch the web application
```bash
python app.py
```

### Step 4 — Open in browser
```
http://127.0.0.1:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/prediction` | Prediction form |
| GET | `/analytics` | Analytics dashboard |
| GET | `/model-metrics` | JSON: all metrics + chart data |
| POST | `/predict/decision-tree` | DT prediction (JSON) |
| POST | `/predict/random-forest` | RF prediction (JSON) |

**Prediction Request (JSON):**
```json
{
  "Age": 17, "Gender": 0, "Ethnicity": 1,
  "ParentalEducation": 2, "StudyTimeWeekly": 15.5,
  "Absences": 3, "Tutoring": 1, "ParentalSupport": 3,
  "Extracurricular": 1, "Sports": 0, "Music": 1, "Volunteering": 0
}
```

**Prediction Response (JSON):**
```json
{
  "prediction": "High",
  "probability": 0.84,
  "probabilities": {"Low": 0.03, "Medium": 0.13, "High": 0.84},
  "algorithm": "Random Forest"
}
```

---

## Screenshots

*(Add screenshots to the `screenshots/` folder after running the app)*

---

## Future Enhancements

1. Add more ML algorithms (SVM, KNN, Gradient Boosting)
2. Hyperparameter tuning with GridSearchCV
3. Cross-validation (k-fold) for more robust evaluation
4. SHAP values for explainable AI
5. Student cohort comparison
6. Export prediction report as PDF
7. Database integration for storing predictions
8. Authentication for teacher/admin portal

---

## Academic Note

This project was developed as a Data Analytics & Visualization (DAV) laboratory mini project demonstrating the complete ML workflow from raw data to a working web application.
