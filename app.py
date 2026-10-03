"""
app.py
======
Student Performance Prediction System
Flask Backend - Data Analytics & Visualization (DAV) Mini Project

Routes:
  GET  /                    -> Dashboard (index.html)
  GET  /prediction          -> Prediction form (prediction.html)
  POST /predict/decision-tree   -> DT prediction (JSON)
  POST /predict/random-forest   -> RF prediction (JSON)
  GET  /analytics           -> Analytics/Visualization page
  GET  /model-metrics       -> Model metrics JSON (for charts)
"""

import os
import json
import pickle
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────
# Flask App Initialization
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
# Load Pre-trained Models & Metrics at Startup
# ─────────────────────────────────────────────
MODELS_DIR = "models"
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

def load_models():
    """Load saved pipelines and metrics. Returns (dt_model, rf_model, feature_cols, metrics)."""
    if not os.path.exists(os.path.join(MODELS_DIR, "decision_tree.pkl")):
        raise FileNotFoundError(
            "Models not found. Please run:  python train_models.py  first."
        )
    with open(os.path.join(MODELS_DIR, "decision_tree.pkl"), "rb") as f:
        dt_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)
    return dt_model, rf_model, feature_cols, metrics

try:
    DT_MODEL, RF_MODEL, FEATURE_COLS, METRICS = load_models()
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    DT_MODEL, RF_MODEL, FEATURE_COLS, METRICS = None, None, [], {}

# ─────────────────────────────────────────────
# Input Validation & Parsing
# ─────────────────────────────────────────────
FEATURE_RANGES = {
    "Age":                   (15, 18),
    "Gender":                (0, 1),
    "Ethnicity":             (0, 3),
    "ParentalEducation":     (0, 4),
    "StudyTimeWeekly":       (0.0, 40.0),
    "Absences":              (0, 30),
    "Tutoring":              (0, 1),
    "ParentalSupport":       (0, 4),
    "Extracurricular":       (0, 1),
    "Sports":                (0, 1),
    "Music":                 (0, 1),
    "Volunteering":          (0, 1),
}

def parse_and_validate(form_data):
    """
    Parse form/JSON data into a validated dict.
    Returns (data_dict, error_message). error_message is None on success.
    """
    parsed = {}
    for feat in FEATURE_COLS:
        val = form_data.get(feat)
        if val is None or str(val).strip() == "":
            return None, f"Missing field: {feat}"
        try:
            val = float(val)
        except (ValueError, TypeError):
            return None, f"Invalid value for {feat}: must be a number."

        lo, hi = FEATURE_RANGES.get(feat, (-1e9, 1e9))
        if not (lo <= val <= hi):
            return None, f"{feat} must be between {lo} and {hi}."
        parsed[feat] = val

    return parsed, None


def build_input_df(data_dict):
    """Convert validated dict to a DataFrame matching training feature order."""
    row = {feat: [data_dict[feat]] for feat in FEATURE_COLS}
    return pd.DataFrame(row)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    """Dashboard page with project overview and key metrics."""
    return render_template("index.html", metrics=METRICS)


@app.route("/prediction")
def prediction():
    """Prediction form page."""
    return render_template("prediction.html")


@app.route("/analytics")
def analytics():
    """Analytics & Visualization dashboard."""
    return render_template("analytics.html", metrics=METRICS)


@app.route("/model-metrics")
def model_metrics():
    """Return full metrics JSON for use by Chart.js."""
    return jsonify(METRICS)


@app.route("/predict/decision-tree", methods=["POST"])
def predict_decision_tree():
    """
    Predict using Decision Tree Classifier.
    Accepts JSON or form data.
    Returns: {"prediction": "High/Medium/Low", "probability": 0.87,
              "probabilities": {"Low": 0.05, "Medium": 0.08, "High": 0.87}}
    """
    if DT_MODEL is None:
        return jsonify({"error": "Models not loaded. Run train_models.py first."}), 503

    try:
        # Accept both JSON and form POST
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form.to_dict()

        data_dict, err = parse_and_validate(form_data)
        if err:
            return jsonify({"error": err}), 400

        input_df = build_input_df(data_dict)

        prediction = DT_MODEL.predict(input_df)[0]
        proba_array = DT_MODEL.predict_proba(input_df)[0]
        classes = DT_MODEL.classes_

        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba_array)}
        max_prob = round(float(np.max(proba_array)), 4)

        return jsonify({
            "prediction": prediction,
            "probability": max_prob,
            "probabilities": proba_dict,
            "algorithm": "Decision Tree"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/random-forest", methods=["POST"])
def predict_random_forest():
    """
    Predict using Random Forest Classifier.
    Returns: {"prediction": "High/Medium/Low", "probability": 0.87,
              "probabilities": {"Low": 0.02, "Medium": 0.11, "High": 0.87}}
    """
    if RF_MODEL is None:
        return jsonify({"error": "Models not loaded. Run train_models.py first."}), 503

    try:
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form.to_dict()

        data_dict, err = parse_and_validate(form_data)
        if err:
            return jsonify({"error": err}), 400

        input_df = build_input_df(data_dict)

        prediction = RF_MODEL.predict(input_df)[0]
        proba_array = RF_MODEL.predict_proba(input_df)[0]
        classes = RF_MODEL.classes_

        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba_array)}
        max_prob = round(float(np.max(proba_array)), 4)

        return jsonify({
            "prediction": prediction,
            "probability": max_prob,
            "probabilities": proba_dict,
            "algorithm": "Random Forest"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Student Performance Prediction System")
    print("  DAV Mini Project - Flask Server")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="127.0.0.1", port=5000)
