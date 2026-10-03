"""
streamlit_app.py
================
Student Performance Prediction System
Streamlit Cloud Deployment — DAV Mini Project

Pages:
  🏠 Dashboard     — Project overview & model metrics
  🔮 Prediction    — Live student performance prediction
  📊 Analytics     — Interactive visualizations
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    margin-top: 4px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Section headers */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(167,139,250,0.4);
}

/* Prediction result */
.result-high {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(6,95,70,0.3));
    border: 1px solid rgba(16,185,129,0.5);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-medium {
    background: linear-gradient(135deg, rgba(251,191,36,0.2), rgba(146,64,14,0.3));
    border: 1px solid rgba(251,191,36,0.5);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(127,29,29,0.3));
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-title {
    font-size: 1rem;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.result-value {
    font-size: 2.8rem;
    font-weight: 800;
    color: #fff;
    margin: 0;
}
.result-prob {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.7);
    margin-top: 6px;
}

/* Info box */
.info-box {
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    color: rgba(255,255,255,0.85);
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Hero */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 12px;
}
.hero-sub {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.65);
    max-width: 600px;
    line-height: 1.7;
}

/* Badge */
.badge {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #a78bfa;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 6px;
    margin-bottom: 6px;
}

/* Streamlit tweaks */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
}
.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #6d28d9, #1d4ed8);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.4);
}
.stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
    color: rgba(255,255,255,0.75) !important;
    font-weight: 500;
}
h1, h2, h3, h4 {
    color: #fff !important;
}
p, li {
    color: rgba(255,255,255,0.8);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Models & Metrics (cached)
# ─────────────────────────────────────────────
MODELS_DIR = "models"

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load trained pipelines and metrics from disk."""
    dt_path = os.path.join(MODELS_DIR, "decision_tree.pkl")
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    fc_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    mt_path = os.path.join(MODELS_DIR, "metrics.json")

    if not all(os.path.exists(p) for p in [dt_path, rf_path, fc_path, mt_path]):
        return None, None, [], {}

    with open(dt_path, "rb") as f:
        dt_model = pickle.load(f)
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
    with open(fc_path, "rb") as f:
        feature_cols = pickle.load(f)
    with open(mt_path, "r") as f:
        metrics = json.load(f)
    return dt_model, rf_model, feature_cols, metrics


DT_MODEL, RF_MODEL, FEATURE_COLS, METRICS = load_models()
MODELS_LOADED = DT_MODEL is not None


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.5rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:700; color:#fff; margin-top:6px;">
            Student Performance
        </div>
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.5); margin-top:2px;">
            Predictor — DAV Mini Project
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.1); margin:12px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🔮 Prediction", "📊 Analytics"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:16px 0;'>", unsafe_allow_html=True)
    if MODELS_LOADED:
        st.markdown("""
        <div style="text-align:center;">
            <span style="color:#34d399; font-size:0.8rem;">✅ Models Loaded</span>
        </div>
        """, unsafe_allow_html=True)

        ds = METRICS.get("dataset", {})
        st.markdown(f"""
        <div style="color:rgba(255,255,255,0.5); font-size:0.75rem; text-align:center; margin-top:8px;">
            {ds.get('total_students', 0):,} students · {ds.get('total_features', 0)} features
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Models not found. Run `train_models.py` first.")

    st.markdown("""
    <div style="position:absolute; bottom:20px; left:0; right:0; text-align:center;">
        <a href="https://github.com/SaiSandeep10/student-performance-prediction"
           target="_blank"
           style="color:rgba(255,255,255,0.4); font-size:0.75rem; text-decoration:none;">
            ⭐ View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    # Hero
    st.markdown("""
    <div style="padding: 40px 0 20px 0;">
        <div class="hero-title">Student Performance<br>Prediction System</div>
        <div class="hero-sub">
            A machine learning system that predicts academic performance (Low / Medium / High)
            using Decision Tree &amp; Random Forest classifiers with interactive visualizations.
        </div>
        <div style="margin-top:16px;">
            <span class="badge">Decision Tree</span>
            <span class="badge">Random Forest</span>
            <span class="badge">Scikit-learn</span>
            <span class="badge">DAV Mini Project</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not MODELS_LOADED:
        st.warning("⚠️ Models are not loaded. Please run `python train_models.py` first.")
        st.stop()

    ds = METRICS["dataset"]
    dt = METRICS["decision_tree"]
    rf = METRICS["random_forest"]

    # Key stats
    st.markdown('<div class="section-header">📈 Key Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{ds['total_students']:,}</div>
            <div class="metric-label">Total Students</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{ds['total_features']}</div>
            <div class="metric-label">Features Used</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{dt['accuracy']*100:.1f}%</div>
            <div class="metric-label">Decision Tree Acc.</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{rf['accuracy']*100:.1f}%</div>
            <div class="metric-label">Random Forest Acc.</div>
        </div>""", unsafe_allow_html=True)

    # Model comparison table
    st.markdown('<div class="section-header">🤖 Model Comparison</div>', unsafe_allow_html=True)
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Decision Tree": [
            f"{dt['accuracy']*100:.2f}%",
            f"{dt['precision']*100:.2f}%",
            f"{dt['recall']*100:.2f}%",
            f"{dt['f1_score']*100:.2f}%",
        ],
        "Random Forest": [
            f"{rf['accuracy']*100:.2f}%",
            f"{rf['precision']*100:.2f}%",
            f"{rf['recall']*100:.2f}%",
            f"{rf['f1_score']*100:.2f}%",
        ],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Radar chart
    st.markdown('<div class="section-header">📡 Performance Radar</div>', unsafe_allow_html=True)
    categories = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[dt["accuracy"], dt["precision"], dt["recall"], dt["f1_score"], dt["accuracy"]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Decision Tree",
        line_color="#a78bfa",
        fillcolor="rgba(167,139,250,0.2)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[rf["accuracy"], rf["precision"], rf["recall"], rf["f1_score"], rf["accuracy"]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Random Forest",
        line_color="#60a5fa",
        fillcolor="rgba(96,165,250,0.2)",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], tickformat=".0%", color="rgba(255,255,255,0.5)"),
            angularaxis=dict(color="rgba(255,255,255,0.7)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)"),
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Performance distribution pie
    st.markdown('<div class="section-header">🎯 Performance Distribution</div>', unsafe_allow_html=True)
    perf_dist = ds["performance_distribution"]
    cols_pie = list(perf_dist.keys())
    vals_pie = list(perf_dist.values())
    color_map = {"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"}
    colors_pie = [color_map.get(k, "#a78bfa") for k in cols_pie]

    fig_pie = go.Figure(go.Pie(
        labels=cols_pie,
        values=vals_pie,
        hole=0.5,
        marker_colors=colors_pie,
        textinfo="label+percent",
        textfont_size=14,
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        legend=dict(bgcolor="rgba(255,255,255,0.05)"),
        height=380,
        annotations=[dict(text="Students", x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#fff")],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Dataset info
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b>📁 Dataset:</b> Student Performance Factors Dataset &nbsp;|&nbsp;
        <b>📊 Records:</b> {ds['total_students']:,} students &nbsp;|&nbsp;
        <b>🔬 Features:</b> {ds['total_features']} input features<br><br>
        <b>Target Variable:</b> <code>Performance_Level</code> derived from GPA thresholds:<br>
        &nbsp;&nbsp;🔴 <b>Low</b> — GPA &lt; 1.5 &nbsp;|&nbsp;
        🟡 <b>Medium</b> — 1.5 ≤ GPA &lt; 3.0 &nbsp;|&nbsp;
        🟢 <b>High</b> — GPA ≥ 3.0<br><br>
        <b>Train/Test Split:</b> {ds['train_size']:,} training / {ds['test_size']:,} testing (80/20, stratified)
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ═══════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown("""
    <div style="padding: 30px 0 16px 0;">
        <div class="hero-title" style="font-size:2.2rem;">🔮 Live Prediction</div>
        <div class="hero-sub">Enter student details below to predict their academic performance level.</div>
    </div>
    """, unsafe_allow_html=True)

    if not MODELS_LOADED:
        st.warning("⚠️ Models are not loaded. Please run `python train_models.py` first.")
        st.stop()

    st.markdown('<div class="section-header">⚙️ Select Algorithm</div>', unsafe_allow_html=True)
    algo = st.radio(
        "Algorithm",
        ["🌳 Decision Tree", "🌲 Random Forest"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-header">📝 Student Information</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.selectbox("Age", [15, 16, 17, 18], index=1,
                               help="Student age (15–18)")
            gender = st.selectbox("Gender", ["Male (0)", "Female (1)"], index=0)
            ethnicity = st.selectbox(
                "Ethnicity",
                ["Group 0", "Group 1", "Group 2", "Group 3"],
                help="Encoded ethnicity group (0–3)"
            )
            parental_edu = st.selectbox(
                "Parental Education",
                ["None (0)", "High School (1)", "Some College (2)", "Bachelor's (3)", "Higher (4)"],
                help="Parents' highest education level"
            )

        with col2:
            study_time = st.slider(
                "Study Time (hrs/week)", 0.0, 40.0, 10.0, 0.5,
                help="Weekly study hours (0–40)"
            )
            absences = st.slider(
                "Absences", 0, 30, 5,
                help="Number of school absences (0–30)"
            )
            tutoring = st.selectbox("Tutoring", ["No (0)", "Yes (1)"])
            parental_support = st.selectbox(
                "Parental Support",
                ["None (0)", "Low (1)", "Moderate (2)", "High (3)", "Very High (4)"]
            )

        with col3:
            extracurricular = st.selectbox("Extracurricular Activities", ["No (0)", "Yes (1)"])
            sports = st.selectbox("Sports", ["No (0)", "Yes (1)"])
            music = st.selectbox("Music", ["No (0)", "Yes (1)"])
            volunteering = st.selectbox("Volunteering", ["No (0)", "Yes (1)"])

        submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True)

    if submitted:
        def extract_num(val_str):
            return int(val_str.split("(")[1].rstrip(")"))

        input_data = {
            "Age": age,
            "Gender": extract_num(gender),
            "Ethnicity": int(ethnicity.split(" ")[1]),
            "ParentalEducation": extract_num(parental_edu),
            "StudyTimeWeekly": study_time,
            "Absences": absences,
            "Tutoring": extract_num(tutoring),
            "ParentalSupport": extract_num(parental_support),
            "Extracurricular": extract_num(extracurricular),
            "Sports": extract_num(sports),
            "Music": extract_num(music),
            "Volunteering": extract_num(volunteering),
        }

        input_df = pd.DataFrame({k: [v] for k, v in input_data.items()})[FEATURE_COLS]

        model = DT_MODEL if "Decision Tree" in algo else RF_MODEL
        algo_name = "Decision Tree" if "Decision Tree" in algo else "Random Forest"

        prediction = model.predict(input_df)[0]
        proba_array = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba_array)}
        max_prob = max(proba_dict.values())

        st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

        css_class = {"High": "result-high", "Medium": "result-medium", "Low": "result-low"}.get(prediction, "result-medium")
        emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(prediction, "")

        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-title">Predicted Performance — {algo_name}</div>
                <div class="result-value">{emoji} {prediction}</div>
                <div class="result-prob">Confidence: {max_prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Probability Breakdown</div>', unsafe_allow_html=True)
        order = ["Low", "Medium", "High"]
        colors_bar = {"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"}
        fig_bar = go.Figure()
        for cls in order:
            if cls in proba_dict:
                fig_bar.add_trace(go.Bar(
                    x=[cls],
                    y=[proba_dict[cls] * 100],
                    name=cls,
                    marker_color=colors_bar[cls],
                    text=[f"{proba_dict[cls]*100:.1f}%"],
                    textposition="outside",
                    textfont_color="#fff",
                ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            yaxis=dict(range=[0, 110], tickformat=".0f", ticksuffix="%",
                       gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            showlegend=False,
            height=320,
            bargap=0.4,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Insights
        st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)
        insights = []
        if study_time >= 20:
            insights.append("✅ High study time (≥20 hrs/week) is strongly associated with better outcomes.")
        elif study_time < 5:
            insights.append("⚠️ Very low study time (&lt;5 hrs/week) — consider increasing study hours.")
        if absences >= 15:
            insights.append("⚠️ High absences (≥15) significantly impact academic performance.")
        elif absences <= 5:
            insights.append("✅ Good attendance (≤5 absences) is a positive indicator.")
        if extract_num(tutoring) == 1:
            insights.append("✅ Tutoring support is a positive factor.")
        if extract_num(parental_support) >= 3:
            insights.append("✅ Strong parental support is associated with higher performance.")
        if not insights:
            insights.append("📊 Factors appear balanced — see Analytics for detailed correlations.")

        st.markdown(f"""
        <div class="info-box">
            {'<br>'.join(insights)}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ═══════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("""
    <div style="padding: 30px 0 16px 0;">
        <div class="hero-title" style="font-size:2.2rem;">📊 Analytics Dashboard</div>
        <div class="hero-sub">Explore data patterns, model performance, and feature importances.</div>
    </div>
    """, unsafe_allow_html=True)

    if not MODELS_LOADED:
        st.warning("⚠️ Models are not loaded. Please run `python train_models.py` first.")
        st.stop()

    viz = METRICS["visualizations"]
    dt = METRICS["decision_tree"]
    rf = METRICS["random_forest"]

    # Feature Importance
    st.markdown('<div class="section-header">🔑 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    fi = rf["feature_importances"]
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1]))
    fig_fi = go.Figure(go.Bar(
        x=list(fi_sorted.values()),
        y=list(fi_sorted.keys()),
        orientation="h",
        marker=dict(
            color=list(fi_sorted.values()),
            colorscale=[[0, "#312e81"], [0.5, "#7c3aed"], [1, "#34d399"]],
            showscale=False,
        ),
        text=[f"{v*100:.1f}%" for v in fi_sorted.values()],
        textposition="outside",
        textfont_color="#fff",
    ))
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        xaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        height=420,
        margin=dict(l=10, r=80, t=10, b=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion Matrices
    st.markdown('<div class="section-header">🔲 Confusion Matrices</div>', unsafe_allow_html=True)
    cm_col1, cm_col2 = st.columns(2)
    labels_cm = ["Low", "Medium", "High"]

    def make_cm_fig(cm_data, title, color):
        cm_arr = np.array(cm_data)
        fig = go.Figure(go.Heatmap(
            z=cm_arr,
            x=[f"Pred {l}" for l in labels_cm],
            y=[f"Act {l}" for l in labels_cm],
            colorscale=[[0, "rgba(0,0,0,0.3)"], [1, color]],
            text=cm_arr.astype(str),
            texttemplate="%{text}",
            showscale=False,
        ))
        fig.update_layout(
            title=dict(text=title, font_color="#fff", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    with cm_col1:
        st.plotly_chart(make_cm_fig(dt["confusion_matrix"], "Decision Tree", "#a78bfa"), use_container_width=True)
    with cm_col2:
        st.plotly_chart(make_cm_fig(rf["confusion_matrix"], "Random Forest", "#60a5fa"), use_container_width=True)

    # Study Time Box Plot
    st.markdown('<div class="section-header">📚 Study Time by Performance Level</div>', unsafe_allow_html=True)
    study = viz["study_hours_by_performance"]
    levels = ["Low", "Medium", "High"]
    box_colors = {"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"}

    fig_box = go.Figure()
    for lvl in levels:
        s = study[lvl]
        fig_box.add_trace(go.Box(
            name=lvl,
            q1=[s["q1"]], median=[s["median"]], q3=[s["q3"]],
            lowerfence=[s["min"]], upperfence=[s["max"]], mean=[s["mean"]],
            marker_color=box_colors[lvl],
            boxmean="sd",
        ))
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        yaxis=dict(title="Hours/Week", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(title="Performance Level"),
        height=360,
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Average Absences Bar
    st.markdown('<div class="section-header">📅 Average Absences by Performance Level</div>', unsafe_allow_html=True)
    abs_data = viz["absences_by_performance"]
    fig_abs = go.Figure(go.Bar(
        x=levels,
        y=[abs_data.get(l, 0) for l in levels],
        marker_color=["#ef4444", "#f59e0b", "#10b981"],
        text=[f"{abs_data.get(l, 0):.1f}" for l in levels],
        textposition="outside",
        textfont_color="#fff",
    ))
    fig_abs.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        yaxis=dict(title="Avg Absences", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(title="Performance Level"),
        height=340,
        showlegend=False,
        bargap=0.4,
    )
    st.plotly_chart(fig_abs, use_container_width=True)

    # Scatter: Study Time vs GPA
    st.markdown('<div class="section-header">🔵 Study Time vs GPA (Sample)</div>', unsafe_allow_html=True)
    scatter = viz["scatter_study_gpa"]
    scatter_df = pd.DataFrame(scatter)
    if not scatter_df.empty:
        fig_sc = px.scatter(
            scatter_df, x="x", y="y", color="level",
            color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"},
            labels={"x": "Study Time (hrs/week)", "y": "GPA", "level": "Performance"},
            opacity=0.75,
        )
        fig_sc.update_traces(marker_size=7)
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            legend=dict(bgcolor="rgba(255,255,255,0.05)"),
            height=400,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # Correlation Heatmap
    st.markdown('<div class="section-header">🌡️ Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_labels = viz["correlation_labels"]
    corr_matrix = viz["correlation_matrix"]
    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=corr_labels,
        y=corr_labels,
        colorscale="RdBu",
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont_size=9,
        showscale=True,
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # GPA Distribution by Level
    st.markdown('<div class="section-header">📈 GPA Distribution by Performance Level</div>', unsafe_allow_html=True)
    gpa = viz["gpa_by_performance"]
    fig_gpa = go.Figure()
    for lvl in levels:
        g = gpa[lvl]
        fig_gpa.add_trace(go.Box(
            name=lvl,
            q1=[g["q1"]], median=[g["median"]], q3=[g["q3"]],
            lowerfence=[g["min"]], upperfence=[g["max"]], mean=[g["mean"]],
            marker_color=box_colors[lvl],
        ))
    fig_gpa.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        yaxis=dict(title="GPA (0–4)", gridcolor="rgba(255,255,255,0.08)"),
        height=360,
        showlegend=False,
    )
    st.plotly_chart(fig_gpa, use_container_width=True)
