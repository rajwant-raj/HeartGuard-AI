import streamlit as st
import joblib
import numpy as np
import os
from pathlib import Path

##################################################
# Page Config – MUST be first Streamlit call
##################################################

st.set_page_config(
    page_title="HeartGuard AI – Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

##################################################
# Custom CSS – Premium Dark UI
##################################################

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:   #0d1117;
    --bg-card:      #161b22;
    --bg-input:     #1c2128;
    --accent:       #e74c3c;
    --accent-light: #ff6b6b;
    --accent-glow:  rgba(231, 76, 60, 0.25);
    --success:      #2ecc71;
    --success-glow: rgba(46, 204, 113, 0.20);
    --warning:      #f39c12;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       #30363d;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 1100px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0a0a 0%, #1c1624 50%, #0d1117 100%);
    border: 1px solid #3d1a1a;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(231,76,60,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff6b6b 0%, #e74c3c 50%, #c0392b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(231,76,60,0.12);
    border: 1px solid rgba(231,76,60,0.35);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #ff6b6b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 1rem;
}

/* ── Section card ── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--accent-light);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Input labels ── */
.stSlider label, .stSelectbox label {
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ── Slider track ── */
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Select box ── */
.stSelectbox > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Predict button ── */
div.stButton > button {
    background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(231,76,60,0.35) !important;
    text-transform: uppercase !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(231,76,60,0.55) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result cards ── */
.result-danger {
    background: linear-gradient(135deg, rgba(231,76,60,0.15), rgba(192,57,43,0.08));
    border: 1px solid rgba(231,76,60,0.45);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: fadeIn 0.5s ease;
}
.result-safe {
    background: linear-gradient(135deg, rgba(46,204,113,0.12), rgba(39,174,96,0.06));
    border: 1px solid rgba(46,204,113,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: fadeIn 0.5s ease;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.6rem; }
.result-label {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.result-danger .result-label { color: #ff6b6b; }
.result-safe   .result-label { color: #2ecc71; }
.result-prob {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 500;
}
.result-prob span {
    font-weight: 700;
    color: var(--text-primary);
}

/* ── Probability bar ── */
.prob-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    height: 10px;
    margin: 1rem auto;
    max-width: 320px;
    overflow: hidden;
}
.prob-bar-fill-danger {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #e74c3c, #ff6b6b);
    transition: width 0.8s ease;
}
.prob-bar-fill-safe {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #2ecc71, #27ae60);
    transition: width 0.8s ease;
}

/* ── Metric pills ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 1rem; }
.metric-pill {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.82rem;
    color: var(--text-muted);
}
.metric-pill b { color: var(--text-primary); }

/* ── Tooltip helper ── */
.info-tip {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: -8px;
    margin-bottom: 4px;
    font-style: italic;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(243,156,18,0.08);
    border: 1px solid rgba(243,156,18,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #f39c12;
    margin-top: 1.5rem;
    text-align: center;
}

/* ── Fade-in animation ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


##################################################
# Load Model
##################################################

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / "models" / "model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"

@st.cache_resource(show_spinner=False)
def load_model():
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error(
        "⚠️ Model not found. Please run **train.py** first to train and save the model.",
        icon="🚨"
    )
    st.stop()


##################################################
# Hero Banner
##################################################

st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🫀 AI-Powered Cardiology Tool</div>
    <p class="hero-title">HeartGuard AI</p>
    <p class="hero-sub">
        Enter your clinical parameters below and let our Random Forest model assess
        your cardiovascular risk in real-time.
    </p>
</div>
""", unsafe_allow_html=True)


##################################################
# Input Form
##################################################

st.markdown('<div class="section-title">🧬 Patient Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    age = st.slider("Age (years)", 20, 100, 54,
                    help="Patient's age in years")
    sex = st.selectbox("Biological Sex",
                       options=[0, 1],
                       format_func=lambda x: "Female (0)" if x == 0 else "Male (1)",
                       help="0 = Female, 1 = Male")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120,
                         help="Measured at hospital admission")

    st.markdown("**Blood Work**")
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240,
                     help="Serum cholesterol level")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                       options=[0, 1],
                       format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
                       help="1 = true, 0 = false")

with col2:
    st.markdown("**Cardiac Symptoms**")
    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 – Typical Angina",
            1: "1 – Atypical Angina",
            2: "2 – Non-anginal Pain",
            3: "3 – Asymptomatic"
        }[x],
        help="Type of chest pain experienced"
    )
    thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150,
                        help="Maximum heart rate during exercise")
    exang = st.selectbox("Exercise-Induced Angina",
                         options=[0, 1],
                         format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
                         help="Angina induced by exercise")
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, step=0.1,
                        help="ST depression induced by exercise relative to rest")

with col3:
    st.markdown("**ECG & Imaging**")
    restecg = st.selectbox(
        "Resting ECG Result",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 – Normal",
            1: "1 – ST-T Wave Abnormality",
            2: "2 – Left Ventricular Hypertrophy"
        }[x],
        help="Resting electrocardiographic results"
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 – Upsloping",
            1: "1 – Flat",
            2: "2 – Downsloping"
        }[x],
        help="Slope of peak exercise ST segment"
    )
    ca = st.slider("Major Vessels Colored (CA)", 0, 4, 0,
                   help="Number of major vessels (0–4) colored by fluoroscopy")
    thal = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 – Normal",
            1: "1 – Fixed Defect",
            2: "2 – Reversible Defect",
            3: "3 – Unknown"
        }[x],
        help="Thalassemia blood disorder type"
    )

st.markdown('</div>', unsafe_allow_html=True)


##################################################
# Build input array aligned to training features
##################################################

raw_values = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}

# Align input order to the features the model was trained on
input_data = np.array([[raw_values.get(f, 0) for f in feature_names]])


##################################################
# Summary pills
##################################################

st.markdown(f"""
<div class="section-card">
<div class="section-title">📋 Input Summary</div>
<div class="metric-row">
  <div class="metric-pill">Age <b>{age} yrs</b></div>
  <div class="metric-pill">Sex <b>{'Male' if sex==1 else 'Female'}</b></div>
  <div class="metric-pill">BP <b>{trestbps} mmHg</b></div>
  <div class="metric-pill">Cholesterol <b>{chol} mg/dl</b></div>
  <div class="metric-pill">Max HR <b>{thalach} bpm</b></div>
  <div class="metric-pill">ST Depression <b>{oldpeak}</b></div>
  <div class="metric-pill">Vessels <b>{ca}</b></div>
</div>
</div>
""", unsafe_allow_html=True)


##################################################
# Predict
##################################################

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔬 Analyse Cardiovascular Risk", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

if predict_clicked:
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    if pred == 1:
        risk_pct  = prob[1] * 100
        safe_pct  = prob[0] * 100
        bar_width = f"{risk_pct:.1f}%"
        st.markdown(f"""
        <div class="result-danger">
            <div class="result-icon">⚠️</div>
            <div class="result-label">Heart Disease Risk Detected</div>
            <div class="result-prob">
                Risk probability: <span>{risk_pct:.2f}%</span>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill-danger" style="width:{bar_width}"></div>
            </div>
            <p style="color:#8b949e;font-size:0.82rem;margin-top:0.5rem;">
                Please consult a cardiologist for a thorough clinical evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        safe_pct  = prob[0] * 100
        bar_width = f"{safe_pct:.1f}%"
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-icon">✅</div>
            <div class="result-label">No Heart Disease Detected</div>
            <div class="result-prob">
                Healthy probability: <span>{safe_pct:.2f}%</span>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill-safe" style="width:{bar_width}"></div>
            </div>
            <p style="color:#8b949e;font-size:0.82rem;margin-top:0.5rem;">
                Maintain a healthy lifestyle and schedule regular check-ups.
            </p>
        </div>
        """, unsafe_allow_html=True)

##################################################
# Disclaimer
##################################################

st.markdown("""
<div class="disclaimer">
    ⚕️ <b>Medical Disclaimer:</b> This tool is for educational and informational purposes only.
    It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the guidance of a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
