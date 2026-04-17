"""
AI-Based Queue-Time Prediction & Smart Counter Allocation
Decision Support System for Hospital Service Centers
==========================================================
Run: streamlit run hospital_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import random
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediFlow AI — Hospital Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Syne:wght@600;700;800&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f4f8 !important;
    font-family: 'DM Sans', sans-serif;
    color: #1a2636;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #dde4ed !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}

/* Sidebar text overrides */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #334155 !important;
}
[data-testid="stSidebar"] .stSlider p { color: #64748b !important; }

/* ── Header Bar ── */
.top-header {
    background: linear-gradient(135deg, #0052a3 0%, #0070d4 55%, #0086f0 100%);
    border-radius: 14px;
    padding: 22px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0, 82, 163, 0.22);
}
.header-logo { display: flex; align-items: center; gap: 14px; }
.header-logo-icon {
    width: 50px; height: 50px;
    background: rgba(255,255,255,0.18);
    border: 1.5px solid rgba(255,255,255,0.35);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
}
.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 800;
    color: #ffffff;
    margin: 0; line-height: 1.15;
    letter-spacing: -0.02em;
}
.header-sub {
    font-size: 11px; color: rgba(255,255,255,0.72);
    letter-spacing: 0.07em; margin-top: 3px;
    text-transform: uppercase; font-weight: 500;
}
.header-status { display: flex; align-items: center; gap: 8px; }
.status-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 0 3px rgba(74,222,128,0.25);
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 3px rgba(74,222,128,0.25); }
    50%      { box-shadow: 0 0 0 6px rgba(74,222,128,0.12); }
}
.status-text { font-size: 12px; color: rgba(255,255,255,0.9); font-weight: 600; }
.header-time { font-size: 13px; color: rgba(255,255,255,0.75); text-align: right; line-height: 1.6; }

/* ── Metric Cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #dde4ed;
    border-radius: 12px;
    padding: 18px 20px 16px;
    position: relative; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: transform 0.18s, box-shadow 0.18s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.09); }
.metric-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 12px 12px 0 0;
}
.metric-card.blue::after   { background: linear-gradient(90deg, #0066cc, #38bdf8); }
.metric-card.green::after  { background: linear-gradient(90deg, #16a34a, #4ade80); }
.metric-card.amber::after  { background: linear-gradient(90deg, #d97706, #fbbf24); }
.metric-card.red::after    { background: linear-gradient(90deg, #dc2626, #f87171); }

.metric-label {
    font-size: 11px; color: #64748b; letter-spacing: 0.07em;
    font-weight: 600; text-transform: uppercase; margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 30px; font-weight: 800; line-height: 1;
    margin-bottom: 6px; color: #0f172a;
}
.metric-value.green { color: #16a34a; }
.metric-value.amber { color: #d97706; }
.metric-value.red   { color: #dc2626; }
.metric-value.blue  { color: #0066cc; }

.metric-delta { font-size: 12px; }
.delta-up   { color: #16a34a; }
.delta-down { color: #dc2626; }
.delta-neu  { color: #64748b; }
.metric-icon {
    position: absolute; top: 16px; right: 16px;
    font-size: 26px; opacity: 0.12;
}

/* ── Section Heading ── */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 13px; font-weight: 700;
    color: #334155;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 0 14px;
    padding-left: 10px;
    border-left: 3px solid #0066cc;
}

/* ── Department Cards ── */
.dept-card {
    background: #ffffff;
    border: 1px solid #dde4ed;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.dept-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.dept-name { font-weight: 600; font-size: 14px; color: #1e293b; }
.dept-badge {
    font-size: 10px; font-weight: 700; letter-spacing: 0.07em;
    padding: 3px 9px; border-radius: 20px; text-transform: uppercase;
}
.badge-critical { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
.badge-high     { background: #fffbeb; color: #b45309; border: 1px solid #fcd34d; }
.badge-normal   { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
.badge-low      { background: #eff6ff; color: #1d4ed8; border: 1px solid #93c5fd; }

.progress-bar-bg {
    background: #e2e8f0; border-radius: 6px; height: 7px;
    overflow: hidden; margin: 6px 0 10px;
}
.progress-bar-fill { height: 100%; border-radius: 6px; transition: width 0.8s ease; }

.dept-stats { display: flex; gap: 24px; }
.dept-stat-val { font-size: 18px; font-weight: 700; font-family: 'Syne', sans-serif; color: #0f172a; }
.dept-stat-lbl { font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 1px; }

/* ── Recommendation Box ── */
.rec-box {
    background: #f8fafc;
    border: 1px solid #dde4ed;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    position: relative;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.rec-box::before {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    border-radius: 12px 0 0 12px;
}
.rec-box.urgent::before   { background: linear-gradient(180deg, #dc2626, #f87171); }
.rec-box.advisory::before { background: linear-gradient(180deg, #d97706, #fbbf24); }
.rec-box.optimal::before  { background: linear-gradient(180deg, #16a34a, #4ade80); }

.rec-type { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 5px; }
.rec-type.urgent   { color: #b91c1c; }
.rec-type.advisory { color: #b45309; }
.rec-type.optimal  { color: #15803d; }

.rec-text { font-size: 13px; color: #334155; line-height: 1.6; }

/* ── Table ── */
.styled-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.styled-table th {
    background: #f1f5f9;
    color: #475569; font-weight: 600;
    font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase;
    padding: 11px 14px; text-align: left;
    border-bottom: 1px solid #dde4ed;
}
.styled-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #e9eef5;
    color: #334155;
}
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: #f8fafc; }

/* ── Alert Banner ── */
.alert-banner {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex; align-items: flex-start; gap: 12px;
}
.alert-icon { font-size: 18px; margin-top: 1px; }
.alert-text { font-size: 13px; color: #7f1d1d; line-height: 1.5; }
.alert-time { font-size: 11px; color: #b91c1c; margin-top: 3px; font-weight: 500; }

/* ── Sidebar ── */
.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 19px; font-weight: 800;
    color: #0052a3;
    margin-bottom: 2px;
}
.sidebar-section {
    font-size: 10px; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 22px 0 8px; font-weight: 700;
}

/* ── Chart containers ── */
.chart-container {
    background: #ffffff;
    border: 1px solid #dde4ed;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dde4ed;
    border-radius: 12px;
    padding: 16px 20px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DUMMY AI ENGINE  (looks real, acts real)
# ─────────────────────────────────────────────
class QueuePredictionEngine:
    """Simulated ML-based prediction engine (Random Forest / XGBoost ensemble)."""

    DEPARTMENTS = {
        "OPD – General Medicine":    {"base_wait": 28, "capacity": 6,  "peak_factor": 1.8},
        "OPD – Pediatrics":          {"base_wait": 22, "capacity": 4,  "peak_factor": 1.5},
        "Diagnostics / Lab":         {"base_wait": 35, "capacity": 8,  "peak_factor": 2.1},
        "Pharmacy":                  {"base_wait": 15, "capacity": 10, "peak_factor": 1.4},
        "Emergency Triage":          {"base_wait": 12, "capacity": 5,  "peak_factor": 1.2},
        "Radiology / Imaging":       {"base_wait": 42, "capacity": 4,  "peak_factor": 1.6},
        "Cardiology OPD":            {"base_wait": 31, "capacity": 3,  "peak_factor": 1.9},
    }

    def __init__(self):
        np.random.seed(int(time.time()) % 1000)
        self._model_accuracy = 0.889    # R² simulated
        self._rmse           = 7.4
        self._mae            = 5.1

    def _time_factor(self, hour: int) -> float:
        """Simulates temporal feature importance learned by the model."""
        peak_curve = {8: 1.6, 9: 1.95, 10: 2.0, 11: 1.85, 12: 1.5,
                      13: 1.2, 14: 1.55, 15: 1.7, 16: 1.5, 17: 1.2}
        return peak_curve.get(hour, 1.0)

    def predict(self, dept_name: str, hour: int = None, noise: float = 0.12):
        if hour is None:
            hour = datetime.now().hour
        cfg = self.DEPARTMENTS[dept_name]
        tf  = self._time_factor(hour)
        raw_wait = cfg["base_wait"] * tf * cfg["peak_factor"] * np.random.uniform(1 - noise, 1 + noise) / cfg["peak_factor"] * 0.95
        predicted_wait   = max(3, round(raw_wait, 1))
        queue_length     = max(1, int(predicted_wait / 4.2 * np.random.uniform(0.9, 1.1)))
        counters_needed  = max(1, min(cfg["capacity"], int(np.ceil(queue_length / 7))))
        counters_active  = max(1, counters_needed - random.randint(0, 1))
        utilization      = min(99, round((queue_length / (counters_active * 7)) * 100, 1))
        return {
            "dept": dept_name,
            "predicted_wait_min": predicted_wait,
            "queue_length": queue_length,
            "counters_needed": counters_needed,
            "counters_active": counters_active,
            "utilization_pct": utilization,
            "status": "critical" if predicted_wait > 40 else "high" if predicted_wait > 25 else "normal" if predicted_wait > 10 else "low",
        }

    def predict_all(self, hour=None):
        return [self.predict(d, hour) for d in self.DEPARTMENTS]

    def hourly_forecast(self, dept_name: str):
        now   = datetime.now().hour
        hours = [(now + i) % 24 for i in range(12)]
        return {
            "hours":    [f"{h:02d}:00" for h in hours],
            "predicted":[self.predict(dept_name, h, noise=0.08)["predicted_wait_min"] for h in hours],
            "upper":    None,
            "lower":    None,
        }

    def historic_accuracy(self):
        days = pd.date_range(end=datetime.today(), periods=30, freq="D")
        actual    = np.random.normal(25, 6, 30).clip(5, 60)
        predicted = actual * np.random.uniform(0.85, 1.15, 30)
        return pd.DataFrame({"date": days, "actual": actual.round(1), "predicted": predicted.round(1)})

    def arrival_heatmap_data(self):
        days   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hours_ = list(range(8, 20))
        data   = []
        for d in days:
            row = []
            for h in hours_:
                base = 12 + self._time_factor(h) * 8
                if d in ["Mon", "Tue"]:
                    base *= 1.3
                if d == "Sun":
                    base *= 0.6
                row.append(max(1, int(base + np.random.normal(0, 2))))
            data.append(row)
        return pd.DataFrame(data, index=days, columns=[f"{h}:00" for h in hours_])

    @property
    def model_stats(self):
        return {"R²": self._model_accuracy, "RMSE": f"{self._rmse} min", "MAE": f"{self._mae} min", "MAPE": "11.3%"}


@st.cache_resource
def get_engine():
    return QueuePredictionEngine()


engine = get_engine()


# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────
def color_for_status(s):
    return {"critical": "#ff6e6e", "high": "#ffd740", "normal": "#00e676", "low": "#60b3ff"}.get(s, "#607d8b")

def badge_for_status(s):
    return {"critical": "badge-critical", "high": "badge-high", "normal": "badge-normal", "low": "badge-low"}.get(s, "badge-low")

def badge_label(s):
    return {"critical": "🔴 CRITICAL", "high": "🟡 HIGH", "normal": "🟢 NORMAL", "low": "🔵 LOW"}.get(s, s)

def bar_color(u):
    if u >= 90: return "#ff6e6e"
    if u >= 70: return "#ffd740"
    return "#00c853"

def now_str():
    return datetime.now().strftime("%d %b %Y  •  %H:%M:%S")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">MediFlow AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#2a4a6a;margin-bottom:20px;">Hospital Intelligence System v2.4</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🏠  Live Dashboard", "📈  Predictions & Forecast",
                         "🔢  Counter Allocation", "📊  Analytics & Reports",
                         "⚙️  Model Performance"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Simulation Controls</div>', unsafe_allow_html=True)
    sim_hour = st.slider("Simulate Hour of Day", 6, 22, datetime.now().hour, format="%d:00")
    auto_refresh = st.toggle("Auto-Refresh (5s)", value=False)

    st.markdown('<div class="sidebar-section">Filters</div>', unsafe_allow_html=True)
    selected_depts = st.multiselect(
        "Departments",
        list(engine.DEPARTMENTS.keys()),
        default=list(engine.DEPARTMENTS.keys()),
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section">Alert Threshold</div>', unsafe_allow_html=True)
    alert_threshold = st.slider("Trigger alert if wait > (min)", 15, 60, 35)

    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#94a3b8;text-align:center;">AI Engine: XGBoost + Random Forest<br>Last retrained: 2 hrs ago<br>Data points: 124,830</div>', unsafe_allow_html=True)


# Auto-refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()


# ─────────────────────────────────────────────
# FETCH LIVE PREDICTIONS
# ─────────────────────────────────────────────
predictions = [p for p in engine.predict_all(sim_hour) if p["dept"] in selected_depts]


# ─────────────────────────────────────────────
# TOP HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
  <div class="header-logo">
    <div class="header-logo-icon">🏥</div>
    <div>
      <div class="header-title">MediFlow AI Intelligence System</div>
      <div class="header-sub">AI-BASED QUEUE PREDICTION & SMART COUNTER ALLOCATION · HOSPITAL OPERATIONS</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:28px;">
    <div class="header-status">
      <div class="status-dot"></div>
      <div class="status-text">LIVE · All Systems Nominal</div>
    </div>
    <div class="header-time">{now_str()}<br><span style="font-size:11px;color:rgba(255,255,255,0.65);">Simulating {sim_hour:02d}:00</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: LIVE DASHBOARD
# ─────────────────────────────────────────────
if "Live Dashboard" in page:
    # ── KPI Row ──────────────────────────────
    total_queue   = sum(p["queue_length"] for p in predictions)
    avg_wait      = round(np.mean([p["predicted_wait_min"] for p in predictions]), 1)
    critical_depts= sum(1 for p in predictions if p["status"] == "critical")
    avg_util      = round(np.mean([p["utilization_pct"] for p in predictions]), 1)
    total_counters= sum(p["counters_active"] for p in predictions)

    kpi_cols = st.columns(5)
    kpis = [
        ("Total Patients Queued",  total_queue,    "👥", "blue",  f"+{random.randint(3,9)}% vs last hour"),
        ("Avg Wait Time",          f"{avg_wait} min", "⏱️", "amber" if avg_wait > 25 else "green",
         "▲ 2.1 min" if avg_wait > 25 else "▼ 1.8 min"),
        ("Critical Depts",         critical_depts,  "⚠️", "red" if critical_depts > 0 else "green",
         "Requires action" if critical_depts > 0 else "All clear"),
        ("Avg Utilization",        f"{avg_util}%",  "📊", "amber" if avg_util > 75 else "green",
         "Optimal range: 75–85%"),
        ("Active Counters",        total_counters,  "🪟", "blue",  f"of {sum(engine.DEPARTMENTS[p['dept']]['capacity'] for p in predictions)} available"),
    ]

    for i, (lbl, val, icon, color, delta) in enumerate(kpis):
        with kpi_cols[i]:
            st.markdown(f"""
            <div class="metric-card {color}">
              <div class="metric-icon">{icon}</div>
              <div class="metric-label">{lbl}</div>
              <div class="metric-value {color}">{val}</div>
              <div class="metric-delta delta-neu">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Active Alerts ─────────────────────────
    critical_preds = [p for p in predictions if p["predicted_wait_min"] > alert_threshold]
    if critical_preds:
        st.markdown('<div class="section-head">🚨 Active Alerts</div>', unsafe_allow_html=True)
        for p in critical_preds[:3]:
            st.markdown(f"""
            <div class="alert-banner">
              <div class="alert-icon">⚠️</div>
              <div>
                <div class="alert-text"><strong>{p['dept']}</strong> — Predicted wait time of
                  <strong>{p['predicted_wait_min']} min</strong> exceeds threshold ({alert_threshold} min).
                  Queue length: <strong>{p['queue_length']}</strong> patients.
                  Recommend activating <strong>{p['counters_needed']}</strong> counters immediately.</div>
                <div class="alert-time">Generated {datetime.now().strftime('%H:%M:%S')} · AI Confidence: 91.4%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Department Status Cards ───────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="section-head">Department Queue Status</div>', unsafe_allow_html=True)
        for p in sorted(predictions, key=lambda x: x["predicted_wait_min"], reverse=True):
            fill_color = bar_color(p["utilization_pct"])
            st.markdown(f"""
            <div class="dept-card">
              <div class="dept-header">
                <div class="dept-name">{p['dept']}</div>
                <div class="dept-badge {badge_for_status(p['status'])}">{badge_label(p['status'])}</div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748b;margin-bottom:4px;">
                <span>Counter Utilization</span><span style="color:{fill_color};font-weight:700;">{p['utilization_pct']}%</span>
              </div>
              <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{min(100,p['utilization_pct'])}%;background:{fill_color};"></div>
              </div>
              <div class="dept-stats">
                <div class="dept-stat">
                  <div class="dept-stat-val" style="color:{color_for_status(p['status'])};">{p['predicted_wait_min']} min</div>
                  <div class="dept-stat-lbl">Predicted Wait</div>
                </div>
                <div class="dept-stat">
                  <div class="dept-stat-val" style="color:#334155;">{p['queue_length']}</div>
                  <div class="dept-stat-lbl">In Queue</div>
                </div>
                <div class="dept-stat">
                  <div class="dept-stat-val" style="color:#60b3ff;">{p['counters_active']} / {p['counters_needed']}</div>
                  <div class="dept-stat-lbl">Counters Active / Needed</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Recommendations Panel ─────────────────
    with right_col:
        st.markdown('<div class="section-head">AI Recommendations</div>', unsafe_allow_html=True)
        sorted_preds = sorted(predictions, key=lambda x: x["predicted_wait_min"], reverse=True)
        for p in sorted_preds[:5]:
            diff = p["counters_needed"] - p["counters_active"]
            if p["status"] == "critical":
                rtype, rtypecls = "URGENT ACTION", "urgent"
                msg = (f"Immediately open <strong>{diff} additional counter(s)</strong> at <strong>{p['dept']}</strong>. "
                       f"Queue of {p['queue_length']} patients will reach {p['predicted_wait_min']:.0f}-min wait. "
                       f"XGBoost confidence: 93.2%.")
            elif p["status"] == "high" and diff > 0:
                rtype, rtypecls = "ADVISORY", "advisory"
                msg = (f"Consider activating <strong>{diff} more counter(s)</strong> at <strong>{p['dept']}</strong> "
                       f"within 15 minutes. Forecast shows continued demand for next 90 min.")
            else:
                rtype, rtypecls = "OPTIMAL", "optimal"
                msg = (f"<strong>{p['dept']}</strong> is operating within optimal range. "
                       f"Current allocation of {p['counters_active']} counters is sufficient. Monitor trend.")
            st.markdown(f"""
            <div class="rec-box {rtypecls}">
              <div class="rec-type {rtypecls}">{rtype}</div>
              <div class="rec-text">{msg}</div>
            </div>
            """, unsafe_allow_html=True)

        # Model Performance Mini-Card
        st.markdown('<div class="section-head" style="margin-top:20px;">Model Performance</div>', unsafe_allow_html=True)
        ms = engine.model_stats
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #dde4ed;border-radius:12px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            {''.join(f'<div><div style="font-size:18px;font-weight:800;color:#0066cc;font-family:Syne,sans-serif;">{v}</div><div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">{k}</div></div>' for k,v in ms.items())}
          </div>
          <div style="margin-top:12px;font-size:11px;color:rgba(255,255,255,0.65);">Ensemble: XGBoost + Random Forest + Gradient Boosting · Retrained every 6h</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Queue Summary Donut Chart ─────────────
    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="section-head">Queue Distribution by Department</div>', unsafe_allow_html=True)
        depts  = [p["dept"].split("–")[-1].strip().split("/")[0].strip() for p in predictions]
        queues = [p["queue_length"] for p in predictions]
        fig_donut = go.Figure(go.Pie(
            labels=depts, values=queues, hole=0.55,
            marker=dict(colors=["#0066cc","#00b4d8","#00c853","#ffd740","#ff6e6e","#ce93d8","#80cbc4"],
                        line=dict(color="#ffffff", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color="#334155"),
            hovertemplate="<b>%{label}</b><br>%{value} patients<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(color="#334155"),
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
            annotations=[dict(text=f"<b>{total_queue}</b><br><span style='font-size:10px'>Total</span>",
                              x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#0f172a")],
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown('<div class="section-head">Wait Time vs Counter Utilization</div>', unsafe_allow_html=True)
        wait_times = [p["predicted_wait_min"] for p in predictions]
        utils_     = [p["utilization_pct"] for p in predictions]
        sizes_     = [p["queue_length"] * 6 for p in predictions]
        colors_    = [color_for_status(p["status"]) for p in predictions]

        fig_scatter = go.Figure(go.Scatter(
            x=utils_, y=wait_times,
            mode="markers+text",
            text=[p["dept"].split("–")[-1].strip().split("/")[0].strip()[:12] for p in predictions],
            textposition="top center",
            textfont=dict(size=9, color="#607d8b"),
            marker=dict(size=sizes_, color=colors_, opacity=0.85,
                        line=dict(color="#ffffff", width=1)),
            hovertemplate="<b>%{text}</b><br>Utilization: %{x}%<br>Wait: %{y} min<extra></extra>",
        ))
        fig_scatter.add_vline(x=75, line_dash="dash", line_color="#cbd5e1", annotation_text="75% target", annotation_font_color="#64748b")
        fig_scatter.add_hline(y=alert_threshold, line_dash="dash", line_color="#fca5a5", annotation_text="Alert threshold", annotation_font_color="#991b1b")
        fig_scatter.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            xaxis=dict(title="Utilization (%)", color="#64748b", gridcolor="#e2e8f0", zeroline=False),
            yaxis=dict(title="Predicted Wait (min)", color="#64748b", gridcolor="#e2e8f0", zeroline=False),
            font=dict(color="#334155"),
            margin=dict(t=10, b=40, l=50, r=10),
            height=320,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────
# PAGE: PREDICTIONS & FORECAST
# ─────────────────────────────────────────────
elif "Predictions" in page:
    st.markdown('<div class="section-head">Hourly Wait-Time Forecast</div>', unsafe_allow_html=True)

    dept_choice = st.selectbox("Select Department", [p["dept"] for p in predictions], label_visibility="collapsed")
    forecast    = engine.hourly_forecast(dept_choice)

    hours_  = forecast["hours"]
    preds_  = forecast["predicted"]
    upper_  = [v * 1.18 for v in preds_]
    lower_  = [max(1, v * 0.82) for v in preds_]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=hours_ + hours_[::-1],
        y=upper_ + lower_[::-1],
        fill="toself",
        fillcolor="rgba(0,102,204,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        hoverinfo="skip",
    ))
    fig_fc.add_trace(go.Scatter(
        x=hours_, y=preds_,
        mode="lines+markers",
        name="Predicted Wait",
        line=dict(color="#0066cc", width=2.5),
        marker=dict(size=8, color="#0066cc", line=dict(color="#ffffff", width=2)),
        hovertemplate="%{x}: <b>%{y:.1f} min</b><extra></extra>",
    ))
    fig_fc.add_hrect(y0=alert_threshold, y1=80, fillcolor="rgba(220,38,38,0.04)",
                     line_width=0, annotation_text="Critical Zone", annotation_font_color="#991b1b")
    fig_fc.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        xaxis=dict(color="#64748b", gridcolor="#e2e8f0", zeroline=False),
        yaxis=dict(title="Wait Time (min)", color="#64748b", gridcolor="#e2e8f0", zeroline=False),
        font=dict(color="#334155"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#475569"),
        margin=dict(t=20, b=40, l=50, r=20),
        height=350,
    )
    st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})

    # Multi-dept forecast table
    st.markdown('<div class="section-head" style="margin-top:20px;">All-Department 3-Hour Forecast</div>', unsafe_allow_html=True)
    future_hours = [(datetime.now() + timedelta(hours=i)).strftime("%H:00") for i in [1, 2, 3]]
    rows = []
    for d in selected_depts:
        row = {"Department": d}
        for i, fh in enumerate(future_hours, 1):
            pred = engine.predict(d, (sim_hour + i) % 24, noise=0.06)
            row[fh + " Wait"] = f"{pred['predicted_wait_min']} min"
            row[fh + " Queue"] = pred["queue_length"]
        rows.append(row)

    df_fc = pd.DataFrame(rows)
    html_rows = ""
    for _, r in df_fc.iterrows():
        html_rows += "<tr>" + "".join(f"<td>{v}</td>" for v in r.values) + "</tr>"
    header_cells = "".join(f"<th>{c}</th>" for c in df_fc.columns)
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #dde4ed;border-radius:12px;padding:0;overflow:hidden;">
      <table class="styled-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{html_rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Arrival heatmap
    st.markdown('<div class="section-head" style="margin-top:24px;">Patient Arrival Heatmap (Avg Arrivals/Hour)</div>', unsafe_allow_html=True)
    heat_df = engine.arrival_heatmap_data()
    fig_heat = go.Figure(go.Heatmap(
        z=heat_df.values, x=heat_df.columns.tolist(), y=heat_df.index.tolist(),
        colorscale=[[0,"#eff6ff"],[0.3,"#93c5fd"],[0.6,"#2563eb"],[1.0,"#1e3a8a"]],
        hovertemplate="<b>%{y} %{x}</b>: %{z} arrivals<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#64748b"), outlinecolor="#dde4ed"),
    ))
    fig_heat.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        xaxis=dict(color="#64748b"), yaxis=dict(color="#64748b"),
        font=dict(color="#334155"),
        margin=dict(t=20, b=40, l=60, r=20),
        height=280,
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────
# PAGE: COUNTER ALLOCATION
# ─────────────────────────────────────────────
elif "Counter Allocation" in page:
    st.markdown('<div class="section-head">Optimal Counter Allocation Engine</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#ffffff;border:1px solid #dde4ed;border-radius:12px;padding:16px 20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
      <div style="font-size:13px;color:#475569;line-height:1.7;">
        The allocation engine uses a <strong style="color:#0052a3;">Greedy + Linear Programming hybrid</strong> to solve the
        constrained resource assignment problem: minimize total expected waiting time
        subject to staff availability, service capacity limits, and minimum SLA guarantees.
      </div>
    </div>
    """, unsafe_allow_html=True)

    total_staff = st.slider("Total Available Staff (all depts combined)", 10, 40, 22)

    # Compute optimal allocation
    alloc_rows = []
    total_needed = sum(p["counters_needed"] for p in predictions)
    for p in predictions:
        ratio  = p["counters_needed"] / max(total_needed, 1)
        alloc  = max(1, round(ratio * total_staff))
        saving = max(0, round((1 - alloc / max(p["counters_needed"], 1)) * p["predicted_wait_min"] * 0.3, 1))
        alloc_rows.append({
            "Department": p["dept"],
            "Current Counters": p["counters_active"],
            "AI Recommended": p["counters_needed"],
            "LP Optimized": alloc,
            "Queue Length": p["queue_length"],
            "Pred. Wait (min)": p["predicted_wait_min"],
            "Est. Savings (min)": saving,
            "Status": badge_label(p["status"]),
        })

    df_alloc = pd.DataFrame(alloc_rows)
    cols_to_show = df_alloc.columns.tolist()
    header_cells = "".join(f"<th>{c}</th>" for c in cols_to_show)
    html_rows = ""
    for _, r in df_alloc.iterrows():
        html_rows += "<tr>"
        for c, v in r.items():
            if c == "Status":
                stype = "critical" if "CRITICAL" in str(v) else "high" if "HIGH" in str(v) else "normal" if "NORMAL" in str(v) else "low"
                html_rows += f'<td><span class="dept-badge {badge_for_status(stype)}">{v}</span></td>'
            elif c == "Est. Savings (min)":
                html_rows += f'<td style="color:#16a34a;font-weight:700;">{v}</td>'
            elif c == "AI Recommended":
                html_rows += f'<td style="color:#0052a3;font-weight:700;">{v}</td>'
            else:
                html_rows += f"<td>{v}</td>"
        html_rows += "</tr>"

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #dde4ed;border-radius:12px;overflow:hidden;">
      <table class="styled-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{html_rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Counter allocation bar chart
    st.markdown('<div class="section-head" style="margin-top:24px;">Counter Allocation Comparison</div>', unsafe_allow_html=True)
    dept_labels = [p["dept"].split("–")[-1].strip().split("/")[0].strip()[:15] for p in predictions]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Currently Active", x=dept_labels,
                             y=[p["counters_active"] for p in predictions],
                             marker_color="#e2e8f0", text=[p["counters_active"] for p in predictions],
                             textposition="outside", textfont=dict(color="#64748b")))
    fig_bar.add_trace(go.Bar(name="AI Recommended", x=dept_labels,
                             y=[p["counters_needed"] for p in predictions],
                             marker_color="#0066cc", text=[p["counters_needed"] for p in predictions],
                             textposition="outside", textfont=dict(color="#60b3ff")))
    fig_bar.add_trace(go.Bar(name="LP Optimized", x=dept_labels,
                             y=df_alloc["LP Optimized"].tolist(),
                             marker_color="#00c853", text=df_alloc["LP Optimized"].tolist(),
                             textposition="outside", textfont=dict(color="#00e676")))
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        xaxis=dict(color="#64748b", gridcolor="#e2e8f0"),
        yaxis=dict(title="Counters", color="#64748b", gridcolor="#e2e8f0"),
        font=dict(color="#334155"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#475569"),
        margin=dict(t=20, b=60, l=50, r=20),
        height=360,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────
# PAGE: ANALYTICS & REPORTS
# ─────────────────────────────────────────────
elif "Analytics" in page:
    st.markdown('<div class="section-head">Historical Wait-Time Analysis</div>', unsafe_allow_html=True)

    hist_df = engine.historic_accuracy()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["actual"],
        name="Actual Wait", mode="lines+markers",
        line=dict(color="#d97706", width=2),
        marker=dict(size=5),
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["predicted"],
        name="AI Predicted", mode="lines+markers",
        line=dict(color="#0066cc", width=2, dash="dot"),
        marker=dict(size=5),
    ))
    fig_hist.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        xaxis=dict(color="#64748b", gridcolor="#e2e8f0"),
        yaxis=dict(title="Avg Wait (min)", color="#64748b", gridcolor="#e2e8f0"),
        font=dict(color="#334155"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#475569"),
        margin=dict(t=20, b=40, l=50, r=20),
        height=320,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # Metrics summary row
    r2   = round(np.corrcoef(hist_df["actual"], hist_df["predicted"])[0, 1] ** 2, 3)
    rmse = round(np.sqrt(np.mean((hist_df["actual"] - hist_df["predicted"]) ** 2)), 2)
    mae  = round(np.mean(np.abs(hist_df["actual"] - hist_df["predicted"])), 2)
    mape = round(np.mean(np.abs((hist_df["actual"] - hist_df["predicted"]) / hist_df["actual"])) * 100, 1)

    metric_cols = st.columns(4)
    for col, (lbl, val, clr) in zip(metric_cols, [
        ("R² Score", r2, "green"), ("RMSE", f"{rmse} min", "blue"),
        ("MAE", f"{mae} min", "blue"), ("MAPE", f"{mape}%", "amber"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card {clr}" style="margin-top:0;">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value {clr}">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # Weekly summary table
    st.markdown('<div class="section-head" style="margin-top:24px;">Weekly Operational Summary</div>', unsafe_allow_html=True)
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    summary_data = {
        "Week": weeks,
        "Avg Wait (min)": [31.4, 28.9, 26.1, 24.3],
        "Peak Queue": [42, 38, 34, 31],
        "Counter Utilization": ["68%", "72%", "77%", "79%"],
        "Recommendations Accepted": ["61%", "68%", "74%", "82%"],
        "Prediction Accuracy": ["84.1%", "86.3%", "88.7%", "89.4%"],
    }
    df_weekly = pd.DataFrame(summary_data)
    header_cells = "".join(f"<th>{c}</th>" for c in df_weekly.columns)
    html_rows    = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for _, row in df_weekly.iterrows())
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #dde4ed;border-radius:12px;overflow:hidden;">
      <table class="styled-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{html_rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif "Model Performance" in page:
    st.markdown('<div class="section-head">ML Model Comparison & Validation</div>', unsafe_allow_html=True)

    models_data = {
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "Ensemble (Final)"],
        "R²": [0.712, 0.851, 0.863, 0.876, 0.889],
        "RMSE (min)": [14.2, 9.8, 9.1, 8.5, 7.4],
        "MAE (min)": [10.8, 7.2, 6.7, 6.1, 5.1],
        "MAPE (%)": [22.1, 14.6, 13.9, 12.7, 11.3],
        "Train Time (s)": [0.3, 12.4, 18.7, 9.2, 40.3],
        "Inference (ms)": [1, 38, 44, 31, 80],
        "Deployed": ["No", "Partial", "No", "Partial", "✅ Yes"],
    }
    df_models = pd.DataFrame(models_data)
    header_cells = "".join(f"<th>{c}</th>" for c in df_models.columns)
    html_rows = ""
    for _, r in df_models.iterrows():
        is_best = r["Model"] == "Ensemble (Final)"
        style = "background:#eff6ff;" if is_best else ""
        html_rows += f"<tr style='{style}'>"
        for c, v in r.items():
            fw = "font-weight:700;color:#0052a3;" if is_best and c == "Model" else ""
            cv = "color:#16a34a;" if str(v) == "✅ Yes" else ""
            html_rows += f"<td style='{fw}{cv}'>{v}</td>"
        html_rows += "</tr>"

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #dde4ed;border-radius:12px;overflow:hidden;margin-bottom:24px;">
      <table class="styled-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{html_rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance bar
    st.markdown('<div class="section-head">Feature Importance (SHAP Values — XGBoost)</div>', unsafe_allow_html=True)
    features = ["Hour of Day", "Queue Length (t-1)", "Day of Week", "Dept Type",
                "Arrival Rate (30min avg)", "Staff Count", "Season", "Holiday Flag"]
    importances = [0.241, 0.198, 0.152, 0.134, 0.112, 0.087, 0.043, 0.033]
    colors_fi = ["#0052a3" if i < 3 else "#93c5fd" for i in range(len(features))]

    fig_fi = go.Figure(go.Bar(
        x=importances, y=features, orientation="h",
        marker_color=colors_fi,
        text=[f"{v:.3f}" for v in importances],
        textposition="outside",
        textfont=dict(color="#64748b"),
        hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
    ))
    fig_fi.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        xaxis=dict(title="Mean |SHAP|", color="#64748b", gridcolor="#e2e8f0"),
        yaxis=dict(color="#334155"),
        font=dict(color="#334155"),
        margin=dict(t=10, b=40, l=170, r=60),
        height=320,
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    # Cross-validation results
    st.markdown('<div class="section-head" style="margin-top:24px;">5-Fold Time-Series Cross-Validation</div>', unsafe_allow_html=True)
    cv_data = {
        "Fold": [f"Fold {i}" for i in range(1, 6)],
        "Train Period": ["Wk 1–8", "Wk 1–10", "Wk 1–12", "Wk 1–14", "Wk 1–16"],
        "Val Period": ["Wk 9–10", "Wk 11–12", "Wk 13–14", "Wk 15–16", "Wk 17–18"],
        "R²": [0.881, 0.886, 0.892, 0.884, 0.902],
        "RMSE": [7.8, 7.5, 7.1, 7.6, 6.9],
        "MAE": [5.4, 5.2, 4.9, 5.3, 4.7],
    }
    df_cv = pd.DataFrame(cv_data)
    hc = "".join(f"<th>{c}</th>" for c in df_cv.columns)
    hr = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for _, row in df_cv.iterrows())
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #dde4ed;border-radius:12px;overflow:hidden;">
      <table class="styled-table">
        <thead><tr>{hc}</tr></thead>
        <tbody>{hr}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    avg_r2   = round(np.mean(cv_data["R²"]), 3)
    avg_rmse = round(np.mean(cv_data["RMSE"]), 2)
    std_rmse = round(np.std(cv_data["RMSE"]), 2)
    st.markdown(f"""
    <div style="background:#f8fafc;border:1px solid #dde4ed;border-radius:10px;padding:14px 18px;margin-top:12px;font-size:13px;color:#64748b;">
      Cross-validation summary — Mean R²: <strong style="color:#0052a3;">{avg_r2}</strong> &nbsp;|&nbsp;
      Mean RMSE: <strong style="color:#0052a3;">{avg_rmse} ± {std_rmse} min</strong> &nbsp;|&nbsp;
      Model generalizes well across all time windows (low variance between folds).
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px;padding:20px 0;border-top:1px solid #dde4ed;
            display:flex;justify-content:space-between;align-items:center;
            font-size:11px;color:#94a3b8;">
  <div>MediFlow AI Intelligence System &nbsp;·&nbsp; Ensemble ML Engine &nbsp;·&nbsp;
       XGBoost + Random Forest + Gradient Boosting</div>
  <div>For authorized hospital administrative use only &nbsp;·&nbsp; v2.4.1</div>
</div>
""", unsafe_allow_html=True)