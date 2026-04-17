"""
Hospital Queue Intelligence Platform - High Readability Version
================================================================
Streamlit dashboard with white background and optimal color contrast

Run with: streamlit run app_improved_readability.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import joblib
import time
from scipy import stats
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import modules
try:
    from modules.module1_data_preprocessing import DataPreprocessor, FeatureEngineer
    from modules.module2_predictive_modelling import AdvancedModelTrainer, DeepNeuralNetworkModel
    from modules.module3_counter_allocation import AllocationEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Modules not available: {e}")
    MODULES_AVAILABLE = False

# Configuration
try:
    from config import paths, allocation_config
except ImportError:
    class DefaultConfig:
        class paths:
            MODELS_DIR = Path("models")
            OUTPUTS_DIR = Path("outputs")
            TRAINED_MODEL = MODELS_DIR / "queue_prediction_model.pkl"
            LABEL_ENCODERS = MODELS_DIR / "label_encoders.pkl"
            FEATURE_NAMES = MODELS_DIR / "feature_names.pkl"
        
        class allocation_config:
            total_available_staff = 20
            DEPARTMENT_CONFIG = {
                'OPD': {'min_counters': 2, 'max_counters': 8, 'priority_weight': 1.2},
                'Diagnostics': {'min_counters': 2, 'max_counters': 6, 'priority_weight': 1.0},
                'Pharmacy': {'min_counters': 2, 'max_counters': 6, 'priority_weight': 0.9},
                'Emergency': {'min_counters': 3, 'max_counters': 10, 'priority_weight': 2.0}
            }
    
    paths = DefaultConfig.paths
    allocation_config = DefaultConfig.allocation_config

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Hospital Queue Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HIGH CONTRAST CSS - WHITE BACKGROUND
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { 
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - Pure white */
    .main { 
        background: #ffffff !important;
    }
    
    /* Global text - Dark for maximum contrast */
    body, .stMarkdown, p, span, div, h1, h2, h3, h4, label, .stMarkdown p, .stMarkdown span {
        color: #1a1a1a !important;
    }
    
    /* Header - High contrast blue with white text */
    .header {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.2);
    }
    
    .header h1 { 
        color: #ffffff !important; 
        font-size: 2.5rem; 
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header p { 
        color: #ffffff !important; 
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Cards - White with dark text and subtle shadow */
    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #3b82f6;
        transition: all 0.2s;
    }
    
    .card:hover { 
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .card h3, .card h4 { 
        color: #1a1a1a !important; 
        margin-top: 0;
        font-weight: 600;
    }
    
    .card p, .card span, .card strong { 
        color: #333333 !important;
    }
    
    /* Metric cards - White background with colored accents */
    .metric {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #4b5563 !important;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    
    /* Department cards with strong border colors */
    .dept-opd { border-left-color: #2563eb; border-left-width: 5px; }
    .dept-diagnostics { border-left-color: #059669; border-left-width: 5px; }
    .dept-pharmacy { border-left-color: #d97706; border-left-width: 5px; }
    .dept-emergency { border-left-color: #dc2626; border-left-width: 5px; }
    
    /* Status indicators - Larger and more visible */
    .status {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-normal { background: #059669; }
    .status-warning { background: #d97706; }
    .status-critical { background: #dc2626; }
    
    /* Alerts - High contrast backgrounds */
    .alert {
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid;
        font-weight: 500;
    }
    
    .alert strong {
        font-weight: 600 !important;
    }
    
    .alert-success {
        background: #d1fae5;
        border-left-color: #059669;
        color: #065f46 !important;
    }
    
    .alert-success strong {
        color: #065f46 !important;
    }
    
    .alert-warning {
        background: #fef3c7;
        border-left-color: #d97706;
        color: #92400e !important;
    }
    
    .alert-warning strong {
        color: #92400e !important;
    }
    
    .alert-error {
        background: #fee2e2;
        border-left-color: #dc2626;
        color: #991b1b !important;
    }
    
    .alert-error strong {
        color: #991b1b !important;
    }
    
    .alert-info {
        background: #dbeafe;
        border-left-color: #2563eb;
        color: #1e40af !important;
    }
    
    .alert-info strong {
        color: #1e40af !important;
    }
    
    /* Sidebar - White with dark text */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] p {
        color: #1a1a1a !important;
    }
    
    /* Buttons - High contrast */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
    }
    
    /* Tables - High contrast headers */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    
    .dataframe thead tr th {
        background: #1e40af !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }
    
    .dataframe tbody tr { 
        color: #1a1a1a !important;
        background: #ffffff !important;
    }
    
    .dataframe tbody tr:hover { 
        background: #f3f4f6 !important;
    }
    
    .dataframe tbody td {
        color: #1a1a1a !important;
        padding: 10px !important;
    }
    
    /* Form elements - Dark labels */
    .stSelectbox label, .stSlider label, .stNumberInput label,
    .stCheckbox label, .stRadio label, .stTextInput label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stSelectbox div, .stNumberInput div {
        color: #1a1a1a !important;
    }
    
    /* Slider values */
    .stSlider div[data-baseweb="slider"] {
        color: #1a1a1a !important;
    }
    
    /* Tabs - High contrast */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f9fafb;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #4b5563 !important;
        font-weight: 500 !important;
        background: transparent;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #1e40af !important;
        font-weight: 600 !important;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    .stRadio div[role="radiogroup"] label {
        color: #1a1a1a !important;
    }
    
    /* Section headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        background: #f9fafb !important;
    }
    
    /* Captions */
    .caption, small {
        color: #6b7280 !important;
    }
    
    /* Markdown text */
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_model_artifacts():
    """Load trained models and preprocessing artifacts"""
    try:
        model = joblib.load(paths.TRAINED_MODEL) if paths.TRAINED_MODEL.exists() else None
        label_encoders = joblib.load(paths.LABEL_ENCODERS) if paths.LABEL_ENCODERS.exists() else {}
        feature_names = joblib.load(paths.FEATURE_NAMES) if paths.FEATURE_NAMES.exists() else []
        return model, label_encoders, feature_names
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return None, {}, []

def generate_synthetic_data(num_points=100):
    """Generate synthetic real-time queue data"""
    np.random.seed(int(time.time()) % 1000)
    departments = list(allocation_config.DEPARTMENT_CONFIG.keys())
    data = []
    base_time = datetime.now() - timedelta(hours=2)
    
    for i in range(num_points):
        timestamp = base_time + timedelta(minutes=i)
        hour = timestamp.hour
        
        for dept in departments:
            # Peak hours simulation
            is_peak = (9 <= hour <= 11) or (15 <= hour <= 17)
            base_queue = np.random.randint(15, 35) if is_peak else np.random.randint(5, 20)
            base_wait = np.random.uniform(20, 45) if is_peak else np.random.uniform(10, 25)
            
            trend = np.sin(i / 10) * 5
            
            data.append({
                'timestamp': timestamp,
                'department': dept,
                'queue_length': max(0, int(base_queue + trend + np.random.normal(0, 3))),
                'wait_time': max(5, base_wait + trend + np.random.normal(0, 5)),
                'arrivals_per_hour': np.random.randint(10, 40),
                'active_counters': allocation_config.DEPARTMENT_CONFIG[dept]['min_counters'] + np.random.randint(0, 3)
            })
    
    return pd.DataFrame(data)

def engineer_features(input_data, label_encoders):
    """Engineer features for prediction"""
    features = {}
    
    # Temporal features
    ts = input_data.get('timestamp', datetime.now())
    features.update({
        'hour': ts.hour,
        'day_of_week': ts.weekday(),
        'is_weekend': 1 if ts.weekday() >= 5 else 0,
        'month': ts.month,
        'day_of_month': ts.day,
        'week_of_year': ts.isocalendar()[1],
        'is_monday': 1 if ts.weekday() == 0 else 0
    })
    
    # Queue features
    features.update({
        'queue_length_at_arrival': input_data.get('queue_length', 10),
        'arrivals_this_hour': input_data.get('arrivals_per_hour', 20),
        'system_utilization': input_data.get('system_utilization', 0.75),
        'total_counters_available': input_data.get('active_counters', 4),
        'doctor_efficiency': input_data.get('doctor_efficiency', 0.95),
        'is_emergency': input_data.get('is_emergency', 0)
    })
    
    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    
    # Interaction features
    features['queue_per_counter'] = features['queue_length_at_arrival'] / max(features['total_counters_available'], 1)
    features['efficiency_load'] = features['doctor_efficiency'] * features['arrivals_this_hour']
    features['utilization_load'] = features['system_utilization'] * features['arrivals_this_hour']
    features['log_queue'] = np.log1p(features['queue_length_at_arrival'])
    features['sqrt_arrivals'] = np.sqrt(features['arrivals_this_hour'])
    
    # Department encoding
    if 'department' in input_data and 'department' in label_encoders:
        le = label_encoders['department']
        dept = str(input_data['department'])
        features['department_enc'] = le.transform([dept])[0] if dept in le.classes_ else 0
    else:
        features['department_enc'] = 0
    
    return features

def make_prediction(model, feature_names, input_features):
    """Make prediction with confidence intervals"""
    try:
        input_df = pd.DataFrame([input_features])
        
        # Ensure all features present
        for feat in feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        X = input_df[feature_names]
        
        if model is not None:
            prediction = float(model.predict(X)[0])
        else:
            # Fallback: simple queueing model
            queue_len = input_features.get('queue_length_at_arrival', 10)
            counters = input_features.get('total_counters_available', 4)
            prediction = (queue_len / counters) * 3.5
        
        # Confidence interval
        std_error = prediction * 0.15
        confidence_lower = max(0, prediction - 1.96 * std_error)
        confidence_upper = prediction + 1.96 * std_error
        
        return {
            'prediction': prediction,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'std_error': std_error
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

def calculate_dept_predictions(realtime_data, model, label_encoders, feature_names):
    """Calculate predictions for all departments"""
    latest_data = realtime_data.groupby('department').tail(1)
    predictions = {}
    
    for _, row in latest_data.iterrows():
        dept = row['department']
        features = engineer_features(row.to_dict(), label_encoders)
        result = make_prediction(model, feature_names, features)
        predictions[dept] = result['prediction'] if result else row['wait_time']
    
    return predictions

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_trend_chart(realtime_data):
    """Create real-time trend chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Wait Time Trends', 'Queue Length Trends'),
        vertical_spacing=0.12
    )
    
    colors = {'OPD': '#2563eb', 'Diagnostics': '#059669', 'Pharmacy': '#d97706', 'Emergency': '#dc2626'}
    
    for dept in realtime_data['department'].unique():
        dept_data = realtime_data[realtime_data['department'] == dept]
        color = colors.get(dept, '#6b7280')
        
        fig.add_trace(go.Scatter(
            x=dept_data['timestamp'], y=dept_data['wait_time'],
            name=dept, mode='lines+markers',
            line=dict(width=2, color=color),
            marker=dict(size=4)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=dept_data['timestamp'], y=dept_data['queue_length'],
            name=dept, mode='lines+markers',
            line=dict(width=2, color=color),
            marker=dict(size=4),
            showlegend=False
        ), row=2, col=1)
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Wait Time (min)", row=1, col=1)
    fig.update_yaxes(title_text="Queue Length", row=2, col=1)
    fig.update_layout(height=600, hovermode='x unified', plot_bgcolor='white', paper_bgcolor='white')
    
    return fig

def create_heatmap(realtime_data):
    """Create department congestion heatmap"""
    pivot = realtime_data.pivot_table(
        values='wait_time',
        index='department',
        columns=realtime_data['timestamp'].dt.hour,
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        text=pivot.values.round(1),
        texttemplate='%{text}',
        colorbar=dict(title="Wait (min)")
    ))
    
    fig.update_layout(
        title='Congestion by Hour',
        xaxis_title='Hour',
        yaxis_title='Department',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_allocation_chart(current, recommended):
    """Create allocation comparison chart"""
    departments = list(current.keys())
    
    fig = go.Figure(data=[
        go.Bar(name='Current', x=departments, y=[current[d] for d in departments],
               marker_color='#2563eb', text=[current[d] for d in departments], textposition='auto'),
        go.Bar(name='Recommended', x=departments, y=[recommended[d] for d in departments],
               marker_color='#059669', text=[recommended[d] for d in departments], textposition='auto')
    ])
    
    fig.update_layout(
        title='Counter Allocation Comparison',
        xaxis_title='Department',
        yaxis_title='Counters',
        barmode='group',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_metric_card(value, label, delta=None, color="#2563eb"):
    """Render a metric card"""
    delta_html = ""
    if delta:
        delta_color = "#059669" if delta.startswith("✓") else "#dc2626"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.9rem; margin-top: 0.5rem; font-weight: 600;">{delta}</div>'
    
    return f"""
    <div class="metric">
        <div class="metric-value" style="color: {color} !important;">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """

def render_dept_card(dept, queue, wait_time, counters, status):
    """Render department status card"""
    dept_colors = {
        'OPD': '#2563eb',
        'Diagnostics': '#059669',
        'Pharmacy': '#d97706',
        'Emergency': '#dc2626'
    }
    
    status_class = {
        'Normal': 'status-normal',
        'Busy': 'status-warning',
        'Critical': 'status-critical'
    }
    
    color = dept_colors.get(dept, '#6b7280')
    dept_class = f"dept-{dept.lower().replace(' ', '')}"
    
    return f"""
    <div class="card {dept_class}" style="border-left-color: {color};">
        <h4>{dept}</h4>
        <div style="font-size: 2rem; font-weight: 700; color: {color} !important; margin: 0.5rem 0;">
            {wait_time:.1f} <span style="font-size: 1rem; font-weight: 500;">min</span>
        </div>
        <p><strong>Queue:</strong> {queue} patients</p>
        <p><strong>Counters:</strong> {counters} active</p>
        <div style="margin-top: 1rem;">
            <span class="status {status_class.get(status, 'status-normal')}"></span>
            <strong>{status}</strong>
        </div>
    </div>
    """

def render_alert(message, alert_type="info"):
    """Render alert message"""
    icons = {
        'success': '✅',
        'warning': '⚠️',
        'error': '🚨',
        'info': 'ℹ️'
    }
    
    return f"""
    <div class="alert alert-{alert_type}">
        <strong>{icons.get(alert_type, 'ℹ️')} {message}</strong>
    </div>
    """

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🏥 Hospital Queue Intelligence</h1>
        <p>AI-Powered Prediction & Resource Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load artifacts
    model, label_encoders, feature_names = load_model_artifacts()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Predictions", "Counter Allocation", "Analytics", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🔄 Status")
        
        status = "Active" if model is not None else "Demo Mode"
        status_class = "status-normal" if model else "status-warning"
        
        st.markdown(f"""
        <div class="card">
            <p><strong>Model:</strong></p>
            <p><span class="status {status_class}"></span> {status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Generate data
    realtime_data = generate_synthetic_data()
    
    # ========================================================================
    # DASHBOARD PAGE
    # ========================================================================
    
    if page == "Dashboard":
        st.markdown("## Real-Time Dashboard")
        
        # Metrics
        latest = realtime_data.groupby('department').tail(1)
        total_queue = latest['queue_length'].sum()
        avg_wait = latest['wait_time'].mean()
        total_counters = latest['active_counters'].sum()
        efficiency = max(0, (1 - avg_wait / 60) * 100)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta = "✓ Normal" if total_queue < 50 else "⚠ High"
            st.markdown(render_metric_card(int(total_queue), "Total Queue", delta, "#2563eb"), unsafe_allow_html=True)
        
        with col2:
            delta = "✓ Good" if avg_wait < 20 else "⚠ Above Target"
            st.markdown(render_metric_card(f"{avg_wait:.1f}", "Avg Wait (min)", delta, "#059669"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(render_metric_card(int(total_counters), "Active Counters", "✓ Operational", "#d97706"), unsafe_allow_html=True)
        
        with col4:
            delta = "✓ Optimal" if efficiency > 70 else "⚠ Below Target"
            st.markdown(render_metric_card(f"{efficiency:.0f}%", "Efficiency", delta, "#6366f1"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Queue Trends")
            st.plotly_chart(create_trend_chart(realtime_data), use_container_width=True)
        
        with col2:
            st.markdown("### 🔥 Congestion Heatmap")
            st.plotly_chart(create_heatmap(realtime_data), use_container_width=True)
        
        st.markdown("---")
        
        # Department Cards
        st.markdown("### 🏥 Department Status")
        
        cols = st.columns(len(allocation_config.DEPARTMENT_CONFIG))
        
        for idx, dept in enumerate(allocation_config.DEPARTMENT_CONFIG.keys()):
            dept_data = latest[latest['department'] == dept]
            
            if len(dept_data) > 0:
                queue = int(dept_data['queue_length'].iloc[0])
                wait = dept_data['wait_time'].iloc[0]
                counters = int(dept_data['active_counters'].iloc[0])
                
                status = "Normal" if wait < 15 else ("Busy" if wait < 30 else "Critical")
                
                with cols[idx]:
                    st.markdown(
                        render_dept_card(dept, queue, wait, counters, status),
                        unsafe_allow_html=True
                    )
        
        # Alerts
        st.markdown("---")
        st.markdown("### 🔔 Alerts")
        
        alerts = []
        for dept in allocation_config.DEPARTMENT_CONFIG.keys():
            dept_data = latest[latest['department'] == dept]
            if len(dept_data) > 0:
                wait = dept_data['wait_time'].iloc[0]
                if wait > 30:
                    alerts.append((f"Critical: {dept} - {wait:.1f} min wait", "error"))
                elif wait > 20:
                    alerts.append((f"Warning: {dept} - {wait:.1f} min wait", "warning"))
        
        if alerts:
            for msg, alert_type in alerts:
                st.markdown(render_alert(msg, alert_type), unsafe_allow_html=True)
        else:
            st.markdown(render_alert("All systems normal", "success"), unsafe_allow_html=True)
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # ========================================================================
    # PREDICTIONS PAGE
    # ========================================================================
    
    elif page == "Predictions":
        st.markdown("## Queue Predictions")
        
        st.markdown(render_alert(
            "AI models analyze patterns to predict wait times with confidence intervals",
            "info"
        ), unsafe_allow_html=True)
        
        # Input form
        st.markdown("### 🎯 Make Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dept = st.selectbox("Department", list(allocation_config.DEPARTMENT_CONFIG.keys()))
            queue = st.slider("Queue Length", 0, 50, 15)
            arrivals = st.slider("Arrivals/Hour", 0, 60, 25)
        
        with col2:
            counters = st.slider("Counters", 1, 10, 4)
            hour = st.slider("Hour", 0, 23, datetime.now().hour)
            is_weekend = st.checkbox("Weekend?")
        
        with col3:
            util = st.slider("Utilization", 0.0, 1.0, 0.75, 0.05)
            efficiency = st.slider("Efficiency", 0.0, 1.0, 0.95, 0.05)
            emergency = st.checkbox("Emergency?")
        
        if st.button("🔮 Predict", use_container_width=True):
            with st.spinner("Analyzing..."):
                input_data = {
                    'department': dept,
                    'queue_length': queue,
                    'arrivals_per_hour': arrivals,
                    'active_counters': counters,
                    'hour': hour,
                    'is_weekend': 1 if is_weekend else 0,
                    'system_utilization': util,
                    'doctor_efficiency': efficiency,
                    'is_emergency': 1 if emergency else 0,
                    'timestamp': datetime.now()
                }
                
                features = engineer_features(input_data, label_encoders)
                result = make_prediction(model, feature_names, features)
                
                if result:
                    pred = result['prediction']
                    lower = result['confidence_lower']
                    upper = result['confidence_upper']
                    
                    st.markdown("### 📊 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(render_metric_card(
                            f"{pred:.1f}", "Predicted Wait (min)", color="#2563eb"
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(render_metric_card(
                            f"{lower:.1f} - {upper:.1f}", "95% Confidence", color="#059669"
                        ), unsafe_allow_html=True)
                    
                    with col3:
                        urgency = "Low" if pred < 15 else ("Medium" if pred < 30 else "High")
                        color = "#059669" if pred < 15 else ("#d97706" if pred < 30 else "#dc2626")
                        st.markdown(render_metric_card(urgency, "Urgency", color=color), unsafe_allow_html=True)
                    
                    # Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Wait Time (min)"},
                        gauge={
                            'axis': {'range': [None, 60]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 15], 'color': "#d1fae5"},
                                {'range': [15, 30], 'color': "#fef3c7"},
                                {'range': [30, 60], 'color': "#fee2e2"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 30}
                        }
                    ))
                    
                    fig.update_layout(height=300, paper_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    if pred > 25:
                        st.markdown(render_alert(
                            "Action recommended: Consider adding counters or optimizing workflow",
                            "warning"
                        ), unsafe_allow_html=True)
                    else:
                        st.markdown(render_alert("Operating normally", "success"), unsafe_allow_html=True)
        
        # Department predictions
        st.markdown("---")
        st.markdown("### 📊 Current Predictions")
        
        predictions = calculate_dept_predictions(realtime_data, model, label_encoders, feature_names)
        
        fig = go.Figure(go.Bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            marker=dict(
                color=list(predictions.values()),
                colorscale='RdYlGn_r',
                showscale=True
            ),
            text=[f"{v:.1f}" for v in predictions.values()],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Predicted Wait Times',
            xaxis_title='Department',
            yaxis_title='Wait Time (min)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # COUNTER ALLOCATION PAGE
    # ========================================================================
    
    elif page == "Counter Allocation":
        st.markdown("## Counter Allocation")
        
        st.markdown(render_alert(
            "Optimization engine recommends optimal counter allocation to minimize wait times",
            "info"
        ), unsafe_allow_html=True)
        
        predictions = calculate_dept_predictions(realtime_data, model, label_encoders, feature_names)
        latest = realtime_data.groupby('department').tail(1)
        current = {row['department']: int(row['active_counters']) for _, row in latest.iterrows()}
        
        if MODULES_AVAILABLE:
            try:
                engine = AllocationEngine(method='linear_programming')
                recommendations = engine.generate_recommendations(
                    predictions, current, allocation_config.total_available_staff
                )
                
                recommended = {rec.department: rec.recommended_counters for rec in recommendations}
                
                # Comparison chart
                st.markdown("### 📊 Allocation Comparison")
                st.plotly_chart(create_allocation_chart(current, recommended), use_container_width=True)
                
                # Recommendations
                st.markdown("### 📋 Recommendations")
                
                for rec in recommendations:
                    action_icons = {'increase': '⬆️', 'decrease': '⬇️', 'maintain': '➡️'}
                    action_colors = {'increase': '#dc2626', 'decrease': '#059669', 'maintain': '#2563eb'}
                    
                    icon = action_icons.get(rec.action, '•')
                    color = action_colors.get(rec.action, '#6b7280')
                    
                    dept_class = f"dept-{rec.department.lower().replace(' ', '')}"
                    
                    st.markdown(f"""
                    <div class="card {dept_class}">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <h4>{rec.department}</h4>
                                <p>Predicted: {rec.predicted_wait_time:.1f} min</p>
                            </div>
                            <div style="font-size: 2rem;">{icon}</div>
                        </div>
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
                            <p><strong>Current:</strong> {rec.current_counters} counters</p>
                            <p style="color: {color} !important;"><strong>Recommended:</strong> {rec.recommended_counters} counters</p>
                            <p><strong>Priority:</strong> {rec.priority_score:.2f}</p>
                        </div>
                        <div style="margin-top: 1rem; padding: 0.75rem; background: #f8fafc; border-radius: 8px;">
                            <p style="margin: 0; font-size: 0.9rem;">{rec.justification}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Alerts
                alerts = engine.generate_alerts(recommendations)
                if alerts:
                    st.markdown("---")
                    st.markdown("### 🔔 Alerts")
                    for alert in alerts:
                        alert_type = "error" if "HIGH WAIT" in alert else "info"
                        st.markdown(render_alert(alert, alert_type), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Allocation error: {e}")
        else:
            st.warning("Allocation module not available")
    
    # ========================================================================
    # ANALYTICS PAGE
    # ========================================================================
    
    elif page == "Analytics":
        st.markdown("## Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Wait Time Distribution")
            
            fig = go.Figure()
            for dept in realtime_data['department'].unique():
                dept_data = realtime_data[realtime_data['department'] == dept]
                fig.add_trace(go.Box(y=dept_data['wait_time'], name=dept))
            
            fig.update_layout(
                title='By Department', 
                yaxis_title='Wait Time (min)', 
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Efficiency Metrics")
            
            metrics = []
            for dept in realtime_data['department'].unique():
                dept_data = realtime_data[realtime_data['department'] == dept]
                metrics.append({
                    'Department': dept,
                    'Avg Wait': dept_data['wait_time'].mean(),
                    'Avg Queue': dept_data['queue_length'].mean(),
                    'Efficiency': max(0, (1 - dept_data['wait_time'].mean() / 60) * 100)
                })
            
            df = pd.DataFrame(metrics)
            st.dataframe(
                df.style.format({
                    'Avg Wait': '{:.1f}',
                    'Avg Queue': '{:.1f}',
                    'Efficiency': '{:.1f}%'
                }).background_gradient(subset=['Efficiency'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("---")
        
        # Correlation
        st.markdown("### Correlation Analysis")
        
        corr_data = realtime_data[['wait_time', 'queue_length', 'active_counters', 'arrivals_per_hour']]
        corr = corr_data.corr()
        
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}'
        ))
        
        fig.update_layout(
            title='Feature Correlations', 
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # SETTINGS PAGE
    # ========================================================================
    
    elif page == "Settings":
        st.markdown("## Settings")
        
        tab1, tab2 = st.tabs(["Department Config", "System Settings"])
        
        with tab1:
            st.markdown("### Department Configuration")
            
            for dept, config in allocation_config.DEPARTMENT_CONFIG.items():
                with st.expander(f"{dept}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.number_input("Min Counters", value=config['min_counters'], key=f"min_{dept}")
                    with col2:
                        st.number_input("Max Counters", value=config['max_counters'], key=f"max_{dept}")
                    with col3:
                        st.slider("Priority", 0.0, 2.0, config['priority_weight'], key=f"pri_{dept}")
        
        with tab2:
            st.markdown("### System Settings")
            
            st.number_input("Confidence Level (%)", 90, 99, 95)
            st.number_input("Refresh Interval (s)", 5, 300, 30)
            st.selectbox("Allocation Method", ["Linear Programming", "Greedy"])
            st.checkbox("Auto-Allocation")
            st.checkbox("Real-time Notifications", value=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem;">
        <p style="margin: 0; color: #4b5563 !important;">
            <strong>Hospital Queue Intelligence v2.0</strong> | 
            Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()