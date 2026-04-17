"""
Configuration Management System
================================
AI-Based Queue-Time Prediction and Smart Counter Allocation Decision Tool

This module centralizes all configuration parameters for the hospital queue
prediction system as specified in the project documentation.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================

class Paths:
    """Centralized path management for all system components"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    LOGS_DIR = BASE_DIR / "logs"
    MODULES_DIR = BASE_DIR / "modules"
    
    # Data files
    RAW_DATA = DATA_DIR / "hospital_queue_data_realistic.csv"
    PROCESSED_DATA = DATA_DIR / "processed_data.csv"
    
    # Model files
    TRAINED_MODEL = MODELS_DIR / "queue_prediction_model.pkl"
    RF_MODEL = MODELS_DIR / "random_forest_model.pkl"
    XGB_MODEL = MODELS_DIR / "xgboost_model.pkl"
    GB_MODEL = MODELS_DIR / "gradient_boosting_model.pkl"
    PROPHET_MODEL = MODELS_DIR / "prophet_model.pkl"
    SCALER = MODELS_DIR / "scaler.pkl"
    FEATURE_NAMES = MODELS_DIR / "feature_names.pkl"
    LABEL_ENCODERS = MODELS_DIR / "label_encoders.pkl"
    
    # Output files
    METRICS_JSON = OUTPUTS_DIR / "model_metrics.json"
    COMPARISON_JSON = OUTPUTS_DIR / "model_comparison.json"
    FEATURE_IMPORTANCE = OUTPUTS_DIR / "feature_importance.csv"
    SHAP_VALUES = OUTPUTS_DIR / "shap_values.pkl"
    ALLOCATION_LOGS = OUTPUTS_DIR / "allocation_recommendations.csv"
    PERFORMANCE_REPORT = OUTPUTS_DIR / "performance_report.json"
    
    # Log files
    TRAINING_LOG = LOGS_DIR / "training.log"
    PREDICTION_LOG = LOGS_DIR / "predictions.log"
    ALLOCATION_LOG = LOGS_DIR / "allocation.log"
    SYSTEM_LOG = LOGS_DIR / "system.log"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.OUTPUTS_DIR, 
                         cls.LOGS_DIR, cls.MODULES_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """
    Machine Learning model hyperparameters and configuration
    Based on project documentation requirements
    """
    
    # Data split ratios (as per Phase 2 methodology)
    train_size: float = 0.70  # 70% training
    validation_size: float = 0.15  # 15% validation
    test_size: float = 0.15  # 15% testing
    random_state: int = 42
    shuffle: bool = False  # Preserve temporal order for time-series
    
    # Outlier removal
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    
    # Performance targets (from section 7.1)
    target_rmse: float = 10.0  # minutes
    target_mae: float = 7.0  # minutes
    target_r2: float = 0.85
    target_mape: float = 15.0  # percentage
    
    # Random Forest parameters
    rf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': 42
    })
    
    # XGBoost parameters
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'random_state': 42
    })
    
    # Gradient Boosting parameters
    gb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'subsample': 0.8,
        'max_features': 'sqrt',
        'random_state': 42
    })
    
    # Linear Regression (baseline model)
    lr_params: Dict = field(default_factory=lambda: {
        'fit_intercept': True,
        'n_jobs': -1
    })
    
    # Prophet (time-series forecasting)
    prophet_params: Dict = field(default_factory=lambda: {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative',
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': False
    })
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_method: str = 'time_series'  # Time-series aware cross-validation


# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

@dataclass
class FeatureConfig:
    """Feature engineering configuration as per documentation"""
    
    # Temporal lag features (critical for predictions)
    LAG_FEATURES = {
        'prev_wait_1': 1,  # Previous patient wait time
        'prev_wait_2': 2,  # 2 patients ago
        'rolling_wait_5': 5,  # Rolling average of 5 patients
        'rolling_wait_10': 10,  # Rolling average of 10 patients
        'rolling_queue_3': 3,  # Rolling queue length
        'rolling_queue_5': 5
    }
    
    # Categorical columns to encode
    CATEGORICAL_COLS = ['department', 'patient_complexity', 'time_period']
    
    # Base temporal features
    TEMPORAL_FEATURES = [
        'hour', 'day_of_week', 'day_of_month', 'month', 
        'week_of_year', 'is_weekend', 'is_monday', 'is_holiday'
    ]
    
    # Service-related features
    SERVICE_FEATURES = [
        'queue_length_at_arrival', 'arrivals_this_hour',
        'system_utilization', 'total_counters_available',
        'doctor_efficiency', 'is_emergency', 'service_duration_expected'
    ]
    
    # Engineered interaction features
    INTERACTION_FEATURES = [
        'workload_index',  # queue * utilization / counters
        'arrival_pressure',  # arrivals / counters
        'efficiency_gap',  # doctor_eff - system_util
        'capacity_ratio',  # queue / total_counters
        'utilization_load'  # utilization * arrivals
    ]
    
    # Transformed features
    TRANSFORM_FEATURES = [
        'log_queue',  # Log transform of queue length
        'sqrt_arrivals',  # Square root of arrivals
        'log_service_time'  # Log of service duration
    ]
    
    # Cyclical encoding features
    CYCLICAL_FEATURES = [
        'hour_sin', 'hour_cos',
        'day_sin', 'day_cos',
        'month_sin', 'month_cos'
    ]
    
    @classmethod
    def get_all_features(cls) -> List[str]:
        """Get complete feature list"""
        features = (
            cls.TEMPORAL_FEATURES +
            cls.SERVICE_FEATURES +
            list(cls.LAG_FEATURES.keys()) +
            cls.INTERACTION_FEATURES +
            cls.TRANSFORM_FEATURES +
            cls.CYCLICAL_FEATURES +
            [f'{col}_enc' for col in cls.CATEGORICAL_COLS]
        )
        return features


# ============================================================================
# COUNTER ALLOCATION CONFIGURATION
# ============================================================================

@dataclass
class AllocationConfig:
    """
    Counter allocation optimization parameters
    Based on Module 3 specifications
    """
    
    # Department configurations
    DEPARTMENT_CONFIG = {
        'OPD': {
            'min_counters': 2,
            'max_counters': 8,
            'avg_service_time': 15,  # minutes
            'priority_weight': 1.0
        },
        'Diagnostics': {
            'min_counters': 2,
            'max_counters': 6,
            'avg_service_time': 20,
            'priority_weight': 1.2
        },
        'Pharmacy': {
            'min_counters': 2,
            'max_counters': 5,
            'avg_service_time': 8,
            'priority_weight': 0.8
        }
    }
    
    # Optimization parameters
    total_available_staff: int = 15
    reallocation_cost: float = 5.0  # minutes (cost of moving staff)
    max_utilization_threshold: float = 0.85
    min_utilization_threshold: float = 0.60
    
    # Alert thresholds
    critical_wait_threshold: float = 30.0  # minutes
    warning_wait_threshold: float = 20.0  # minutes
    
    # Optimization method
    optimization_method: str = 'linear_programming'  # 'greedy' or 'linear_programming'


# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

@dataclass
class DashboardConfig:
    """Streamlit dashboard configuration (Module 4)"""
    
    # Server settings
    port: int = 8501
    host: str = 'localhost'
    
    # Update intervals (seconds)
    prediction_update_interval: int = 60
    metrics_update_interval: int = 300
    
    # Visualization settings
    chart_height: int = 400
    chart_width: int = 800
    color_scheme: str = 'viridis'
    
    # Alert settings
    enable_alerts: bool = True
    alert_sound: bool = False
    
    # Display options
    show_shap_values: bool = True
    show_feature_importance: bool = True
    show_historical_trends: bool = True
    max_historical_hours: int = 24


# ============================================================================
# PERFORMANCE MONITORING CONFIGURATION
# ============================================================================

@dataclass
class MonitoringConfig:
    """Performance monitoring settings (Module 5)"""
    
    # Metrics to track
    tracked_metrics: List[str] = field(default_factory=lambda: [
        'rmse', 'mae', 'r2_score', 'mape',
        'avg_waiting_time', 'queue_length',
        'counter_utilization', 'prediction_latency'
    ])
    
    # Performance targets (from section 7.2)
    operational_targets = {
        'waiting_time_reduction': 0.25,  # 25% reduction
        'utilization_target': (0.75, 0.85),  # 75-85% range
        'queue_reduction': 0.25,  # 25% reduction
        'prediction_latency': 2.0  # seconds
    }
    
    # Reporting intervals
    daily_report: bool = True
    weekly_report: bool = True
    monthly_report: bool = True
    
    # Model retraining triggers
    accuracy_drop_threshold: float = 0.05  # Trigger retraining
    min_samples_for_retrain: int = 1000
    retrain_schedule: str = 'weekly'  # 'daily', 'weekly', 'monthly'


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class LogConfig:
    """Logging configuration for all modules"""
    
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Module-specific log levels
    MODULE_LEVELS = {
        'data_preprocessing': logging.INFO,
        'model_training': logging.INFO,
        'prediction': logging.DEBUG,
        'allocation': logging.INFO,
        'dashboard': logging.WARNING
    }


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """Overall system configuration"""
    
    # System metadata
    project_name: str = "AI Queue-Time Prediction System"
    version: str = "1.0.0"
    environment: str = "production"  # 'development', 'testing', 'production'
    
    # Database settings (for HIS integration)
    enable_database: bool = False
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_name: Optional[str] = None
    
    # API settings
    enable_api: bool = False
    api_port: int = 8000
    api_host: str = 'localhost'
    
    # Security
    require_authentication: bool = False
    session_timeout: int = 3600  # seconds


# ============================================================================
# EXPORT CONFIGURATION INSTANCES
# ============================================================================

# Create singleton instances
paths = Paths()
model_config = ModelConfig()
feature_config = FeatureConfig()
allocation_config = AllocationConfig()
dashboard_config = DashboardConfig()
monitoring_config = MonitoringConfig()
log_config = LogConfig()
system_config = SystemConfig()

# Create directories on import
paths.create_directories()

# Setup logging
logging.basicConfig(
    level=log_config.LOG_LEVEL,
    format=log_config.LOG_FORMAT,
    datefmt=log_config.DATE_FORMAT,
    handlers=[
        logging.FileHandler(paths.SYSTEM_LOG),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded for {system_config.project_name} v{system_config.version}")