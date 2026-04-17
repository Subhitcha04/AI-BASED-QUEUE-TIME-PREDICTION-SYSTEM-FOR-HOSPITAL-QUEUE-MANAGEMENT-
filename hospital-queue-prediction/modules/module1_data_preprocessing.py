"""
MODULE 1: Data Collection and Preprocessing
============================================
AI-Based Queue-Time Prediction System

This module handles:
- Data extraction from HIS/manual logs
- Data cleaning pipeline
- Feature engineering
- Data validation and quality checks

As specified in project documentation Section 5, Module 1
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from pathlib import Path
import json

from config import paths, feature_config, model_config

# Setup logging
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Handles data extraction from various sources
    """
    
    def __init__(self):
        self.data_sources = []
        logger.info("DataCollector initialized")
    
    def load_from_csv(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        filepath = filepath or paths.RAW_DATA
        
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Convert datetime columns
            datetime_cols = ['arrival_time', 'service_start_time', 'service_end_time']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Sort by arrival time (critical for temporal features)
            df = df.sort_values('arrival_time').reset_index(drop=True)
            
            logger.info(f"[OK] Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_from_database(self, query: str, connection_string: str) -> pd.DataFrame:
        """
        Load data from hospital information system database
        
        Args:
            query: SQL query to extract data
            connection_string: Database connection string
            
        Returns:
            DataFrame with loaded data
        """
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql(query, engine)
            logger.info(f"[OK] Loaded {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Database load error: {str(e)}")
            raise
    
    def validate_data_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that dataframe has required columns
        
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        required_columns = [
            'arrival_time', 'waiting_time', 'queue_length_at_arrival',
            'department', 'system_utilization', 'total_counters_available',
            'doctor_efficiency', 'is_emergency'
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        is_valid = len(missing) == 0
        
        if not is_valid:
            logger.warning(f"Missing columns: {missing}")
        
        return is_valid, missing


class DataCleaner:
    """
    Handles data cleaning operations
    """
    
    def __init__(self):
        self.cleaning_stats = {}
        logger.info("DataCleaner initialized")
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        original_size = len(df)
        df = df.drop_duplicates()
        removed = original_size - len(df)
        
        self.cleaning_stats['duplicates_removed'] = removed
        logger.info(f"Removed {removed} duplicate rows ({removed/original_size*100:.2f}%)")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies
        """
        missing_before = df.isnull().sum().sum()
        
        # For temporal features, forward fill then backward fill
        temporal_cols = ['hour', 'day_of_week', 'month']
        for col in temporal_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # For numeric features, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # For categorical features, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        missing_after = df.isnull().sum().sum()
        self.cleaning_stats['missing_values_filled'] = missing_before - missing_after
        
        logger.info(f"Handled {missing_before - missing_after} missing values")
        return df
    
    def filter_outliers(self, df: pd.DataFrame, column: str, 
                       method: str = 'iqr') -> pd.DataFrame:
        """
        Filter outliers using IQR or quantile method
        
        Args:
            df: DataFrame
            column: Column to filter
            method: 'iqr' or 'quantile'
            
        Returns:
            Filtered DataFrame
        """
        original_size = len(df)
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        elif method == 'quantile':
            lower_limit = df[column].quantile(model_config.lower_quantile)
            upper_limit = df[column].quantile(model_config.upper_quantile)
            df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
        
        removed = original_size - len(df)
        self.cleaning_stats[f'{column}_outliers_removed'] = removed
        
        logger.info(f"Removed {removed} outliers from {column} ({removed/original_size*100:.2f}%)")
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality checks
        
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'negative_waiting_times': (df['waiting_time'] < 0).sum() if 'waiting_time' in df else 0,
            'future_dates': (df['arrival_time'] > pd.Timestamp.now()).sum() if 'arrival_time' in df else 0,
            'invalid_departments': 0,
            'data_quality_score': 0.0
        }
        
        # Calculate quality score (0-100)
        issues = (quality_report['missing_values'] + 
                 quality_report['duplicate_rows'] +
                 quality_report['negative_waiting_times'] +
                 quality_report['future_dates'])
        
        quality_report['data_quality_score'] = max(0, 100 - (issues / len(df) * 100))
        
        logger.info(f"Data quality score: {quality_report['data_quality_score']:.2f}%")
        return quality_report


class FeatureEngineer:
    """
    Handles feature engineering operations
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        logger.info("FeatureEngineer initialized")
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features as per documentation
        """
        logger.info("Creating temporal features...")
        
        if 'arrival_time' not in df.columns:
            logger.error("arrival_time column not found")
            return df
        
        # Basic temporal features
        df['hour'] = df['arrival_time'].dt.hour
        df['day_of_week'] = df['arrival_time'].dt.dayofweek
        df['day_of_month'] = df['arrival_time'].dt.day
        df['month'] = df['arrival_time'].dt.month
        df['week_of_year'] = df['arrival_time'].dt.isocalendar().week
        
        # Binary indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        
        # Time period classification
        df['time_period'] = df['hour'].apply(self._classify_time_period)
        
        logger.info(f"[OK] Created {len(feature_config.TEMPORAL_FEATURES)} temporal features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features (critical for prediction accuracy)
        """
        logger.info("Creating lag features...")
        
        if 'waiting_time' not in df.columns:
            logger.warning("waiting_time column not found, skipping lag features")
            return df
        
        # Previous wait times
        for lag_name, lag_period in feature_config.LAG_FEATURES.items():
            if 'wait' in lag_name:
                if 'rolling' in lag_name:
                    # Rolling average
                    df[lag_name] = (
                        df['waiting_time']
                        .shift(1)
                        .rolling(window=lag_period, min_periods=1)
                        .mean()
                        .bfill()
                    )
                else:
                    # Simple lag
                    df[lag_name] = df['waiting_time'].shift(lag_period).bfill()
            
            elif 'queue' in lag_name:
                # Queue length lags
                df[lag_name] = (
                    df['queue_length_at_arrival']
                    .rolling(window=lag_period, min_periods=1)
                    .mean()
                    .bfill()
                )
        
        logger.info(f"[OK] Created {len(feature_config.LAG_FEATURES)} lag features")
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical encoding for temporal features
        """
        logger.info("Creating cyclical features...")
        
        # Hour (0-23)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (0-6)
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (1-12)
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        logger.info(f"[OK] Created {len(feature_config.CYCLICAL_FEATURES)} cyclical features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and derived features
        """
        logger.info("Creating interaction features...")
        
        # Workload index
        df['workload_index'] = (
            df['queue_length_at_arrival'] * df['system_utilization']
        ) / (df['total_counters_available'] + 1)
        
        # Arrival pressure
        df['arrival_pressure'] = (
            df['arrivals_this_hour'] / (df['total_counters_available'] + 1)
        )
        
        # Efficiency gap
        df['efficiency_gap'] = df['doctor_efficiency'] - df['system_utilization']
        
        # Capacity ratio
        df['capacity_ratio'] = df['queue_length_at_arrival'] / (df['total_counters_available'] + 1)
        
        # Utilization load
        df['utilization_load'] = df['system_utilization'] * df['arrivals_this_hour']
        
        logger.info(f"[OK] Created {len(feature_config.INTERACTION_FEATURES)} interaction features")
        return df
    
    def create_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transformed features for skewed distributions
        """
        logger.info("Creating transformed features...")
        
        # Log transforms
        df['log_queue'] = np.log1p(df['queue_length_at_arrival'])
        df['sqrt_arrivals'] = np.sqrt(df['arrivals_this_hour'])
        
        if 'service_duration_expected' in df.columns:
            df['log_service_time'] = np.log1p(df['service_duration_expected'])
        
        logger.info(f"[OK] Created {len(feature_config.TRANSFORM_FEATURES)} transformed features")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: DataFrame
            fit: Whether to fit new encoders or use existing
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        for col in feature_config.CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            
            if fit:
                le = LabelEncoder()
                df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  Encoded {col}: {len(le.classes_)} classes")
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[f'{col}_enc'] = df[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
        
        return df
    
    @staticmethod
    def _classify_time_period(hour: int) -> str:
        """Classify hour into time period"""
        if 6 <= hour < 9:
            return 'early_morning'
        elif 9 <= hour < 12:
            return 'peak_morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 20:
            return 'evening'
        else:
            return 'night'
    
    def get_feature_list(self) -> List[str]:
        """Get list of all engineered features"""
        return feature_config.get_all_features()


class DataPreprocessor:
    """
    Complete data preprocessing pipeline integrating all components
    """
    
    def __init__(self):
        self.collector = DataCollector()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.preprocessing_stats = {}
        logger.info("="*80)
        logger.info("MODULE 1: Data Collection and Preprocessing")
        logger.info("="*80)
    
    def preprocess(self, filepath: Optional[Path] = None, 
                   fit_encoders: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to data file
            fit_encoders: Whether to fit new encoders
            
        Returns:
            Tuple of (X features, y target)
        """
        logger.info("\n[PHASE 2] Data Preprocessing and Exploratory Analysis")
        logger.info("-"*80)
        
        # Step 1: Load data
        df = self.collector.load_from_csv(filepath)
        self.preprocessing_stats['original_size'] = len(df)
        
        # Step 2: Validate schema
        is_valid, missing = self.collector.validate_data_schema(df)
        if not is_valid:
            raise ValueError(f"Invalid data schema. Missing columns: {missing}")
        
        # Step 3: Clean data
        df = self.cleaner.remove_duplicates(df)
        df = self.cleaner.handle_missing_values(df)
        df = self.cleaner.filter_outliers(df, 'waiting_time', method='quantile')
        
        # Step 4: Quality check
        quality_report = self.cleaner.validate_data_quality(df)
        self.preprocessing_stats['quality_report'] = quality_report
        
        # Step 5: Feature engineering
        df = self.engineer.create_temporal_features(df)
        df = self.engineer.create_lag_features(df)
        df = self.engineer.create_cyclical_features(df)
        df = self.engineer.create_interaction_features(df)
        df = self.engineer.create_transform_features(df)
        df = self.engineer.encode_categorical_features(df, fit=fit_encoders)
        
        # Step 6: Prepare X and y
        feature_list = self.engineer.get_feature_list()
        available_features = [f for f in feature_list if f in df.columns]
        
        X = df[available_features].copy()
        y = df['waiting_time'].copy()
        
        self.preprocessing_stats['final_size'] = len(X)
        self.preprocessing_stats['num_features'] = len(available_features)
        self.preprocessing_stats['removed_records'] = (
            self.preprocessing_stats['original_size'] - 
            self.preprocessing_stats['final_size']
        )
        
        logger.info("\n[OK] Preprocessing Complete")
        logger.info(f"  Original records: {self.preprocessing_stats['original_size']:,}")
        logger.info(f"  Final records: {self.preprocessing_stats['final_size']:,}")
        logger.info(f"  Removed: {self.preprocessing_stats['removed_records']:,} "
                   f"({self.preprocessing_stats['removed_records']/self.preprocessing_stats['original_size']*100:.2f}%)")
        logger.info(f"  Features created: {self.preprocessing_stats['num_features']}")
        logger.info(f"  Data quality score: {quality_report['data_quality_score']:.2f}%")
        
        return X, y
    
    def save_preprocessor(self):
        """Save preprocessing artifacts"""
        import joblib
        
        # Save label encoders
        joblib.dump(self.engineer.label_encoders, paths.LABEL_ENCODERS)
        logger.info(f"[OK] Saved label encoders to {paths.LABEL_ENCODERS}")
        
        # Save feature names
        feature_names = self.engineer.get_feature_list()
        joblib.dump(feature_names, paths.FEATURE_NAMES)
        logger.info(f"[OK] Saved feature names to {paths.FEATURE_NAMES}")
        
        # Save preprocessing stats
        stats_path = paths.OUTPUTS_DIR / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            # Convert non-serializable values
            stats = self.preprocessing_stats.copy()
            if 'quality_report' in stats:
                stats['quality_report'] = {k: float(v) if isinstance(v, (np.integer, np.floating)) 
                                          else v for k, v in stats['quality_report'].items()}
            json.dump(stats, f, indent=2)
        logger.info(f"[OK] Saved preprocessing statistics")
    
    def load_preprocessor(self):
        """Load saved preprocessing artifacts"""
        import joblib
        
        # Load label encoders
        if paths.LABEL_ENCODERS.exists():
            self.engineer.label_encoders = joblib.load(paths.LABEL_ENCODERS)
            logger.info("[OK] Loaded label encoders")
        
        return self


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_and_preprocess_data(filepath: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, DataPreprocessor]:
    """
    Convenience function for complete preprocessing
    
    Returns:
        Tuple of (X, y, preprocessor)
    """
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(filepath)
    preprocessor.save_preprocessor()
    
    return X, y, preprocessor


if __name__ == "__main__":
    # Test Module 1
    logger.info("Testing Module 1: Data Collection and Preprocessing")
    X, y, preprocessor = load_and_preprocess_data()
    
    print(f"\nFeature Matrix Shape: {X.shape}")
    print(f"Target Vector Shape: {y.shape}")
    print(f"\nFirst 5 features:\n{X.columns[:5].tolist()}")
    print(f"\nTarget statistics:\n{y.describe()}")