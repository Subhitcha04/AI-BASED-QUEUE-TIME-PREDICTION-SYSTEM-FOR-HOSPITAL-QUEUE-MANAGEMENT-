"""
Data Processing Pipeline for Hospital Queue Prediction
Handles data loading, cleaning, feature engineering, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import paths, feature_config, department_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(paths.TRAINING_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Comprehensive data processing pipeline for hospital queue data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load hospital queue data from CSV
        
        Args:
            filepath: Path to CSV file. Uses default if None.
            
        Returns:
            DataFrame with loaded data
        """
        if filepath is None:
            filepath = paths.TRAIN_DATA
            
        logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Convert timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from timestamp or existing time columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features added
        """
        logger.info("Creating temporal features")
        
        # If timestamp column exists, extract features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['day_of_month'] = df['timestamp'].dt.day
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Create cyclical features for better periodicity representation
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_week' in df.columns:
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Peak hours indicator
        if 'hour' in df.columns:
            df['is_peak_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 12)).astype(int)
            df['is_evening'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
        
        logger.info(f"Added {len([c for c in df.columns if c.endswith(('_sin', '_cos', 'is_peak', 'is_weekend', 'is_evening'))])} temporal features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for time-series patterns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features added
        """
        logger.info("Creating lag features")
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Lag features for waiting time
        if 'wait_time_minutes' in df.columns:
            for lag in feature_config.LAG_FEATURES:
                df[f'prev_wait_{lag}'] = df['wait_time_minutes'].shift(lag)
        
        # Lag features for queue length
        if 'queue_length_at_arrival' in df.columns:
            for lag in feature_config.LAG_FEATURES:
                df[f'prev_queue_{lag}'] = df['queue_length_at_arrival'].shift(lag)
        
        # Rolling statistics
        if 'wait_time_minutes' in df.columns:
            for window in feature_config.ROLLING_WINDOWS:
                df[f'rolling_wait_{window}'] = df['wait_time_minutes'].rolling(
                    window=window, min_periods=1
                ).mean()
                df[f'rolling_wait_std_{window}'] = df['wait_time_minutes'].rolling(
                    window=window, min_periods=1
                ).std()
        
        if 'queue_length_at_arrival' in df.columns:
            for window in feature_config.ROLLING_WINDOWS:
                df[f'rolling_queue_{window}'] = df['queue_length_at_arrival'].rolling(
                    window=window, min_periods=1
                ).mean()
        
        # Fill NaN values from lag features with 0 or median
        lag_cols = [c for c in df.columns if c.startswith(('prev_', 'rolling_'))]
        for col in lag_cols:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        logger.info(f"Added {len(lag_cols)} lag and rolling features")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            fit: If True, fit label encoders. If False, use existing encoders.
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical variables")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col == 'timestamp':
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    known_categories = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        if not feature_config.CREATE_INTERACTIONS:
            return df
            
        logger.info("Creating interaction features")
        
        # Queue workload indicators
        if 'queue_length_at_arrival' in df.columns and 'system_utilization' in df.columns:
            df['queue_pressure'] = df['queue_length_at_arrival'] * df['system_utilization']
        
        if 'queue_length_at_arrival' in df.columns and 'total_counters_available' in df.columns:
            df['counters_per_patient'] = df['total_counters_available'] / (df['queue_length_at_arrival'] + 1)
        
        # Workload index
        if all(col in df.columns for col in ['queue_length_at_arrival', 'system_utilization', 'total_counters_available']):
            df['workload_index'] = (df['queue_length_at_arrival'] * df['system_utilization']) / (df['total_counters_available'] + 1)
        
        # Time-based workload
        if 'hour' in df.columns and 'queue_length_at_arrival' in df.columns:
            df['hour_queue_interaction'] = df['hour'] * df['queue_length_at_arrival']
        
        # Service efficiency
        if 'doctor_efficiency' in df.columns and 'system_utilization' in df.columns:
            df['efficiency_gap'] = df['doctor_efficiency'] - df['system_utilization']
        
        # Department-specific features
        if 'department' in df.columns:
            # Create one-hot encoding for departments
            dept_dummies = pd.get_dummies(df['department'], prefix='dept')
            df = pd.concat([df, dept_dummies], axis=1)
        
        logger.info(f"Created {len([c for c in df.columns if c.endswith(('_interaction', '_index', '_gap', '_pressure'))])} interaction features")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, target_col: str = 'wait_time_minutes') -> pd.DataFrame:
        """
        Remove outliers using statistical methods
        
        Args:
            df: Input DataFrame
            target_col: Target column to check for outliers
            
        Returns:
            DataFrame with outliers removed
        """
        if not feature_config.REMOVE_OUTLIERS or target_col not in df.columns:
            return df
            
        logger.info("Removing outliers")
        
        initial_count = len(df)
        
        # Z-score method
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
        df = df[z_scores < feature_config.OUTLIER_STD_THRESHOLD]
        
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} outliers ({removed_count/initial_count*100:.2f}%)")
        
        return df.reset_index(drop=True)
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add domain-specific derived features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Adding derived features")
        
        # Log transformations for skewed features
        if 'queue_length_at_arrival' in df.columns:
            df['log_queue'] = np.log1p(df['queue_length_at_arrival'])
        
        # Arrival rate indicators
        if 'arrival_rate' in df.columns:
            df['high_arrival_rate'] = (df['arrival_rate'] > df['arrival_rate'].quantile(0.75)).astype(int)
        
        # Capacity indicators
        if 'total_counters_available' in df.columns:
            df['low_capacity'] = (df['total_counters_available'] <= df['total_counters_available'].quantile(0.25)).astype(int)
        
        # Utilization categories
        if 'system_utilization' in df.columns:
            df['util_category'] = pd.cut(
                df['system_utilization'],
                bins=[0, 0.5, 0.75, 0.9, 1.0],
                labels=['low', 'medium', 'high', 'critical']
            )
            util_dummies = pd.get_dummies(df['util_category'], prefix='util')
            df = pd.concat([df, util_dummies], axis=1)
            df.drop('util_category', axis=1, inplace=True)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'wait_time_minutes', 
                    test_size: float = 0.15, val_size: float = 0.15) -> Tuple:
        """
        Complete data preparation pipeline
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Starting data preparation pipeline")
        
        # Step 1: Remove missing values in target
        if target_col in df.columns:
            df = df.dropna(subset=[target_col])
            logger.info(f"Dataset size after removing missing targets: {len(df)}")
        
        # Step 2: Feature engineering
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        df = self.create_interaction_features(df)
        df = self.add_derived_features(df)
        
        # Step 3: Encode categorical variables
        df = self.encode_categorical(df, fit=True)
        
        # Step 4: Remove outliers
        df = self.remove_outliers(df, target_col)
        
        # Step 5: Separate features and target
        exclude_cols = [target_col, 'timestamp', 'patient_id'] if 'patient_id' in df.columns else [target_col, 'timestamp']
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Store feature names
        self.feature_names = list(X.columns)
        logger.info(f"Total features: {len(self.feature_names)}")
        
        # Step 6: Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Step 7: Validation split from training data
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Step 8: Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        logger.info("Data preparation complete")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def process_single_input(self, input_data: dict) -> pd.DataFrame:
        """
        Process a single input for prediction
        
        Args:
            input_data: Dictionary of input features
            
        Returns:
            Processed DataFrame ready for prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Apply same transformations (without fitting)
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        df = self.add_derived_features(df)
        df = self.encode_categorical(df, fit=False)
        
        # Ensure all required features are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Select only trained features in correct order
        df = df[self.feature_names]
        
        # Scale
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.feature_names
        )
        
        return df_scaled


def generate_sample_data(n_samples: int = 1000, save_path: str = None) -> pd.DataFrame:
    """
    Generate realistic sample hospital queue data for testing
    
    Args:
        n_samples: Number of samples to generate
        save_path: Path to save CSV. If None, doesn't save.
        
    Returns:
        DataFrame with sample data
    """
    logger.info(f"Generating {n_samples} sample records")
    
    np.random.seed(42)
    
    # Departments
    departments = list(department_config.DEPARTMENTS.keys())
    dept_probs = [0.5, 0.25, 0.15, 0.1]  # OPD, Diagnostics, Pharmacy, Emergency
    
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='5min'),
        'department': np.random.choice(departments, n_samples, p=dept_probs),
        'queue_length_at_arrival': np.random.poisson(8, n_samples),
        'total_counters_available': np.random.randint(2, 8, n_samples),
        'arrival_rate': np.random.uniform(0.5, 3.0, n_samples),
        'system_utilization': np.random.beta(5, 2, n_samples),
        'doctor_efficiency': np.random.normal(0.75, 0.1, n_samples).clip(0.4, 1.0),
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic wait times based on features
    df['wait_time_minutes'] = (
        5 +
        df['queue_length_at_arrival'] * 2.5 +
        (8 - df['total_counters_available']) * 3 +
        df['system_utilization'] * 20 +
        np.random.normal(0, 3, n_samples)
    ).clip(1, 120)
    
    # Add hour-based variations
    df['hour'] = df['timestamp'].dt.hour
    peak_hours_mask = (df['hour'] >= 9) & (df['hour'] <= 12)
    df.loc[peak_hours_mask, 'wait_time_minutes'] *= 1.3
    
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Sample data saved to {save_path}")
    
    return df


if __name__ == "__main__":
    # Test data processor
    processor = DataProcessor()
    
    # Generate sample data if not exists
    if not paths.TRAIN_DATA.exists():
        logger.info("Training data not found. Generating sample data...")
        generate_sample_data(5000, paths.TRAIN_DATA)
    
    # Load and process data
    df = processor.load_data()
    
    print("\nOriginal Data Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.prepare_data(df)
    
    print(f"\nProcessed Data Shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    
    print(f"\nFeature names ({len(processor.feature_names)}):")
    for i, name in enumerate(processor.feature_names, 1):
        print(f"{i}. {name}")