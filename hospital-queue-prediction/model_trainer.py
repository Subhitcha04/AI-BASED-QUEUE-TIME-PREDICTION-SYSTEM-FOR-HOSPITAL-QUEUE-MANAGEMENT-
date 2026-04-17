"""
Model Training Module for Hospital Queue Prediction
Trains ensemble models and provides prediction capabilities
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

from config import paths, model_config

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Advanced ensemble model trainer for queue prediction
    """
    
    def __init__(self):
        self.model = None
        self.base_models = {}
        self.metrics = {}
        self.feature_importance = None
        
    def build_stacking_model(self):
        """Build stacking ensemble with multiple base models"""
        logger.info("Building stacking ensemble model")
        
        # Base models
        self.base_models = {
            'xgboost': XGBRegressor(**model_config.XGB_PARAMS),
            'lightgbm': LGBMRegressor(**model_config.LGB_PARAMS),
            'catboost': CatBoostRegressor(**model_config.CAT_PARAMS),
            'extratrees': ExtraTreesRegressor(**model_config.ET_PARAMS)
        }
        
        # Meta-learner
        if model_config.META_MODEL == 'ridge':
            meta_learner = Ridge(alpha=1.0)
        elif model_config.META_MODEL == 'lasso':
            meta_learner = Lasso(alpha=1.0)
        else:
            meta_learner = ElasticNet(alpha=1.0)
        
        # Create stacking regressor
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        logger.info(f"Stacking model created with {len(self.base_models)} base models")
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        logger.info("Starting model training")
        
        # Apply log transformation if configured
        if model_config.USE_LOG_TRANSFORM:
            y_train_transformed = np.log1p(y_train)
            if y_val is not None:
                y_val_transformed = np.log1p(y_val)
        else:
            y_train_transformed = y_train
            y_val_transformed = y_val if y_val is not None else None
        
        # Build model if not already built
        if self.model is None:
            self.build_stacking_model()
        
        # Train
        self.model.fit(X_train, y_train_transformed)
        
        logger.info("Model training complete")
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train)
        
        return self
    
    def _calculate_feature_importance(self, X):
        """Calculate feature importance from base models"""
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_dict = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
        
        # Average importance across models
        if importance_dict:
            avg_importance = np.mean(list(importance_dict.values()), axis=0)
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        
        # Inverse transform if log was used
        if model_config.USE_LOG_TRANSFORM:
            predictions = np.expm1(predictions)
        
        return np.maximum(predictions, 0)  # Ensure non-negative
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }
        
        return metrics
    
    def save_model(self):
        """Save model and artifacts"""
        logger.info("Saving model artifacts")
        
        joblib.dump(self.model, paths.MODEL_FILE)
        logger.info(f"Model saved to {paths.MODEL_FILE}")
        
        if self.feature_importance is not None:
            self.feature_importance.to_csv(paths.FEATURE_IMPORTANCE_FILE, index=False)
            logger.info(f"Feature importance saved to {paths.FEATURE_IMPORTANCE_FILE}")
    
    def load_model(self):
        """Load saved model"""
        logger.info(f"Loading model from {paths.MODEL_FILE}")
        self.model = joblib.load(paths.MODEL_FILE)
        return self


if __name__ == "__main__":
    print("Model trainer module - use main.py to train models")