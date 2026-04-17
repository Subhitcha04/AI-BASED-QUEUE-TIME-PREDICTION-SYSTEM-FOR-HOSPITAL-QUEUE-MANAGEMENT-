"""
MODULE 2: Advanced Deep Learning Prediction Engine
===================================================
AI-Based Queue-Time Prediction System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =========================
# Sklearn
# =========================
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

# =========================
# Gradient Boosting
# =========================
import lightgbm as lgb

try:
    import catboost as cb
except ImportError:
    cb = None

# =========================
# PyTorch (SAFE IMPORT)
# =========================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# =========================
# TabNet
# =========================
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
except ImportError:
    TabNetRegressor = None

from config import paths, model_config

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

logger.info(f"Using device: {device}")


# ============================================================
# DATASET
# ============================================================

if TORCH_AVAILABLE:
    class QueueDataset(Dataset):
        def __init__(self, X, y):
            import numpy as np
            
            # Convert to numpy if pandas
            if hasattr(X, "values"):
                X = X.values
            if hasattr(y, "values"):
                y = y.values

            self.X = torch.FloatTensor(np.array(X))
            self.y = torch.FloatTensor(np.array(y)).view(-1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]



# ============================================================
# DEEP NEURAL NETWORK
# ============================================================

if TORCH_AVAILABLE:
    class DeepQueuePredictor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()

            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.model(x).squeeze()


# ============================================================
# BASE MODEL
# ============================================================

class BaseModel:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = None
        self.metrics = {}

    def evaluate(self, X, y):
        preds = self.predict(X)
        self.metrics = {
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "mae": mean_absolute_error(y, preds),
            "r2": r2_score(y, preds)
        }
        return self.metrics


# ============================================================
# DEEP LEARNING WRAPPER
# ============================================================

class DeepNeuralNetworkModel(BaseModel):
    def __init__(self):
        super().__init__("Deep Neural Network")
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_val=None, y_val=None):

        if not TORCH_AVAILABLE:
            logger.warning("Torch not available")
            return

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        train_loader = DataLoader(
            QueueDataset(X_train, y_train),
            batch_size=128,
            shuffle=True
        )

        val_loader = DataLoader(
            QueueDataset(X_val, y_val),
            batch_size=128
        )

        self.model = DeepQueuePredictor(X_train.shape[1]).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, patience=5)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(50):

            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(self.model(xb), yb).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                break

    def predict(self, X):
        X = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            return self.model(X_tensor).cpu().numpy()


# ============================================================
# LIGHTGBM
# ============================================================

class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__("LightGBM")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        self.model = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 64,
                "verbose": -1
            },
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50)]
        )

    def predict(self, X):
        return self.model.predict(X)


# ============================================================
# CATBOOST
# ============================================================

class CatBoostModel(BaseModel):
    def __init__(self):
        super().__init__("CatBoost")

    def train(self, X_train, y_train, X_val=None, y_val=None):

        if cb is None:
            return

        self.model = cb.CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            task_type="GPU" if TORCH_AVAILABLE and torch.cuda.is_available() else "CPU",
            verbose=False
        )

        self.model.fit(X_train, y_train, eval_set=(X_val, y_val))

    def predict(self, X):
        return self.model.predict(X)


# ============================================================
# TRAINER
# ============================================================

class AdvancedModelTrainer:
    def __init__(self):
        self.models = {}

    def train_and_compare(self, X, y):

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        self.models = {
            "deep_nn": DeepNeuralNetworkModel(),
            "lightgbm": LightGBMModel()
        }

        if cb:
            self.models["catboost"] = CatBoostModel()

        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}")
            model.train(X_train, y_train, X_val, y_val)
            results[name] = model.evaluate(X_test, y_test)

        best = min(results.items(), key=lambda x: x[1]["rmse"])

        logger.info(f"Best Model: {best[0]}")

        return best


# ============================================================
# MAIN FUNCTION
# ============================================================

def train_and_compare_models(X: pd.DataFrame, y: pd.Series):
    trainer = AdvancedModelTrainer()
    return trainer.train_and_compare(X, y)


if __name__ == "__main__":
    logger.info("Advanced DL Engine Ready")
