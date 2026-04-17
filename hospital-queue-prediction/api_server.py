"""
FastAPI Server for Hospital Queue Prediction System
Provides REST API endpoints for predictions and allocations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from config import api_config, paths, department_config
from counter_allocation import CounterAllocator, DepartmentState
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Queue Prediction API",
    description="AI-powered queue prediction and counter allocation system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
model = None
scaler = None
feature_names = None
allocator = None


class PredictionInput(BaseModel):
    """Single prediction input"""
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    queue_length_at_arrival: int = Field(..., ge=0)
    total_counters_available: int = Field(..., ge=1)
    arrival_rate: float = Field(..., ge=0)
    system_utilization: float = Field(..., ge=0, le=1)
    doctor_efficiency: Optional[float] = Field(0.75, ge=0, le=1)
    department: str = Field("OPD")


class AllocationRequest(BaseModel):
    """Request for counter allocation"""
    department_predictions: Dict[str, Dict]
    current_allocations: Dict[str, int]
    total_staff: Optional[int] = None


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global model, scaler, feature_names, allocator
    
    try:
        model = joblib.load(paths.MODEL_FILE)
        scaler = joblib.load(paths.SCALER_FILE)
        feature_names = joblib.load(paths.FEATURE_NAMES_FILE)
        allocator = CounterAllocator()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Models will need to be trained first")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hospital Queue Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/allocate", "/metrics"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """Get model metrics"""
    try:
        with open(paths.METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics file not found")


@app.get("/feature-importance")
async def get_feature_importance():
    """Get feature importance"""
    try:
        df = pd.read_csv(paths.FEATURE_IMPORTANCE_FILE)
        return df.to_dict(orient='records')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Feature importance file not found")


@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Single prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dataframe
        df = pd.DataFrame([input_data.dict()])
        
        # Add derived features  
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['workload_index'] = (df['queue_length_at_arrival'] * df['system_utilization']) / df['total_counters_available']
        df['log_queue'] = np.log1p(df['queue_length_at_arrival'])
        
        # Ensure all features are present
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Select and order features
        df = df[feature_names]
        
        # Scale
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_scaled)[0]
        prediction = max(0, prediction)
        
        # Calculate confidence interval
        confidence_lower = max(0, prediction * 0.85)
        confidence_upper = prediction * 1.15
        
        return {
            "predicted_wait_time": round(prediction, 2),
            "confidence_interval": {
                "lower": round(confidence_lower, 2),
                "upper": round(confidence_upper, 2)
            },
            "department": input_data.department,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/allocate")
async def allocate_counters(request: AllocationRequest):
    """Get counter allocation recommendations"""
    if allocator is None:
        raise HTTPException(status_code=503, detail="Allocator not initialized")
    
    try:
        results = allocator.recommend_allocation(
            request.department_predictions,
            request.current_allocations,
            request.total_staff
        )
        
        # Convert to dict
        results_dict = {}
        for dept, result in results.items():
            results_dict[dept] = {
                "current_counters": result.current_counters,
                "recommended_counters": result.recommended_counters,
                "predicted_wait_time": round(result.predicted_wait_time, 2),
                "queue_length": result.predicted_queue_length,
                "utilization": round(result.utilization, 3),
                "alert_level": result.alert_level,
                "reasoning": result.reasoning
            }
        
        summary = allocator.get_allocation_summary(results)
        
        return {
            "recommendations": results_dict,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Allocation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/departments")
async def get_departments():
    """Get list of departments"""
    return {
        "departments": list(department_config.DEPARTMENTS.keys()),
        "details": department_config.DEPARTMENTS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=api_config.HOST,
        port=api_config.PORT,
        reload=api_config.RELOAD,
        log_level=api_config.LOG_LEVEL
    )