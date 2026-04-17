# Project Structure Documentation

## Overview

This document explains the architecture and organization of the Hospital Queue Prediction System.

## Directory Structure

```
hospital-queue-prediction/
│
├── 📄 Core Python Files
│   ├── config.py                 # Configuration management
│   ├── data_processor.py         # Data processing pipeline
│   ├── model_trainer.py          # Model training and evaluation
│   ├── main.py                   # Training pipeline entry point
│   ├── api_server.py            # FastAPI REST API server
│   └── predict.py               # Standalone prediction utility
│
├── 🎨 Frontend Files
│   ├── dashboard.jsx            # Main React dashboard component
│   ├── main.jsx                 # React entry point
│   ├── index.html               # HTML template
│   ├── vite.config.js           # Vite build configuration
│   └── package.json             # Node.js dependencies
│
├── 📦 Configuration Files
│   ├── requirements.txt         # Python dependencies
│   ├── .gitignore              # Git ignore patterns
│   ├── setup.sh                # Unix setup script
│   └── setup.bat               # Windows setup script
│
├── 📚 Documentation
│   ├── README.md               # Main documentation
│   └── PROJECT_STRUCTURE.md    # This file
│
└── 📁 Data Directories
    ├── data/                   # Dataset storage
    │   └── hospital_queue_data_realistic.csv
    ├── models/                 # Trained models
    │   ├── queue_prediction_model.pkl
    │   ├── scaler.pkl
    │   └── feature_names.pkl
    ├── outputs/                # Model outputs
    │   ├── model_metrics.json
    │   └── feature_importance.csv
    └── logs/                   # Training logs
        └── training.log
```

## Component Descriptions

### 1. Configuration Layer (`config.py`)

**Purpose**: Centralized configuration management

**Key Classes**:
- `Paths`: File and directory path management
- `ModelConfig`: ML model hyperparameters
- `FeatureConfig`: Feature engineering settings
- `APIConfig`: API server configuration

**Benefits**:
- Single source of truth for all settings
- Easy to modify parameters without code changes
- Type-safe configuration with dataclasses

### 2. Data Processing (`data_processor.py`)

**Purpose**: Handle all data-related operations

**Key Class**: `DataProcessor`

**Responsibilities**:
1. Load data from CSV files
2. Create temporal lag features
3. Encode categorical variables
4. Generate cyclical time features
5. Create interaction features
6. Remove outliers

**Pipeline Flow**:
```
Raw CSV → Load Data → Feature Engineering → Train/Test Split → Model Ready Data
```

### 3. Model Training (`model_trainer.py`)

**Purpose**: Train and evaluate ML models

**Key Class**: `ModelTrainer`

**Features**:
- Stacking ensemble with 4 base models:
  - XGBoost
  - LightGBM
  - CatBoost
  - ExtraTrees
- Target transformation (log/expm1)
- Cross-validation
- Feature importance calculation
- Model persistence

**Training Flow**:
```
Data → Base Models → Stacking → Target Transform → Trained Model
```

### 4. Main Pipeline (`main.py`)

**Purpose**: Orchestrate the complete training workflow

**Workflow**:
1. Load and process data
2. Engineer features
3. Train model
4. Evaluate performance
5. Save artifacts
6. Log results

### 5. API Server (`api_server.py`)

**Purpose**: Serve predictions via REST API

**Technology**: FastAPI

**Endpoints**:
- `GET /`: Root endpoint
- `GET /health`: Health check
- `GET /metrics`: Model metrics
- `GET /feature-importance`: Feature rankings
- `POST /predict`: Single prediction
- `POST /batch-predict`: Batch predictions

**Features**:
- CORS support for frontend
- Input validation with Pydantic
- Automatic API documentation
- Error handling

### 6. Prediction Utility (`predict.py`)

**Purpose**: Command-line prediction interface

**Use Cases**:
- Quick predictions without API
- Testing model performance
- Integration with other systems

### 7. React Dashboard (`dashboard.jsx`)

**Purpose**: Professional web interface for predictions

**Features**:
- Real-time predictions
- Model metrics visualization
- Feature importance charts
- Confidence intervals
- Smart recommendations
- Responsive design

**Technology Stack**:
- React 18
- Recharts for visualizations
- Lucide React for icons
- Vite for build tooling

## Data Flow

### Training Pipeline

```
CSV File
  ↓
DataProcessor
  ↓
Feature Engineering
  ↓
Train/Test Split
  ↓
ModelTrainer
  ↓
Stacking Ensemble
  ↓
Evaluation
  ↓
Save Model & Metrics
```

### Prediction Pipeline

```
User Input (Dashboard/API)
  ↓
FastAPI Server
  ↓
Input Validation
  ↓
Feature Engineering
  ↓
Loaded Model
  ↓
Prediction
  ↓
Response (JSON)
  ↓
Dashboard Display
```

## Key Design Decisions

### 1. Separation of Concerns

Each module has a single, well-defined responsibility:
- `config.py`: Configuration only
- `data_processor.py`: Data operations only
- `model_trainer.py`: Model training only
- `api_server.py`: API serving only

### 2. Type Safety

- Type hints throughout codebase
- Pydantic models for API validation
- Dataclasses for configuration

### 3. Logging

- Comprehensive logging at each step
- Both console and file output
- Debug, info, warning, and error levels

### 4. Error Handling

- Try-catch blocks for critical operations
- Meaningful error messages
- Graceful degradation

### 5. Modularity

- Each component can be imported independently
- Easy to test individual modules
- Simple to extend functionality

## Feature Engineering Pipeline

### Base Features (from data)
- `hour`, `day_of_week`, `month`
- `queue_length_at_arrival`
- `system_utilization`
- `total_counters_available`

### Temporal Features (lag-based)
- `prev_wait_1`: Previous patient's wait time
- `rolling_wait_5`: 5-patient rolling average
- `rolling_queue_3`: 3-patient queue average

### Cyclical Features
- `hour_sin`, `hour_cos`: Circular time encoding

### Interaction Features
- `workload_index`: queue × utilization / counters
- `arrival_pressure`: arrivals / counters
- `efficiency_gap`: doctor efficiency - utilization

### Transformed Features
- `log_queue`: Log-transformed queue length

## Model Architecture

```
Input Features (20+)
       ↓
┌──────────────────────────┐
│   Base Models (Parallel) │
├──────────────────────────┤
│  • XGBoost              │
│  • LightGBM             │
│  • CatBoost             │
│  • ExtraTrees           │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│   Meta-Learner          │
│   (Ridge Regression)    │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│  Target Transform       │
│  (Inverse Log)          │
└──────────────────────────┘
       ↓
  Final Prediction
```

## API Architecture

```
React Frontend (Port 3000)
       ↓ HTTP Request
FastAPI Backend (Port 8000)
       ↓
┌──────────────────┐
│  Input Validation│
│   (Pydantic)     │
└──────────────────┘
       ↓
┌──────────────────┐
│ Feature Engineer │
└──────────────────┘
       ↓
┌──────────────────┐
│  Loaded Model    │
│   (joblib)       │
└──────────────────┘
       ↓
┌──────────────────┐
│   Prediction     │
└──────────────────┘
       ↓ HTTP Response
React Frontend (Display)
```

## Performance Optimization

### Data Processing
- Vectorized operations with NumPy/Pandas
- Efficient rolling window calculations
- Minimal data copying

### Model Training
- Parallel processing (`n_jobs=-1`)
- Early stopping in gradient boosting
- Optimized hyperparameters

### API Server
- Async endpoints (FastAPI)
- Model loaded once at startup
- Efficient JSON serialization

### Frontend
- Lazy loading of charts
- Debounced API calls
- Optimized re-renders

## Testing Strategy

### Unit Tests
- Test individual functions
- Mock external dependencies
- Validate data transformations

### Integration Tests
- Test pipeline end-to-end
- Validate API endpoints
- Check model predictions

### Performance Tests
- Measure training time
- Test prediction latency
- Monitor memory usage

## Deployment Considerations

### Development
```bash
# Python backend
python api_server.py

# React frontend
npm run dev
```

### Production
```bash
# Python backend with Gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker

# React frontend
npm run build
serve -s dist
```

### Docker
- Separate containers for API and frontend
- Docker Compose for orchestration
- Volume mounts for models

## Scalability

### Horizontal Scaling
- API server: Multiple instances behind load balancer
- Frontend: CDN distribution
- Model: Shared storage or model registry

### Vertical Scaling
- Increase API server resources
- Use GPU for model inference
- Optimize model size

## Security

### API
- CORS configured for allowed origins
- Input validation with Pydantic
- Rate limiting (future enhancement)

### Data
- No sensitive data stored
- Model versioning
- Audit logging

## Future Enhancements

1. **Real-time Updates**: WebSocket for live predictions
2. **Model Versioning**: A/B testing different models
3. **Monitoring**: Prometheus + Grafana
4. **Authentication**: JWT-based auth
5. **Caching**: Redis for frequent predictions
6. **Database**: Store predictions and feedback
7. **CI/CD**: Automated testing and deployment

## Maintenance

### Regular Tasks
- Monitor model performance
- Retrain with new data
- Update dependencies
- Review logs
- Backup models

### Version Control
- Semantic versioning
- Tag releases
- Maintain changelog

---

**Last Updated**: 2024
**Version**: 1.0.0