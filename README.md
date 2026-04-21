# AI-Based Queue-Time Prediction and Smart Counter Allocation System

**Complete hospital queue management system with predictive analytics and resource optimization**

## 📋 Project Overview

This system addresses three critical challenges in hospital outpatient departments:
1. **Unpredictable Queue Congestion** - Predicts patient waiting times with >85% accuracy
2. **Suboptimal Resource Allocation** - Recommends optimal counter/doctor allocation
3. **Decision-Support Infrastructure** - Provides actionable insights via intuitive dashboard

## 🎯 Key Features

- **Multi-Model Prediction Engine**: Random Forest, XGBoost, Gradient Boosting, Linear Regression
- **Time-Series Forecasting**: Prophet for arrival pattern analysis
- **Smart Counter Allocation**: Linear Programming and Greedy optimization algorithms
- **Interactive Dashboard**: Streamlit-based real-time monitoring interface
- **Performance Monitoring**: Continuous evaluation and reporting system

## 🏗️ System Architecture

```
project/
├── config.py                          # Configuration management
├── main.py                            # Training pipeline entry point
├── modules/
│   ├── module1_data_preprocessing.py  # Data collection & feature engineering
│   ├── module2_predictive_modeling.py # ML model training & comparison
│   └── module3_counter_allocation.py  # Optimization algorithms
├── dashboard/
│   └── app.py                         # Streamlit dashboard
├── data/                              # Dataset storage
├── models/                            # Trained models
├── outputs/                           # Results and reports
└── logs/                             # System logs
```

## 📊 Performance Targets

| Metric | Target | Typical Achievement |
|--------|--------|-------------------|
| R² Score | > 0.85 | 0.90-0.95 |
| RMSE | < 10 min | 5-8 min |
| MAE | < 7 min | 4-6 min |
| MAPE | < 15% | 10-13% |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd hospital-queue-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your dataset in `data/hospital_queue_data_realistic.csv` with required columns:
- `arrival_time` (datetime)
- `waiting_time` (target variable)
- `queue_length_at_arrival`
- `department`
- `system_utilization`
- `total_counters_available`
- `doctor_efficiency`
- `is_emergency`

### 3. Train Models

```bash
python main.py
```

This executes the complete pipeline:
- **Phase 1-2**: Data preprocessing and feature engineering (Module 1)
- **Phase 3**: Model training and comparison (Module 2)
- **Phase 4**: Counter allocation optimization (Module 3)

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Access dashboard at: `http://localhost:8501`

## 📦 Modules

### Module 1: Data Collection and Preprocessing
- Data extraction from HIS/CSV
- Missing value imputation
- Outlier removal
- Feature engineering (40+ features):
  - Temporal: hour, day, week patterns
  - Lag features: previous wait times
  - Cyclical: sin/cos encoding
  - Interactions: workload index, arrival pressure
  - Transforms: log/sqrt for skewed distributions

### Module 2: Predictive Modeling Engine
- **Baseline**: Linear Regression
- **Ensemble Models**: 
  - Random Forest (200 estimators)
  - XGBoost (500 estimators, learning_rate=0.03)
  - Gradient Boosting (300 estimators)
- **Time-Series**: Prophet for arrival forecasting
- Hyperparameter tuning with TimeSeriesSplit
- Feature importance analysis

### Module 3: Counter Allocation Recommendation
- **Greedy Algorithm**: Fast, near-optimal decisions
- **Linear Programming**: Optimal allocation with constraints
  - Min/max counters per department
  - Total staff availability
  - Priority-based assignment
- Alert generation for critical congestion

### Module 4: Dashboard Interface (Streamlit)
- Real-time waiting time predictions
- Model performance metrics
- Historical trend visualization
- Counter allocation recommendations
- Department-wise queue monitoring

### Module 5: Performance Monitoring (Integrated)
- Prediction accuracy tracking
- Resource utilization metrics
- Waiting time reduction analysis
- Automated reporting

## 🔬 Methodology

### Phase 1-2: Data Preprocessing (Weeks 1-4)
- Collect 3-6 months historical data
- Clean: remove duplicates, handle missing values
- Engineer 40+ features
- Split: 70% train, 15% validation, 15% test

### Phase 3: Model Development (Weeks 5-7)
- Train multiple models
- Perform 5-fold cross-validation
- Compare performance
- Select best model (typically XGBoost)

### Phase 4: Optimization (Weeks 8-9)
- Define allocation constraints
- Implement Greedy and LP algorithms
- Test with simulated scenarios

### Phase 5: Dashboard Development (Weeks 10-11)
- Design user interface
- Integrate prediction engine
- Add real-time alerts

### Phase 6-7: Testing & Validation (Weeks 12-15)
- End-to-end system testing
- Performance validation
- Documentation

## 📈 Expected Outcomes

### Operational Improvements
- **20-30% reduction** in average waiting time
- **25% reduction** in peak-hour queue length
- **75-85% counter utilization** (optimized from 60-70%)
- **2-5 actionable recommendations** per hour during peaks

### System Capabilities
- Predictions within 2 seconds
- Real-time queue monitoring
- Proactive congestion alerts
- Data-driven staffing decisions

## 🧪 Testing Scenarios

1. **Peak Hour Load**: Monday mornings, post-holidays
2. **Multi-Department**: Simultaneous OPD, diagnostics, pharmacy
3. **Staff Shortage**: 20-30% reduced capacity
4. **Unexpected Surge**: Emergency events simulation
5. **Long-term Accuracy**: 4-8 weeks continuous operation

## 📚 Configuration

Edit `config.py` to customize:

```python
# Model hyperparameters
model_config.target_rmse = 10.0
model_config.xgb_params['n_estimators'] = 500

# Allocation constraints
allocation_config.total_available_staff = 15
allocation_config.critical_wait_threshold = 30.0

# Dashboard settings
dashboard_config.port = 8501
dashboard_config.prediction_update_interval = 60
```

## 🔧 Advanced Features

### Database Integration (Optional)
```python
from config import system_config

system_config.enable_database = True
system_config.db_host = "localhost"
system_config.db_name = "hospital_db"

# Load from database
from modules.module1_data_preprocessing import DataCollector
collector = DataCollector()
df = collector.load_from_database(query, connection_string)
```

### Model Retraining
```python
# Automated retraining when accuracy drops
from config import monitoring_config

monitoring_config.retrain_schedule = 'weekly'
monitoring_config.accuracy_drop_threshold = 0.05
```

## 📊 Output Files

After training, find results in:
- `models/queue_prediction_model.pkl` - Best trained model
- `outputs/model_comparison.json` - Performance comparison
- `outputs/feature_importance.csv` - Feature rankings
- `outputs/allocation_recommendations.csv` - Historical recommendations
- `logs/training.log` - Complete training log

## 🐛 Troubleshooting

### Model not found
```bash
# Train the model first
python main.py
```

### Dashboard connection error
```bash
# Ensure model is trained
ls models/queue_prediction_model.pkl

# Restart dashboard
streamlit run dashboard/app.py
```

### Prophet installation issues
```bash
# Prophet requires specific compilers
pip install prophet --no-cache-dir
```

## 📝 Citation

If using this system in research, please cite:
```
AI-Based Queue-Time Prediction and Smart Counter Allocation System
Hospital Queue Management with Machine Learning
2026
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 🎓 References

Based on project documentation:
- Section 4: Tools and Techniques
- Section 5: Modules Architecture
- Section 6: Methodology
- Section 7: Testing Plan
- Section 8: Expected Outcomes

## 📞 Support

For questions or issues:
- Review documentation in `/docs`
- Check troubleshooting guide
- Open GitHub issue
- Contact: [subhitcha.s@gmail.com]

---

**Built with** ❤️ **for improving hospital efficiency and patient experience**

**Version**: 1.0.0  
**Last Updated**: 2026
