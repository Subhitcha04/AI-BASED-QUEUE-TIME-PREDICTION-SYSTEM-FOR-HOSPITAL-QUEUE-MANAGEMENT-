# Quick Start Guide - Hospital Queue Prediction System

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (1 minute)

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

This will:
- Create Python virtual environment
- Install all Python packages
- Install Node.js packages
- Generate sample training data
- Train initial ML model

### Step 2: Start the Backend (30 seconds)

**Linux/Mac:**
```bash
source venv/bin/activate
python api_server.py
```

**Windows:**
```cmd
venv\Scripts\activate
python api_server.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 3: Start the Frontend (30 seconds)

Open a **new terminal** and run:

```bash
npm run dev
```

You should see:
```
VITE ready in XXX ms

➜  Local:   http://localhost:3000/
```

### Step 4: Open Dashboard

Open your browser and go to: **http://localhost:3000**

You should see the Hospital Queue Intelligence dashboard!

## 🎯 Making Your First Prediction

1. **Select Department**: Choose from OPD, Diagnostics, Pharmacy, or Emergency
2. **Enter Current Time**: Set the hour (0-23)
3. **Set Queue Parameters**:
   - Queue Length: Number of patients waiting
   - Active Counters: How many service counters are open
   - Arrival Rate: Patients arriving per hour
   - System Utilization: 0-1 (how busy the system is)
4. **Click "Predict Wait Time"**

The system will show:
- Predicted wait time in minutes
- Confidence interval
- Automated recommendations

## 📊 Getting Counter Allocation Recommendations

1. After making predictions for departments, click **"Calculate Optimal Allocation"**
2. The system will show:
   - Current vs. recommended counter allocation for each department
   - Alert levels (normal/warning/critical)
   - Reasoning for each recommendation
   - Summary statistics

## 🔧 Troubleshooting

### Backend won't start
- **Error**: "Address already in use"
  - **Solution**: Port 8000 is busy. Change port in `config.py`:
    ```python
    api_config.PORT = 8001
    ```

### Frontend won't start
- **Error**: "EADDRINUSE"
  - **Solution**: Port 3000 is busy. Change in `vite.config.js`:
    ```javascript
    server: { port: 3001 }
    ```

### "Model not found" error
- **Solution**: Train the model first:
  ```bash
  python main.py
  ```

### Dashboard shows "Error making prediction"
- **Check**: Is the API server running?
- **Check**: Browser console for error messages
- **Fix**: Ensure API server is on http://localhost:8000

## 📝 Using the Command Line Tool

For quick predictions without the dashboard:

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Make a prediction
python predict.py --hour 10 --queue-length 12

# Full example
python predict.py \
  --hour 14 \
  --day-of-week 1 \
  --queue-length 15 \
  --counters 4 \
  --arrival-rate 2.5 \
  --utilization 0.85 \
  --verbose
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the API at http://localhost:8000/docs
- Customize configuration in `config.py`
- Retrain with your own data

## 🆘 Need Help?

- Check `logs/training.log` for training issues
- Check `logs/api.log` for API errors
- Review browser console for frontend issues
- Open an issue on GitHub

---

**Happy Predicting!** 🏥