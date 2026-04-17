@echo off
REM Hospital Queue Prediction System - Setup Script (Windows)

echo ==============================================
echo Hospital Queue Prediction System - Setup
echo ==============================================
echo.

REM Check Python
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating Python virtual environment...
python -m venv venv
echo Virtual environment created

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install Python dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt
echo Python dependencies installed

REM Create directories
echo.
echo Creating project directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist outputs mkdir outputs
if not exist logs mkdir logs
echo Directories created

REM Check Node.js
echo.
echo Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Node.js found
    echo.
    echo Installing frontend dependencies...
    call npm install
    echo Frontend dependencies installed
) else (
    echo WARNING: Node.js not found. Please install Node.js 18+ for the dashboard.
    echo Download from: https://nodejs.org/
)

REM Generate sample data
echo.
echo Generating sample training data...
python -c "from data_processor import generate_sample_data; from config import paths; generate_sample_data(5000, paths.TRAIN_DATA)"
echo Sample data generated

REM Train initial model
echo.
echo Training initial model (this may take a few minutes)...
python main.py

REM Final instructions
echo.
echo ==============================================
echo Setup Complete!
echo ==============================================
echo.
echo To start the system:
echo.
echo 1. Start the API server:
echo    venv\Scripts\activate
echo    python api_server.py
echo.
echo 2. In a new command prompt, start the dashboard:
echo    npm run dev
echo.
echo 3. Open http://localhost:3000 in your browser
echo.
echo ==============================================
pause