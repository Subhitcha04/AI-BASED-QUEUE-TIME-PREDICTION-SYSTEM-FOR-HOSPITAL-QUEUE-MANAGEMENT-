#!/bin/bash

# Hospital Queue Prediction System - Setup Script (Unix/Linux/Mac)

echo "=============================================="
echo "Hospital Queue Prediction System - Setup"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install Python dependencies
echo ""
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Create necessary directories
echo ""
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p data models outputs logs
echo -e "${GREEN}✓ Directories created${NC}"

# Check Node.js
echo ""
echo -e "${YELLOW}Checking Node.js installation...${NC}"
if command -v node &> /dev/null; then
    node_version=$(node --version)
    echo "Found Node.js $node_version"
    
    # Install frontend dependencies
    echo ""
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${RED}⚠ Node.js not found. Please install Node.js 18+ to use the dashboard.${NC}"
    echo "Download from: https://nodejs.org/"
fi

# Generate sample data
echo ""
echo -e "${YELLOW}Generating sample training data...${NC}"
python3 -c "from data_processor import generate_sample_data; from config import paths; generate_sample_data(5000, paths.TRAIN_DATA)"
echo -e "${GREEN}✓ Sample data generated${NC}"

# Train initial model
echo ""
echo -e "${YELLOW}Training initial model (this may take a few minutes)...${NC}"
python3 main.py

# Final instructions
echo ""
echo "=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "To start the system:"
echo ""
echo "1. Start the API server:"
echo "   source venv/bin/activate"
echo "   python3 api_server.py"
echo ""
echo "2. In a new terminal, start the dashboard:"
echo "   npm run dev"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "=============================================="