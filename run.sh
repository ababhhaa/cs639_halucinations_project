#!/bin/bash

echo "========================================"
echo " Starting INSIDE-HALoGEN Project Setup  "
echo "========================================"

# Step 1: Install requirements silently
echo "[1/2] Installing required Python packages..."
pip install -q datasets pandas matplotlib seaborn numpy

# Step 2: Run Exploratory Data Analysis
echo ""
echo "[2/2] Running Exploratory Data Analysis (EDA)..."
python eda_halogen.py

echo ""
echo "========================================"
echo " Finished! Check the 'eda_results' folder"
echo " and 'demo_results.csv' for the outputs."
echo "========================================"