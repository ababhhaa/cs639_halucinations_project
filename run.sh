#!/bin/bash

echo "========================================"
echo " Starting INSIDE-HALoGEN Project Setup  "
echo "========================================"

# Step 1: Install requirements silently
echo "[1/3] Installing required Python packages..."
pip install -q datasets pandas matplotlib seaborn numpy

# Step 2: Run Exploratory Data Analysis
echo ""
echo "[2/3] Running Exploratory Data Analysis (EDA)..."
python eda_halogen.py

echo ""
echo "[3/3] Running the inside implementation  on halogen with 1 prompt..."
python -m pipeline.generate --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset halogen --device cpu --fraction_of_data_to_use 0.001

echo ""
echo "========================================"
echo " Finished! Check the 'eda_results' folder"
echo " and 'data/output/<model_name>' for the outputs."
echo "========================================"
