#!/bin/bash

# CNN Image Classification Project Setup Script

echo "🚀 Setting up CNN Image Classification Project..."
echo "=================================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv cnn_env

# Activate virtual environment
echo "🔌 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source cnn_env/Scripts/activate
else
    # Linux/Mac
    source cnn_env/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements_detailed.txt

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data
mkdir -p models
mkdir -p plots  
mkdir -p results
mkdir -p logs
mkdir -p notebooks

# Download CIFAR-10 dataset (will be done automatically on first run)
echo "📊 CIFAR-10 dataset will be downloaded automatically on first run"

# Make scripts executable
chmod +x main.py

echo "✅ Setup completed successfully!"
echo ""
echo "To activate the environment and run the project:"
echo "1. Activate environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source cnn_env/Scripts/activate"
else
    echo "   source cnn_env/bin/activate"  
fi
echo "2. Run training and evaluation:"
echo "   python main.py --mode both --epochs 50"
echo ""
echo "For help with command line options:"
echo "   python main.py --help"
