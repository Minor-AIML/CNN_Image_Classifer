@echo off

echo 🚀 Setting up CNN Image Classification Project...
echo ==================================================

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv cnn_env

REM Activate virtual environment  
echo 🔌 Activating virtual environment...
call cnn_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing requirements...
pip install -r requirements_detailed.txt

REM Create directory structure
echo 📁 Creating directory structure...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "plots" mkdir plots
if not exist "results" mkdir results  
if not exist "logs" mkdir logs
if not exist "notebooks" mkdir notebooks

echo ✅ Setup completed successfully!
echo.
echo To run the project:
echo 1. Activate environment: call cnn_env\Scripts\activate.bat
echo 2. Run training: python main.py --mode both --epochs 50
echo.
echo For help: python main.py --help

pause
