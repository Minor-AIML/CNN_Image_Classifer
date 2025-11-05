#!/usr/bin/env python3
"""
Script to run the GUI with GPU support
Checks system configuration and launches GUI
"""
import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    required = ['torch', 'torchvision', 'PIL', 'tkinter']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:", ', '.join(missing))
        print("\n📦 Please install dependencies:")
        print("   pip install torch torchvision pillow")
        print("   Note: tkinter usually comes with Python")
        return False
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        print("\n🔍 GPU Configuration Check:")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print("   ✅ GPU will be used for inference")
        else:
            print("   ⚠️  CUDA not available - using CPU")
            print("   Tip: Install CUDA-enabled PyTorch for GPU support:")
            print("   Visit: https://pytorch.org/get-started/locally/")
        print()
    except Exception as e:
        print(f"   Error checking GPU: {e}")

def check_model():
    """Check if model file exists"""
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("\n📋 To train the model first, run:")
        print("   python main.py --mode train --epochs 50")
        return False
    print(f"✅ Model file found: {model_path}")
    return True

def run_gui():
    """Launch the GUI application"""
    print("\n🚀 Launching Image Classification GUI...")
    print("   The GUI window should open shortly.")
    print("   You can upload any image to classify it.\n")
    
    # Import and run GUI
    import gui
    # The GUI will run via gui.py's root.mainloop()

if __name__ == "__main__":
    print("=" * 60)
    print("   CNN Image Classification - GUI Launcher")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Check model
    if not check_model():
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run GUI
    try:
        run_gui()
    except Exception as e:
        print(f"\n❌ Error running GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
