#!/usr/bin/env python3
"""
GPU PyTorch Installation Helper
Guides user through installing GPU-enabled PyTorch
"""
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible with CUDA PyTorch"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor in [11, 12]:
        print("✅ Python version compatible with CUDA PyTorch")
        return True
    elif version.major == 3 and version.minor == 13:
        print("⚠️  Python 3.13 detected - CUDA PyTorch support is limited")
        print("   Recommendation: Use Python 3.11 or 3.12 for GPU support")
        return False
    else:
        print("❌ Unsupported Python version")
        return False

def detect_cuda():
    """Try to detect CUDA installation"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected via nvidia-smi")
            print(result.stdout[:500])  # First 500 chars
            return True
        else:
            print("❌ nvidia-smi not found - CUDA may not be installed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - CUDA may not be installed")
        return False

def suggest_installation():
    """Suggest appropriate PyTorch installation command"""
    print("\n" + "="*60)
    print("RECOMMENDED INSTALLATION COMMANDS")
    print("="*60)
    
    print("\n1. For CUDA 12.1 (Latest, recommended for RTX 30xx/40xx):")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n2. For CUDA 11.8 (Older GPUs, more stable):")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n3. To keep CPU version (current):")
    print("   No action needed - already installed")
    
    print("\n" + "="*60)
    print("After installation, run: python check_gpu.py")
    print("="*60)

def main():
    print("="*60)
    print("  GPU PyTorch Installation Helper")
    print("="*60)
    
    # Check Python version
    print("\n📌 Checking Python version...")
    compatible = check_python_version()
    
    if not compatible:
        print("\n⚠️  GPU PyTorch installation may not work with your Python version")
        print("   Consider using Python 3.11 or 3.12 for best compatibility")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Check for NVIDIA GPU
    print("\n📌 Checking for NVIDIA GPU...")
    has_gpu = detect_cuda()
    
    if not has_gpu:
        print("\n⚠️  No NVIDIA GPU detected or CUDA not installed")
        print("   GPU-enabled PyTorch will not provide benefits without a GPU")
        print("\nOptions:")
        print("   1. Install NVIDIA GPU drivers and CUDA toolkit")
        print("   2. Continue with CPU version (current)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Provide installation suggestions
    suggest_installation()
    
    # Ask if user wants automatic installation
    print("\n" + "="*60)
    response = input("\nWould you like to install GPU PyTorch now? (y/n): ")
    if response.lower() != 'y':
        print("Installation skipped. You can manually run the commands above.")
        return
    
    # Ask which CUDA version
    print("\nWhich CUDA version would you like to install?")
    print("1. CUDA 12.1 (recommended for newer GPUs)")
    print("2. CUDA 11.8 (recommended for older GPUs)")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        url = "https://download.pytorch.org/whl/cu121"
        cuda_version = "CUDA 12.1"
    elif choice == '2':
        url = "https://download.pytorch.org/whl/cu118"
        cuda_version = "CUDA 11.8"
    else:
        print("Invalid choice. Exiting...")
        return
    
    print(f"\n🚀 Installing PyTorch with {cuda_version}...")
    print("This may take several minutes...\n")
    
    try:
        # Uninstall existing PyTorch
        print("Uninstalling existing PyTorch...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 
                       'torch', 'torchvision', 'torchaudio'])
        
        # Install GPU version
        print(f"\nInstalling PyTorch with {cuda_version}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', url
        ])
        
        if result.returncode == 0:
            print("\n✅ Installation completed successfully!")
            print("\nVerifying installation...")
            subprocess.run([sys.executable, 'check_gpu.py'])
        else:
            print("\n❌ Installation failed. Please try manual installation.")
            
    except Exception as e:
        print(f"\n❌ Error during installation: {e}")
        print("Please try manual installation using the commands above.")

if __name__ == "__main__":
    main()
