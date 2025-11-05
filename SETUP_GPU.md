# GPU Setup Instructions

## Current Status
✅ PyTorch is installed (CPU version)  
⚠️ CUDA support not available (CPU-only version installed)

## Why CPU-only?
Python 3.13 has limited support for CUDA-enabled PyTorch packages. The current installation uses PyTorch 2.9.0+cpu.

## Options for GPU Support

### Option 1: Use Python 3.11 or 3.12 (Recommended)
1. Install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/)
2. Create a new virtual environment:
   ```bash
   python -m venv cnn_env_gpu
   .\cnn_env_gpu\Scripts\activate
   ```
3. Install PyTorch with CUDA:
   ```bash
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # OR for CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Continue with CPU (Current Setup)
The project works perfectly fine on CPU, just slower for training. Inference (GUI) is still fast.

**Performance Comparison:**
- **Training on GPU:** ~2-5 minutes per epoch
- **Training on CPU:** ~10-30 minutes per epoch  
- **Inference (GUI):** Fast on both (< 1 second per image)

## Verify GPU Support

Run this command to check if GPU is available:
```bash
python check_gpu.py
```

Expected output with GPU:
```
PyTorch Version: 2.9.0+cu121
CUDA Available: True
CUDA Version: 12.1
GPU Device: NVIDIA GeForce RTX 3060
GPU Count: 1
```

## Running the Project

### With Current CPU Setup
```bash
# Run GUI
python run_gui.py

# Run training (will use CPU)
python main.py --mode train --epochs 50

# Run evaluation
python main.py --mode evaluate
```

### If You Install GPU Version
The same commands will automatically use GPU when available!

## Troubleshooting

### CUDA Not Found
1. Check if NVIDIA GPU drivers are installed
2. Verify CUDA toolkit is installed
3. Ensure PyTorch CUDA version matches your CUDA toolkit

### Import Errors
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Performance Tips

### For Training (Even on CPU)
1. Reduce batch size: `--batch-size 64` instead of 128
2. Reduce epochs: `--epochs 25` for quick testing
3. Use smaller model if needed

### For GUI
- Works great on both CPU and GPU
- Inference is fast regardless of device
- Pre-trained model is already available
