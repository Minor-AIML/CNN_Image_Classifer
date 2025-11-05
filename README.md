# CNN Image Classification Project

VGG-inspired Convolutional Neural Network for CIFAR-10 image classification with GUI interface.

**Authors:** Akshat Jain, Amartya Singh, Mrityunjaya Sharma  
**Institution:** Manipal Institute of Technology

## 🚀 Quick Start

### Run the GUI (Easiest Way)
```bash
python run_gui.py
```

The GUI will automatically:
- ✅ Check all dependencies
- ✅ Verify GPU/CPU configuration
- ✅ Load the trained model
- ✅ Open the classification interface

### Upload and Classify Images
1. Click "Upload Image" button
2. Select any image file (JPG, PNG, etc.)
3. Get instant classification results!

## 📋 Features

### Model Architecture
- **VGG-inspired CNN** with 5 convolutional blocks
- Batch normalization and dropout for regularization
- Adaptive pooling for flexible input sizes
- ~15M trainable parameters

### Training Features
- Data augmentation (flips, rotations, color jitter)
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Comprehensive training visualization

### Evaluation & Analysis
- Confusion matrix
- Per-class performance metrics
- Confidence analysis
- Misclassification analysis
- Multiple evaluation visualizations

### Interfaces
1. **GUI Application** (Tkinter) - User-friendly image classification
2. **Web API** (Flask) - REST API for predictions
3. **Command-line** - Full training and evaluation pipeline

## 🖥️ System Requirements

### Minimum
- Python 3.11+ (3.13 supported but CPU-only PyTorch)
- 4GB RAM
- 2GB disk space

### Recommended
- Python 3.11 or 3.12
- 8GB+ RAM
- NVIDIA GPU with CUDA support
- 10GB disk space

### Current Installation Status
- ✅ PyTorch 2.9.0 installed
-  CPU+GPU version (see SETUP_GPU.md for GPU setup)
- ✅ All dependencies installed
- ✅ Pre-trained model available

## 📦 Installation

### Dependencies Already Installed
```
✅ torch 2.9.0
✅ torchvision 0.24.0
✅ numpy, pandas, matplotlib
✅ scikit-learn, seaborn
✅ tqdm, pyyaml, flask
✅ Pillow (for image processing)
```

### If You Need to Reinstall
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### 1. GUI Application (Recommended for beginners)
```bash
python run_gui.py
```

### 2. Training a New Model
```bash
# Full training (50 epochs)
python main.py --mode train --epochs 50

# Quick training (25 epochs)
python main.py --mode train --epochs 25 --batch-size 64

# With specific settings
python main.py --mode train --epochs 50 --lr 0.001 --dropout 0.5
```

### 3. Evaluate Existing Model
```bash
python main.py --mode evaluate
```

### 4. Train and Evaluate
```bash
python main.py --mode both --epochs 50
```

### 5. Web API Server
```bash
python server.py
```

Then use curl or any HTTP client:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

## 📊 Model Performance

Based on pre-trained model in `models/best_model.pth`:

- **Test Accuracy:** ~85-90%
- **Training Time:** 2-5 min/epoch (GPU) or 10-30 min/epoch (CPU)
- **Inference Time:** <1 second per image

### CIFAR-10 Classes
1. Airplane ✈️
2. Automobile 🚗
3. Bird 🐦
4. Cat 🐱
5. Deer 🦌
6. Dog 🐕
7. Frog 🐸
8. Horse 🐴
9. Ship 🚢
10. Truck 🚚

## 📁 Project Structure

```
minor/
├── main.py              # Main training/evaluation script
├── run_gui.py          # GUI launcher with checks
├── gui.py              # Tkinter GUI application
├── server.py           # Flask web server
├── vgg_model.py        # VGG model architecture
├── trainer.py          # Training logic
├── evaluator.py        # Evaluation and metrics
├── data_loader.py      # Data loading and preprocessing
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
├── SETUP_GPU.md       # GPU setup instructions
│
├── models/             # Saved model checkpoints
│   └── best_model.pth
├── data/               # CIFAR-10 dataset
│   └── cifar-10-batches-py/
├── plots/              # Training and evaluation plots
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── ...
└── results/            # Metrics and analysis
    ├── evaluation_metrics.json
    ├── class_performance.csv
    └── ...
```

## 🎨 GUI Features

- **Upload Images:** Support for JPG, PNG, and other formats
- **Live Classification:** Instant prediction results
- **Visual Display:** See your uploaded image
- **Easy to Use:** Simple, intuitive interface

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation settings
- Paths and directories

## 📈 Training Output

The training process generates:
1. **Model checkpoints** in `models/`
2. **Training curves** in `plots/training_history.png`
3. **Training logs** in `results/training_history.json`

## 📊 Evaluation Output

Evaluation generates:
1. **Confusion matrix** - `plots/confusion_matrix.png`
2. **Per-class metrics** - `plots/class_performance.png`
3. **Confidence analysis** - `plots/confidence_analysis.png`
4. **Detailed report** - `results/evaluation_report.txt`
5. **Metrics JSON** - `results/evaluation_metrics.json`

## 🐛 Troubleshooting

### GUI won't start
```bash
# Check dependencies
python run_gui.py
```

### Model not found
```bash
# Train a new model
python main.py --mode train --epochs 50
```

### CUDA errors
See `SETUP_GPU.md` for GPU setup instructions

### Out of memory (OOM)
```bash
# Reduce batch size
python main.py --mode train --batch-size 64
```

### Import errors
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Performance Tips

### For Faster Training
1. Use GPU (see SETUP_GPU.md)
2. Increase batch size: `--batch-size 256` (if enough memory)
3. Use fewer workers: `num_workers=2` in config

### For Better Accuracy
1. Train longer: `--epochs 100`
2. Try different learning rates: `--lr 0.0001`
3. Adjust dropout: `--dropout 0.3`
4. Experiment with data augmentation in config.yaml

### For Quick Testing
```bash
# Fast test with smaller settings
python main.py --mode train --epochs 5 --batch-size 64
```

## 📚 Documentation

- `main.py` - Comprehensive docstrings for all functions
- `config.yaml` - Detailed configuration comments
- `SETUP_GPU.md` - GPU setup guide

## 🤝 Contributing

This is an academic project. For issues or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 Citation

If you use this code, please cite:
```
Image Classification using Convolutional Neural Networks (CNNs)
Authors: Akshat Jain, Amartya Singh, Mrityunjaya Sharma
Institution: Manipal Institute of Technology
```

## 📄 License

Academic project for educational purposes.

## 🆘 Support

For questions or issues:
1. Check the troubleshooting section
2. Read SETUP_GPU.md for GPU issues
3. Review the configuration in config.yaml
4. Check the output logs for error messages

## ✨ Acknowledgments

- VGG architecture inspired by the VGGNet paper
- CIFAR-10 dataset from University of Toronto
- PyTorch framework for deep learning
- Manipal Institute of Technology for academic support

---

**Status:** ✅ Ready to use!  
**Last Updated:** October 2025
