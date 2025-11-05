#!/usr/bin/env python3
"""
CNN Image Classification Project - Main Execution Script
Implementation of VGG-inspired CNN for CIFAR-10 classification

Based on the Computational Intelligence minor specialization project:
"Image Classification using Convolutional Neural Networks (CNNs)"

Authors: Akshat Jain, Amartya Singh, Mrityunjaya Sharma
Institution: Manipal Institute of Technology
"""

import torch
import torch.nn as nn
import os
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from vgg_model import VGGNet, print_model_summary
from data_loader import CIFAR10DataLoader
from trainer import CNNTrainer
from evaluator import ModelEvaluator

def main():
    """Main execution function"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CNN Image Classification on CIFAR-10')
    parser.add_argument('--batch-size', type=int, default=128, 
                       help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='weight decay (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='dropout rate (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed (default: 42)')
    parser.add_argument('--model-path', type=str, default='best_model.pth',
                       help='path to save/load model')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], 
                       default='both', help='execution mode')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("=" * 60)
    print("CNN IMAGE CLASSIFICATION PROJECT")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if use_cuda:
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("=" * 60)

    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Initialize data loader
    print("\n1. Loading and preprocessing CIFAR-10 dataset...")
    data_loader = CIFAR10DataLoader(
        batch_size=args.batch_size,
        validation_split=0.1,
        num_workers=4 if use_cuda else 2
    )

    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    print(f"   ✓ Training samples: {len(train_loader.dataset)}")
    print(f"   ✓ Validation samples: {len(val_loader.dataset)}")
    print(f"   ✓ Test samples: {len(test_loader.dataset)}")
    print(f"   ✓ Batch size: {args.batch_size}")
    print(f"   ✓ Classes: {data_loader.classes}")

    # Visualize sample data
    print("\n   Creating sample visualization...")
    try:
        fig = data_loader.visualize_samples(train_loader)
        fig.savefig('plots/cifar10_samples.png', dpi=300, bbox_inches='tight')
        print("   ✓ Sample images saved to plots/cifar10_samples.png")
    except Exception as e:
        print(f"   ⚠ Could not create sample visualization: {e}")

    # Initialize model
    print("\n2. Initializing VGG-inspired CNN model...")
    model = VGGNet(num_classes=10, dropout=args.dropout)

    # Print model summary
    print_model_summary(model)

    # Training phase
    if args.mode in ['train', 'both']:
        print("\n3. Starting training phase...")

        trainer = CNNTrainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.lr,
            weight_decay=args.weight_decay
        )

        # Train the model
        trainer.train(
            num_epochs=args.epochs,
            early_stopping_patience=10,
            save_path=os.path.join('models', args.model_path)
        )

        # Plot training history
        print("\n   Creating training visualizations...")
        try:
            fig = trainer.plot_training_history()
            fig.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
            print("   ✓ Training history saved to plots/training_history.png")
        except Exception as e:
            print(f"   ⚠ Could not create training plots: {e}")

        # Save training history
        history_path = os.path.join('results', 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(trainer.history, f, indent=2)
        print(f"   ✓ Training history saved to {history_path}")

    # Evaluation phase
    if args.mode in ['evaluate', 'both']:
        print("\n4. Starting evaluation phase...")

        # Load best model if in evaluate-only mode
        if args.mode == 'evaluate':
            model_path = os.path.join('models', args.model_path)
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"   ✓ Loaded model from {model_path}")
            else:
                print(f"   ✗ Model file not found: {model_path}")
                return

        # Initialize evaluator
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            test_loader=test_loader,
            classes=data_loader.classes
        )

        # Evaluate model
        metrics = evaluator.evaluate_model()

        print("\n   EVALUATION RESULTS:")
        print("   " + "=" * 40)
        print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print("   " + "=" * 40)

        # Generate comprehensive report
        report = evaluator.generate_evaluation_report('results/evaluation_report.txt')
        print("   ✓ Detailed report saved to results/evaluation_report.txt")

        # Create evaluation visualizations
        print("\n   Creating evaluation visualizations...")

        try:
            # Confusion matrix
            fig_cm = evaluator.plot_confusion_matrix(normalize=True)
            fig_cm.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("   ✓ Confusion matrix saved to plots/confusion_matrix.png")

            # Per-class performance
            fig_perf, df_perf = evaluator.plot_class_performance()
            fig_perf.savefig('plots/class_performance.png', dpi=300, bbox_inches='tight')
            df_perf.to_csv('results/class_performance.csv', index=False)
            print("   ✓ Class performance plots saved to plots/class_performance.png")
            print("   ✓ Class performance data saved to results/class_performance.csv")

            # Confidence analysis
            fig_conf = evaluator.plot_prediction_confidence()
            fig_conf.savefig('plots/confidence_analysis.png', dpi=300, bbox_inches='tight')
            print("   ✓ Confidence analysis saved to plots/confidence_analysis.png")

            # Misclassification analysis
            misclass_df = evaluator.analyze_misclassifications(num_samples=50)
            if misclass_df is not None:
                misclass_df.to_csv('results/misclassifications.csv', index=False)
                print("   ✓ Misclassification analysis saved to results/misclassifications.csv")

        except Exception as e:
            print(f"   ⚠ Could not create some visualizations: {e}")

        # Save evaluation metrics
        metrics_path = os.path.join('results', 'evaluation_metrics.json')
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        }

        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        print(f"   ✓ Evaluation metrics saved to {metrics_path}")

    # Project summary
    print("\n" + "=" * 60)
    print("PROJECT EXECUTION COMPLETED")
    print("=" * 60)
    print("\nGenerated files:")
    print("📁 models/     - Trained model checkpoints")
    print("📁 plots/      - Training and evaluation visualizations")
    print("📁 results/    - Metrics, reports, and analysis data")
    print("\nFiles created:")

    file_list = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.png', '.json', '.csv', '.txt', '.pth')):
                file_list.append(os.path.join(root, file))

    for file_path in sorted(file_list)[-10:]:  # Show last 10 files
        print(f"   ✓ {file_path}")

    print("\n🎉 CNN Image Classification project completed successfully!")
    print("\nImplementation based on:")
    print("   Paper: 'Image Classification using Convolutional Neural Networks'")
    print("   Authors: Akshat Jain, Amartya Singh, Mrityunjaya Sharma")
    print("   Institution: Manipal Institute of Technology")

if __name__ == '__main__':
    main()
